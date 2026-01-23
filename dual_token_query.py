import torch
import torch.nn as nn
import torch.nn.functional as F


class _DualTokenBlock(nn.Module):
    """Q-Former-like: (LN -> self-attn on queries) + (LN -> cross-attn queries->vision) + (LN -> FFN)"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        topk_ratio: float = 0.25,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.topk_ratio = topk_ratio

        self.ln_q1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.ln_q2 = nn.LayerNorm(hidden_size)
        # global cross-attn for token-A
        self.cross_attn_global = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        # salient cross-attn for token-B (attend only top-k patches)
        self.cross_attn_salient = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.ln_q3 = nn.LayerNorm(hidden_size)
        inner = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
            nn.Dropout(dropout),
        )

    @torch.no_grad()
    def _topk_patches(self, kv: torch.Tensor) -> torch.Tensor:
        """
        kv: (1, N, D). Use L2-norm as saliency proxy and pick top-k tokens.
        """
        n = kv.shape[1]
        k = max(1, int(n * self.topk_ratio))
        score = kv.norm(dim=-1)  # (1, N)
        topk_idx = score.topk(k, dim=1, largest=True, sorted=False).indices  # (1, k)
        # gather
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, kv.shape[-1])  # (1, k, D)
        return torch.gather(kv, dim=1, index=idx)



    def _select_regions_from_residual(
        self,
        residual_2d: torch.Tensor,   # (h, w)
        num_regions: int,
        region_size: int,
        suppress: bool = True,
    ):
        """
        Greedy pick top residual centers, expand to squares, optionally suppress overlap.
        Return: list[Tensor] each is 1D flat indices in [0, h*w).
        """
        assert residual_2d.dim() == 2
        h, w = residual_2d.shape
        r = region_size // 2

        # work on a copy
        score = residual_2d.clone()
        regions = []

        # use -inf to suppress
        neg_inf = torch.finfo(score.dtype).min

        for _ in range(num_regions):
            flat = score.view(-1)
            maxv, arg = flat.max(dim=0)
            if torch.isinf(maxv) or (maxv <= neg_inf / 2):
                break  # no valid region left

            cy = (arg // w).item()
            cx = (arg % w).item()

            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)

            ys = torch.arange(y0, y1, device=score.device)
            xs = torch.arange(x0, x1, device=score.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            idx = (yy * w + xx).reshape(-1)  # flat indices

            regions.append(idx)

            if suppress:
                score[y0:y1, x0:x1] = neg_inf

        return regions


    def _build_salient_kv_and_max_init(
        self,
        frame_tokens: torch.Tensor,        # (N, D)
        regions: list[torch.Tensor],       # list of flat indices
    ):
        """
        For each region: max-pool tokens inside -> 1 region token
        KV for salient token: (1, R, D)
        Max-init token: max over union(region tokens) OR union(patch tokens)
        """
        device = frame_tokens.device
        dtype = frame_tokens.dtype

        if len(regions) == 0:
            # fallback: use all tokens
            kv_salient = frame_tokens.unsqueeze(0)           # (1, N, D)
            max_init = frame_tokens.max(dim=0).values        # (D,)
            return kv_salient, max_init

        region_tokens = []
        union = []

        for idx in regions:
            idx = idx.to(device=device)
            # guard
            idx = idx.clamp_(0, frame_tokens.shape[0] - 1)
            tok = frame_tokens.index_select(0, idx)          # (m, D)
            region_tok = tok.max(dim=0).values               # (D,)
            region_tokens.append(region_tok)
            union.append(idx)

        region_tokens = torch.stack(region_tokens, dim=0).to(dtype=dtype)  # (R, D)
        kv_salient = region_tokens.unsqueeze(0)                              # (1, R, D)

        # max init over union patches (更贴近“在正方块上max pooling”)
        union_idx = torch.unique(torch.cat(union, dim=0))
        union_tok = frame_tokens.index_select(0, union_idx)
        max_init = union_tok.max(dim=0).values

        return kv_salient, max_init

    
    
    def forward(self, q, kv_global, kv_salient):
        # 1) self-attn among queries
        q_sa, _ = self.self_attn(self.ln_q1(q), self.ln_q1(q), self.ln_q1(q), need_weights=False)
        q = q + q_sa

        # 2) role-differentiated cross-attn
        q_ln = self.ln_q2(q)
        qA = q_ln[:, :1, :]   # global token
        qB = q_ln[:, 1:, :]   # salient token

        qA_ca, _ = self.cross_attn_global(qA, kv_global, kv_global, need_weights=False)
        qB_ca, _ = self.cross_attn_salient(qB, kv_salient, kv_salient, need_weights=False)

        q = q + torch.cat([qA_ca, qB_ca], dim=1)

        # 3) FFN
        q = q + self.ffn(self.ln_q3(q))
        return q


class DualTokenQFormer(nn.Module):
    """
    Produce 2 tokens per frame:
      - token 0: global impression (mean init + global cross-attn)
      - token 1: representative/salient (max init + top-k cross-attn)
    Role separation mechanisms:
      (a) different init (mean vs max)
      (b) different cross-attn receptive field (all patches vs top-k patches)
      (c) token-type embedding (A/B)
      (d) optional temporal embedding (frame index)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        num_regions: int = 4,
        region_size: int = 5, # new hyperparameters for salient region selection
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        topk_ratio: float = 0.25,
        max_frames: int = 4096,
        use_temporal_embed: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_temporal_embed = use_temporal_embed
        self.num_regions = num_regions
        self.region_size = region_size

        # learned base query (keeps "Q-Former query token" spirit)
        self.query_base = nn.Parameter(torch.zeros(1, 2, hidden_size))
        nn.init.trunc_normal_(self.query_base, std=0.02)

        # token-type/role embedding: 0 for global, 1 for salient
        self.role_embed = nn.Embedding(2, hidden_size)
        nn.init.trunc_normal_(self.role_embed.weight, std=0.02)

        # temporal embedding (optional)
        if use_temporal_embed:
            self.time_embed = nn.Embedding(max_frames, hidden_size)
            nn.init.trunc_normal_(self.time_embed.weight, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _DualTokenBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    topk_ratio=topk_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_ln = nn.LayerNorm(hidden_size)

    
    def forward_frame_with_kv(
        self,
        frame_tokens: torch.Tensor,   # (N, D) patches of ONE frame
        kv_global: torch.Tensor,      # (1, N, D) (usually = frame_tokens[None])
        kv_salient: torch.Tensor,     # (1, K, D) salient subset / region pooled
        max_init: torch.Tensor,       # (D,) init for salient query (your “representative” init)
        frame_idx: int = 0,
    ) -> torch.Tensor:
        """
        Return: (2, D) for this frame:
          token0 = global-impression
          token1 = representative/salient
        """
        assert frame_tokens.dim() == 2 and frame_tokens.size(-1) == self.dim
        device, dtype = frame_tokens.device, frame_tokens.dtype

        # ---- 1) per-frame init (data-dependent) ----
        mean_tok = frame_tokens.mean(dim=0)                  # (D,)
        q_init = torch.stack([mean_tok, max_init], dim=0)    # (2, D)
        q_init = q_init.unsqueeze(0)                         # (1, 2, D)

        # ---- 2) role/type + temporal + learnable base ----
        role = self._make_role(device, dtype)                # (1, 2, D)
        temporal = self._make_temporal(frame_idx, device, dtype)  # (1, 2, D)

        q = q_init + self.query_base.to(dtype=dtype) + role + temporal  # (1,2,D)

        # ---- 3) cross-attn refinement ----
        # Each block is expected to:
        #   - self-attend within q (optional)
        #   - cross-attend q -> kv_global
        #   - cross-attend q -> kv_salient (or fuse both attentions)
        for blk in self.blocks:
            q = blk(q, kv_global=kv_global, kv_salient=kv_salient)

        q = self.out_ln(q)  # (1,2,D)
        return q.squeeze(0) # (2,D)
    


    def forward_frame(self, frame_tokens: torch.Tensor, frame_idx: int = 0) -> torch.Tensor:
        """
        frame_tokens: (N, D)  patch tokens for ONE frame
        returns: (2, D)
        """
        assert frame_tokens.dim() == 2 and frame_tokens.shape[-1] == self.hidden_size
        kv = frame_tokens.unsqueeze(0)  # (1, N, D)

        # init queries from statistics of the frame
        mean_tok = frame_tokens.mean(dim=0)
        max_tok = frame_tokens.max(dim=0).values
        q_init = torch.stack([mean_tok, max_tok], dim=0).unsqueeze(0)  # (1, 2, D)

        # role embedding
        role_ids = torch.tensor([0, 1], device=frame_tokens.device, dtype=torch.long)
        role = self.role_embed(role_ids).unsqueeze(0)  # (1, 2, D)

        q = q_init + self.query_base.to(q_init.dtype) + role

        if self.use_temporal_embed:
            idx = min(int(frame_idx), self.time_embed.num_embeddings - 1)
            t_emb = self.time_embed(
                torch.tensor([idx], device=frame_tokens.device, dtype=torch.long)
            )  # (1, D)
            q = q + t_emb.unsqueeze(1)  # broadcast to (1,2,D)

        for blk in self.blocks:
            q = blk(q, kv)

        q = self.out_ln(q)
        return q.squeeze(0)  # (2, D)

    def compress_video_tokens(
        self,
        video_embeds: torch.Tensor,                 # (sum(T*Hw'), D)
        deepstack_video_embeds: list[torch.Tensor], # list of (sum(T*Hw'), D)
        video_grid_thw: torch.LongTensor,           # (num_videos, 3) with (T,H,W) in LLM grid
        spatial_merge_size: int,
    ):
        """
        Returns:
          video_embeds_new: (sum(T*2), D)
          deepstack_video_embeds_new: list of (sum(T*2), D)
        """
        device = video_embeds.device
        dtype = video_embeds.dtype

        # per-video token counts (T*H*W / merge^2)
        split_sizes = (video_grid_thw.prod(-1) // (spatial_merge_size**2)).tolist()
        vid_chunks = torch.split(video_embeds, split_sizes, dim=0)

        # deepstack split similarly
        deep_chunks_per_layer = []
        for layer_feat in deepstack_video_embeds:
            deep_chunks_per_layer.append(torch.split(layer_feat, split_sizes, dim=0))

        out_main = []
        out_deep = [[] for _ in range(len(deepstack_video_embeds))]

        for vid_idx, chunk in enumerate(vid_chunks):
            T, H, W = video_grid_thw[vid_idx].tolist()
            tokens_per_frame = (H * W) // (spatial_merge_size**2)
            expected = T * tokens_per_frame
            if chunk.shape[0] != expected:
                raise ValueError(
                    f"[DualTokenQFormer] token count mismatch for video {vid_idx}: "
                    f"got {chunk.shape[0]} vs expected {expected} (=T*H*W/merge^2). "
                    f"Check video_grid_thw and spatial_merge_size."
                )

            chunk = chunk.view(T, tokens_per_frame, -1)

        #     # deepstack: view per frame too
        #     deep_view = []
        #     for li in range(len(deepstack_video_embeds)):
        #         d = deep_chunks_per_layer[li][vid_idx]
        #         if d.shape[0] != expected:
        #             raise ValueError(
        #                 f"[DualTokenQFormer] deepstack token mismatch at layer {li}, video {vid_idx}: "
        #                 f"{d.shape[0]} vs {expected}."
        #             )
        #         deep_view.append(d.view(T, tokens_per_frame, -1))

        #     # per-frame compression
        #     for t in range(T):
        #         q2 = self.forward_frame(chunk[t], frame_idx=t)  # (2,D)
        #         out_main.append(q2)

        #         for li in range(len(deepstack_video_embeds)):
        #             q2d = self.forward_frame(deep_view[li][t], frame_idx=t)  # (2,D)
        #             out_deep[li].append(q2d)

        # video_embeds_new = torch.cat(out_main, dim=0).to(device=device, dtype=dtype)
        # deep_new = []
        # for li in range(len(deepstack_video_embeds)):
        #     deep_new.append(torch.cat(out_deep[li], dim=0).to(device=device, dtype=dtype))

        # return video_embeds_new, deep_new

        #增加自己额外的残差选中区域的处理
            
            # 假设我们希望在压缩后保留一些残差信息·

            deep_view = []
            for li in range(len(deepstack_video_embeds)):
                d = deep_chunks_per_layer[li][vid_idx]          # (T * tokens_per_frame, D)
                d = d.view(T, tokens_per_frame, -1)             # -> (T, tokens_per_frame, D)
                deep_view.append(d)
            
            # 假设已得到 h_tok, w_tok，且 tokens_per_frame == h_tok*w_tok
            h_tok = H // spatial_merge_size
            w_tok = W // spatial_merge_size

            for t in range(T):
                cur = chunk[t]  # (N, D)

                # --- 1) residual map (feature space) ---
                if t == 0:
                    if T > 1:
                        res_1d = (chunk[1] - chunk[0]).norm(dim=-1)     # (N,)
                    else:
                        res_1d = chunk[0].norm(dim=-1)
                else:
                    res_1d = (chunk[t] - chunk[t-1]).norm(dim=-1)

                residual_2d = res_1d.view(h_tok, w_tok)

                # 可选：轻微平滑让区域更连贯（推荐）
                residual_2d = F.avg_pool2d(residual_2d[None, None], kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)

                # --- 2) pick k square regions by residual ---
                regions = self._select_regions_from_residual(
                    residual_2d=residual_2d,
                    num_regions=self.num_regions,      # 你新增的超参
                    region_size=self.region_size,      # 你新增的超参（奇数，比如 3/5/7）
                    suppress=True,
                )

                # --- 3) build salient KV + max-init restricted to regions ---
                kv_salient, max_init = self._build_salient_kv_and_max_init(cur, regions)

                # kv_global: all patches of this frame
                kv_global = cur.unsqueeze(0)  # (1, N, D)

                # --- 4) build 2 queries for this frame (mean + region-max) and run QFormer blocks ---
                q2 = self.forward_frame_with_kv(
                    frame_tokens=cur,          # for mean init etc.
                    kv_global=kv_global,
                    kv_salient=kv_salient,
                    max_init=max_init,
                    frame_idx=t,
                )  # (2, D)

                out_main.append(q2)

                # deepstack：用同一组 regions（由主干 residual 决定），在 deep 特征上也做相同区域池化
                for li in range(len(deepstack_video_embeds)):
                    cur_d = deep_view[li][t]
                    kv_salient_d, max_init_d = self._build_salient_kv_and_max_init(cur_d, regions)
                    kv_global_d = cur_d.unsqueeze(0)
                    q2d = self.forward_frame_with_kv(cur_d, kv_global_d, kv_salient_d, max_init_d, frame_idx=t)
                    out_deep[li].append(q2d)
