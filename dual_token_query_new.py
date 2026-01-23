import torch
import torch.nn as nn
import torch.nn.functional as F


class _DualTokenBlock(nn.Module):
    """Q-Former-like block: (LN -> self-attn on queries) + (LN -> cross-attn) + (LN -> FFN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.ln_q1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.ln_q2 = nn.LayerNorm(hidden_size)
        self.cross_attn_global = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
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

    def forward(
        self,
        q: torch.Tensor,
        kv_global: torch.Tensor,
        kv_salient: torch.Tensor,
    ) -> torch.Tensor:
        # 1) self-attn among queries
        q_sa, _ = self.self_attn(self.ln_q1(q), self.ln_q1(q), self.ln_q1(q), need_weights=False)
        q = q + q_sa

        # 2) role-differentiated cross-attn
        q_ln = self.ln_q2(q)
        q_global = q_ln[:, :1, :]
        q_salient = q_ln[:, 1:, :]

        q_global_ca, _ = self.cross_attn_global(q_global, kv_global, kv_global, need_weights=False)
        q_salient_ca, _ = self.cross_attn_salient(q_salient, kv_salient, kv_salient, need_weights=False)

        q = q + torch.cat([q_global_ca, q_salient_ca], dim=1)

        # 3) FFN
        q = q + self.ffn(self.ln_q3(q))
        return q


class DualTokenQFormer(nn.Module):
    """
    Produce 2 tokens per frame:
      - token 0: global impression (mean init + global cross-attn)
      - token 1: representative/salient (max init + salient cross-attn)
    Role separation mechanisms:
      (a) different init (mean vs max)
      (b) different cross-attn receptive field (all patches vs salient patches)
      (c) token-type embedding (A/B)
      (d) optional temporal embedding (frame index)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        num_regions: int = 4,
        region_size: int = 5,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        topk_ratio: float = 0.25,
        max_frames: int = 4096,
        use_temporal_embed: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_temporal_embed = use_temporal_embed
        self.num_regions = num_regions
        self.region_size = region_size
        self.topk_ratio = topk_ratio

        # learned base query (keeps Q-Former query token spirit)
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
                )
                for _ in range(num_layers)
            ]
        )

        self.out_ln = nn.LayerNorm(hidden_size)

    def _make_role(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        role_ids = torch.tensor([0, 1], device=device, dtype=torch.long)
        role = self.role_embed(role_ids).to(dtype=dtype)
        return role.unsqueeze(0)

    def _make_temporal(
        self, frame_idx: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if not self.use_temporal_embed:
            return torch.zeros((1, 2, self.hidden_size), device=device, dtype=dtype)
        idx = min(int(frame_idx), self.time_embed.num_embeddings - 1)
        t_emb = self.time_embed(torch.tensor([idx], device=device, dtype=torch.long))
        return t_emb.unsqueeze(1).to(dtype=dtype)

    def _topk_patches(self, kv: torch.Tensor) -> torch.Tensor:
        """kv: (N, D). Use L2-norm as saliency proxy and pick top-k tokens."""
        n = kv.shape[0]
        k = max(1, int(n * self.topk_ratio))
        score = kv.norm(dim=-1)
        topk_idx = score.topk(k, dim=0, largest=True, sorted=False).indices
        return kv.index_select(0, topk_idx).unsqueeze(0)

    def _select_regions_from_residual(
        self,
        residual_2d: torch.Tensor,
        num_regions: int,
        region_size: int,
        suppress: bool = True,
    ) -> list[torch.Tensor]:
        """
        Greedy pick top residual centers, expand to squares, optionally suppress overlap.
        Returns list of 1D flat indices in [0, h*w).
        """
        if residual_2d.dim() != 2:
            raise ValueError("residual_2d must be 2D.")
        h, w = residual_2d.shape
        r = region_size // 2

        score = residual_2d.clone()
        regions = []
        neg_inf = torch.finfo(score.dtype).min

        for _ in range(num_regions):
            flat = score.view(-1)
            maxv, arg = flat.max(dim=0)
            if torch.isinf(maxv) or (maxv <= neg_inf / 2):
                break

            cy = (arg // w).item()
            cx = (arg % w).item()

            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)

            ys = torch.arange(y0, y1, device=score.device)
            xs = torch.arange(x0, x1, device=score.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            idx = (yy * w + xx).reshape(-1)

            regions.append(idx)

            if suppress:
                score[y0:y1, x0:x1] = neg_inf

        return regions

    def _build_salient_kv_and_max_init(
        self,
        frame_tokens: torch.Tensor,
        regions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each region: max-pool tokens inside -> 1 region token.
        KV for salient token: (1, R, D)
        Max-init token: max over union of region tokens.
        """
        device = frame_tokens.device
        dtype = frame_tokens.dtype

        if len(regions) == 0:
            kv_salient = frame_tokens.unsqueeze(0)
            max_init = frame_tokens.max(dim=0).values
            return kv_salient, max_init

        region_tokens = []
        union = []

        for idx in regions:
            idx = idx.to(device=device)
            idx = idx.clamp_(0, frame_tokens.shape[0] - 1)
            tok = frame_tokens.index_select(0, idx)
            region_tok = tok.max(dim=0).values
            region_tokens.append(region_tok)
            union.append(idx)

        region_tokens = torch.stack(region_tokens, dim=0).to(dtype=dtype)
        kv_salient = region_tokens.unsqueeze(0)

        union_idx = torch.unique(torch.cat(union, dim=0))
        union_tok = frame_tokens.index_select(0, union_idx)
        max_init = union_tok.max(dim=0).values

        return kv_salient, max_init

    def forward_frame_with_kv(
        self,
        frame_tokens: torch.Tensor,
        kv_global: torch.Tensor,
        kv_salient: torch.Tensor,
        max_init: torch.Tensor,
        frame_idx: int = 0,
    ) -> torch.Tensor:
        """
        Return: (2, D) for this frame:
          token0 = global-impression
          token1 = representative/salient
        """
        if frame_tokens.dim() != 2 or frame_tokens.size(-1) != self.hidden_size:
            raise ValueError("frame_tokens must be (N, D) with D == hidden_size.")
        device, dtype = frame_tokens.device, frame_tokens.dtype

        mean_tok = frame_tokens.mean(dim=0)
        q_init = torch.stack([mean_tok, max_init], dim=0).unsqueeze(0)

        role = self._make_role(device, dtype)
        temporal = self._make_temporal(frame_idx, device, dtype)
        q = q_init + self.query_base.to(dtype=dtype) + role + temporal

        for blk in self.blocks:
            q = blk(q, kv_global=kv_global, kv_salient=kv_salient)

        q = self.out_ln(q)
        return q.squeeze(0)

    def forward_frame(self, frame_tokens: torch.Tensor, frame_idx: int = 0) -> torch.Tensor:
        """
        frame_tokens: (N, D) patch tokens for ONE frame
        returns: (2, D)
        """
        if frame_tokens.dim() != 2 or frame_tokens.shape[-1] != self.hidden_size:
            raise ValueError("frame_tokens must be (N, D) with D == hidden_size.")

        kv_global = frame_tokens.unsqueeze(0)
        kv_salient = self._topk_patches(frame_tokens)
        max_init = frame_tokens.max(dim=0).values

        return self.forward_frame_with_kv(
            frame_tokens=frame_tokens,
            kv_global=kv_global,
            kv_salient=kv_salient,
            max_init=max_init,
            frame_idx=frame_idx,
        )

    def compress_video_tokens(
        self,
        video_embeds: torch.Tensor,
        deepstack_video_embeds: list[torch.Tensor],
        video_grid_thw: torch.LongTensor,
        spatial_merge_size: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns:
          video_embeds_new: (sum(T*2), D)
          deepstack_video_embeds_new: list of (sum(T*2), D)
        """
        device = video_embeds.device
        dtype = video_embeds.dtype

        split_sizes = (video_grid_thw.prod(-1) // (spatial_merge_size**2)).tolist()
        vid_chunks = torch.split(video_embeds, split_sizes, dim=0)

        deep_chunks_per_layer = [
            torch.split(layer_feat, split_sizes, dim=0) for layer_feat in deepstack_video_embeds
        ]

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

            deep_view = []
            for li in range(len(deepstack_video_embeds)):
                d = deep_chunks_per_layer[li][vid_idx]
                if d.shape[0] != expected:
                    raise ValueError(
                        f"[DualTokenQFormer] deepstack token mismatch at layer {li}, video {vid_idx}: "
                        f"{d.shape[0]} vs {expected}."
                    )
                deep_view.append(d.view(T, tokens_per_frame, -1))

            h_tok = H // spatial_merge_size
            w_tok = W // spatial_merge_size

            for t in range(T):
                cur = chunk[t]

                # --- 1) residual map (feature space) ---
                if t == 0:
                    if T > 1:
                        res_1d = (chunk[1] - chunk[0]).norm(dim=-1)
                    else:
                        res_1d = chunk[0].norm(dim=-1)
                else:
                    res_1d = (chunk[t] - chunk[t - 1]).norm(dim=-1)

                residual_2d = res_1d.view(h_tok, w_tok)
                residual_2d = (
                    F.avg_pool2d(residual_2d[None, None], kernel_size=3, stride=1, padding=1)
                    .squeeze(0)
                    .squeeze(0)
                )

                # --- 2) pick k square regions by residual ---
                regions = self._select_regions_from_residual(
                    residual_2d=residual_2d,
                    num_regions=self.num_regions,
                    region_size=self.region_size,
                    suppress=True,
                )

                # --- 3) build salient KV + max-init restricted to regions ---
                kv_salient, max_init = self._build_salient_kv_and_max_init(cur, regions)
                kv_global = cur.unsqueeze(0)

                # --- 4) build 2 queries for this frame and run blocks ---
                q2 = self.forward_frame_with_kv(
                    frame_tokens=cur,
                    kv_global=kv_global,
                    kv_salient=kv_salient,
                    max_init=max_init,
                    frame_idx=t,
                )
                out_main.append(q2)

                # deepstack: use same regions on deep features
                for li in range(len(deepstack_video_embeds)):
                    cur_d = deep_view[li][t]
                    kv_salient_d, max_init_d = self._build_salient_kv_and_max_init(cur_d, regions)
                    kv_global_d = cur_d.unsqueeze(0)
                    q2d = self.forward_frame_with_kv(
                        frame_tokens=cur_d,
                        kv_global=kv_global_d,
                        kv_salient=kv_salient_d,
                        max_init=max_init_d,
                        frame_idx=t,
                    )
                    out_deep[li].append(q2d)

        video_embeds_new = torch.cat(out_main, dim=0).to(device=device, dtype=dtype)
        deep_new = [torch.cat(layer_out, dim=0).to(device=device, dtype=dtype) for layer_out in out_deep]

        return video_embeds_new, deep_new
