import os
import math
import torch
import torch.nn as nn
import warnings
from clip.clip_maple import tokenize
from clip.clip_maple.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        if torch.isnan(prompts).any() or torch.isinf(prompts).any():
            print(f"[DEBUG TextEncoder] NaN/Inf in prompts: {prompts.shape}")
            raise ValueError("prompts contains NaN/Inf")
        for i, deep_prompt in enumerate(compound_prompts_deeper_text):
            if torch.isnan(deep_prompt).any() or torch.isinf(deep_prompt).any():
                print(f"[DEBUG TextEncoder] NaN/Inf in compound_prompts_deeper_text[{i}]: {deep_prompt.shape}")
                raise ValueError(f"compound_prompts_deeper_text[{i}] contains NaN/Inf")

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        combined_input = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined_input)
        x_from_transformer = outputs[0]

        if torch.isnan(x_from_transformer).any() or torch.isinf(x_from_transformer).any():
            print(f"[DEBUG TextEncoder] NaN/Inf in x_from_transformer: {x_from_transformer.shape}")
            raise ValueError("x_from_transformer contains NaN/Inf")

        x_permuted = x_from_transformer.permute(1, 0, 2)
        x_after_ln = self.ln_final(x_permuted).type(self.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1)
        x_eot_selected = x_after_ln[torch.arange(x_after_ln.shape[0]), eot_indices]
        x_eot_selected = torch.clamp(x_eot_selected, min=-1e4, max=1e4)
        text_projection_clipped = torch.clamp(self.text_projection, min=-1e4, max=1e4)

        if torch.is_autocast_enabled() and x_eot_selected.dtype == torch.float16:
            x_eot_selected_fp32 = x_eot_selected.float()
            text_projection_fp32 = text_projection_clipped.float()
            final_output = x_eot_selected_fp32 @ text_projection_fp32
        else:
            final_output = x_eot_selected @ text_projection_clipped

        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            print(f"[DEBUG TextEncoder] NaN/Inf in final_output: {final_output.shape}")
            raise ValueError("final_output contains NaN/Inf")
        return final_output


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.coop_n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[
            0]
        visual_width = clip_model.visual.conv1.weight.shape[
            0]

        self.compound_prompts_depth = cfg.maple_prompt_depth

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.proj = nn.Linear(ctx_dim, visual_width, dtype=dtype)
        nn.init.normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None: nn.init.zeros_(self.proj.bias)
        self.ctx = nn.Parameter(ctx_vectors)
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
                                                       for _ in range(self.compound_prompts_depth - 1)])

        for prompt in self.compound_prompts_text:
            nn.init.normal_(prompt, std=0.01)
            if torch.isnan(prompt).any() or torch.isinf(prompt).any():
                raise ValueError("compound_prompts_text initialization contains NaN/Inf")
        self.compound_prompt_projections = nn.ModuleList([
            nn.Linear(ctx_dim, visual_width, dtype=dtype)
            for _ in range(self.compound_prompts_depth - 1)
        ])

        for proj_layer in self.compound_prompt_projections:
            nn.init.normal_(proj_layer.weight, std=0.02)
            if proj_layer.bias is not None:
                nn.init.zeros_(proj_layer.bias)
        classnames_processed = [name.replace("_", " ") for name in classnames]
        prompts_texts = [prompt_prefix + " " + name + "." for name in classnames_processed]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts_texts])

        with torch.no_grad():
            token_embedding_module = clip_model.token_embedding
            target_device = clip_model.positional_embedding.device
            token_embedding_module.to(target_device)
            tokenized_prompts_on_device = tokenized_prompts.to(target_device)
            embedding = token_embedding_module(tokenized_prompts_on_device).type(dtype)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        ctx_expanded = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts_for_text_encoder = self.construct_prompts(ctx_expanded, prefix, suffix)
        shared_ctx_for_vision = self.proj(self.ctx)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            deep_text_prompt_param = self.compound_prompts_text[index]
            projected_visual_prompt = layer(deep_text_prompt_param)
            visual_deep_prompts.append(projected_visual_prompt)

        return prompts_for_text_encoder, shared_ctx_for_vision, self.compound_prompts_text, visual_deep_prompts

class MDPRPluginMaPLe(nn.Module):
    def __init__(self, cfg, classnames, clip_model,
                 matrix_prior_data: torch.Tensor
                 ):
        super().__init__()
        self.cfg = cfg
        self.num_classes = len(classnames)
        self.device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        encoded_knowledge_path = getattr(cfg, 'encoded_knowledge_path', None)
        if not encoded_knowledge_path or not os.path.exists(encoded_knowledge_path):
            raise FileNotFoundError(
                f"Path to pre-encoded semantic knowledge ('encoded_knowledge_path') not provided in cfg or file not found: {encoded_knowledge_path}"
            )

        print(f"[MDPR-MaPLe] Loading pre-encoded semantic knowledge from: {encoded_knowledge_path}")
        knowledge_data = torch.load(encoded_knowledge_path, map_location='cpu', weights_only=True)

        required_keys = ["encoded_prompt_sem", "prompt_sem_cls_features", "num_phrases_per_class"]
        if not all(k in knowledge_data for k in required_keys):
            raise KeyError(
                f"Encoded knowledge file {encoded_knowledge_path} is missing one or more required keys: {required_keys}")

        self.register_buffer("encoded_prompt_sem",
                             knowledge_data["encoded_prompt_sem"].type(self.dtype).to(self.device))
        self.register_buffer("prompt_sem_cls_features",
                             knowledge_data["prompt_sem_cls_features"].type(self.dtype).to(self.device))
        self.num_phrases_per_class = knowledge_data["num_phrases_per_class"]

        if self.encoded_prompt_sem.shape[0] != self.num_classes or \
                self.encoded_prompt_sem.shape[1] != self.num_phrases_per_class:
            raise ValueError(f"Shape mismatch for loaded encoded_prompt_sem from {encoded_knowledge_path}. "
                             f"Expected ({self.num_classes}, {self.num_phrases_per_class}, ...), "
                             f"got {self.encoded_prompt_sem.shape}")
        if self.prompt_sem_cls_features.shape[0] != self.num_classes:
            raise ValueError(f"Shape mismatch for loaded prompt_sem_cls_features from {encoded_knowledge_path}.")

        embed_dim_knowledge = self.encoded_prompt_sem.shape[-1]
        print(
            f"[MDPR-MaPLe] Loaded pre-encoded semantic knowledge: {self.num_classes} classes, {self.num_phrases_per_class} phrases/class, embed_dim={embed_dim_knowledge}")

        expected_matrix_shape = (self.num_classes, self.num_phrases_per_class)
        if matrix_prior_data.shape != expected_matrix_shape:
            warnings.warn(
                f"matrix_prior_data shape mismatch. Expected {expected_matrix_shape}, got {matrix_prior_data.shape}. Attempting reshape/padding.")
            current_rows, current_cols = matrix_prior_data.shape
            target_rows, target_cols = expected_matrix_shape
            new_matrix = torch.zeros((target_rows, target_cols), dtype=matrix_prior_data.dtype,
                                     device=self.device)

            copy_rows = min(current_rows, target_rows)
            copy_cols = min(current_cols, target_cols)
            new_matrix[:copy_rows, :copy_cols] = matrix_prior_data[:copy_rows, :copy_cols].to(self.device)

            if target_rows > current_rows and copy_rows > 0:
                fill_row_source = new_matrix[copy_rows - 1, :]
                new_matrix[current_rows:target_rows, :] = fill_row_source.unsqueeze(0).expand(
                    target_rows - current_rows, target_cols)
            elif target_rows > current_rows and copy_rows == 0:
                new_matrix[current_rows:target_rows, :] = (
                        torch.ones(target_rows - current_rows, target_cols, device=self.device) / target_cols)

            if target_cols > current_cols and copy_rows > 0:
                fill_col_source = new_matrix[:copy_rows, :copy_cols].mean(dim=1, keepdim=True) if copy_cols > 0 else (
                        torch.ones(copy_rows, 1, device=self.device) / target_cols)
                new_matrix[:copy_rows, current_cols:target_cols] = fill_col_source.expand(copy_rows,
                                                                                          target_cols - current_cols)
            elif target_cols > current_cols and copy_rows == 0:
                new_matrix[:, current_cols:target_cols] = (
                        torch.ones(target_rows, target_cols - current_cols, device=self.device) / target_cols)

            matrix_prior_data_processed = new_matrix
            print(f"Matrix prior adjusted to shape: {matrix_prior_data_processed.shape}")
        else:
            matrix_prior_data_processed = matrix_prior_data.to(self.device)

        self.register_buffer("matrix_prior", matrix_prior_data_processed.type(self.dtype))
        print(f"[MDPR-MaPLe] Loaded matrix_prior with final shape: {self.matrix_prior.shape}")

        self.attn_embed_dim = embed_dim_knowledge
        self.attn_num_heads = getattr(cfg, 'attn_num_heads', 8)
        self.attn_dropout = getattr(cfg, 'attn_dropout', 0.0)

        self.semantic_attention = nn.MultiheadAttention(
            embed_dim=self.attn_embed_dim,
            num_heads=self.attn_num_heads,
            dropout=self.attn_dropout,
            batch_first=False
        ).to(self.device)
        nn.init.xavier_uniform_(self.semantic_attention.in_proj_weight)
        if self.semantic_attention.in_proj_bias is not None:
            nn.init.zeros_(self.semantic_attention.in_proj_bias)

        self.ka_projection_dim = getattr(cfg, 'ka_projection_dim', 128)
        self.ka_projection = nn.Linear(self.attn_embed_dim, self.ka_projection_dim).type(self.dtype).to(self.device)
        nn.init.normal_(self.ka_projection.weight, std=0.02)
        nn.init.zeros_(self.ka_projection.bias)

        print("\n[MDPR-MaPLe] Setting parameter gradients (initial pass)...")

        for param in clip_model.parameters():
            param.requires_grad = False

        for param in self.prompt_learner.parameters():
            param.requires_grad = True

        for param in self.semantic_attention.parameters():
            param.requires_grad = True
        for param in self.ka_projection.parameters():
            param.requires_grad = True

        if isinstance(self.logit_scale, nn.Parameter):
            self.logit_scale.requires_grad = True
        else:
            warnings.warn("logit_scale is not an nn.Parameter. Cannot set requires_grad.")

    def forward(self, image, target_for_ka=None):
        batch_size = image.shape[0]
        with torch.no_grad():
            if isinstance(self.logit_scale, nn.Parameter):
                self.logit_scale.data.clamp_(max=math.log(100.0))
            else:
                self.logit_scale.clamp_(max=math.log(100.0))

        current_logit_scale = self.logit_scale.exp()

        epsilon = 1e-6

        maple_prompts_embeddings, shared_ctx_for_visual, \
            deep_compound_prompts_text_params, visual_deep_prompts_tensors = self.prompt_learner()
        tokenized_maple_prompts_dev = self.prompt_learner.tokenized_prompts.to(self.device)
        text_features_maple = self.text_encoder(
            maple_prompts_embeddings.to(self.device),
            tokenized_maple_prompts_dev,
            deep_compound_prompts_text_params
        )
        text_norm = text_features_maple.norm(dim=-1, keepdim=True)
        image_features_maple = self.image_encoder(
            image.type(self.dtype),
            shared_ctx_for_visual.to(self.device),
            [p.to(self.device) for p in visual_deep_prompts_tensors]
        )

        image_features_maple_norm = image_features_maple / (image_features_maple.norm(dim=-1, keepdim=True) + epsilon)
        text_features_maple_norm = text_features_maple / (text_features_maple.norm(dim=-1, keepdim=True) + epsilon)
        logits_base = current_logit_scale * image_features_maple_norm @ text_features_maple_norm.t()
        encoded_prompt_sem_dev = self.encoded_prompt_sem
        query_img_maple = image_features_maple_norm

        q_for_mha = query_img_maple.unsqueeze(1).repeat(1, self.num_classes, 1)
        q_for_mha = q_for_mha.view(batch_size * self.num_classes, 1, -1)
        q_for_mha = q_for_mha.permute(1, 0, 2)

        k_v_for_mha = encoded_prompt_sem_dev.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        k_v_for_mha = k_v_for_mha.view(batch_size * self.num_classes, self.num_phrases_per_class, -1)
        k_v_for_mha = k_v_for_mha.permute(1, 0, 2)

        attn_output_batched, attn_weights_batched = self.semantic_attention(
            q_for_mha, k_v_for_mha, k_v_for_mha
        )

        x_sem_all_cls = attn_output_batched.squeeze(0)
        x_sem_all_cls = x_sem_all_cls.view(batch_size, self.num_classes, -1)

        pa_matrix_finetune = attn_weights_batched.squeeze(1)
        pa_matrix_finetune = pa_matrix_finetune.view(batch_size, self.num_classes, -1)

        x_sem_norm_all_cls = x_sem_all_cls / (x_sem_all_cls.norm(dim=-1, keepdim=True) + epsilon)
        logits_sem = current_logit_scale * torch.einsum('bd,bcd->bc', image_features_maple_norm, x_sem_norm_all_cls)
        prompt_sem_cls_features_dev = self.prompt_sem_cls_features

        return (logits_base, logits_sem, pa_matrix_finetune,
                x_sem_norm_all_cls, prompt_sem_cls_features_dev,
                image_features_maple_norm)