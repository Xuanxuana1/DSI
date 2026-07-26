from __future__ import annotations

import logging
import math
import random
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _spec_map(description: list[tuple[str, Any, str]]) -> dict[str, tuple[Any, str]]:
    return {name: (size, kind) for name, size, kind in description}


def _vector_cardinalities(description: list[tuple[str, Any, str]], name: str) -> list[int]:
    spec = _spec_map(description)
    if name not in spec:
        return []
    size, _ = spec[name]
    if isinstance(size, (list, tuple)):
        return [max(int(value), 1) for value in size]
    return [max(int(size), 1)]


def _scalar_cardinality(description: list[tuple[str, Any, str]], name: str) -> int:
    spec = _spec_map(description)
    if name not in spec:
        return 1
    size, _ = spec[name]
    return max(int(size), 1)


def _pad_cards(values: list[int], target: int) -> list[int]:
    padded = list(values[:target])
    padded.extend([1] * max(int(target) - len(padded), 0))
    return padded


class PackedSlotEncoder(nn.Module):
    def __init__(self, description: list[tuple[str, Any, str]], feat_dim: int, duration_buckets: int, include_duration_in_item: bool = False) -> None:
        super().__init__()
        logger.info("Initializing the generative baseline slot encoder. feat_dim=%d duration_buckets=%d", int(feat_dim), int(duration_buckets))

        self.feat_dim = int(feat_dim)
        self.include_duration_in_item = bool(include_duration_in_item)
        self.user_id_card = _scalar_cardinality(description, "user_id")
        self.item_id_card = _scalar_cardinality(description, "video_id")
        self.user_feature_cards = _pad_cards(_vector_cardinalities(description, "raw_sparse"), 18)
        item_cards = _pad_cards(_vector_cardinalities(description, "content_category"), 4)
        self.item_feature_card = max(max(item_cards), 1)
        self.duration_card = max(int(duration_buckets), 1)

        self.user_id_embedding = nn.Embedding(self.user_id_card, self.feat_dim)
        self.user_fea_embedding = nn.ModuleList([
            nn.Embedding(cardinality, self.feat_dim)
            for cardinality in self.user_feature_cards
        ])
        self.item_id_embedding = nn.Embedding(self.item_id_card, self.feat_dim)
        self.item_fea_embedding = nn.Embedding(self.item_feature_card, self.feat_dim)
        self.item_duration_embedding = nn.Embedding(self.duration_card, self.feat_dim)
        logger.info("Generative baseline slot encoder initialized.")

    def _pack_slots(self, features: dict[str, torch.Tensor], duration_bucket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.debug("Packing generative baseline slots.")

        batch_size = int(features["user_id"].shape[0])
        device = features["user_id"].device
        x = torch.zeros((batch_size, 25), dtype=torch.long, device=device)
        q_mask = torch.ones((batch_size, 19), dtype=torch.float32, device=device)
        i_mask = torch.ones((batch_size, 5), dtype=torch.float32, device=device)

        x[:, 0] = features["user_id"].view(-1).long().clamp(0, self.user_id_card - 1)
        raw_sparse = features.get("raw_sparse")
        if raw_sparse is not None and raw_sparse.numel() > 0:
            raw_values = raw_sparse.long()
            for idx, embedding in enumerate(self.user_fea_embedding):
                if idx >= raw_values.shape[1]:
                    break
                x[:, idx + 1] = raw_values[:, idx].clamp(0, embedding.num_embeddings - 1)

        x[:, 19] = features["video_id"].view(-1).long().clamp(0, self.item_id_card - 1)
        content = features.get("content_category")
        if content is not None and content.numel() > 0:
            content_values = content.long()
            for idx in range(min(4, content_values.shape[1])):
                value = content_values[:, idx].clamp(0, self.item_feature_card - 1)
                x[:, 20 + idx] = value
                i_mask[:, idx + 1] = (value > 0).float()
        x[:, 24] = duration_bucket.view(-1).long().clamp(0, self.duration_card - 1)
        logger.debug("Generative baseline slots packed.")
        return x, q_mask, i_mask

    def encode_source(self, features: dict[str, torch.Tensor], duration_bucket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.debug("Encoding generative baseline source features.")

        x, q_mask, i_mask = self._pack_slots(features, duration_bucket)

        user_id = self.user_id_embedding(x[:, 0].unsqueeze(1))
        user_features = torch.cat(
            [embedding(x[:, idx + 1].unsqueeze(1)) for idx, embedding in enumerate(self.user_fea_embedding)],
            dim=1,
        )
        item_id = self.item_id_embedding(x[:, 19].unsqueeze(1))
        item_features = torch.cat([self.item_fea_embedding(x[:, idx].unsqueeze(1)) for idx in range(20, 24)], dim=1)
        item_duration = self.item_duration_embedding(x[:, 24].unsqueeze(1))

        user_stack = torch.cat([user_id, user_features], dim=1)
        item_stack = torch.cat([item_id, item_features], dim=1)
        q_weight = q_mask.unsqueeze(-1)
        i_weight = i_mask.unsqueeze(-1)
        if self.include_duration_in_item:
            item_stack = torch.cat([item_stack, item_duration], dim=1)
            i_weight = torch.cat([i_weight, torch.ones_like(i_weight[:, :1])], dim=1)
        query_sum = (user_stack * q_weight).sum(dim=1) / q_weight.sum(dim=1).clamp_min(1.0)
        item_sum = (item_stack * i_weight).sum(dim=1) / i_weight.sum(dim=1).clamp_min(1.0)
        logger.debug("Generative baseline source features encoded.")
        return query_sum, item_sum, item_duration.sum(dim=1), x


class GRSourceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim) * 2, int(hidden_dim) * 2)
        self.dropout1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return x.unsqueeze(0).contiguous()


class ScaleDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        _, _, _, d_tensor = k.size()
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(float(d_tensor))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score)
        return torch.matmul(score, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.n_head = int(n_head)
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(int(d_model), int(d_model))
        self.w_k = nn.Linear(int(d_model), int(d_model))
        self.w_v = nn.Linear(int(d_model), int(d_model))
        self.w_concat = nn.Linear(int(d_model), int(d_model))

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

    def _concat(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, head, length, d_tensor = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, length, head * d_tensor)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self._split(self.w_q(q)), self._split(self.w_k(k)), self._split(self.w_v(v))
        out = self.attention(q, k, v, mask=mask)
        return self.w_concat(self._concat(out))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(int(d_model), int(hidden))
        self.linear2 = nn.Linear(int(hidden), int(d_model))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class GRDecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(int(d_model))
        self.dropout1 = nn.Dropout(float(dropout))
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(int(d_model))
        self.dropout2 = nn.Dropout(float(dropout))
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm3 = nn.LayerNorm(int(d_model))
        self.dropout3 = nn.Dropout(float(dropout))

    def forward(self, dec: torch.Tensor, enc: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.self_attention(dec, dec, dec, trg_mask))
        x = self.norm1(x + dec)
        y = self.dropout2(self.enc_dec_attention(x, enc, enc, src_mask))
        y = self.norm2(y + x)
        z = self.dropout3(self.ffn(y))
        return self.norm3(z + y)


class GRDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, ffn_hidden: int, n_head: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GRDecoderLayer(d_model, ffn_hidden, n_head, dropout)
            for _ in range(int(n_layers))
        ])
        self.linear = nn.Linear(int(d_model), int(vocab_size))
        self.tok_emb = nn.Embedding(int(vocab_size), int(d_model))
        self.drop_out = nn.Dropout(float(dropout))
        self.output_dim = int(vocab_size)

    def _embed_mixed(self, items: list[torch.Tensor]) -> torch.Tensor:
        embedding = self.tok_emb.weight
        embedded_items = []
        for item in items:
            if item.shape[-1] == self.output_dim:
                probs = F.softmax(item, dim=-1)
                embedded_items.append(self.drop_out(torch.einsum("bv,ve->be", probs, embedding)).unsqueeze(1))
            else:
                embedded_items.append(self.drop_out(self.tok_emb(item)))
        return torch.cat(embedded_items, dim=1)

    def forward(self, trg: torch.Tensor, enc_src: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trg_emb = self.drop_out(self.tok_emb(trg))
        for layer in self.layers:
            trg_emb = layer(trg_emb, enc_src, trg_mask, src_mask)
        return self.linear(trg_emb), trg_emb

    def weight_forward(self, trg, enc_src: torch.Tensor, trg_mask: torch.Tensor, src_mask: torch.Tensor, use_embedding_mixup: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if use_embedding_mixup:
            trg_emb = self._embed_mixed(trg)
        else:
            trg_emb = self.drop_out(self.tok_emb(trg))
        for layer in self.layers:
            trg_emb = layer(trg_emb, enc_src, trg_mask, src_mask)
        return self.linear(trg_emb), trg_emb


class GR(nn.Module):
    def __init__(
        self,
        description: list[tuple[str, Any, str]],
        vocab_size: int,
        duration_buckets: int,
        feat_dim: int = 128,
        hidden_dim: int = 128,
        n_head: int = 8,
        decoder_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_embedding_mixup: bool = False,
    ) -> None:
        super().__init__()
        logger.info("Initializing GR baseline. vocab_size=%d", int(vocab_size))
        self.slot_encoder = PackedSlotEncoder(description, feat_dim, duration_buckets, include_duration_in_item=True)
        self.encoder = GRSourceEncoder(feat_dim, hidden_dim, dropout)
        self.decoder = GRDecoder(vocab_size, hidden_dim, ffn_dim, n_head, decoder_layers, dropout)
        self.use_embedding_mixup = bool(use_embedding_mixup)
        self.device_ref: torch.device | None = None
        logger.info("GR baseline initialized.")

    def make_trg_mask(self, trg: torch.Tensor, trg_pad_mask: torch.Tensor) -> torch.Tensor:
        device = trg.device
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        return (trg_pad_mask.unsqueeze(1).bool() & trg_sub_mask.unsqueeze(0)).unsqueeze(1)

    def get_encoder_result(self, features: dict[str, torch.Tensor], duration_bucket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_sum, item_sum, _, _ = self.slot_encoder.encode_source(features, duration_bucket)
        src_input = torch.cat([query_sum, item_sum], dim=1)
        enc_src = self.encoder(src_input).transpose(0, 1)
        src_mask = torch.ones(query_sum.shape[0], 1, device=query_sum.device).unsqueeze(1).unsqueeze(2)
        return enc_src, src_mask

    def forward(
        self,
        features: dict[str, torch.Tensor],
        duration_bucket: torch.Tensor,
        trg: torch.Tensor,
        true_label_mask: torch.Tensor,
        teacher_force_ratio: float,
    ) -> torch.Tensor:
        logger.debug("Running the GR forward pass.")
        enc_src, src_mask = self.get_encoder_result(features, duration_bucket)
        trg_mask = self.make_trg_mask(trg, true_label_mask)
        output, _ = self.decoder(trg, enc_src, trg_mask, src_mask)
        prob = output.argmax(dim=-1)
        input_teacher = []
        teacher_force = [random.random() > float(teacher_force_ratio) for _ in range(1, output.shape[1])]
        for idx in range(output.shape[1]):
            if idx != 0 and teacher_force[idx - 1]:
                input_teacher.append(output[:, idx - 1, :] if self.use_embedding_mixup else prob[:, idx - 1].unsqueeze(1))
            else:
                input_teacher.append(trg[:, idx].unsqueeze(1))
        if not self.use_embedding_mixup:
            input_teacher = torch.cat(input_teacher, dim=1)
        output, _ = self.decoder.weight_forward(input_teacher, enc_src, trg_mask, src_mask, self.use_embedding_mixup)
        logger.debug("GR forward pass complete.")
        return output

    def autoregressive_logits(self, features: dict[str, torch.Tensor], duration_bucket: torch.Tensor, max_len: int) -> torch.Tensor:
        logger.debug("Running GR autoregressive inference. max_len=%d", int(max_len))
        enc_src, src_mask = self.get_encoder_result(features, duration_bucket)
        batch_size = int(features["user_id"].shape[0])
        device = features["user_id"].device
        tokens = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        outputs = []
        mixed_inputs: list[torch.Tensor] = [tokens]
        for _ in range(max(int(max_len) - 1, 1)):
            trg_pad_mask = torch.ones_like(tokens, dtype=torch.bool)
            trg_mask = self.make_trg_mask(tokens, trg_pad_mask)
            if self.use_embedding_mixup:
                logits, _ = self.decoder.weight_forward(mixed_inputs, enc_src, trg_mask, src_mask, True)
            else:
                logits, _ = self.decoder(tokens, enc_src, trg_mask, src_mask)
            step_logits = logits[:, -1, :]
            next_word = torch.argmax(step_logits, dim=-1).unsqueeze(1)
            tokens = torch.cat([tokens, next_word], dim=1)
            mixed_inputs.append(step_logits if self.use_embedding_mixup else next_word)
            outputs.append(step_logits.unsqueeze(1))
        logger.debug("GR autoregressive inference complete.")
        return torch.cat(outputs, dim=1)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(int(input_dim), 1, bias=False) for _ in range(int(num_layers))])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(int(input_dim))) for _ in range(int(num_layers))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        out = x
        for layer, bias in zip(self.layers, self.bias):
            out = x0 * layer(out) + bias + out
        return out


class DIFL(nn.Module):
    def __init__(self, description: list[tuple[str, Any, str]], duration_buckets: int, feat_dim: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        logger.info("Initializing DIFL baseline. feat_dim=%d", int(feat_dim))
        self.feat_dim = int(feat_dim)
        self.slot_encoder = PackedSlotEncoder(description, feat_dim, duration_buckets)
        hidden_dim = int(feat_dim) * 2
        self.cn = CrossNetwork(hidden_dim, 3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.fc1 = nn.Linear(int(feat_dim) * 5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(float(dropout))
        logger.info("DIFL baseline initialized.")

    def forward(self, features: dict[str, torch.Tensor], duration_bucket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logger.debug("Running the DIFL forward pass.")
        query_sum, item_sum, item_duration, _ = self.slot_encoder.encode_source(features, duration_bucket)
        x = torch.cat([query_sum, item_sum], dim=1)
        cross = self.cn(x)
        deep = self.mlp(x)
        representation = torch.cat([cross, deep], dim=1)
        out = torch.cat([representation, item_duration], dim=1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        logits = self.fc4(out)
        logger.debug("DIFL forward pass complete.")
        return representation, torch.sigmoid(logits)

    def forward_with_hidden(self, features: dict[str, torch.Tensor], duration_bucket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        representation, pred = self.forward(features, duration_bucket)
        return representation, pred, representation
