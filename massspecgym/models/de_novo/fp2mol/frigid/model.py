"""
FRIGID decoder: Masked Diffusion Language Model for fingerprint-to-molecule generation.

Implements the FRIGID-base architecture: a BERT backbone with cross-attention
conditioning on chemical formula and molecular fingerprint, trained with MDLM
(Masked Diffusion Language Model) objective over SAFE token sequences.

Key architectural details (matching FRIGID paper):
- BERT backbone: 12 layers, hidden 768, 12 heads, FFN 3072, vocab ~1880 SAFE BPE
- Formula conditioning: 30-element sequence with atom+count+position embeddings
- Fingerprint conditioning: active-bit set (max 256) with 3 self-attention layers
- Shared cross-attention: formula + FP concatenated for joint conditioning
- Training: MDLM loss + optional differentiable formula loss
"""

import random
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig

from massspecgym.models.base import Stage
from massspecgym.models.de_novo.fp2mol.base import FP2MolDeNovoModel
from massspecgym.models.de_novo.fp2mol.formula_utils import FormulaEncoder
from .mdlm import MDLM, LogLinearExpNoiseSchedule
from .components import (
    CrossAttentionFormulaConditioner,
    CrossAttentionFingerprintConditioner,
)
from .bert_cross_attention import BertForMaskedLMWithCrossAttention


def _safe_to_smiles(safe_string: str) -> Optional[str]:
    """Convert SAFE string to SMILES, returning None on failure."""
    try:
        from safe import decode as safe_decode
        smiles = safe_decode(safe_string)
        if smiles:
            return smiles
    except Exception:
        pass
    try:
        smi = safe_string.replace(".", "")
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


class _FallbackSafeTokenizer:
    """Small deterministic tokenizer used when the SAFE HF tokenizer is unavailable."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=256):
        if isinstance(texts, str):
            texts = [texts]
        encoded = []
        for text in texts:
            ids = [self.bos_token_id]
            ids.extend([(ord(ch) % (self.vocab_size - 4)) + 4 for ch in str(text)])
            ids.append(self.eos_token_id)
            if truncation:
                ids = ids[:max_length]
                ids[-1] = self.eos_token_id
            encoded.append(ids)
        max_len = max(len(ids) for ids in encoded) if padding else None
        if padding:
            encoded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids):
        chars = []
        for tid in token_ids:
            tid = int(tid)
            if tid in {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id}:
                continue
            chars.append(chr((tid - 4) % 95 + 32))
        return "".join(chars)

    def batch_decode(self, batch_token_ids, skip_special_tokens=True):
        return [self.decode(ids) for ids in batch_token_ids]


class FRIGIDDecoder(FP2MolDeNovoModel):
    """FRIGID Masked Diffusion Language Model decoder.

    Generates SAFE molecular sequences conditioned on Morgan fingerprints
    and chemical formulas using iterative confidence-based unmasking.

    Args:
        hidden_size: BERT hidden dimension.
        num_hidden_layers: Number of BERT layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: BERT FFN intermediate dimension.
        max_position_embeddings: Maximum sequence length.
        vocab_size: SAFE BPE vocabulary size.
        tokenizer_name: HuggingFace tokenizer name.
        fingerprint_bits: Number of fingerprint bits.
        fingerprint_seq_max_len: Maximum fingerprint sequence length.
        fingerprint_activation_threshold: Threshold for active bits.
        fingerprint_num_self_attention_layers: Self-attention layers in FP encoder.
        use_formula_conditioning: Whether to use formula cross-attention.
        use_fingerprint_conditioning: Whether to use fingerprint cross-attention.
        use_shared_cross_attention: Whether to concatenate formula+FP for shared cross-attn.
        formula_dropout_prob: Probability of dropping formula conditioning during training.
        fingerprint_dropout_prob: Probability of dropping FP conditioning during training.
        antithetic_sampling: Use antithetic time sampling in MDLM.
        formula_loss_weight: Weight for differentiable formula matching loss.
        softmax_temp: Sampling temperature.
        randomness: Gumbel noise scale for confidence sampling.
        ema_decay: EMA decay rate (0 to disable).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 256,
        vocab_size: int = 1880,
        tokenizer_name: str = "datamol-io/safe-gpt",
        fingerprint_bits: int = 4096,
        fingerprint_seq_max_len: int = 256,
        fingerprint_activation_threshold: float = 0.0,
        fingerprint_num_self_attention_layers: int = 3,
        use_formula_conditioning: bool = True,
        use_fingerprint_conditioning: bool = True,
        use_shared_cross_attention: bool = True,
        formula_dropout_prob: float = 0.0,
        fingerprint_dropout_prob: float = 0.25,
        antithetic_sampling: bool = True,
        formula_loss_weight: float = 0.0,
        softmax_temp: float = 0.8,
        randomness: float = 0.5,
        ema_decay: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(fingerprint_bits=fingerprint_bits, *args, **kwargs)

        self.hidden_size = hidden_size
        self.use_formula_conditioning = use_formula_conditioning
        self.use_fingerprint_conditioning = use_fingerprint_conditioning
        self.use_shared_cross_attention = use_shared_cross_attention
        self.formula_dropout_prob = formula_dropout_prob
        self.fingerprint_dropout_prob = fingerprint_dropout_prob
        self.antithetic_sampling = antithetic_sampling
        self.formula_loss_weight = formula_loss_weight
        self.softmax_temp = softmax_temp
        self.randomness = randomness

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = _FallbackSafeTokenizer(vocab_size)
        actual_vocab_size = self.tokenizer.vocab_size
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.pad_index = self.tokenizer.pad_token_id

        self.formula_encoder = FormulaEncoder(normalize="none")

        if self.use_formula_conditioning:
            self.formula_conditioner = CrossAttentionFormulaConditioner(
                num_atom_types=self.formula_encoder.vocab_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_layers=num_hidden_layers,
                dropout=0.1,
                use_count_embedding=True,
            )

        if self.use_fingerprint_conditioning:
            create_fp_cross_attn = not use_shared_cross_attention
            self.fingerprint_conditioner = CrossAttentionFingerprintConditioner(
                num_bits=fingerprint_bits,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_layers=num_hidden_layers,
                max_seq_len=fingerprint_seq_max_len,
                activation_threshold=fingerprint_activation_threshold,
                dropout=0.1,
                num_self_attention_layers=fingerprint_num_self_attention_layers,
                create_cross_attention_layers=create_fp_cross_attn,
            )

        bert_config = BertConfig(
            vocab_size=actual_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        formula_ca = self.formula_conditioner.cross_attention_layers if self.use_formula_conditioning else None
        fp_ca = None
        if self.use_fingerprint_conditioning and not use_shared_cross_attention:
            fp_ca = self.fingerprint_conditioner.cross_attention_layers

        self.backbone = BertForMaskedLMWithCrossAttention(
            bert_config,
            cross_attention_layers=formula_ca,
            fingerprint_cross_attention_layers=fp_ca,
            use_shared_cross_attention=use_shared_cross_attention,
        )

        self.mdlm = MDLM(
            mask_token_id=self.mask_index,
            vocab_size=actual_vocab_size,
            noise_schedule=LogLinearExpNoiseSchedule(),
            sampling_eps=1e-3,
        )

        self.max_position_embeddings = max_position_embeddings

    def _build_token_embeddings(self, x):
        emb = self.backbone.embeddings
        token_emb = emb.word_embeddings(x)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        pos_emb = emb.position_embeddings(pos_ids)
        type_emb = emb.token_type_embeddings(torch.zeros_like(x))
        return token_emb + pos_emb + type_emb

    def _prepare_formula_embeddings(self, formula, x):
        if not self.use_formula_conditioning:
            return None, None
        if formula is not None:
            fv = self.formula_encoder.encode_batch(formula).to(x.device)
        else:
            fv = self.formula_encoder.encode_batch(["C"] * x.size(0)).to(x.device)
            return fv.new_zeros(x.size(0), self.formula_encoder.vocab_size, self.hidden_size), \
                   fv.new_zeros(x.size(0), self.formula_encoder.vocab_size)
        emb, mask = self.formula_conditioner.encode_formula(fv)
        if self.training and formula is not None and random.random() < self.formula_dropout_prob:
            emb = emb * 0.0
        return emb, mask

    def _prepare_fingerprint_embeddings(self, fingerprint, x):
        if not self.use_fingerprint_conditioning:
            return None, None
        if fingerprint is None:
            return None, None
        fp = fingerprint.to(device=x.device, dtype=torch.float32)
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        if fp.size(0) == 1 and x.size(0) > 1:
            fp = fp.expand(x.size(0), -1)
        emb, mask = self.fingerprint_conditioner.encode_fingerprint(fp)
        if self.training and random.random() < self.fingerprint_dropout_prob:
            emb = emb * 0.0
        return emb, mask

    def forward_model(self, x, attention_mask=None, formula=None, fingerprint=None):
        """Forward pass through BERT backbone with conditioning.

        Matches the reference GenMol.forward() exactly:
        1. Build token + position + type embeddings.
        2. Apply LayerNorm + dropout.
        3. Compute extended attention mask.
        4. In shared mode: concatenate formula + FP embeddings for single cross-attn path.
           In independent mode: pass them separately.
        5. Run encoder layers (self-attn -> cross-attn -> FFN per layer).
        6. Apply MLM head (cls) to get logits.

        Returns logits [batch, seq_len, vocab_size].
        """
        with torch.amp.autocast("cuda", dtype=torch.float32):
            formula_emb, formula_mask = self._prepare_formula_embeddings(formula, x)
            fp_emb, fp_mask = self._prepare_fingerprint_embeddings(fingerprint, x)

            embeddings = self._build_token_embeddings(x)
            embeddings = self.backbone.embeddings.LayerNorm(embeddings)
            embeddings = self.backbone.embeddings.dropout(embeddings)

            if attention_mask is None:
                attention_mask = torch.ones_like(x)
            ext_attn = self.backbone.get_extended_attention_mask(
                attention_mask, x.shape, x.device
            )

            if self.use_shared_cross_attention:
                cond_emb = formula_emb
                cond_mask = formula_mask
                if fp_emb is not None:
                    if cond_emb is None:
                        cond_emb, cond_mask = fp_emb, fp_mask
                    else:
                        cond_emb = torch.cat([cond_emb, fp_emb], dim=1)
                        cond_mask = torch.cat([cond_mask, fp_mask], dim=1)
                enc_out = self.backbone.bert.encoder(
                    embeddings, attention_mask=ext_attn,
                    condition_embeddings=cond_emb, condition_mask=cond_mask,
                )
            else:
                enc_out = self.backbone.bert.encoder(
                    embeddings, attention_mask=ext_attn,
                    condition_embeddings=formula_emb, condition_mask=formula_mask,
                    fingerprint_embeddings=fp_emb, fingerprint_mask=fp_mask,
                )

            sequence_output = enc_out[0]
            return self.backbone.cls(sequence_output)

    def compute_decoder_loss(self, batch: dict) -> torch.Tensor:
        """Compute MDLM training loss from a batch.

        In fp2mol_pretrain mode, batch contains: fingerprint, formula, mol_repr.
        In spec2mol mode, batch contains spectrum data + mol.
        """
        fingerprint = batch.get("fingerprint", None)
        formula = batch.get("formula", None)
        mol_repr = batch.get("mol_repr", batch.get("mol", None))

        if isinstance(mol_repr, (list, tuple)):
            encoded = self.tokenizer(
                list(mol_repr), return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_position_embeddings,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        else:
            input_ids = mol_repr
            attention_mask = (mol_repr != self.pad_index).long()

        self.mdlm.to_device(self.device)
        t = self.mdlm.sample_time(input_ids.shape[0], antithetic=self.antithetic_sampling)
        t = t.to(self.device)
        xt = self.mdlm.forward_process(input_ids, t)

        logits = self.forward_model(xt, attention_mask, formula=formula, fingerprint=fingerprint)
        loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)

        if self.formula_loss_weight > 0 and formula is not None:
            gt_fv = self.formula_encoder.encode_batch(formula, normalize="none").to(self.device)
            probs = torch.softmax(logits, dim=-1)
            token_atom_counts = self._get_token_atom_counts()
            expected = torch.matmul(probs, token_atom_counts)
            if attention_mask is not None:
                expected = expected * attention_mask.unsqueeze(-1)
            predicted = expected.sum(dim=1)
            f_loss = F.mse_loss(predicted, gt_fv)
            loss = loss + self.formula_loss_weight * f_loss

        return loss

    def _get_token_atom_counts(self):
        if not hasattr(self, "_token_atom_counts_buf"):
            n_atoms = self.formula_encoder.vocab_size
            v_size = self.tokenizer.vocab_size
            buf = torch.zeros(v_size, n_atoms, dtype=torch.float32, device=self.device)
            for tid in range(v_size):
                tok_str = self.tokenizer.decode([tid])
                try:
                    counts = self.formula_encoder.formula_to_counts(tok_str)
                    vec = self.formula_encoder.counts_to_vector(counts, normalize="none")
                    buf[tid] = vec.to(self.device)
                except Exception:
                    pass
            self._token_atom_counts_buf = buf
        return self._token_atom_counts_buf

    @torch.no_grad()
    def decode_from_fingerprint(
        self,
        fingerprint: torch.Tensor,
        formula=None,
        num_samples: int = 1,
    ) -> list:
        """Generate molecules from fingerprint + formula via iterative unmasking."""
        self.eval()
        self.mdlm.to_device(self.device)
        batch_size = fingerprint.shape[0]
        all_preds = []

        for b_idx in range(batch_size):
            fp_single = fingerprint[b_idx:b_idx + 1]
            form_single = [formula[b_idx]] if formula is not None else None

            x = torch.full((1, 1), self.bos_index, device=self.device)
            x = torch.cat([x, torch.full((1, 1), self.eos_index, device=self.device)], dim=1)

            target_len = 50
            x_batch = self._insert_masks(x, num_samples, target_len)
            x_batch = x_batch.to(self.device)

            fp_expanded = fp_single.expand(num_samples, -1)
            form_expanded = form_single * num_samples if form_single else None

            num_steps = max(self.mdlm.get_num_steps_confidence(x_batch), 2)
            attn_mask = (x_batch != self.pad_index).long()

            for i in range(num_steps):
                logits = self.forward_model(
                    x_batch, attn_mask, formula=form_expanded, fingerprint=fp_expanded
                )
                x_batch = self.mdlm.step_confidence(
                    logits, x_batch, i, num_steps, self.softmax_temp, self.randomness
                )

            decoded = self.tokenizer.batch_decode(x_batch, skip_special_tokens=True)
            smiles_list = []
            for safe_str in decoded:
                smi = _safe_to_smiles(safe_str)
                smiles_list.append(smi)
            all_preds.append(smiles_list)

        return all_preds

    def _insert_masks(self, x, num_samples, target_len):
        """Create num_samples copies of x with mask tokens inserted."""
        x_seq = x[0]
        results = []
        for _ in range(num_samples):
            add_len = max(target_len - len(x_seq), 10)
            add_len = min(add_len, self.max_position_embeddings - len(x_seq))
            add_len = max(add_len, 10)
            new_x = torch.cat([
                x_seq[:-1],
                torch.full((add_len,), self.mask_index, device=x.device),
                x_seq[-1:],
            ])
            results.append(new_x)
        max_len = max(len(r) for r in results)
        padded = []
        for r in results:
            pad_len = max_len - len(r)
            if pad_len > 0:
                r = torch.cat([r, torch.full((pad_len,), self.pad_index, device=x.device)])
            padded.append(r)
        return torch.stack(padded)
