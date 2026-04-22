"""
MIST-CF MistNet model: Formula scoring from MS/MS spectra.

Ported from external/mist-cf/src/mist_cf/mist_cf_score/mist_cf_model.py.

The model uses a FormulaTransformer to encode subformulae-annotated spectra
and scores candidate (formula, adduct) pairs. Key differences from the
MIST fingerprint encoder:
- Uses "abs-sines" form embedder (not "float" or "pos-cos").
- Uses "cls" pooling by default.
- Includes rel_mass_diff as an input feature.
- Includes num_peaks as a scaled feature.
- Output is a single score (not a fingerprint vector).
"""

import torch
import torch.nn as nn
import numpy as np

from massspecgym.models.encoders.mist.chem_constants import (
    max_instr_idx as MAX_INSTR_IDX,
    ION_LST,
)
from massspecgym.models.encoders.mist.form_embedders import get_embedder
from massspecgym.models.encoders.mist.modules import MLPBlocks, _get_clones
from massspecgym.models.encoders.mist import transformer_layer


class MistCFFormulaTransformer(nn.Module):
    """FormulaTransformer for MIST-CF formula scoring.

    Differs from the MIST fingerprint encoder's FormulaTransformer:
    - Input includes: form_embedded, diff_embedded, cls_flag, intensity,
      num_peaks (scaled), rel_mass_diff, and optionally ion/instrument.
    - No explicit "types" one-hot; uses cls_flag (1 dim) instead.
    - Uses "abs-sines" embedder by default.
    - Pool methods: cls (default), intensity, mean.

    Args:
        form_encoder: Name of formula embedder ('abs-sines', etc.).
        hidden_size: Hidden dimension.
        peak_attn_layers: Number of transformer layers.
        set_pooling: Pooling type ('cls', 'intensity', 'mean').
        spectra_dropout: Dropout rate.
        additive_attn: Use additive attention.
        pairwise_featurization: Use pairwise features.
        num_heads: Number of attention heads.
        ion_info: Include ion type as input feature.
        instrument_info: Include instrument type as input feature.
        num_valid_ion: Number of valid ion types.
        num_valid_instrument: Number of valid instrument types.
    """

    def __init__(
        self,
        form_encoder: str = "abs-sines",
        hidden_size: int = 256,
        peak_attn_layers: int = 2,
        set_pooling: str = "cls",
        spectra_dropout: float = 0.1,
        additive_attn: bool = False,
        pairwise_featurization: bool = False,
        num_heads: int = 8,
        ion_info: bool = False,
        instrument_info: bool = False,
        num_valid_ion: int = len(ION_LST),
        num_valid_instrument: int = MAX_INSTR_IDX,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.set_pooling = set_pooling
        self.ion_info = ion_info
        self.instrument_info = instrument_info
        self.pairwise_featurization = pairwise_featurization

        self.form_encoder_mod = get_embedder(form_encoder)
        self.formula_dim = self.form_encoder_mod.full_dim
        self.cls_type = 1  # matches mist_cf_data.cls_type in original

        input_dim = self.formula_dim * 2 + 1 + 1 + 1 + 1  # form, diff, cls_flag, inten, num_peak, rel_mass_diff
        if ion_info:
            input_dim += num_valid_ion
        if instrument_info:
            input_dim += num_valid_instrument

        self.formula_encoder = MLPBlocks(
            input_size=input_dim,
            hidden_size=hidden_size,
            dropout=spectra_dropout,
            num_layers=2,
        )

        if pairwise_featurization:
            self.pairwise_featurizer = MLPBlocks(
                input_size=self.formula_dim,
                hidden_size=hidden_size,
                dropout=spectra_dropout,
                num_layers=2,
            )

        peak_attn_layer = transformer_layer.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=spectra_dropout,
            additive_attn=additive_attn,
            pairwise_featurization=pairwise_featurization,
        )
        self.peak_attn_layers = _get_clones(peak_attn_layer, peak_attn_layers)

    def forward(
        self,
        num_peaks,
        peak_types,
        form_vec,
        ion_vec,
        instrument_vec,
        intens,
        rel_mass_diffs,
        return_aux=False,
    ):
        device = num_peaks.device
        batch_size, peak_dim, _form_dim = form_vec.shape

        cls_type_mask = peak_types == self.cls_type
        cls_tokens = form_vec[cls_type_mask]
        diff_vec = cls_tokens[:, None, :] - form_vec

        diff_embedded = self.form_encoder_mod(diff_vec)
        form_embedded = self.form_encoder_mod(form_vec)
        cat_input = [form_embedded, diff_embedded, cls_type_mask[:, :, None].float()]

        if self.ion_info:
            cat_input.append(ion_vec)
        if self.instrument_info:
            cat_input.append(instrument_vec)

        inten_tensor = intens[:, :, None]
        rel_mass_diff_tensor = rel_mass_diffs[:, :, None]
        num_peak_feat = num_peaks[:, None, None].expand(batch_size, peak_dim, 1) / 10.0

        cat_input.extend([inten_tensor, num_peak_feat, rel_mass_diff_tensor])
        input_vec = torch.cat(cat_input, -1)
        peak_tensor = self.formula_encoder(input_vec)

        peak_tensor = peak_tensor.transpose(0, 1)
        peaks_aranged = torch.arange(peak_dim, device=device)
        attn_mask = ~(peaks_aranged[None, :] < num_peaks[:, None])

        pairwise_features = None
        if self.pairwise_featurization:
            form_diffs = form_vec[:, None, :, :] - form_vec[:, :, None, :]
            same_sign = torch.all(form_diffs >= 0, -1) | torch.all(form_diffs <= 0, -1)
            form_diffs[~same_sign].fill_(0)
            form_diffs = torch.abs(form_diffs)
            encoded_diffs = self.form_encoder_mod(form_diffs)
            pairwise_features = self.pairwise_featurizer(encoded_diffs)

        for layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = layer(
                peak_tensor,
                pairwise_features=pairwise_features,
                src_key_padding_mask=attn_mask,
            )

        output = self._pool_out(
            peak_tensor, inten_tensor, rel_mass_diff_tensor,
            peak_types, attn_mask, batch_size,
        )

        if return_aux:
            return output, {}
        return output

    def _pool_out(self, peak_tensor, inten_tensor, rel_mass_diff_tensor,
                  peak_types, attn_mask, batch_size):

        """_pool_out.

        pool the output of the network

        Return:
            (output (B x H), peak_tensor : L x B x H)

        """
        EPS = 1e-22

        zero_mask = attn_mask[:, :, None].repeat(1, 1, self.hidden_size).transpose(0, 1)
        peak_tensor[zero_mask] = 0

        if self.set_pooling == "cls":
            pool_factor = (peak_types == self.cls_type).float()
        elif self.set_pooling == "intensity":
            inten_flat = inten_tensor.reshape(batch_size, -1)
            intensities_sum = inten_flat.sum(1).reshape(-1, 1) + EPS
            pool_factor = (inten_flat / intensities_sum) * ~attn_mask
        elif self.set_pooling == "mean":
            inten_flat = inten_tensor.reshape(batch_size, -1)
            pool_factor = torch.clone(inten_flat).fill_(1)
            pool_factor = pool_factor * ~attn_mask
            pool_factor[pool_factor == 0] = 1
            pool_factor = pool_factor / pool_factor.sum(1).reshape(-1, 1)
        elif self.set_pooling == "rel_mass_diff":
            rel_mass_diff_tensor = rel_mass_diff_tensor.reshape(batch_size, -1)
            pool_factor = torch.clone(rel_mass_diff_tensor).fill_(1)
            pool_factor = pool_factor * ~attn_mask
            pool_factor[pool_factor == 0] = 1
            pool_factor = pool_factor / pool_factor.sum(1).reshape(-1, 1)
        else:
            raise NotImplementedError(f"Unknown pooling: {self.set_pooling}")

        output = torch.einsum("nbd,bn->bd", peak_tensor, pool_factor)
        return output


class MistCFNet(nn.Module):
    """MIST-CF scoring network.

    Combines MistCFFormulaTransformer with a linear scoring head.

    Args:
        hidden_size: Hidden dimension.
        layers: Number of transformer layers.
        dropout: Dropout rate.
        ion_info: Include ion type features.
        instrument_info: Include instrument type features.
        cls_mass_diff: Include mass diff in cls token.
        form_encoder: Formula embedder name.
        max_subpeak: Maximum number of subpeaks.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
        ion_info: bool = False,
        instrument_info: bool = False,
        cls_mass_diff: bool = False,
        form_encoder: str = "abs-sines",
        max_subpeak: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_subpeak = max_subpeak

        self.xformer = MistCFFormulaTransformer(
            form_encoder=form_encoder,
            hidden_size=hidden_size,
            peak_attn_layers=layers,
            set_pooling="cls",
            spectra_dropout=dropout,
            ion_info=ion_info,
            instrument_info=instrument_info,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(
        self,
        num_peaks,
        peak_types,
        form_vec,
        ion_vec,
        instrument_vec,
        intens,
        rel_mass_diffs,
    ):
        """Score a batch of (spectrum, candidate_formula) pairs.

        Returns:
            Tensor of shape [batch] with scores for each candidate.
        """
        output = self.xformer(
            num_peaks, 
            peak_types, 
            form_vec, 
            ion_vec, 
            instrument_vec, 
            intens, 
            rel_mass_diffs,
            return_aux=False,
        )
        return self.output_layer(output)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "MistCFNet":
        """Load MistCFNet from a checkpoint.

        Handles both plain state_dict files and PyTorch Lightning
        checkpoint format (which wraps the state dict under a state_dict
        key and prefixes keys with model. or xformer.).

        Args:
            checkpoint_path: Path to the checkpoint file.
            **kwargs: Forwarded to MistCFNet.__init__ when hparams are
                not available in the checkpoint.

        Returns:
            MistCFNet with loaded weights in eval mode.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Extract hparams if saved by Lightning
        hparams = ckpt.get("hyper_parameters", {})
        hparams.update(kwargs)

        model = cls(**hparams)

        state_dict = ckpt.get("state_dict", ckpt)
        # Lightning prefixes keys with the attribute name (e.g. "xformer.")
        # but may also have a top-level "model." prefix — strip both.
        stripped = {}
        for k, v in state_dict.items():
            # Remove leading "model." if present
            new_k = k[len("model."):] if k.startswith("model.") else k
            stripped[new_k] = v

        missing, unexpected = model.load_state_dict(stripped, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).warning(
                f"MistCFNet.from_pretrained: missing keys: {missing}"
            )
        if unexpected:
            import logging
            logging.getLogger(__name__).warning(
                f"MistCFNet.from_pretrained: unexpected keys: {unexpected}"
            )
        return model.eval()
