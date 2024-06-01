import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel


class SmilesTransformer(DeNovoMassSpecGymModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        smiles_tokenizer,
        max_len: int = 100,
        k_predictions: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.smiles_tokenizer = smiles_tokenizer
        self.vocab_size = smiles_tokenizer.get_vocab_size()
        self.src_encoder = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(self.vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            norm_first=True,  # TODO: maybe post-norm will be stable with more parameters.
        )
        self.decoder = nn.Linear(d_model, self.vocab_size)
        self.d_model = d_model
        self.max_len = max_len
        self.k_predictions = k_predictions
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        src = self.src_encoder(src) * (
            self.d_model**0.5
        )  # (seq_len, batch_size, d_model)
        tgt = self.tgt_embedding(tgt) * (
            self.d_model**0.5
        )  # (seq_len, batch_size, d_model)

        memory = self.transformer.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )
        output = self.transformer.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.decoder(output)  # (seq_len, batch_size, vocab_size)
        return output

    def step(
        self, batch: dict, metric_pref: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = batch["spec"].float()  # (batch_size, seq_len , in_dim)
        smiles = batch["mol"]  # List of SMILES of length batch_size

        smiles = self.smiles_tokenizer.encode_batch(smiles)
        smiles = [s.ids for s in smiles]
        smiles = torch.tensor(smiles)  # (batch_size, seq_len)

        # Generating padding masks for variable-length sequences
        src_key_padding_mask = self.generate_src_padding_mask(spec)
        tgt_key_padding_mask = self.generate_tgt_padding_mask(smiles)

        # Create target mask (causal mask)
        tgt_seq_len = smiles.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            smiles.device
        )

        spec = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
        smiles = smiles.permute(1, 0)  # (seq_len, batch_size)

        smiles_pred = self(
            src=spec,
            tgt=smiles[:-1, :],
            tgt_mask=tgt_mask[:-1, :-1],
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
        )

        loss = self.criterion(
            smiles_pred.view(-1, self.vocab_size), smiles[1:, :].contiguous().view(-1)
        )
        return dict(loss=loss, mols_pred=None)

    def validation_step(self, batch: dict, batch_idx: torch.Tensor) -> tuple:
        outputs = self.step(batch)
        spec = batch["spec"].float()  # (batch_size, seq_len, in_dim)

        start_symbol = self.smiles_tokenizer.token_to_id("<s>")  # TODO: no hardcoded value

        decoded_smiles_str = []
        for i in range(self.k_predictions):
            decoded_smiles = self.greedy_decode(
                spec,
                max_len=self.max_len,
                start_symbol=start_symbol,
                temperature=self.temperature,
            )
            decoded_smiles_str.append(
                [self.smiles_tokenizer.decode(seq.tolist()) for seq in decoded_smiles]
            )

        # Transpose from (k, batch_size) to (batch_size, k)
        decoded_smiles_str = list(map(list, zip(*decoded_smiles_str)))

        return dict(loss=outputs["loss"], mols_pred=decoded_smiles_str)

    def generate_src_padding_mask(self, spec):
        return spec.sum(-1) == 0

    def generate_tgt_padding_mask(self, smiles):
        return smiles == self.smiles_tokenizer.token_to_id("<pad>")  # TODO: no hardocded value

    def greedy_decode(self, spec, max_len, start_symbol, temperature=1.0):
        # Ensure the input shape is (batch_size, seq_len, in_dim)
        assert (
            len(spec.shape) == 3
        ), "spec input should have shape (batch_size, seq_len, in_dim)"

        # Permute spec to match the expected shape (seq_len, batch_size, in_dim)
        spec = spec.permute(1, 0, 2)

        # Encode the source sequence
        src = self.src_encoder(spec) * (
            self.d_model**0.5
        )  # (seq_len, batch_size, d_model)
        memory = self.transformer.encoder(src)

        batch_size = src.size(1)
        ys = (
            torch.ones(1, batch_size)
            .fill_(start_symbol)
            .type(torch.long)
            .to(spec.device)
        )

        for i in range(max_len - 1):
            tgt = self.tgt_embedding(ys) * (self.d_model**0.5)
            out = self.transformer.decoder(tgt, memory)
            out = self.decoder(out[-1, :])  # (batch_size, vocab_size)

            # Apply softmax with temperature to get probabilities
            probs = F.softmax(out / temperature, dim=-1)

            # Sample from the probability distribution
            next_word = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size,)

            next_word = next_word.unsqueeze(0)  # (1, batch_size)

            ys = torch.cat([ys, next_word], dim=0)
            if torch.all(next_word == self.smiles_tokenizer.token_to_id("</s>")):
                break

        return ys.permute(1, 0)  # (batch_size, seq_len)
