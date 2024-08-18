import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T
from torch_geometric.nn import MLP
from massspecgym.models.tokenizers import SpecialTokensBaseTokenizer
from massspecgym.data.transforms import MolToFormulaVector
from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


class SmilesTransformer(DeNovoMassSpecGymModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        smiles_tokenizer: SpecialTokensBaseTokenizer,
        start_token: str = SOS_TOKEN,
        end_token: str = EOS_TOKEN,
        pad_token: str = PAD_TOKEN,
        dropout: float = 0.1,
        max_smiles_len: int = 200,
        k_predictions: int = 1,
        temperature: T.Optional[float] = 1.0,
        pre_norm: bool = False,
        chemical_formula: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.smiles_tokenizer = smiles_tokenizer
        self.vocab_size = smiles_tokenizer.get_vocab_size()
        for token in [start_token, end_token, pad_token]:
            assert token in smiles_tokenizer.get_vocab(), f"Token {token} not found in tokenizer vocabulary."
        self.start_token_id = smiles_tokenizer.token_to_id(start_token)
        self.end_token_id = smiles_tokenizer.token_to_id(end_token)
        self.pad_token_id = smiles_tokenizer.token_to_id(pad_token)

        self.d_model = d_model
        self.max_smiles_len = max_smiles_len
        self.k_predictions = k_predictions
        self.temperature = temperature
        if self.k_predictions == 1:  # TODO: this logic should be changed because sampling with k = 1 also makes sense
            self.temperature = None

        self.src_encoder = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(self.vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            norm_first=pre_norm
        )
        self.tgt_decoder = nn.Linear(d_model, self.vocab_size)

        self.chemical_formula = chemical_formula
        if self.chemical_formula:
            self.formula_mlp = MLP(
                in_channels=MolToFormulaVector.num_elements(),
                hidden_channels=MolToFormulaVector.num_elements(),
                out_channels=d_model,
                num_layers=1,
                dropout=dropout,
                norm=None
            )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        
        spec = batch["spec"]  # (batch_size, seq_len, in_dim)
        smiles = batch["mol"]  # List of SMILES of length batch_size

        smiles = self.smiles_tokenizer.encode_batch(smiles)
        smiles = [s.ids for s in smiles]
        smiles = torch.tensor(smiles, device=spec.device)  # (batch_size, seq_len)

        # Generating padding masks for variable-length sequences
        src_key_padding_mask = self.generate_src_padding_mask(spec)
        tgt_key_padding_mask = self.generate_tgt_padding_mask(smiles)

        # Create target mask (causal mask)
        tgt_seq_len = smiles.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(smiles.device)

        # Preapre inputs for transformer teacher forcing
        src = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
        smiles = smiles.permute(1, 0)  # (seq_len, batch_size)
        tgt = smiles[:-1, :]
        tgt_mask = tgt_mask[:-1, :-1]
        src_key_padding_mask = src_key_padding_mask
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

        # Input and output embeddings
        src = self.src_encoder(src)  # (seq_len, batch_size, d_model)
        if self.chemical_formula:
            formula_emb = self.formula_mlp(batch["formula"])  # (batch_size, d_model)
            src = src + formula_emb.unsqueeze(0)  # (seq_len, batch_size, d_model) + (1, batch_size, d_model)
        src = src * (self.d_model**0.5)
        tgt = self.tgt_embedding(tgt) * (self.d_model**0.5)  # (seq_len, batch_size, d_model)

        # Transformer forward pass
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # Logits to vocabulary
        output = self.tgt_decoder(output)  # (seq_len, batch_size, vocab_size)

        # Reshape before returning
        smiles_pred = output.view(-1, self.vocab_size)
        smiles = smiles[1:, :].contiguous().view(-1)
        return smiles_pred, smiles

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:

        # Forward pass
        smiles_pred, smiles = self.forward(batch)

        # Compute loss
        loss = self.criterion(smiles_pred, smiles)

        # Generate SMILES strings
        if stage in self.log_only_loss_at_stages:
            mols_pred = None
        else:
            mols_pred = self.decode_smiles(batch)

        return dict(loss=loss, mols_pred=mols_pred)

    def generate_src_padding_mask(self, spec):
        return spec.sum(-1) == 0

    def generate_tgt_padding_mask(self, smiles):
        return smiles == self.pad_token_id

    def decode_smiles(self, spec):

        decoded_smiles_str = []
        for _ in range(self.k_predictions):
            decoded_smiles = self.greedy_decode(
                spec,  # (batch_size, seq_len, in_dim) 
                max_len=self.max_smiles_len,
                temperature=self.temperature,
            )

            decoded_smiles = [seq.tolist() for seq in decoded_smiles]
            decoded_smiles_str.append(self.smiles_tokenizer.decode_batch(decoded_smiles))

        # Transpose from (k, batch_size) to (batch_size, k)
        decoded_smiles_str = list(map(list, zip(*decoded_smiles_str)))

        return decoded_smiles_str

    def greedy_decode(self, batch, max_len, temperature):

        with torch.inference_mode():

            spec = batch["spec"]
            src_key_padding_mask = self.generate_src_padding_mask(spec)   

            spec = spec.permute(1, 0, 2)  # (seq_len, batch_size, in_dim)
            src = self.src_encoder(spec)  # (seq_len, batch_size, d_model)
            if self.chemical_formula:
                formula_emb = self.formula_mlp(batch["formula"])  # (batch_size, d_model)
                src = src + formula_emb.unsqueeze(0)  # (seq_len, batch_size, d_model) + (1, batch_size, d_model)
            src = src * (self.d_model**0.5)
            memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask,)

            batch_size = src.size(1)
            out_tokens = torch.ones(1, batch_size).fill_(self.start_token_id).type(torch.long).to(spec.device)

            for _ in range(max_len - 1):
                tgt = self.tgt_embedding(out_tokens) * (self.d_model**0.5)
                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
                out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
                out = self.tgt_decoder(out[-1, :])  # (batch_size, vocab_size)

                # Select next token
                if self.temperature is None:
                    probs = F.softmax(out, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)  # (batch_size,)
                else:
                    probs = F.softmax(out / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # (batch_size,)

                next_token = next_token.unsqueeze(0)  # (1, batch_size)

                out_tokens = torch.cat([out_tokens, next_token], dim=0)
                if torch.all(next_token == self.end_token_id):
                    break

            out_tokens = out_tokens.permute(1, 0)  # (batch_size, seq_len)
            return out_tokens  
