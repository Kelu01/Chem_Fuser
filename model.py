import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, dropout=0.1, max_len=140, num_steps=1000, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_steps = num_steps
        self.pad_token_id = pad_token_id

        # Embedding layers
        self.token_embed = nn.Embedding(vocab_size, d_model)     # Token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)          # Positional embedding (like in transformers)
        self.time_embed = nn.Embedding(num_steps, d_model)       # Diffusion timestep embedding

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,                          # Number of attention heads
            dim_feedforward=4 * d_model,            # Feedforward layer size inside each transformer block
            dropout=dropout,
            batch_first=True                        # Input format: (batch, seq_len, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final linear projection from hidden to vocab space
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, t=None):
        """
        Forward pass through the transformer model.
        Combines token, position, and time embeddings, and outputs logits for each token.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Positional indices (0 to T-1)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        tok_emb = self.token_embed(input_ids)           # (B, T, d_model)
        pos_emb = self.pos_embed(positions)             # (B, T, d_model)

        # Embed timestep t (for diffusion awareness)
        if t is not None:
            t = t.to(device)
            t_ids = (t * (self.num_steps - 1)).long()   # Scale to integer ID range
            t_emb = self.time_embed(t_ids).unsqueeze(1) # (B, 1, d_model)
        else:
            t_emb = torch.zeros_like(tok_emb[:, :1])    # If no t, default to zeros

        x = tok_emb + pos_emb + t_emb                   # (B, T, d_model)

        # Create mask(attention) for [PAD] tokens
        pad_mask = input_ids == self.pad_token_id

        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Final prediction layer: (B, T, vocab_size)
        logits = self.output_head(x)
        return logits


def forward_diffusion(x0, t, mask_token_id, pad_token_id):
    """
    Applies forward diffusion to input sequence x0:
    randomly masks tokens based on a masking probability t.
    """
    device = x0.device
    B, T = x0.shape

    # Make t a tensor if it's a scalar
    if isinstance(t, float) or isinstance(t, int):
        t = torch.full((B, 1), t, device=device)

    t = t.to(device)

    # Expand t to shape (B, T) if needed
    if t.dim() == 1:
        t = t.unsqueeze(1).expand(B, T)

    # Generate a random boolean mask based on t
    prob_mask = torch.rand(B, T, device=device) < t

    # Prevent masking([MASK]) to [PAD] tokens
    non_pad_mask = (x0 != pad_token_id)
    final_mask = prob_mask & non_pad_mask

    # Apply mask
    xt = x0.clone()
    xt[final_mask] = mask_token_id

    return xt, final_mask


class DiffusionModel(nn.Module):
    def __init__(self, voc, model_config=None):
        super().__init__()
        self.voc = voc
        self.mask_token = voc.vocab['[MASK]']
        self.pad_token = voc.vocab['[PAD]'] 

        if model_config is None:
            model_config = {}

        # Initialize the main transformer model
        self.model = DiffusionTransformer(
            vocab_size=voc.vocab_size,
            d_model=model_config.get("d_model", 256),      # Embedding and hidden size
            n_heads=model_config.get("n_heads", 8),         # Attention heads
            n_layers=model_config.get("n_layers", 7),       # Transformer layers
            dropout=model_config.get("dropout", 0.1),
            max_len=model_config.get("max_len", 140),       # Max sequence length
            num_steps=model_config.get("num_steps", 1000),  # Total diffusion steps
            pad_token_id=self.pad_token
        ).to(device)

    def likelihood(self, x0, t):
        """
        Compute the average negative log-likelihood over masked tokens.
        Returns the mean scaled loss and the number of masked tokens.
        """
        device = next(self.parameters()).device
        x0 = x0.to(device)
        t = t.to(device)

        # Forward diffusion (apply noise/masking)
        x_t, mask = forward_diffusion(x0, t, self.mask_token, self.pad_token)

        # Run transformer and get token-level predictions
        logits = self.model(x_t, t=t)
        log_probs = F.log_softmax(logits, dim=-1)

        # Only calculate loss for positions that were masked
        masked_log_probs = log_probs[mask].clone()
        masked_targets = x0[mask].clone()
        loss = F.nll_loss(masked_log_probs, masked_targets, reduction='none')

        # Optionally scale loss by timestep
        if t.dim() == 1:
            t = t.unsqueeze(1).expand_as(x0)
        t_masked = t[mask].clone()
        scaled_loss = loss / (t_masked + 1e-6)

        return scaled_loss.mean(), mask.sum().item()

    def sample(self, voc, batch, scheduler=None, current_step=None, num_show=5):
        """
        Takes a batch of sequences, masks them partially,
        predicts unmasked sequences, and returns both masked and recovered SMILES strings.
        """
        self.eval()
        with torch.no_grad():
            x0 = batch[:num_show].to(device)

            # Decide timestep for masking
            if scheduler is not None and current_step is not None:
                t_val = scheduler.get_t(current_step)
            else:
                t_val = 0.3  # default mask level

            t = torch.full((x0.size(0),), t_val, device=device)

            # Apply forward diffusion (mask)
            x_t, mask = forward_diffusion(x0, t, self.mask_token, self.pad_token)

            # Predict masked tokens
            logits = self.model(x_t, t=t)
            probs = F.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

            # Replace only masked positions with predictions
            unmasked = x_t.clone()
            unmasked[mask] = pred_ids[mask]

            pad_idx = voc.vocab['[PAD]']

            # Helper to remove PADs before decoding
            def strip_pad(seq):
                return [i for i in seq if i != pad_idx]

            # Decode to SMILES
            masked_smiles = [voc.decode(strip_pad(x_t[i].tolist())) for i in range(x_t.size(0))]
            unmasked_smiles = [voc.decode(strip_pad(unmasked[i].tolist())) for i in range(unmasked.size(0))]

            return masked_smiles, unmasked_smiles


class MaskingScheduler:
    def __init__(self, total_steps, min_mask=0.15, max_mask=1.0, schedule='cosine'):
        self.total_steps = total_steps
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.schedule = schedule

    def get_t(self, current_step):
        """
        Returns masking probability for the current training step,
        using one of several scheduling functions.
        """
        progress = current_step / self.total_steps
        if self.schedule == 'cosine':
            # Slowly increases then plateaus
            return self.min_mask + (self.max_mask - self.min_mask) * (1 - math.cos(math.pi * progress)) / 2
        elif self.schedule == 'linear':
            return self.min_mask + (self.max_mask - self.min_mask) * progress
        elif self.schedule == 'reverse_cosine':
            return self.max_mask - (self.max_mask - self.min_mask) * (1 - math.cos(math.pi * progress)) / 2
        elif self.schedule == 'reverse_linear':
            return self.max_mask - (self.max_mask - self.min_mask) * progress
        else:
            raise ValueError("Unknown masking schedule")