import math

from custom_logging import Logger

import lightning as L

import torch.nn as nn
import torch
import torch.nn.functional as F

from config import HyperParams
import torchmetrics


class Head(nn.Module):
    """one head of self-attention """

    def __init__(self, params: HyperParams):
        super().__init__()
        self.key = nn.Linear(params.n_embd, params.n_embd // params.n_head, bias=False)
        self.query = nn.Linear(params.n_embd, params.n_embd // params.n_head, bias=False)
        self.value = nn.Linear(params.n_embd, params.n_embd // params.n_head, bias=False)

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, params: HyperParams):
        super().__init__()
        self.heads = nn.ModuleList([Head(params) for _ in range(params.n_head)])
        self.proj = nn.Linear(params.n_embd, params.n_embd)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, params: HyperParams):
        super().__init__()
        # Define individual layers
        self.l1 = nn.Linear(params.n_embd, params.n_hidden)
        self.batchnorm1 = nn.BatchNorm1d(params.context_length)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(params.dropout)
        self.l2 = nn.Linear(params.n_hidden, params.n_embd)
        self.batchnorm2 = nn.BatchNorm1d(params.context_length)
        self.elu2 = nn.ELU()

    def forward(self, x):
        # Manually apply each layer
        x = self.l1(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.l2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, params: HyperParams):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.lnorm1 = nn.LayerNorm(params.n_embd)
        self.dropout1 = nn.Dropout(p=params.dropout)
        self.sa = MultiHeadAttention(params)
        self.lnorm2 = nn.LayerNorm(params.n_embd)
        self.ffwd = FeedFoward(params)
        self.dropout2 = nn.Dropout(p=params.dropout)

    def forward(self, x):
        # self-attention
        _x = x
        x = self.sa(x)
        x = self.dropout1(x)
        x = self.lnorm1(_x + x)

        # feedforward
        _x = x
        x = self.ffwd(x)
        x = self.dropout2(x)
        x = self.lnorm2(_x + x)
        return x

class TransfEncModel(L.LightningModule):
    def __init__(self, params: HyperParams, n_classes=2):
        super().__init__()
        self.save_hyperparameters()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.lossi = []
        self.custom_logger: Logger = None
        self.params = params

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(params.vocab_size, params.n_embd)
        self.token_embedding_table.weight = nn.Parameter(self.token_embedding_table.weight // (params.n_embd ** 0.5), requires_grad=True)

        # Create sinusoidal positional embedding
        self.position_embedding_table = self.create_sinusoidal_positional_embedding(params.context_length, params.n_embd)

        self.blocks = nn.Sequential(*[Block(params) for _ in range(self.params.n_blocks)])
        self.lnorm_f = nn.LayerNorm(params.n_embd)  # final layer norm
        self.l_out = nn.Linear(params.n_embd, n_classes)
        self.l_out.weight = nn.Parameter(self.l_out.weight * 0.1, requires_grad=True)

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.predictions = []
        self.labels = []

    def create_sinusoidal_positional_embedding(self, seq_len, embd, n = 10000):
        """Create sinusoidal positional embedding matrix as per Vaswani et al."""

        if embd % 2 != 0:
            raise ValueError(
                "Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(
                    embd))

        T = seq_len
        d = embd  # d_model=head_num*d_k, not d_q, d_k, d_v

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2 * torch.arange(0, d // 2) / d)  # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions / denominators)  # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions / denominators)  # cos(pos/10000^(2i/d_model))

        return nn.Parameter(embeddings, requires_grad=False)

    def set_custom_logger(self, logger: Logger):
        self.custom_logger = logger

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.params.lr, weight_decay=0.01)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table  # (1, T, C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.lnorm_f(x)  # (B,T,C)
        # B, T, C = x.shape
        # x = x.view(B, T * C)  # (B,T,C) -> (B, T*C)
        logits = self.l_out(x)  # (B,n_classes)
        logits = logits[:, -1, :]  # becomes (B, C)

        return logits

    def set_val_data_loader(self, val_dataloader):
        self.val_dataloader = val_dataloader

    def compute_val_loss(self):
        # Function to compute validation loss before any weights are updated
        val_loss = 0.0
        num_batches = 0
        for batch in self.val_dataloader:
            inputs, target = batch
            output = self(inputs)
            loss = torch.nn.functional.cross_entropy(output, target.view(-1))
            val_loss += loss.item()
            num_batches += 1
        val_loss /= num_batches
        return val_loss

    def on_train_start(self) -> None:
        self.custom_logger.log_model_arch()
        self.custom_logger.log_params()
        # Compute validation loss before any weights are updated
        if self.val_dataloader:
            val_loss = self.compute_val_loss()  # You need to implement this method
            self.logger.experiment.add_scalar("val_loss", val_loss, self.global_step)
            
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        opt = self.optimizers()
        opt.zero_grad()
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        # Call backward with retain_graph=True
        self.manual_backward(loss, retain_graph=True)
        # Ensure that logger has the log_ud method
        self.training_step_log()
        opt.step()
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss
    
    def training_step_log(self):
        if self.custom_logger:
            self.custom_logger.log_ud()
            self.custom_logger.log_activation_out_sat()

    def on_train_epoch_end(self) -> None:
        if self.custom_logger:
            self.custom_logger.log_out_on_epoch()

    def validation_step(self, batch, batch_idx):
        inputs, y = batch
        logits = self(inputs)
        
        # Update metrics
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, y)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, y.view(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # Log metrics to TensorBoard
        self.log('val_accuracy', self.accuracy.compute(), on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(y.cpu().numpy())

        # Update metrics
        self.accuracy.update(preds, y)
        return {'test_loss': F.cross_entropy(logits, y)}
    
    def on_test_epoch_end(self):
        # Log metrics to TensorBoard
        self.log('test_accuracy', self.accuracy.compute(), on_epoch=True)
