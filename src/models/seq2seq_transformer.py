import torch
from torch import nn
from torch.nn import Transformer
from transformers.optimization import Adafactor
from models.positional_encoding import PositionalEncoding
import metrics
import math


def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz,  sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == 3).transpose(0, 1)
    tgt_padding_mask = (tgt == 3).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,  model_config, target_tokenizer, source_tokenizer, device='cuda', num_decoder_layers=3, num_encoder_layers=3, dropout=0.3):
        super(Seq2SeqTransformer, self).__init__()
        self.target_tokenizer = target_tokenizer
        self.source_tokenizer = source_tokenizer
        self.device = device
        dim_feedforward = model_config["hidden_size"]
        d_model = model_config["embedding_size"]
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(device)
        self.transformer = Transformer(d_model=d_model,
                                       nhead=8,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers,
                                     dim_feedforward=dim_feedforward,
                                     dropout=dropout).to(device)
        self.generator = nn.Linear(d_model, tgt_vocab_size).to(device)
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model).to(device)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.d_model = d_model
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                    lr=model_config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=125, gamma=0.99)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=3)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.pos_encoder(self.src_tok_emb(src))
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.pos_encoder(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.pos_encoder(self.tgt_tok_emb(tgt)), memory, tgt_mask)


    def training_step(self, batch):
        src, tgt = batch
        src = src.transpose(1, 0).to(self.device)
        tgt = tgt.transpose(1, 0).to(self.device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.device)

        logits = self.forward(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        self.optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_step(self, batch):
        src, tgt = batch
        src = src.transpose(1, 0).to(self.device)
        tgt = tgt.transpose(1, 0).to(self.device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.device)

        with torch.no_grad():
            logits = self.forward(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]

        loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        return loss.item()

    def eval_bleu(self, source, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        source = source.detach().cpu().numpy()
        bleu_score, source_sentences, actual_sentences, predicted_sentences = metrics.bleu_scorer(source=source,
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer, source_tokenizer=self.source_tokenizer
        )
        return bleu_score, source_sentences, actual_sentences, predicted_sentences

    def greedy_decode(self, src, max_len=15):
        num_tokens = src.shape[1]
        start_symbol = 1
        src = src.transpose(1, 0).to(self.device)
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(self.device)

        memory = self.encode(src, src_mask)
        ys = torch.ones(1, src.shape[1]).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len - 1):
            memory = memory.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), self.device).type(torch.bool)).to(self.device)
            out = self.decode(ys, memory, tgt_mask)
            prob = self.generator(out[-1, :])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(0).data
            ys = torch.cat([ys, next_word], dim=0)
        return ys





