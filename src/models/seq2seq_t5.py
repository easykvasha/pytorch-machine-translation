import torch
import metrics
from transformers.optimization import Adafactor
from transformers import T5ForConditionalGeneration


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, model_config, device, tokenizer):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
        self.model.resize_token_embeddings(len(tokenizer))
        self.optimizer = Adafactor(self.model.parameters(), lr=model_config["learning_rate"], relative_step=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)

    def forward(self, batch):
        output = self.model.generate(input_ids=batch[0], attention_mask=batch[1], max_new_tokens=15)

        return output


    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2]).loss

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()


    def validation_step(self, batch):
        self.model.eval()
        result_dict = dict()
        with torch.no_grad():
            output = self.model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = output.loss
        result_dict["loss"] = loss.item()

        decoded_result_list = torch.argmax(output.logits, dim=-1).cpu().numpy()

        decoded_query_list = list()
        for sample in decoded_result_list:
            decoded_query_tokens = self.tokenizer.decode(sample)
            query = " ".join(decoded_query_tokens)
            decoded_query_list.append(query)

        result_dict['predicted_query'] = decoded_query_list
        return loss.item()


    def eval_bleu(self, source, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, source_sentences, actual_sentences, predicted_sentences = metrics.bleu_scorer(source=source,
            predicted=predicted, actual=actuals, target_tokenizer=self.tokenizer, source_tokenizer=self.tokenizer
        )
        return bleu_score, source_sentences, actual_sentences, predicted_sentences




