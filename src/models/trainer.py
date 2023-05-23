from tqdm import tqdm


class Trainer:
    def __init__(self, model, model_config, logger):
        self.model = model
        self.epoch_num = model_config['epoch_num']
        self.logger = logger

        self.logger.log(model_config)

    def train(self, train_dataloader, val_dataloader):
        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.model.train()
                for batch in train_dataloader:
                    train_loss = self.model.training_step(batch)
                    train_epoch_loss += train_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss, val_epoch_bleu = 0, 0
                self.model.eval()
                for batch in val_dataloader:
                    val_loss = self.model.validation_step(batch)
                    val_epoch_loss += val_loss
                val_epoch_loss = val_epoch_loss / len(val_dataloader)

                # predicted_samples = self.model.forward(batch) # T5
                predicted_samples = self.model.greedy_decode(batch[0])
                # bleu_score, source_sentences, actual_sentences, predicted_sentences = self.model.eval_bleu(batch[0], [predicted_samples.cpu().detach()[:, i] for i in range(predicted_samples.shape[1])], batch[2]) # T5
                bleu_score, source_sentences, actual_sentences, predicted_sentences = self.model.eval_bleu(batch[0], predicted_samples, batch[1])

                print('Current BLEU: ', bleu_score)
                for i, (a, b) in zip(source_sentences[:10], zip(actual_sentences[:10], predicted_sentences[:10])):
                    print(f"{i} ---> {a} ---> {b}")
                print('##############################')

                self.logger.log({"train_loss": train_epoch_loss,
                                "val_loss": val_epoch_loss,
                                 "bleu_score": bleu_score})

        except KeyboardInterrupt:
            pass

        print(f"Last {epoch} epoch train loss: ", train_epoch_loss)
        print(f"Last {epoch} epoch val loss: ", val_epoch_loss)
        print(f"Last {epoch} epoch val bleu: ", bleu_score)
