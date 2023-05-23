import sys
sys.path.append("./src")
import torch
import yaml
from models import trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger
from models.seq2seq_transformer import Seq2SeqTransformer


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    data_config = yaml.load(open("./configs/data_config.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader, src_vocab_size, tgt_vocab_size, target_tokenizer, source_tokenizer = dm.prepare_data()

    model_config = yaml.load(open("./configs/model_config.yaml", 'r'),   Loader=yaml.Loader)

    model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, model_config=model_config, target_tokenizer=target_tokenizer, source_tokenizer=source_tokenizer, device=DEVICE)

    logger = TXTLogger('./training_logs_seq2seq')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)

    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)




