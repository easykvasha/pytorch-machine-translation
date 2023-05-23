from torch.utils.data import DataLoader
from transformers import T5Tokenizer, AutoTokenizer
from data.mt_dataset import MTDataset
from data.utils import TextUtils, short_text_filter_function
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class DataManager:
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.input_lang_n_words = None
        self.output_lang_n_words = None
        self.device = device

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
        prefix_filter = self.config['prefix_filter']
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        source_sentences,target_sentences = [], []
        # dataset is ambiguous -> i lied -> я солгал/я соврала
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if short_text_filter_function(pair, self.config['max_length'], prefix_filter) and source not in unique_sources:
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)
        train_size = int(len(source_sentences)*self.config["train_size"])
        source_train_sentences, source_val_sentences = source_sentences[:train_size], source_sentences[train_size:]
        target_train_sentences, target_val_sentences = target_sentences[:train_size], target_sentences[train_size:]

        self.tokenizer_1 = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(vocab_size=10000)

        self.tokenizer_1.pre_tokenizer = Whitespace()
        self.tokenizer_1.train_from_iterator(source_sentences + target_sentences, trainer)
        print(len(self.tokenizer_1.get_vocab()))
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")

        # set_of_new_words = set()
        # for seq in target_sentences:
        #     for tok in seq.split():
        #         if 2 in self.tokenizer.encode(tok):
        #             set_of_new_words.update({tok})
        # print(len(set_of_new_words))

        print(len(self.tokenizer.get_vocab()))
        print(len(set(self.tokenizer_1.get_vocab()) - set(self.tokenizer.get_vocab())))
        self.tokenizer.add_tokens(list(set(self.tokenizer_1.get_vocab()) - set(self.tokenizer.get_vocab())))
        # self.tokenizer.add_tokens(list(set_of_new_words - set(self.tokenizer.get_vocab())))
        print(len(self.tokenizer.get_vocab()))

        tokenized_source_train_sentences = self.tokenizer(source_train_sentences, padding="longest", max_length=self.config['max_length'], truncation=True, return_tensors="pt")
        tokenized_source_val_sentences = self.tokenizer(source_val_sentences, padding="longest", max_length=self.config['max_length'], truncation=True, return_tensors="pt")
        tokenized_target_train_sentences = self.tokenizer(target_train_sentences, padding="longest", max_length=self.config['max_length'], truncation=True, return_tensors="pt")
        tokenized_target_val_sentences = self.tokenizer(target_val_sentences, padding="longest", max_length=self.config['max_length'], truncation=True, return_tensors="pt")

        train_dataset = MTDataset(tokenized_source_list=tokenized_source_train_sentences,
                                  tokenized_target_list=tokenized_target_train_sentences, dev=self.device)

        val_dataset = MTDataset(tokenized_source_list=tokenized_source_val_sentences,
                                tokenized_target_list=tokenized_target_val_sentences, dev=self.device)
        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      batch_size=self.config["batch_size"], drop_last=True
        )
        val_dataloader = DataLoader(val_dataset, shuffle=True,
                                    batch_size=self.config["batch_size"], drop_last=True)
        return train_dataloader, val_dataloader
