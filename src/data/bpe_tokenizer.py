from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import processors
import numpy as np


class BPETokenizer:
    def __init__(self, sentence_list):
        """
        sentence_list - список предложений для обучения
        """
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], vocab_size=10000)

        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train_from_iterator(sentence_list, trainer)

        sos_token_id = self.tokenizer.token_to_id("[SOS]")
        eos_token_id = self.tokenizer.token_to_id("[EOS]")

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[SOS]:0 $A:0 [EOS]:0",
            special_tokens=[("[EOS]", eos_token_id), ("[SOS]", sos_token_id)]
        )



    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        return self.pad_sequence(self.tokenizer.encode(sentence).ids)


    def pad_sequence(self, ids):
        ids = ids[:15]
        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        seq = np.ones(15)*pad_token_id
        seq[:len(ids)] = ids
        return seq

    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tokenizer.decode(token_list)