import torch
from abc import ABC, abstractmethod
from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification, AutoTokenizer

class LLM(ABC, torch.nn.Module):
    """Abstract class for all LLMs."""

    def __init__(self, device: torch.device):
        super(LLM, self).__init__()
        self.device = device

    @property
    def embeddings(self) -> torch.Tensor:
        """Returns the token embeddings with gradients required."""
        return self.model.get_input_embeddings()

    @abstractmethod
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        pass

    @abstractmethod
    def inputs_embeds_forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Returns the output of the token embeddings."""
        pass

class BertModelWrapper(LLM):
    def __init__(self, out_dim: int, device: torch.device, model_name: str = 'google-bert/bert-base-uncased'):
        super(BertModelWrapper, self).__init__(device)
        self.config = BertConfig.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        # Get size of the last hidden state
        self.bert_last_layer_size = self.config.hidden_size
        middle_classification_head_layer_size = self.bert_last_layer_size // 2
        self.intermediate_layer = torch.nn.Linear(self.bert_last_layer_size, middle_classification_head_layer_size)
        self.act = torch.nn.ReLU()
        self.final_layer = torch.nn.Linear(middle_classification_head_layer_size, out_dim)
        self.probability_normalizer = torch.nn.Softmax(dim=-1) if out_dim > 1 else torch.nn.Sigmoid()

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-trained model forward pass
        pre_trained_y = self.model(token_ids, attention_mask=attention_mask).pooler_output

        # Classification head forward pass
        y = self.intermediate_layer(pre_trained_y)
        y = self.act(y)
        y = self.final_layer(y)
        output = self.probability_normalizer(y)

        # Apply softmax / sigmoid
        return output

    def inputs_embeds_forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.pooler_output
        x = self.intermediate_layer(pooled)
        x = self.act(x)
        logits = self.final_layer(x)
        return self.probability_normalizer(logits)

class Tokenizer(ABC):

    def __init__(self, spurious_words: list[str]):
        assert hasattr(self, 'tokenizer'), "Tokenizer not initialized."
        self.vocab = self.tokenizer.get_vocab()
        self.spurious_words_token_ids = torch.tensor([]).to(torch.int64)
        for s_w in spurious_words:
            s_w_token_id = list(self.tokenize(s_w)["input_ids"].flatten())[1:-1]
            self.spurious_words_token_ids = torch.cat((self.spurious_words_token_ids, torch.tensor(s_w_token_id)), dim=0).unique()

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenizes the input text."""
        pass

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        assert hasattr(self, 'tokenizer'), "Tokenizer not initialized."
        return self.tokenizer.vocab_size

    def get_tokens_given_ids(self, token_ids: torch.Tensor) -> list:
        assert hasattr(self, 'tokenizer'), "Tokenizer not initialized."
        batch_tokens = []
        for ids in token_ids:
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            batch_tokens.append(tokens)
        return batch_tokens

    def sample_token_ids_from_vocab(self, num_samples: int) -> torch.Tensor:
        """Samples token ids from the vocabulary."""
        token_id_indices = torch.randint(0, self.vocab_size, (num_samples,))
        return torch.tensor(list(self.vocab.values()))[token_id_indices]

    def sample_token_ids_from_spur_words(self, num_samples: int) -> torch.Tensor:
        """Samples token ids from the set of spurious words."""
        token_id_indices = torch.randint(0, len(self.spurious_words_token_ids), (num_samples,))
        return torch.tensor(self.spurious_words_token_ids)[token_id_indices]

class BertTokenizerWrapper(Tokenizer):
    def __init__(self, spurious_words: list[str], model_name: str = 'google-bert/bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        super(BertTokenizerWrapper, self).__init__(spurious_words)

    def tokenize(self, text: str) -> torch.Tensor:
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            # stride=128,
            max_length=512, # BERT max length
            return_overflowing_tokens=False # Not sure if we want this, though
        )
        return encoding
