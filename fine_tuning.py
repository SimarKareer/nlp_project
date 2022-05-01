from lib2to3.pgen2 import token
from transformers import LukeTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import LukeForEntitySpanClassification, AdamW
import pytorch_lightning as pl
from luke_utils import compute_labels

# def pad_tensor(vec, pad, dim):
#     """
#     args:
#         vec - tensor to pad
#         pad - the size to pad to
#         dim - dimension to pad

#     return:
#         a new tensor padded to 'pad' in dimension 'dim'
#     """
#     pad_size = list(vec.shape)
#     pad_size[dim] = pad - vec.size(dim)
#     return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # print("Batch: ", len(batch))
        # # find longest sequence
        # max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # # pad according to max_len
        # batch = map(lambda x, y: (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # # stack all
        # xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        # ys = torch.LongTensor(map(lambda x: x[1], batch))
        # return xs, ys



        td_batch = {}

        for key in batch[0].keys():
            list_of_key = [i[key] for i in batch]
            print("First two shapes: ", list_of_key[0].shape, list_of_key[1].shape)
            padded = torch.nn.utils.rnn.pad_sequence(list_of_key, batch_first=True).type(torch.LongTensor).cuda()
            print("Key: ", key, "padded shape: ", padded.shape)
            td_batch[key] = padded
        
        return td_batch

    def __call__(self, batch):
        return self.pad_collate(batch)


class NERDataset(Dataset):
    """Relation extraction dataset."""

    def __init__(self, data, tokenizer, tag2i, device_str="cuda"):
        """
        Args:
            data : Pandas dataframe.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.tag2i = tag2i
        self.device_str = device_str

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # sentence = item.sentence
        # entity_spans = [tuple(x) for x in item.entity_spans]

        # encoding = tokenizer(sentence, entity_spans=entity_spans, padding="max_length", truncation=True, return_tensors="pt")

        # for k,v in encoding.items():
        #   encoding[k] = encoding[k].squeeze()

        # encoding["label"] = torch.tensor(label2id[item.string_id])

        texts = ex["text"]
        entity_spans = ex['entity_spans']
        labels = ex['labels']
        e_labels = torch.tensor(
                    list(compute_labels(labels, ex["original_word_spans"], self.tag2i)),
                    requires_grad=False
                ).cuda()
        
        # with torch.no_grad():
        tokenized_batch_examples = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True).to(self.device_str)
        e_labels_padded = e_labels.long().to(self.device_str) #torch.nn.utils.rnn.pad_sequence(e_labels, batch_first=True).type(torch.LongTensor).to(self.device_str)
        tokenized_batch_examples["label"] = e_labels_padded

        for key in tokenized_batch_examples.keys():
            # print("P1: ", key, tokenized_batch_examples[key].shape)
            if key != "label":
                assert(tokenized_batch_examples[key].shape[0] == 1)
                tokenized_batch_examples[key] = tokenized_batch_examples[key][0]

        return tokenized_batch_examples

class LUKE(pl.LightningModule):

    def __init__(self, train_ds, val_ds, test_ds):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003").to("cuda")

    def forward(self, model_dict):     
        # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, 
        #                      entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        # return outputs

        return self.model(**model_dict)
    
    def common_step(self, batch, batch_idx):
        labels = batch['label']
        del batch['label']
        print("batch: ", batch)
        outputs = self(batch)
        logits = outputs.logits

        loss = outputs.loss
        accuracy = 0

        # criterion = torch.nn.CrossEntropyLoss() # multi-class classification
        # loss = criterion(logits, labels)
        # predictions = logits.argmax(-1)
        # correct = (predictions == labels).sum().item()
        # accuracy = correct/batch['input_ids'].shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=4, shuffle=True, collate_fn=PadCollate(dim=0))
    #     return train_dataloader

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=2, collate_fn=PadCollate(dim=0))

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=2, collate_fn=PadCollate(dim=0))