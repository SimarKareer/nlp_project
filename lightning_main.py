from luke_utils import read_GloVe, load_documents, load_examples, train_loop, read_conll_format
from collections import Counter
from transformers import LukeTokenizer, LukeConfig, LukeForEntitySpanClassification
from pytorch_memlab import LineProfiler
from fine_tuning import LUKE, NERDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

def main():
    
    tag2i = {'B-ORG': 3, 'B-LOC': 4, 'I-PER': 2, 'O': 0, 'I-MISC': 1, 'B-MISC': 1, 'I-ORG': 3, 'I-LOC': 4}

    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
    train_documents = load_documents("eng.train")
    train_examples = load_examples(tokenizer, train_documents)
    
    val_documents = load_documents("eng.testa")
    val_examples = load_examples(tokenizer, val_documents)

    test_documents = load_documents("eng.testb")
    test_examples = load_examples(tokenizer, test_documents)

    train_ds = NERDataset(train_examples, tokenizer, tag2i)
    val_ds = NERDataset(val_examples, tokenizer, tag2i)
    test_ds = NERDataset(test_examples, tokenizer, tag2i)


    model = LUKE(train_ds, val_ds, test_ds)

# wandb_logger = WandbLogger(name='luke-first-run-12000-articles-bis', project='LUKE')
# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
# early_stop_callback = EarlyStopping(
#     monitor='val_loss',
#     patience=2,
#     strict=False,
#     verbose=False,
#     mode='min'
# )

    trainer = Trainer(gpus=1)#, logger=wandb_logger, callbacks=[EarlyStopping(monitor='validation_loss')])
    trainer.fit(model)



main()