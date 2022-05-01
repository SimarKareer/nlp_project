from luke_utils import read_GloVe, load_documents, load_examples, train_loop, read_conll_format
from collections import Counter
from transformers import LukeTokenizer, LukeConfig, LukeForEntitySpanClassification
from pytorch_memlab import LineProfiler

def main():
    tag2i = {'B-ORG': 3, 'B-LOC': 4, 'I-PER': 2, 'O': 0, 'I-MISC': 1, 'B-MISC': 1, 'I-ORG': 3, 'I-LOC': 4}

    # (sentences_train, tags_train) = read_conll_format("train")
    # (sentences_dev, tags_dev)     = read_conll_format("dev")

    # wordCounts = Counter([w for l in sentences_train for w in l])
    # GloVe = read_GloVe("glove.840B.300d.conll_filtered.txt")

    # singletons = set([w for (w,c) in wordCounts.items() if c == 1 and not w in GloVe.keys()])
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
    test_documents = load_documents("eng.testb")
    test_examples = load_examples(tokenizer, test_documents)

    model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003").to("cuda")

    params = {'BATCH_SIZE': 1,
    'LR':1e-5,
    'WEIGHT_DECAY':0,
    'N_EPOCHS':3,
    'CLIP':1.0}
    model.to("cuda")


    with LineProfiler(train_loop) as prof:
        train_loop(params, model, tokenizer, test_examples, tag2i)
    
    print(prof.display())




main()