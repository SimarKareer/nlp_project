from transformers import LukeTokenizer, LukeConfig, ReformerConfig, LukeForEntitySpanClassification, ReformerModel, LukePreTrainedModel
from transformers.models.luke.modeling_luke import BaseLukeModelOutputWithPooling, EntitySpanClassificationOutput
import torch
from typing import Dict, Optional, Tuple, Union
from transformers.models import reformer

from collections import Counter
import random

import unicodedata

import numpy as np
import seqeval.metrics
import spacy
from torch import optim
from tqdm import tqdm, trange
from transformers import LukeTokenizer
from pynvml import *
from pytorch_memlab import MemReporter


device_str = "cuda"

#Read in the training data
def read_conll_format(filename):
    (words, tags, currentSent, currentTags) = ([],[],['-START-'],['START'])
    for line in open(filename).readlines():
        line = line.strip()
        #print(line)
        if line == "":
            currentSent.append('-END-')
            currentTags.append('END')
            words.append(currentSent)
            tags.append(currentTags)
            (currentSent, currentTags) = (['-START-'], ['START'])
        else:
            (word, tag) = line.split()
            currentSent.append(word)
            currentTags.append(tag)
    return (words, tags)

def sentences2char(sentences):
    return [[['start'] + [c for c in w] + ['end'] for w in l] for l in sentences]

def read_GloVe(filename):
    embeddings = {}
    for line in open(filename).readlines():
        #print(line)
        fields = line.strip().split(" ")
        word = fields[0]
        embeddings[word] = [float(x) for x in fields[1:]]
    return embeddings

#When training, randomly replace singletons with UNK tokens sometimes to simulate situation at test time.
def getDictionaryRandomUnk(w, dictionary, singletons, train=False):
    if train and (w in singletons and random.random() > 0.5):
        return 1
    else:
        return dictionary.get(w, 1)

#Map a list of sentences from words to indices.
def sentences2indices(words, dictionary, singletons, train=False):
    #1.0 => UNK
    return [[getDictionaryRandomUnk(w,dictionary, singletons, train=train) for w in l] for l in words]

def sentences2indices_e(words, dictionary, singletons, train=False):
    #1.0 => UNK
    return [[[getDictionaryRandomUnk(w,dictionary, singletons, train=train) for w in l] for l in x] for x in words]


#Map a list of sentences containing to indices (character indices)
def sentences2indicesChar(chars, dictionary):
    #1.0 => UNK
    return [[[dictionary.get(c,1) for c in w] for w in l] for l in chars]

#Pad inputs to max sequence length (for batching)
def prepare_input(X_list):
    X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l) for l in X_list], batch_first=True).type(torch.LongTensor) # padding the sequences with 0
    X_mask   = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l)) for l in X_list], batch_first=True).type(torch.FloatTensor) # consisting of 0 and 1, 0 for padded positions, 1 for non-padded positions
    return (X_padded, X_mask)

#Maximum word length (for character representations)
MAX_CLEN=32

def prepare_input_char(X_list):
    MAX_SLEN = max([len(l) for l in X_list])
    X_padded  = [l + [[]]*(MAX_SLEN-len(l))  for l in X_list]
    X_padded  = [[w[0:MAX_CLEN] for w in l] for l in X_padded]
    X_padded  = [[w + [1]*(MAX_CLEN-len(w)) for w in l] for l in X_padded]
    return torch.as_tensor(X_padded).type(torch.LongTensor)

#Pad outputs using one-hot encoding
def prepare_output_onehot(Y_list, tag2i):
    NUM_TAGS=max(tag2i.values())+1
    Y_onehot = [torch.zeros(len(l), NUM_TAGS) for l in Y_list]
    for i in range(len(Y_list)):
        for j in range(len(Y_list[i])):
            Y_onehot[i][j,Y_list[i][j]] = 1.0
    Y_padded = torch.nn.utils.rnn.pad_sequence(Y_onehot, batch_first=True).type(torch.FloatTensor)
    return Y_padded

def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = []
    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(dict(
                        words=words,
                        labels=labels,
                        sentence_boundaries=sentence_boundaries
                    ))
                    words = []
                    labels = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

    if words:
        documents.append(dict(
            words=words,
            labels=labels,
            sentence_boundaries=sentence_boundaries
        ))
        
    return documents


def load_examples(tokenizer, documents):
    examples = []
    max_token_length = 510
    max_mention_length = 30

    for document in tqdm(documents):
        words = document["words"]
        labels = document["labels"]
        subword_lengths = [len(tokenizer.tokenize(w)) for w in words]
        total_subword_length = sum(subword_lengths)
        sentence_boundaries = document["sentence_boundaries"]

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i:i+2]
            if total_subword_length <= max_token_length:
                # if the total sequence length of the document is shorter than the
                # maximum token length, we simply use all words to build the sequence
                context_start = 0
                context_end = len(words)
            else:
                # if the total sequence length is longer than the maximum length, we add
                # the surrounding words of the target sentence　to the sequence until it
                # reaches the maximum length
                context_start = sentence_start
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_start:context_end])
                while True:
                    if context_start > 0:
                        if cur_length + subword_lengths[context_start - 1] <= max_token_length:
                            cur_length += subword_lengths[context_start - 1]
                            context_start -= 1
                        else:
                            break
                    if context_end < len(words):
                        if cur_length + subword_lengths[context_end] <= max_token_length:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break

            text = ""
            for word in words[context_start:sentence_start]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "

            sentence_words = words[sentence_start:sentence_end]
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            for word in words[sentence_end:context_end]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "
            text = text.rstrip()

            entity_spans = []
            original_word_spans = []
            e_labels = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start:word_end]) <= max_mention_length:
                        entity_spans.append(
                            (word_start_char_positions[word_start], word_end_char_positions[word_end])
                        )
                        e_labels.append(labels[word_start:word_end])
                        # e_labels.append('-')
                        original_word_spans.append(
                            (word_start, word_end + 1)
                        )

            examples.append(dict(
                text=text,
                words=sentence_words,
                entity_spans=entity_spans,
                original_word_spans=original_word_spans,
                labels=labels,
                e_labels=e_labels
            ))

    return examples


def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def compute_labels(labels, entity_spans, tag2i):
    span_label = {}
    for span in entity_spans:
        span_label[span] = tag2i['O']

    entered_span_idx = -1
    entered_span_label = ''
#   print(span_label)
    for i, l in enumerate(labels):
        if i == 0 and l != 'O':
            entered_span_idx = i
            entered_span_label = tag2i[l]
        elif labels[i-1][0] == 'O' and l[0] == 'I':
            entered_span_idx = i
            entered_span_label = tag2i[l]
        elif labels[i-1] != l:
            if (entered_span_idx, i) in span_label:
                span_label[(entered_span_idx, i)] = entered_span_label
            if l[0] != 'O':
                entered_span_idx = i
                entered_span_label = tag2i[l]
        elif i == (len(labels) - 1) and l != 'O':
            if (entered_span_idx, i+1) in span_label:
                span_label[(entered_span_idx, i+1)] = entered_span_label
    
    return span_label.values()

def train_loop(params, model, tokenizer, test_examples, tag2i):
    optimizer = optim.AdamW(model.parameters(), lr=params['LR'])

    reporter = MemReporter(model)

    params["N_EPOCHS"] = 1

    for epoch in range(params['N_EPOCHS']):
        model.train()

        epoch_loss = 0

        for batch_start_idx in trange(0, len(test_examples), params['BATCH_SIZE']):
            # if batch_start_idx > 30:
            #     break      
            batch_examples = test_examples[batch_start_idx:batch_start_idx + params['BATCH_SIZE']]
            texts = [item['text'] for item in batch_examples]
            entity_spans = [item['entity_spans'] for item in batch_examples]
            labels = [item['labels'] for item in batch_examples]
            e_labels = []
            for item in batch_examples:
                e_labels.append(
                    torch.tensor(
                        list(compute_labels(item['labels'], item['original_word_spans'], tag2i)),
                        requires_grad=False
                    ).to(device_str)
                )
            
            with torch.no_grad():
                tokenized_batch_examples = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True).to(device_str)
                e_labels_padded = torch.nn.utils.rnn.pad_sequence(e_labels, batch_first=True).type(torch.LongTensor).to(device_str)
            
            # print_args(**tokenized_batch_examples)
            print_gpu_utilization()
            # reporter.report()
            optimizer.zero_grad()

            output = model(**tokenized_batch_examples, labels = e_labels_padded)
            # output = model(**tokenized_batch_examples, labels = labels)
            # loss = torch.nn.functional.cross_entropy(output.logits.view(-1, 5), torch.tensor(labels)[:,0:len(entity_spans[0])].view(-1).to(device_str))
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['CLIP'])
            
            optimizer.step()
            
                
        #   epoch_loss += loss.item()
        # print(epoch_loss/len(test_examples))

def eval(luke_model, tokenizer, test_examples, test_documents):
    batch_size = 2
    all_logits = []
    total_batches = len(range(0, len(test_examples), batch_size))
    print("Calculating logits....Total batches =", len(range(0, len(test_examples), batch_size)))
    idx = 0
    for batch_start_idx in tqdm(range(0, len(test_examples), batch_size)):
        batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
        texts = [example["text"] for example in batch_examples]
        entity_spans = [example["entity_spans"] for example in batch_examples]
        inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = luke_model(inputs)
        #if idx % 10 == 0:
          #print("{:.2f}% completed".format(idx/total_batches * 100))
        idx = idx + 1
        all_logits.extend(outputs.logits.tolist())
    final_labels = [label for document in test_documents for label in document["labels"]]
    final_predictions = []
    total_example_size = len(test_examples)
    print("Computing predictions....Total Examples =", len(test_examples))
    idx = 0
    for example_index, example in tqdm(enumerate(test_examples)):
        logits = all_logits[example_index]
        max_logits = np.max(logits, axis=1)
        max_indices = np.argmax(logits, axis=1)
        original_spans = example["original_word_spans"]
        predictions = []
        for logit, index, span in zip(max_logits, max_indices, original_spans):
            if index != 0:  # the span is not NIL
                predictions.append((logit, span, luke_model.model.config.id2label[index]))
        #if idx % 10 == 0:
          #print("{:.2f}% completed".format(idx/len(test_examples) * 100))
        idx = idx + 1
        # construct an IOB2 label sequence
        predicted_sequence = ["O"] * len(example["words"])
        for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)
        final_predictions += predicted_sequence
    trunc_labels = final_labels[0:len(final_predictions)]
    print(len(final_predictions))
    print(seqeval.metrics.classification_report([trunc_labels], [final_predictions], digits=4))

def main():
    (sentences_train, tags_train) = read_conll_format("train")
    (sentences_dev, tags_dev)     = read_conll_format("dev")

    print(sentences_train[2])
    print(tags_train[2])

    sentencesChar = sentences2char(sentences_train)

    print(sentencesChar[2])
    GloVe = read_GloVe("glove.840B.300d.conll_filtered.txt")

    #Will need this later to remove 50% of words that only appear once in the training data from the vocabulary (and don't have GloVe embeddings).
    wordCounts = Counter([w for l in sentences_train for w in l])
    charCounts = Counter([c for l in sentences_train for w in l for c in w])
    singletons = set([w for (w,c) in wordCounts.items() if c == 1 and not w in GloVe.keys()])
    charSingletons = set([w for (w,c) in charCounts.items() if c == 1])

    #Build dictionaries to map from words, characters to indices and vice versa.
    #Save first two words in the vocabulary for padding and "UNK" token.
    word2i = {w:i+2 for i,w in enumerate(set([w for l in sentences_train for w in l] + list(GloVe.keys())))}
    char2i = {w:i+2 for i,w in enumerate(set([c for l in sentencesChar for w in l for c in w]))}
    i2word = {i:w for w,i in word2i.items()}
    i2char = {i:w for w,i in char2i.items()}

    vocab_size = max(word2i.values()) + 1
    char_vocab_size = max(char2i.values()) + 1

    #Tag dictionaries.
    tag2i = {w:i for i,w in enumerate(set([t for l in tags_train for t in l]))}

    tag2i = {'B-ORG': 3, 'B-LOC': 4, 'I-PER': 2, 'O': 0, 'I-MISC': 1, 'B-MISC': 1, 'I-ORG': 3, 'I-LOC': 4}
    i2tag = {i:t for t,i in tag2i.items()}

    #Indices
    X       = sentences2indices(sentences_train, word2i, singletons, train=True)
    X_char  = sentences2indicesChar(sentencesChar, char2i)
    Y       = sentences2indices(tags_train, tag2i, singletons)

    print("vocab size:", vocab_size)
    print("char vocab size:", char_vocab_size)
    print()

    print("index of word 'the':", word2i["the"])
    print("word of index 253:", i2word[253])
    print()

    print("tag2i: ", tag2i)

    (X_padded, X_mask) = prepare_input(X)
    X_padded_char      = prepare_input_char(X_char)
    Y_onehot           = prepare_output_onehot(Y)


    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

    test_documents = load_documents("eng.testb.1")
    test_examples = load_examples(tokenizer, test_documents)
    print(test_examples[0].keys())

    test_documents = load_documents("eng.testb.1")
    test_examples = load_examples(tokenizer, test_documents)


    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
    model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003").to(device_str)

# main()