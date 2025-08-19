#!/usr/bin/env python3

from transformers import AutoTokenizer
from transformers import BertTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "Yo to come check this crazy flow!"
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors = "pt")
print(inputs)


tokenizer2 = BertTokenizer.from_pretrained("bert-base-cased")
inputs = tokenizer2(raw_inputs, padding=True, truncation=True, return_tensors = "pt")
print(inputs)

print(tokenizer2.tokenize("Yo yo come check the crazy flow!!"))
print(tokenizer2.tokenize("Availability is what makes stuff available"))

