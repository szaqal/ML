#!/usr/bin/env python3

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "Yo to come check this crazy flow!"
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors = "pt")
print(inputs)


