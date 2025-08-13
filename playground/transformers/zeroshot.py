#!/usr/bin/env python3
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
print(classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
))
