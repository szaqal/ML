from transformers import pipeline

classifer = pipeline("sentiment-analysis")
print(classifer("What the hell?"))
print(classifer("It was so so"))
print(classifer("That was cool"))