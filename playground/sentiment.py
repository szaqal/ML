from transformers import pipeline

# https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m
classifer = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
with open("messages.csv", "r") as f:

    msg =f.readline()
    while(msg):
        if msg.strip() != "" and len(msg.split(" ")) > 1:
            msg = msg.replace("\n", "")
            print(f'MSG: {msg}: {classifer(msg)}')
        msg = f.readline()
