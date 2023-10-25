from transformers import pipeline

classifer = pipeline("sentiment-analysis")
with open("messages.csv", "r") as f:

    msg =f.readline()
    while(msg):
        if msg.strip() != "" and len(msg.split(" ")) > 1:
            msg = msg.replace("\n", "")
            print(f'MSG: {msg}: {classifer(msg)}')
        msg = f.readline()
