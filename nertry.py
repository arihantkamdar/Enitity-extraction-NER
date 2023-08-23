
from transformers import AutoTokenizer, BertForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("gagan3012/bert-tiny-finetuned-ner")

model = BertForTokenClassification.from_pretrained("gagan3012/bert-tiny-finetuned-ner")

inputs = tokenizer("HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt")

with torch.no_grad():

    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)

# Note that tokens are classified rather then input words which means that

# there might be more predicted token classes than words.

# Multiple token classes might account for the same word

predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

print(predicted_tokens_classes)
print("HuggingFace is a company based in Paris and New York".split(" "))
labels = predicted_token_class_ids

loss = model(**inputs, labels=labels).loss

round(loss.item(), 2)