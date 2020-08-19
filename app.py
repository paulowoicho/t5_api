from flask import Flask, jsonify, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import string
import gdown
import os

if os.path.isfile('t5_model/pytorch_model.bin'):
  pass

else:
  gdown.download(
    "https://drive.google.com/uc?id=1_16_KnESJO7Y46arbHa_DenuaCMuqLnF",
    "t5_model/pytorch_model.bin",
  )



app = Flask(__name__)
tokenizer = T5Tokenizer.from_pretrained('t5_model/')
model = T5ForConditionalGeneration.from_pretrained('t5_model/')


def clean_up(text):
  head, _ , _ = text.partition(' ---')
  head = head.strip()
  first_letter = head[0].capitalize()
  head = first_letter + head[1:]
  if head[-1] in string.punctuation:
    pass
  else:
    head += '.'
  return head

def t5_inference(transcript):
  threshold = 6500
  t5_form = 'summarize: ' + transcript
  tokenized_text = tokenizer.encode(t5_form, return_tensors="pt")
  if len(tokenized_text[0]) > threshold:
    revised_text = sent_tokenize(t5_form)
    length = len(revised_text)
    final_text = revised_text[:int(length/2)] #maybe they talk about content in the first half? find proof
    text = ' '.join(final_text)
    return t5_inference(text)
  summary_ids = model.generate(tokenized_text, max_length=150, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  output = clean_up(output)
  return output


@app.route("/")
def home():
    return "API is up!"

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    podcast = request.json
    transcript = podcast['transcript']
    summary = t5_inference(transcript)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(port = 5000, debug=True)