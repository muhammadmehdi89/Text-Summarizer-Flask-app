from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)





app = Flask(__name__)





@app.route('/', methods=["GET", "POST"])
def index():
    summary = ""  # Initialize summary variable
    if request.method == "POST":
        inputtext = request.form["inputtext_"]
        input_text = "summarize: " + inputtext

        # Generate summary
        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    return render_template('index.html', data={"summary": summary})






if __name__ == "__main__":
    app.run(debug=True)
