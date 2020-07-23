from flask import Flask, render_template, request
import json

from predict import predict_n_keywords

app = Flask(__name__)

saved_models = ["bert_term_30", "scibert_term_20", "bert_mostdepth_30", "scibert_mostdepth_30", "bert_mixed_30", "scibert_mixed_30", ]

@app.route('/')
def index():
	return render_template('keywords_generation.html')

@app.route('/submit', methods=['POST'])
def submit():
	"""Api pointer of "submit" button after clicking it

	Returns:
	json: json file containing predictions
	"""
	print("request.data is")
	print(request.data)
	submitted_text = request.data.decode('utf-8')
	print(submitted_text)
	num_keywords = 20
	result = {}
	for model in saved_models:
		if model.split("_")[0] == "bert":
			vocab_path = 'bert_vocab_path'
		else:
			vocab_path = 'scibert_vocab_path'
		option = model.split("_")[1]
		print("Computing keywords for " + model)
		result[model] = predict_n_keywords(submitted_text, model, vocab_path, 512, True, num_keywords, option)
	return json.dumps(result)

if __name__ == "__main__":
	app.run()
