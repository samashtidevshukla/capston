# from flask.ext.api import FlaskAPI
from flask import Flask, request, current_app, abort,render_template
from functools import wraps
import model
from html import unescape


app = Flask(__name__)
# app.config.from_object('settings')


@app.route('/predict', methods=['POST'])
def predict():
    # from engine import content_engine
	reviews_username = request.form['reviews_username']
	#reviews_username = request.data.get('reviews_username').lower()
	# num_predictions = request.data.get('num',10)
	if not reviews_username:
		return []
	obj = model.ContenEngine()
	Result = obj.predict_final_products(str(reviews_username))
	return render_template('recommend.html', Result = Result[['name']].to_html())


#Welcome Page
@app.route("/")
def welcome():
    return render_template('welcome.html')





# @app.route('/train')
# @token_auth
# def train():
# 	from engine import content_engine
# 	data_url = request.data.get('data-url', None)
# 	content_engine.train(data_url)
# 	return{"message": "Success!", "success": 1}

if __name__ == "__main__":
	app.debug = True
	app.run()