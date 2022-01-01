from flask import Flask, render_template, request,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import pickle
import joblib
app = Flask(__name__)

dic = {0 : 'tb', 1 : 'no tb'}

model = load_model('tb_model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = model.predict_classes(i)
	return dic[p[0]]

#def predict_label(img_path):
#	img1 = image.load_img(img_path, target_size=(150,150,1),color_mode="grayscale")
#	Y = image.img_to_array(img1)
#	X= np.expand_dims(Y,axis=0)
#	p = model.predict(X)
#	if p==0:
#		return("Normal")
#	else:
#		return("TB")


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/heart")
def heart():
	return render_template("heart.html")

@app.route("/tb")
def tb():
	return render_template("tb.html")

@app.route("/about")
def about_page():
	return render_template("about.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("tb_predict.html", prediction = p, img_path = img_path)


@app.route("/heart_predict", methods=['GET','POST'])
def heart_predict():
	if request.method == 'POST':
		age = float(request.form['age'])
		sex = float(request.form['sex'])
		cp = float(request.form['cp'])
		trestbps = float(request.form['trestbps'])
		chol = float(request.form['chol'])
		fbs= float(request.form['fbs'])
		restecg = float(request.form['restecg'])
		thalach = float(request.form['thalach'])
		exang = float(request.form['exang'])
		oldpeak = float(request.form['oldpeak'])
		slope = float(request.form['slope'])
		#ca = float(request.form['ca'])
		#thal = float(request.form['thal'])

#		pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

		pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope]

		mul_reg = open('heart_model.pkl','rb')
		ml_model = joblib.load(mul_reg)
		model_prediction = ml_model.predict([pred_args])
		res = "some default value to avoid error"
		
		if model_prediction == 1:
			res = 'Affected'
		else:
			res = 'Not affected'
		#return res
	return render_template('heart_predict.html', prediction = res)

if __name__ =='__main__':
	app.debug = True
	app.run()