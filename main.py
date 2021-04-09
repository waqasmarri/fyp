import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from Image_Captioning_InceptionV3 import predict_captions_manual
import json
import shutil
import requests
# from file import test_fun
#app = Flask('app')

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('/index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))
		source = os.getcwd()+'\\'+f.filename
		destination = 'static/'+f.filename
		shutil.move(source, destination)
		data ={"name": destination,"caption": predict_captions_manual(destination)}
		response = app.response_class(response=json.dumps(data),mimetype='application/json')
		return response

@app.route('/link', methods = ['GET', 'POST'])
def link():
	response = requests.get("https://i.imgur.com/ExdKOOz.png")
	file = open("sample_image.png", "wb")
	file.write(response.content)
	file.close()
	return 'saved'

if __name__ == '__main__':
    app.run(debug=True, host='127.1.1.1', port=9000)