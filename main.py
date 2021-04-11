import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from Image_Captioning_InceptionV3 import predict_captions_manual, beam_search_predictions_manual
import json
import shutil
import requests

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('/index2.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        source = os.getcwd()+'\\'+f.filename
        destination = 'static/'+f.filename
        shutil.move(source, destination)
        return returnResponse(destination)

@app.route('/link', methods = ['GET', 'POST'])
def link():
    if request.method == 'POST':
        link = request.get_data()
        link = link.decode('UTF-8')
        response = requests.get(link)
        destination = "static/"+link[8:12]+".jpg"
        file = open(destination, "wb")
        file.write(response.content)
        file.close()
        return returnResponse(destination)

@app.route('/uploader-beam', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        source = os.getcwd()+'\\'+f.filename
        destination = 'static/'+f.filename
        shutil.move(source, destination)
        data ={"name": destination,"caption": beam_search_predictions_manual(destination, beam_index=5)}
        response = app.response_class(response=json.dumps(data),mimetype='application/json')
        return response

@app.route('/link-beam', methods = ['GET', 'POST'])
def link2():
    if request.method == 'POST':
        link = request.get_data()
        link = link.decode('UTF-8')
        response = requests.get(link)
        destination = "static/"+link[8:12]+".jpg"
        file = open(destination, "wb")
        file.write(response.content)
        file.close()
        data ={"name": destination,"caption": beam_search_predictions_manual(destination, beam_index=5)}
        response = app.response_class(response=json.dumps(data),mimetype='application/json')
        return response

def returnResponse(destination):
    data ={"name": destination,"caption": predict_captions_manual(destination)}
    response = app.response_class(response=json.dumps(data),mimetype='application/json')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='127.1.1.1', port=9000)
