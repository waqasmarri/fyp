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
   return render_template('/index2.html')
   # return predict_captions_manual('images/download.jpg')

# @app.route('/index2')
# def home():
#    return render_template('/index2.html')

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

def returnResponse(destination):
    data ={"name": destination,"caption": predict_captions_manual(destination)}
    response = app.response_class(response=json.dumps(data),mimetype='application/json')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.1.1.1', port=9000)
