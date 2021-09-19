import os
from flask import Flask, request, render_template,Blueprint,jsonify,Response,send_file
from werkzeug.utils import secure_filename
from Image_Captioning_InceptionV3 import predict_captions_manual, beam_search_predictions_manual
import json
import shutil
import requests
import random as rd
from PIL import Image
import io
import jsonpickle
import numpy as np
import base64
import sqlite3

app = Flask(__name__)


@app.route('/')
def home():
   return render_template('/index2.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename.replace(" ","")
        f.save(secure_filename(name))
        source = os.getcwd()+'\\'+name
        destination = 'static/'+name
        shutil.move(source, destination)
        return returnResponse(destination)

@app.route('/link', methods = ['GET', 'POST'])
def link():
    if request.method == 'POST':
        link = request.get_data()
        link = link.decode('UTF-8')
        response = requests.get(link)
        destination = "static/"+random()+".jpg"
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
        destination = "static/"+random()+".jpg"
        file = open(destination, "wb")
        file.write(response.content)
        file.close()
        data ={"name": destination,"caption": beam_search_predictions_manual(destination, beam_index=5)}
        response = app.response_class(response=json.dumps(data),mimetype='application/json')
        return response

def returnResponse(destination):
    data ={"name": destination,"caption": predict_captions_manual(destination)}
    response = app.response_class(response=json.dumps(data),mimetype='application/json')
    flag = saveToDb(data)
    return response

def random():
    random = rd.randint(1,1000000000)
    rdlist = []
    for _ in range(500):
        while random not in rdlist:
            rdlist.append(random)
        while random in rdlist:
            random = rd.randint(1,1000000000)
    return str(random)

@app.route('/api', methods=['GET','POST'])
def apiHome():
    # r = request.method
    # if(r=="GET"):
    #     with open("text/data.json") as f:
    #         data=json.load(f)
    #     return data
    # elif(r=='POST'):
    #     with open('static/sample.jpg',"wb") as fh:
    #         fh.write(base64.decodebytes(request.data))
    #     captions=beam_search_predictions_manual('static/sample.jpg', beam_index=5)
    #     cap={"captions":captions}
    #     with open("text/data.json","w") as fjson:
    #         json.dump(cap,fjson)
    # else:
    #     return jsonify({
    #     "captions":"Refresh again !"
    #     })  

    r = request.method
    if(r=="GET"):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        with open("text/data.json") as f:
            data=json.load(f)
            f.close()
        print(data)
        print()
        print()
        return jsonify(data) 
    elif(r=='POST'):
        with open('static/sample.jpg',"wb") as fh:
            fh.write(base64.decodebytes(request.data))
            fh.close()
        print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        # captions=beam_search_predictions_manual('static/sample.jpg', beam_index=3)
        captions=predict_captions_manual('static/sample.jpg')
        print("DONEEEEEEEE")
        cap={"captions":captions}
        with open("text/data.json","w") as fjson:
            json.dump(cap,fjson)
        print()
        print()
        return True
    else:
        return jsonify({
        "captions":"Refresh again !"
        }) 

@app.route('/result')
def sendImage():
    return send_file('static/sample.jpg',mimetype='image/gif')

# @app.route('/test1')
def saveToDb(data):
    # data ={"name": 'dummy dest',"caption": 'dummy caps'}
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()

    cur.execute("SELECT id FROM images order by id desc limit 1;")
    result = cur.fetchmany(1)
    image_id = str(result[0][0] + 1)

    cur.execute("INSERT INTO images(image_path)VALUES('"+data['name']+"')")
    cur.execute("INSERT INTO objects(object_name,image_id)VALUES('"+data['name']+"','"+image_id+"')")
    cur.execute("INSERT INTO captions(caption,accuracy,image_id)VALUES('"+data['caption']+"','"+str(rd.randint(50,90))+"','"+image_id+"')")
    conn.commit()
    return True


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)