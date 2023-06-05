from flask import Flask,request, jsonify, abort, send_file
from flask_cors import CORS
import json

# import firebase_admin
# from firebase_admin import credentials, firestore, storage

from datetime import datetime, timedelta
from backend import *

### Connect Firebase
# cred = firebase_admin.credentials.Certificate("project-nlp-9b41d-firebase-adminsdk-w4jxt-038c435e97.json")
# firebase_admin.initialize_app(cred,{
#     'storageBucket': 'project-nlp-9b41d.appspot.com'},name='bicarapilpres')
# db = firestore.client()  # this connects to our Firestore database



app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return '''<h1>TalentEase Machine Learning API v1</h1>'''
	

@app.route('/api/extract', methods=['POST'])
def extract():
    

@app.route('/api/summary', methods=['POST'])
def summary():
    

@app.route('/api/wordcloudsent', methods=['POST'])
def get_wordcloudsent():
    if not request.json or not 'status' in request.json:
        abort(400)
    data = request.get_json()
    if data['status'] != 'minta datanya dong':
            abort(400)
    else:
        credentials = service_account.Credentials.from_service_account_file("project-nlp-9b41d-firebase-adminsdk-w4jxt-038c435e97.json")
        calon = data['calon']
        waktu = data['waktu']
        sent = data['sentiment']
        storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob(f'wordcloud/{calon}/wordcloud_mention_{calon}_{waktu}_{sent}.jpg').download_to_filename(f'wordcloud/{calon}/wordcloud_mention_{calon}_{waktu}_{sent}.jpg')
        return send_file(f'wordcloud/{calon}/wordcloud_mention_{calon}_{waktu}_{sent}.jpg', mimetype=f'image/png')

if __name__ == '__main__':
    app.run(debug=True)