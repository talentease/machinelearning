from flask import Flask,request, jsonify, abort, send_file
from flask_cors import CORS
from backend import *

# # Connect Firebase
# cred = firebase_admin.credentials.Certificate("talentease-project-firebase-adminsdk-db1kq-66426f47e0.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()  # this connects to our Firestore database



app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return '''<h1>TalentEase Machine Learning API v1</h1>'''

@app.route('/api/upload', methods=['POST'])
def upload():
    if not request.json:
        abort(400)
    else:
        data = request.get_json()
        summary_pred(data['id'])
        return jsonify({"response":"Hasil prediksi berhasil masuk ke database!"}),201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)