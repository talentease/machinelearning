# Import Library
import pandas as pd


import PyPDF2

## Firebase
import firebase_admin
from firebase_admin import credentials, firestore, storage

## Model
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer


hg_token = "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"
cred = firebase_admin.credentials.Certificate("talentease-project-firebase-adminsdk-db1kq-66426f47e0.json")
firebase_admin.initialize_app(cred)
db = firestore.client()  # this connects to our Firestore database


# Get PDF CV Data


# Extract Data


# Summary model predict
def summary_pred(text, id):
    model = TFAutoModelForSeq2SeqLM.from_pretrained('walkerrose/cv_summarization',use_auth_token=hg_token)
    tokenizer = AutoTokenizer.from_pretrained('walkerrose/cv_summarization',use_auth_token=hg_token)
    pred_token = tokenizer(text, max_length=512, padding="max_length", return_tensors="np").input_ids
    print("Generate Summary...")
    outputs = model.generate(pred_token,min_length=50, max_length=150,early_stopping=True)
    hasil = {"summary": tokenizer.decode(outputs[0], skip_special_tokens=True)}
    db.collection('ml test').document(id).set(hasil)
    print("Data berhasil disimpan di database")
