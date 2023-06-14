import nltk
nltk.download("stopwords")

# Import Library
import nltk
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
import PyPDF2 as pdf
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re
import pandas as pd

import os


## Firebase & google cloud
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from google.oauth2 import service_account


# Model
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
## Model Load
model = TFAutoModelForSeq2SeqLM.from_pretrained('walkerrose/cv_summarization-distilbart-cnn-16-6')
tokenizer = AutoTokenizer.from_pretrained('walkerrose/cv_summarization-distilbart-cnn-16-6')

# Firebase & Google Cloud auth
cred = firebase_admin.credentials.Certificate("talentease-project-firebase-adminsdk-db1kq-66426f47e0.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
credentials = service_account.Credentials.from_service_account_file("talentease-project-firebase-adminsdk-db1kq-66426f47e0.json")

# Skills Data
skills_data = pd.read_csv('skills.csv')
skills_data['Skill'] = skills_data['Skill'].apply(lambda x: re.sub('\\r\\n', '', x))


def clean_summ(res):
  res = res.replace(" | "," ")
  res = res.replace("•","")
  res = res.replace("_","")  
  res = res.strip().replace('\n', '')
  res = re.sub(' +', ' ', res)
  res = re.sub(r'\●', ' ', res)
  return res

# Get PDF CV Data
def get_pdf(id):
    cloud_path = f"CV/{id}.pdf"
    temp_path = f"{id}.pdf"
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket("talentease-project.appspot.com")
    blob = bucket.blob(cloud_path).download_to_filename(temp_path)

# Extract Skills
def get_skills(resume_text):
    # Extract skills using multiple datasets
    skills_datasets = [skills_data]
    skills = []
    for dataset in skills_datasets:
        skills_pattern = r"\b(?:{})\b".format("|".join(map(re.escape, dataset['Skill'].dropna())))
        skills_match = re.findall(skills_pattern, resume_text, re.IGNORECASE)
        skills.extend(skills_match)

    return list(set(skills))

# Extract
def extract_with_ocr(file,id):
    # pretrained model
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    # reading files
    document = DocumentFile.from_pdf(file)

    # analyze
    result = model(document)

    # export to json
    output = result.export()

    # grouping detected words
    separated_words = []
    for page in output["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    separated_words.append(word["value"])
    
    # combining separated words into sentences
    text = " ".join(separated_words)
    skills = get_skills(text)
    text = clean_summ(text)
    os.remove(f"{id}.pdf")
    return text, skills

def text_extractor(id):
    # read file
    file = f"{id}.pdf"
    reader = pdf.PdfReader(file)
    page = reader.pages[0]
    
    # checks for text
    # if chars detected more than 100 chars, it will be considered scanned pdf
    if len(page.extract_text()) > 100: # tweak this number for character count if you want
        # extract text
        text = page.extract_text()
        if len(text.split(" ")) > 1000:
            text, skills = extract_with_ocr(file,id)
        text = clean_summ(text)
        skills = get_skills(text.lower())
        os.remove(file)
        return text, skills
    else:
        return extract_with_ocr(file,id)

# Summary model predict
def summary_pred(id):
    get_pdf(id)
    text, skills = text_extractor(id)
    text = (text[:4600]) if len(text) > 4600 else text
    print("Generate Summary...")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")
    output = summarizer(text,min_length=50,max_length=128)
    hasil = {"skills":", ".join(skills),
            "summary": output[0]['summary_text']}
    db.collection('prediction').document(id).set(hasil)
    print("Data berhasil disimpan di database")
    return hasil
    