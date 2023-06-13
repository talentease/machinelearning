# Import Library
import PyPDF2 as pdf
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re
from pyresparser import ResumeParser
from fpdf import FPDF
import os

## Firebase & google cloud
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from google.oauth2 import service_account


## Model
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Firebase & Google Cloud auth
cred = firebase_admin.credentials.Certificate("talentease-project-firebase-adminsdk-db1kq-66426f47e0.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
credentials = service_account.Credentials.from_service_account_file("talentease-project-firebase-adminsdk-db1kq-66426f47e0.json")


def clean_summ(res):
  res = res.replace(" | "," ")
  res = res.replace("•","")
  res = res.replace("_","")  
  res = res.strip().replace('\n', '')
  res = re.sub(' +', ' ', res)
  return res

# Get PDF CV Data
def get_pdf(id):
    cloud_path = f"CV/{id}.pdf"
    temp_path = f"temp/{id}.pdf"
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket("talentease-project.appspot.com")
    blob = bucket.blob(cloud_path).download_to_filename(temp_path)


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
    pdf = FPDF()
    # Add a page
    pdf.add_page()

    # set style and size of font
    # that you want in the pdf
    pdf.set_font("Arial", size = 11)
    
    # create a cell
    pdf.cell(200, 10, txt = text,
            ln = 1, align = 'J')
    
    # save the pdf with name .pdf
    pdf.output(f"temp/temp_{id}.pdf")  
    data = ResumeParser(f"temp/temp_{id}.pdf").get_extracted_data()
    text = clean_summ(text)
    os.remove(f"temp/temp_{id}.pdf")
    return text, data['skills'],data['experience']

def text_extractor(id):
    # read file
    file = f"temp/{id}.pdf"
    reader = pdf.PdfReader(file)
    page = reader.pages[0]
    
    # checks for text
    # if chars detected more than 100 chars, it will be considered scanned pdf
    if len(page.extract_text()) > 100: # tweak this number for character count if you want
        # extract text
        text = page.extract_text()
        text = clean_summ(text)
        data = ResumeParser(file).get_extracted_data()
        os.remove(file)
        return text, data['skills'],data['experience']
    else:
        return extract_with_ocr(file,id)




# Summary model predict
def summary_pred(id):
    get_pdf(id)
    text, skills, experience = text_extractor(id)
    model = TFAutoModelForSeq2SeqLM.from_pretrained('walkerrose/cv_summarization-distilbart-cnn-16-6')
    tokenizer = AutoTokenizer.from_pretrained('walkerrose/cv_summarization-distilbart-cnn-16-6')
    pred_token = tokenizer(text, max_length=512, padding="max_length", return_tensors="np").input_ids
    print("Generate Summary...")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")
    output = summarizer(text,min_length=50,max_length=128)
    hasil = {"skills":skills,
            "experience":experience,
            "summary": output[0]['summary_text']}
    db.collection('ml test').document(id).set(hasil)
    print("Data berhasil disimpan di database")

    


