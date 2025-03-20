from flask import Flask, request, jsonify, render_template
from roboflow import Roboflow
import cv2
import pytesseract
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Set the correct Tesseract-OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Roboflow
try:
    rf = Roboflow(api_key="Y10vh3ZwlwXaBpUEcDJ3")
    VERSION = 13
    workspace = rf.workspace()
    project = workspace.project("resume-parse")
    model = project.version(VERSION).model
except Exception as e:
    print("Error initializing Roboflow:", str(e))

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

# Function to extract text from resume image
def extract_resume_text(image_path):
    try:
        prediction = model.predict(image_path, confidence=40, overlap=30)
        image = cv2.imread(image_path)
        prediction_data = prediction.json()
        resume_text = ""

        for item in prediction_data.get('predictions', []):
            x1 = int(item['x'] - item['width'] / 2)
            y1 = int(item['y'] - item['height'] / 2)
            x2 = int(item['x'] + item['width'] / 2)
            y2 = int(item['y'] + item['height'] / 2)
            
            cropped_image = image[y1:y2, x1:x2]
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            extracted_text = pytesseract.image_to_string(threshold_image)
            resume_text += extracted_text

        return preprocess_text(resume_text)
    
    except Exception as e:
        print("Error extracting text from resume:", str(e))
        return ""

# Function to calculate similarity
def calculate_similarity(resume_text, job_description_text):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description_text])
        cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        set1 = set(resume_text.split())
        set2 = set(job_description_text.split())
        overlap_score = len(set1.intersection(set2)) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0.0

        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings1 = bert_model.encode(resume_text, convert_to_tensor=True)
        embeddings2 = bert_model.encode(job_description_text, convert_to_tensor=True)
        bert_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()

        return {
            "Cosine Similarity": cosine_score,
            "Overlap Coefficient": overlap_score,
            "BERT Semantic Similarity": bert_score,
            "Average Score": (cosine_score + overlap_score + bert_score) / 3
        }
    except Exception as e:
        print("Error calculating similarity:", str(e))
        return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    
    resume_text = extract_resume_text(filepath)
    job_descriptions = [request.form.get(f'job{i}') for i in range(1, 6)]
    results = []

    for i, job_desc in enumerate(job_descriptions):
        if job_desc:
            scores = calculate_similarity(resume_text, preprocess_text(job_desc))
            results.append({"Job Description": f"Job {i + 1}", **scores})
    
    sorted_results = sorted(results, key=lambda x: x["Average Score"], reverse=True)
    return jsonify(sorted_results)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
