from flask import Flask, request, jsonify
from model import VQADemo
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'backend/temp'

# Create temp folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model once
model_path = 'models/best_vqa_model.pth'
vqa = VQADemo(model_path=model_path, answers_vocab_path='data/answer_vocab.json')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image and question
        image = request.files['image']
        question = request.form['question']

        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Get prediction
        answer = vqa.get_answer(image_path, question)

        # Remove the uploaded file
        os.remove(image_path)

        return jsonify({'question': question, 'answer': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
