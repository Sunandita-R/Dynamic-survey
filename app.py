from flask import Flask, request, jsonify, render_template
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
question_count=0
# Load survey questions
with open('questions.json') as f:
    questions = json.load(f)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Define the logic to determine the next question
def get_next_question(current_question_id, answer):
    # Tokenize the input answer
    inputs = tokenizer(answer, return_tensors='pt')
    
    # Get the sentiment prediction from the model
    outputs = model(**inputs)
    sentiment_score = torch.argmax(outputs.logits).item()
    
    # Define the sentiment mapping based on the model's output
    sentiment_mapping = {
        0: 'negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'positive'
    }
    
    # Map the sentiment score to a sentiment label
    sentiment_label = sentiment_mapping.get(sentiment_score, 'neutral')  # Default to 'neutral' if score is out of bounds
    
    # Debug print statements to check intermediate values
    print(f"Sentiment score: {sentiment_score}")
    print(f"Sentiment label: {sentiment_label}")
    global question_count
    question_count+= 1
    # Determine the next question based on sentiment label
    next_question_id = questions[str(current_question_id)]['next'].get(sentiment_label)
    if question_count >= 7:
        return None
    
    # Debug print statement to check the next question ID
    print(f"Next question ID: {next_question_id}")
    
    return next_question_id


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/next-question', methods=['POST'])
def next_question():
    data = request.json
    current_question_id = str(data['question_id'])
    answer = data['answer']
    
    next_question_id = get_next_question(current_question_id, answer)
    if next_question_id:
        next_question = questions[str(next_question_id)]
        next_question['id'] = next_question_id  # Include the id in the response
        return jsonify(next_question)
    else:
        return jsonify({"message": "Survey complete. Thank you for your feedback!"})

if __name__ == '__main__':
    app.run(debug=True)
