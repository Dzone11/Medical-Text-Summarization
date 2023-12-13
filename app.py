# app.py

from flask import Flask, render_template, request
import pandas as pd
import pickle
from modules.preprocessing import preprocess_text_data
from modules.summarization import extractive_summarization_bert

app = Flask(__name__)

# Load the medical transcription dataset
df = pd.read_csv('data/mtsamples.csv')

# Preprocess the data using the module
df = preprocess_text_data(df, column='transcription')

# Save preprocessed data to a pickle file
with open('preprocessed_data.pkl', 'wb') as file:
    pickle.dump(df['clean_tokens'], file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        doc_id = int(request.form['doc_id'])
        document = df.loc[doc_id, 'transcription']
        # Use the preprocessed data instead of preprocessing again
        tokens = df.loc[doc_id, 'clean_tokens']
        summary = extractive_summarization_bert(' '.join(tokens), ratio=0.2)
        return render_template('result.html', document=document, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
