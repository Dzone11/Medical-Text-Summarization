from modules.preprocessing import preprocess_text_data
from summarizer import Summarizer  # Import Summarizer from the bert-extractive-summarizer library

def extractive_summarization_bert(text):
    summarizer = Summarizer()
    sentences = preprocess_text_data(text)['clean_tokens'].tolist()  # Assuming 'clean_tokens' is a column in your DataFrame
    summary = summarizer(sentences)
    return summary
