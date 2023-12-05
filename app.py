# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset and preprocess it (you can replace this with your actual data loading logic)
df = pd.read_excel(r"C:\Users\bhush\Downloads\chatbot_dataset.xlsx")
# ... (your data preprocessing code)

df['Question'].iloc[0] = 'how are you doing?'

new_data= {'Question': ['what is the capital of india',
                        'what is your name?',
                        'who is the most followed athlete in india',
                       'Hi,hie,hii',
                        'hello',
                       'Nothing',
                       'thank you',
                       'nice',
                       'I am also fine',
                       'Exit',
                       'gender',
                       'quite',
                       'hi',
                       'what is going on',
                       'i am fine',
                       'quite','are you male?'], 
           'Answer': ['New Delhi',
                      'chatbot',
                      'virat kohli',
                      'hello',
                     'hi,how can i help you',
                     'okay, thank you!',
                     'You are welcome! Let me know if you need anything else.',
                     'thank you for your compliment',
                     'okay',
                     'sure',
                     'As i am chatbot i dont have gender',
                     'sure',
                     'hello',
                     'nothing',
                     'okay',
                     'sure',"no"]}
new_df = pd.DataFrame(new_data)
df = pd.concat([df, new_df], ignore_index=True)

# Train the model (you can replace this with your actual training logic)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(train_data['Question'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define the get_response function
def get_response(user_input):
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    idx = similarities.argsort()[-1]
    return train_data['Answer'].iloc[idx]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['user_input']
        bot_response = get_response(user_input)
        return render_template('index.html', user_input=user_input, bot_response=bot_response)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

