import pandas as pd
import numpy as np
import faiss

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer

# Load dataset
data = pd.read_csv("shopping_data.csv")

questions = data["question"].tolist()
answers = data["answer"].tolist()

# Load MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
question_embeddings = model.encode(questions)

question_embeddings = np.array(question_embeddings).astype('float32')

# Create FAISS index
dimension = question_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(question_embeddings)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    bot_response = ""

    if request.method == "POST":

        user_question = request.form["question"]

        user_embedding = model.encode([user_question])

        user_embedding = np.array(user_embedding).astype('float32')

        k = 1

        distances, indices = index.search(user_embedding, k)

        best_match_index = indices[0][0]

        bot_response = answers[best_match_index]

    return render_template("index.html", response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
