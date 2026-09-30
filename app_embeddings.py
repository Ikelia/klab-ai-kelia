import streamlit as st
import pandas as pd
import re
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- FAQ DATA (same 40 Python Q&A pairs) ----------
faq_data = {
    'question': [
        "What is Python?",
        "How do I install Python?",
        "What is a variable in Python?",
        "What are the main data types in Python?",
        "How do I write a comment in Python?",
        "What is a list in Python?",
        "What is a tuple in Python?",
        "What is a dictionary in Python?",
        "What is a set in Python?",
        "How do I create a function in Python?",
        "What is the difference between a list and a tuple?",
        "How do I handle errors in Python?",
        "What is a Python library?",
        "How do I install a Python library?",
        "What is pip?",
        "What is a virtual environment?",
        "What is PEP 8?",
        "How do I read a file in Python?",
        "How do I write to a file in Python?",
        "What is a module in Python?",
        "How do I import a module in Python?",
        "What is the difference between == and is in Python?",
        "What is a lambda function?",
        "What are list comprehensions?",
        "What is the purpose of __init__ in Python?",
        "What is a class in Python?",
        "How do I create an object in Python?",
        "What is inheritance in Python?",
        "What is the difference between append and extend?",
        "How do I sort a list in Python?",
        "How do I reverse a list in Python?",
        "How do I check the type of a variable?",
        "What is a docstring?",
        "How do I format strings in Python?",
        "What is the difference between deepcopy and shallow copy?",
        "What are decorators in Python?",
        "What is a generator in Python?",
        "What is the GIL in Python?",
        "What is the difference between Python 2 and Python 3?",
        "How do I make a Python script executable?"
    ],
    'answer': [
        "Python is a high-level interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        "Download Python from python.org and run the installer. On Windows make sure to check Add Python to PATH during installation.",
        "A variable is a named container that stores a value. You create one by assigning a value like x = 5 or name = Alice.",
        "The main data types are int float str bool list tuple dict and set.",
        "Use the hash symbol for single-line comments like # this is a comment. For multi-line comments use triple quotes.",
        "A list is an ordered mutable collection of items enclosed in square brackets. Example: numbers = [1 2 3 4].",
        "A tuple is an ordered immutable collection of items enclosed in parentheses. Example: point = (3 4).",
        "A dictionary is an unordered collection of key-value pairs enclosed in curly braces.",
        "A set is an unordered collection of unique items enclosed in curly braces. Example: unique_nums = {1 2 3}.",
        "Use the def keyword followed by the function name and parentheses. Example: def greet(name): return Hello + name.",
        "Lists are mutable (can be changed) and use square brackets. Tuples are immutable (cannot be changed) and use parentheses.",
        "Use try and except blocks. Example: try: x = 1 / 0 except ZeroDivisionError: print Cannot divide by zero.",
        "A library is a collection of pre-written code that you can import and use. Examples include NumPy pandas and requests.",
        "Use pip from the command line. Example: pip install numpy. Make sure your virtual environment is active.",
        "pip is Python's package installer. It downloads and installs libraries from the Python Package Index (PyPI).",
        "A virtual environment is an isolated Python environment for a project. Create one with: python -m venv venv.",
        "PEP 8 is Python's official style guide. It recommends 4-space indentation snake_case for variables and keeping lines under 79 characters.",
        "Use the open() function with a context manager. Example: with open(file.txt r) as f: content = f.read().",
        "Open the file in write mode. Example: with open(file.txt w) as f: f.write(Hello).",
        "A module is a .py file containing Python code that can be imported into other files using the import statement.",
        "Use the import keyword. Example: import math then use math.sqrt(16). You can also use from math import sqrt.",
        "== checks if values are equal. is checks if two variables point to the same object in memory.",
        "A lambda is a small anonymous function defined with the lambda keyword. Example: square = lambda x: x * x.",
        "A concise way to create lists. Example: squares = [x * x for x in range(10)].",
        "__init__ is a constructor method in a class. It runs automatically when you create a new object from the class.",
        "A class is a blueprint for creating objects. It bundles data (attributes) and behavior (methods) together.",
        "Call the class like a function. Example: if you have class Dog: you create an object with my_dog = Dog().",
        "Inheritance lets a class inherit attributes and methods from another class. Example: class Dog(Animal): means Dog inherits from Animal.",
        "append adds one item to a list. extend adds multiple items from an iterable. Example: [1 2].append(3) gives [1 2 3].",
        "Use the sort() method for in-place sorting or sorted() for a new sorted list. Example: numbers.sort() or sorted(numbers).",
        "Use the reverse() method or slicing. Example: numbers.reverse() or numbers[::-1].",
        "Use the type() function. Example: type(5) returns class int. For checking use isinstance(x int).",
        "A docstring is a string literal that documents a module function class or method. It appears as the first statement and uses triple quotes.",
        "Use f-strings (Python 3.6+): name = Alice then f Hello {name}. You can also use .format() or percent formatting.",
        "A shallow copy creates a new object but references the same nested objects. A deepcopy recursively copies all nested objects.",
        "Decorators are functions that modify other functions. They use the @ symbol. Example: @staticmethod @classmethod @property.",
        "A generator is a function that yields values one at a time using yield instead of return. It is memory-efficient for large data.",
        "The Global Interpreter Lock (GIL) allows only one thread to execute Python bytecode at a time. It affects multi-threaded CPU-bound programs.",
        "Python 3 is the current version. Python 2 reached end of life in 2020. Key differences include print() as a function and Unicode string handling.",
        "On Linux or Mac add a shebang line #!/usr/bin/env python3 then run chmod +x script.py. On Windows use py script.py."
    ]
}

faq = pd.DataFrame(faq_data)

# ---------- PREPROCESSING ----------
STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her','it','its',
    'they','them','their','what','which','who','this','that','these','those',
    'am','is','are','was','were','be','been','being','have','has','had','do',
    'does','did','a','an','the','and','but','if','or','because','as','of','at',
    'by','for','with','about','to','from','in','out','on','off','over','under',
    'again','then','once','here','there','when','where','why','how','all','any',
    'both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','can','will','just','should','now'
}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens)

# ---------- LOAD SBERT MODEL (cached) ----------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---------- BUILD FAQ EMBEDDINGS (cached) ----------
@st.cache_data
def build_faq_embeddings(questions_tuple):
    processed = [preprocess(q) for q in questions_tuple]
    return model.encode(processed, convert_to_numpy=True, show_progress_bar=False)

questions = faq['question'].tolist()
answers = faq['answer'].tolist()
faq_embeddings = build_faq_embeddings(tuple(questions))

# ---------- CHATBOT FUNCTION ----------
def get_answer(user_question, threshold=0.35):
    processed_input = preprocess(user_question)
    user_embedding = model.encode([processed_input], convert_to_numpy=True)
    similarities = cosine_similarity(user_embedding, faq_embeddings).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    if best_score < threshold:
        return "Sorry, I don't understand that question. Could you rephrase?", best_score
    return answers[best_idx], best_score

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Python FAQ Chatbot (Embeddings)", page_icon="🧠")

st.title("🧠 Python FAQ Chatbot — Embeddings Version")
st.markdown("Powered by Sentence-BERT (`all-MiniLM-L6-v2`). Handles paraphrases much better than TF-IDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a Python question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response, score = get_answer(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
        if score >= 0.35:
            st.caption(f"Match confidence: {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": response})
