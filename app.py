from flask import Flask, render_template, request
import pickle
from re import *


with open('emails.csv','r') as f: #email data from kaggle, stores in f, r is read mode
    lines = [] # create an empty array to keep track of lines
    for line in f.readlines():
        lines.append(line.strip())

header = lines[0]
header = header.split(',')[1:]
data = lines[1:]
data = [','.join(line.split(',')[1:]) for line in data]


def get_word_counts(text):
    text = text.lower() # convert to lowercase
    counts = [] # create blank array to store counts
    for word in header[:-1]:
        counts.append(len(findall(word, text))) # find all instances of word in text and store in counts
    return counts

with open('jar.pkl', 'rb') as jar:
    rf = pickle.load(jar)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/spamorham', methods=['GET', 'POST'])
def spamorham():
    data = request.args.get('email')
    spamcounts = get_word_counts(data)
    data = rf.predict([spamcounts])
    if data == ['1']:
        data = 'Spam!'
    elif data == ['0']:
        data = 'Ham!'
    print(data)
    return render_template('spamorham.html', data=data)

if __name__ == '__main__':
    app.run(port=5050, debug=True)