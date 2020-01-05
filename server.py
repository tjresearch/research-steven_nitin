from flask import Flask, render_template, redirect
import os

OUTPUT_FOLDER = os.path.join('static', 'output')
full_filename = ""

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/')
def index():
    print(full_filename)
    return render_template('index.html', image_src = full_filename)

@app.route('/cv')
def cv():
    global full_filename
    os.system('python3 growth_detection.py -f /mnt/c/steven/research/pickles/ -o /mnt/c/steven/research/output/')
    print("edges analyzed")
    full_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'output1.png')
    return render_template('index.html', image_src = full_filename)



@app.route('/nn')
def nn():
    print("you have analyzed using a neural network")
    return "no more analysis"

if __name__ == '__main__':
    app.run()
