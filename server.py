from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    print("you have analyzed")
    return "no more analysis"

if __name__ == '__main__':
    app.run()
