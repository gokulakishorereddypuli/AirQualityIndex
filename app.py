from flask import Flask, render_template, request, redirect, url_for, session
import ibm_db
import json
import requests
app = Flask(__name__)
@app.route('/')    
@app.route('/home')
def login():
    return render_template('index.html')
@app.route('/news')    
def news():
    return render_template('news.html')
@app.route('/contact')    
def contact():
    return render_template('contact.html')
@app.route('/live-cameras')    
def live_cameras():
    return render_template('live-cameras.html')
@app.route('/photos')    
def photos():
    return render_template('photos.html')

if __name__ == "__main__":
    app.run(debug=False)