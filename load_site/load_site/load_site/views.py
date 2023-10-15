"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
from load_site import app
import os
import tempfile
#from madul1 import toggle_case_in_file
from class1 import toggle_case_in_file


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/upload', methods=['POST'])
def process_file():
    uploaded_file = request.files['file']
    
    if uploaded_file is not None:
        result = toggle_case_in_file(uploaded_file)
        return jsonify(message=result)

    return jsonify({"error": "Nofile"}, 400)
    
    return "Error", 400

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
