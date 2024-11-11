import json
import os

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, jsonify, request, send_file, send_from_directory

os.environ["GOOGLE_API_KEY"] = "AIzaSyAYekVc945Ld70rwSu1a2hH5f9_g5S5lvI"; 

app = Flask(__name__)


if __name__ == "__main__":
    app.run(port=int(os.environ.get('PORT', 80)))
