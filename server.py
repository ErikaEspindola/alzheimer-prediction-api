from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
import json
from flask_jsonpify import jsonify
import sys
import os
import base64

app = Flask(__name__)
api = Api(app)

CORS(app)

parser = reqparse.RequestParser()

class UploadFile(Resource):
    def post(self):
        target = os.path.join('/tmp')
        file = request.files['file']
        file_name = file.filename or ''
        destination = '/'.join([target, file_name])
        file.save(destination)

        with open(destination, "rb") as image_file:
            a = base64.b64encode(image_file.read()).decode("utf-8")
        return a

api.add_resource(UploadFile, '/upload_file') 


if __name__ == '__main__':
    app.run(port=5002, debug=True)
