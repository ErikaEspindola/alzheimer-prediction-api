import os
import sys
import json
import base64
import tempfile

from flask import Flask, request
from flask_jsonpify import jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse

from fsl.Helper_aux import helper_aux as hp
from fsl.Brain import brain
from fsl.Fsl_helper import fsl_helper 

app = Flask(__name__)
api = Api(app)

CORS(app)

parser = reqparse.RequestParser()

class UploadFile(Resource):

    def post(self):

        target = tempfile.gettempdir()
        file = request.files['file']
        file_name = file.filename or ''
        
        cerebro_original   = brain(target + '/' + hp.randomiza_nome(file_name))
        cerebro_sem_cranio = brain(target + '/' + hp.randomiza_nome(file_name))

        file.save(cerebro_original.caminho)

        print("Iniciando a remoção do crânio")
        fsl_helper.remove_cranio(cerebro_original, cerebro_sem_cranio)
        print("Remoção do crânio efetuada com sucesso")


        print("Iniciando normalização do cérebro para o espaço MNI")
        cerebro_normalizado = brain(target + '/' + hp.randomiza_nome(file_name))
        fsl_helper.normaliza_cerebro(cerebro_sem_cranio, cerebro_normalizado)
        print("Normalização do cérebro concluída com sucesso")
        
        resultado = []
        with open(cerebro_normalizado.caminho + '.gz', "rb") as image_file:
            resultado.append(base64.b64encode(image_file.read()).decode("utf-8"))

        with open(fsl_helper.fsl_caminho + '/' + fsl_helper.fsl_mascara_ref, "rb") as image_file:
            resultado.append(base64.b64encode(image_file.read()).decode("utf-8"))
        
        return resultado

api.add_resource(UploadFile, '/upload_file') 


if __name__ == '__main__':
    app.run(port=5002, debug=True)
