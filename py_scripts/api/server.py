import time
import tempfile

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse

import extract as ext
import show_slices as sl
import classification as classify

from fsl.Brain import brain
from fsl.Fsl_helper import fsl_helper
from fsl.Helper_aux import helper_aux as hp


app = Flask(__name__)
api = Api(app)

CORS(app)

parser = reqparse.RequestParser()

class UploadFile(Resource):

    def post(self):

        target = tempfile.gettempdir()
        file = request.files['file']
        file_name = file.filename or ''

        cerebro_original = brain(target + '/' + hp.randomiza_nome(file_name))
        cerebro_sem_cranio = brain(target + '/' + hp.randomiza_nome(file_name))

        file.save(cerebro_original.caminho)

        sl.show_slices(cerebro_original.caminho, 'Imagem do cérebro original')

        print("Iniciando a remoção do crânio")
        start_rem = time.time()
        fsl_helper.remove_cranio(cerebro_original, cerebro_sem_cranio)
        end_rem = time.time()
        print("Remoção do crânio efetuada com sucesso")
        tot_rem = end_rem - start_rem

        caminho_sem_cranio = ext.extract(cerebro_sem_cranio.caminho)

        sl.show_slices(caminho_sem_cranio, 'Imagem do cérebro sem crânio')

        print("Iniciando normalização do cérebro para o espaço MNI")
        cerebro_normalizado = brain(
            target + '/' + hp.randomiza_nome(file_name))
        start_norm = time.time()
        fsl_helper.normaliza_cerebro(cerebro_sem_cranio, cerebro_normalizado)
        end_norm = time.time()
        print("Normalização do cérebro concluída com sucesso")
        tot_norm = end_norm - start_norm

        caminho_normalizado = ext.extract(cerebro_normalizado.caminho)

        sl.show_slices(caminho_normalizado, 'Imagem do cérebro normalizado')

        start_class = time.time()
        res = int(classify.classification(caminho_normalizado))
        end_class = time.time()
        tot_class = end_class - start_class

        resultado = {
            'resultado': res,
            'tempo': {
                'remocao': tot_rem,
                'normalizacao': tot_norm,
                'classificacao': tot_class

            }
        }

        # resultado = []
        # with open(cerebro_normalizado.caminho + '.gz', "rb") as image_file:
        #     resultado.append(base64.b64encode(image_file.read()).decode("utf-8"))

        # with open(fsl_helper.fsl_caminho + '/' + fsl_helper.fsl_mascara_ref, "rb") as image_file:
        #     resultado.append(base64.b64encode(image_file.read()).decode("utf-8"))

        return resultado


api.add_resource(UploadFile, '/upload_file')


if __name__ == '__main__':
    app.run(port=5002, debug=True)
