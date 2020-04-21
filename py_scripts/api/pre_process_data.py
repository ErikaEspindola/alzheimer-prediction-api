import glob
import os

from fsl.Helper_aux import helper_aux as hp
from fsl.Fsl_helper import fsl_helper
from fsl.Brain import brain
from fsl.Recorta_ROI import Recorta_ROI as roi

def processar_dados():
    for caminho_original in glob.iglob('/home/erika/New_ADNI2/**/**', recursive=False):
        id = str(caminho_original.split('/')[4:5][0])

        os.makedirs('/home/erika/RemocaoCranio/' + id)
        os.makedirs('/home/erika/NormalizacaoCerebro/' + id)
        os.makedirs('/home/erika/PreProcessamento/' + id)

        caminho_remove_cranio     = '/home/erika/RemocaoCranio/' + id + '/' + id + '.nii'
        caminho_normaliza_cerebro = '/home/erika/NormalizacaoCerebro/' + id + '/' + id + '.nii'
        caminho_regiao_interesse  = '/home/erika/PreProcessamento/' + id + '/' + id + '.nii'

        print('Iniciando remoção do crânio')
        fsl_helper.remove_cranio(brain(caminho_original), brain(caminho_remove_cranio))
        print('Crânio removido')

        print('Iniciando normalização do cérebro')
        fsl_helper.normaliza_cerebro(brain(caminho_remove_cranio), brain(caminho_normaliza_cerebro))
        print('Cérebro normalizado')

        print('Iniciando recorte da região de interesse')
        roi.cutBrainMask(caminho_normaliza_cerebro + '.gz', caminho_regiao_interesse)
        print('Recorte da região de interesse finalizado')

processar_dados()