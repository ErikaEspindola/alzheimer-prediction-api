from Brain import brain
import subprocess
import os

class fsl_helper:
    fsl_caminho      = '/usr/local/fsl'
    fsl_arquivo_ref  = 'data/standard/MNI152_T1_1mm_brain'
    fsl_mascara_ref  = 'data/atlases/Juelich/Juelich-maxprob-thr25-1mm.nii.gz'

    @staticmethod
    def remove_cranio(pre_brain, pos_brain):
        bash_dir = os.getcwd().split('/')
        bash_dir.pop()

        comando = '/'.join(bash_dir) + '/bash_scripts/remove_cranio.sh'
        comando = comando + ' ' + pre_brain.caminho + ' ' + pos_brain.caminho

        subprocess.run(comando.split())

    @staticmethod
    def normaliza_cerebro(pre_brain, pos_brain):
        bash_dir = os.getcwd().split('/')
        bash_dir.pop()

        comando = '/'.join(bash_dir) + '/bash_scripts/normaliza_cerebro.sh '       
        comando += fsl_helper.fsl_caminho + '/bin/flirt '
        comando += pre_brain.caminho + ' '
        comando += pos_brain.caminho + ' '
        comando += fsl_helper.fsl_caminho + '/' + fsl_helper.fsl_arquivo_ref

        subprocess.run(comando.split())



        
    


