class brain:
    caminho = ''
    nome    = ''

    def __init__(self, caminho):
        self.caminho = caminho
        self.nome    = caminho.split('/')[-1]   
