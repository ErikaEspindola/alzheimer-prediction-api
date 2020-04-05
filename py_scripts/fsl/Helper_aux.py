import string
import random

class helper_aux:

    @staticmethod
    def randomiza_nome(nome):
        res = nome
        pre = ''

        for i in range(6):
            pre = pre + (random.choice(string.ascii_uppercase + string.digits))
        res = pre + '_' + res
      
        return res       