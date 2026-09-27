from configparser import ConfigParser
from huggingface_hub import login

def load_config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)
    # get section, default to postgresql
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return config

def hf_login(filename='database.ini'):

    try:
        hf_config = load_config(filename=filename, section='huggingface')
        hf_token = hf_config.get('token')
        if hf_token:
            login(token=hf_token)
            print("Autenticado en HuggingFace con el token")
            return True
        print("Sin token de HuggingFace configurado, continuando sin autenticar.")
        return False
    except Exception as e:
        print(f"No se pudo leer/usar el token de HuggingFace ({e}), continuando sin autenticar.")
        return False
 
 
if __name__ == '__main__':
    config = load_config()
    print(config)