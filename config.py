from configparser import ConfigParser
import os
from huggingface_hub import login

def load_config(filename='database.ini', section='postgresql'):
    # Prioridad 1: Leer variables de entorno (útil para GitHub Actions o Producción)
    if os.environ.get('DB_HOST'):
        return {
            'host': os.environ.get('DB_HOST'),
            'database': os.environ.get('DB_NAME'),
            'user': os.environ.get('DB_USER'),
            'password': os.environ.get('DB_PASSWORD')
        }
    
    # Prioridad 2: Leer archivo database.ini local
    parser = ConfigParser()
    if not os.path.exists(filename):
        raise Exception(f'El archivo {filename} no existe y no hay variables de entorno configuradas.')
    
    parser.read(filename)
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
        # Intentar obtener el token de variable de entorno o del archivo ini
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token and os.path.exists(filename):
            hf_config = load_config(filename=filename, section='huggingface')
            hf_token = hf_config.get('token')
            
        if hf_token:
            login(token=hf_token)
            print("Autenticado en HuggingFace")
            return True
        print("Sin token de HuggingFace configurado.")
        return False
    except Exception as e:
        print(f"No se pudo autenticar en HuggingFace ({e})")
        return False
