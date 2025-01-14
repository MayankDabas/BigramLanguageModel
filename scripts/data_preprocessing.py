import json
import os

with open('./data/raw/the_merchant_of_venice.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabulary = sorted(set(text))
char_to_int = { char:indx for indx, char in enumerate(vocabulary) }
int_to_char = { indx:char for indx, char in enumerate(vocabulary) }

def character_encoding(string):
    encode = [ char_to_int[char] for char in string ]
    return encode

def character_decoding(encode):
    decode = [int_to_char[i] for i in encode]
    string = ''.join(decode)

    return string

def save_data(data):
    data_dict = {'Data': data}
    PATH = './data/preprocessed_data'
    filename = 'data.json'

    os.makedirs(PATH, exist_ok=True)
    json_filepath = os.path.join(PATH, filename)

    with open(json_filepath, 'w') as f:
        json.dump(data_dict, f)
    
    print(f"Preprocessed data saved: {json_filepath}...")



def preprocess_data():
    data = character_encoding(text)
    save_data(data)

