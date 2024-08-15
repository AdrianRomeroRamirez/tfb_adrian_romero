from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import pickle
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')

def limpiar_texto(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = word_tokenize(texto)
    palabras_limpias = [palabra for palabra in palabras if palabra not in stopwords.words('spanish')]
    texto_limpio = ' '.join(palabras_limpias)
    return texto_limpio

def obtener_vector_promedio(texto, model_w2v):
    tamano_vector = model_w2v.vector_size
    vector_promedio = np.zeros(tamano_vector)
    palabras_validas = [palabra for palabra in texto if palabra in model_w2v.wv.key_to_index]
    if palabras_validas:
        vectores_palabras = np.array([model_w2v.wv[palabra] for palabra in palabras_validas])
        vector_promedio = vectores_palabras.mean(axis=0)
    return vector_promedio

def predecir_tipo_pregunta(input_text):
    with open('../model/model.pkl', 'rb') as file:
      modelo_rf_unbalanced = pickle.load(file)

    model_w2v = Word2Vec.load("../models/word2vec.model")

    texto_limpio = limpiar_texto(input_text)
    input_data = obtener_vector_promedio(texto_limpio, model_w2v)
    input_data = np.array([input_data])
    prediction = modelo_rf_unbalanced.predict(input_data)
    return prediction[0]

def generar_respuesta(input_text, tipo_pregunta):
    tokenizer = GPT2Tokenizer.from_pretrained('../models/myModelgpt2')
    model = GPT2LMHeadModel.from_pretrained('../models/myModelgpt2')

    prompt = (
        f"Answer the following question in a helpful and concise manner, considering the specified question type. "
        f"Question: {input_text}. "
        f"Question type: {tipo_pregunta}. "
        f"Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=25,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta = respuesta.replace(prompt, "").strip()

    # Post-procesamiento básico para encontrar el último punto
    ultimo_punto = respuesta.rfind('.')
    if ultimo_punto != -1:
        respuesta = respuesta[:ultimo_punto+1]
    return respuesta