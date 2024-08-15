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

# Cargar el modelo GPT-2 para generación de texto
tokenizer = GPT2Tokenizer.from_pretrained('./models/myModelgpt2')
model = GPT2LMHeadModel.from_pretrained('./models/myModelgpt2')

def generar_respuesta(input_text, tipo_pregunta):
    prompt = (
        f"Responde a la siguiente pregunta de manera útil y concisa, considerando el tipo de pregunta especificado. "
        f"Pregunta: {input_text}. "
        f"Tipo de pregunta: {tipo_pregunta}. "
        f"Respuesta:"
    )
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=30,
            top_p=0.95,
            temperature=0.7,  # Lower temperature for more deterministic results
            pad_token_id=tokenizer.eos_token_id  # Use eos_token_id for padding
        )
    
    # Decode the response
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta.strip()

def predecir_tipo_pregunta(input_text, modelo_rf_unbalanced, model_w2v):
    texto_limpio = limpiar_texto(input_text)
    input_data = obtener_vector_promedio(texto_limpio, model_w2v)
    input_data = np.array([input_data])
    prediction = modelo_rf_unbalanced.predict(input_data)
    return prediction[0]

def main():
    # Cargar el modelo desde el archivo
    with open('model/model.pkl', 'rb') as file:
        modelo_rf_unbalanced = pickle.load(file)
        
    # Cargar el modelo Word2Vec
    model_w2v = Word2Vec.load("./models/word2vec.model")
    
    input_text = input("Ingresa el texto para hacer la predicción: ")
    
    tipo_pregunta = predecir_tipo_pregunta(input_text, modelo_rf_unbalanced, model_w2v)
    
    # Generar una respuesta usando GPT-2
    respuesta = generar_respuesta(input_text, tipo_pregunta)

    print("Respuesta:", respuesta)

if __name__ == "__main__":
    main()