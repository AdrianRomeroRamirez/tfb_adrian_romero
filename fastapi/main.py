from fastapi import FastAPI, Body
from pydantic import BaseModel
from functions import predecir_tipo_pregunta, generar_respuesta

class PreguntaMedica(BaseModel):
    pregunta: str

app = FastAPI()

@app.post("/responder-pregunta")
async def responder_pregunta(pregunta: PreguntaMedica = Body(...)):
  """
  Esta ruta recibe una pregunta médica por el cuerpo de la solicitud y lo devuelve.

  Args:
      pregunta (PreguntaMedica): La pregunta a recibir.

  Returns:
      str: La respuesta.
  """

  tipo_pregunta = predecir_tipo_pregunta(pregunta.pregunta)

  respuesta = generar_respuesta(pregunta.pregunta, tipo_pregunta)

  return {"respuesta": respuesta}




