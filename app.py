import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

def calcular_probabilidades_posteriori(datos_paciente, subgrupos, priors, medias, desvios):
    """
    Calcula las probabilidades a posteriori de cada subgrupo usando el Teorema de Bayes.
    """
    likelihoods = {}
    evidencias = []
    
    for subgrupo in subgrupos:
        # Calcular la probabilidad condicional para cada variable usando la distribución normal
        prob_condicional = 1
        for var in datos_paciente:
            media = medias[subgrupo][var]
            desvio = desvios[subgrupo][var]
            probabilidad = norm.pdf(datos_paciente[var], media, desvio)
            prob_condicional *= probabilidad
        
        # Probabilidad del subgrupo dado los datos
        likelihoods[subgrupo] = prob_condicional * priors[subgrupo]
        evidencias.append(likelihoods[subgrupo])
    
    # Normalizar para obtener probabilidades a posteriori
    evidencia_total = sum(evidencias)
    probabilidades_posteriori = {subgrupo: likelihoods[subgrupo] / evidencia_total for subgrupo in subgrupos}
    
    return probabilidades_posteriori

# Definir los subgrupos en el orden correcto con sus definiciones
subgrupos = [
    "SAID (Diabetes Autoinmune Severa)",
    "SIDD (Diabetes Severa Deficiente en Insulina)",
    "SIRD (Diabetes Severa Resistente a la Insulina)",
    "MOD (Diabetes Leve Relacionada con la Obesidad)",
    "MARD (Diabetes Leve Relacionada con la Edad)"
]

# Probabilidades previas basadas en el estudio ANDIS (2018)
priors = {
    "SAID (Diabetes Autoinmune Severa)": 0.064,
    "SIDD (Diabetes Severa Deficiente en Insulina)": 0.175,
    "SIRD (Diabetes Severa Resistente a la Insulina)": 0.153,
    "MOD (Diabetes Leve Relacionada con la Obesidad)": 0.216,
    "MARD (Diabetes Leve Relacionada con la Edad)": 0.391
}

# Conversión de HbA1c de mmol/mol a %
def convertir_hba1c_mmol_a_porcentaje(mmol):
    return (mmol / 10.929) + 2.15

# Medias y desviaciones estándar de cada subgrupo en %
medias = {
    "SAID (Diabetes Autoinmune Severa)": {"HbA1c": convertir_hba1c_mmol_a_porcentaje(85), "IMC": 25, "Edad": 45, "Filtrado_Glomerular": 90},
    "SIDD (Diabetes Severa Deficiente en Insulina)": {"HbA1c": convertir_hba1c_mmol_a_porcentaje(85.5), "IMC": 31.9, "Edad": 53.6, "Filtrado_Glomerular": 85},
    "SIRD (Diabetes Severa Resistente a la Insulina)": {"HbA1c": convertir_hba1c_mmol_a_porcentaje(47.5), "IMC": 36.4, "Edad": 59.2, "Filtrado_Glomerular": 70},
    "MOD (Diabetes Leve Relacionada con la Obesidad)": {"HbA1c": convertir_hba1c_mmol_a_porcentaje(49.0), "IMC": 31.5, "Edad": 48.6, "Filtrado_Glomerular": 80},
    "MARD (Diabetes Leve Relacionada con la Edad)": {"HbA1c": convertir_hba1c_mmol_a_porcentaje(46.3), "IMC": 27.9, "Edad": 66.9, "Filtrado_Glomerular": 75}
}

desvios = {
    "SAID (Diabetes Autoinmune Severa)": {"HbA1c": 1.0, "IMC": 3, "Edad": 5, "Filtrado_Glomerular": 10},
    "SIDD (Diabetes Severa Deficiente en Insulina)": {"HbA1c": 1.3, "IMC": 5.7, "Edad": 10.7, "Filtrado_Glomerular": 12},
    "SIRD (Diabetes Severa Resistente a la Insulina)": {"HbA1c": 0.7, "IMC": 6.1, "Edad": 9.9, "Filtrado_Glomerular": 15},
    "MOD (Diabetes Leve Relacionada con la Obesidad)": {"HbA1c": 0.8, "IMC": 5.1, "Edad": 8.1, "Filtrado_Glomerular": 10},
    "MARD (Diabetes Leve Relacionada con la Edad)": {"HbA1c": 0.6, "IMC": 4.0, "Edad": 6.6, "Filtrado_Glomerular": 12}
}

# Aplicación web con Streamlit
st.title("Calculadora Bayesiana de Subgrupos de Diabetes")
st.write("Ingrese los valores del paciente para predecir su subgrupo de diabetes.")

# Inputs de usuario
hbA1c = st.number_input("Hemoglobina Glicosilada (HbA1c en %)", min_value=4.0, max_value=15.0, value=6.5)
imc = st.number_input("Índice de Masa Corporal (IMC)", min_value=15.0, max_value=50.0, value=29.0)
edad = st.number_input("Edad", min_value=18, max_value=100, value=55)
filtrado_glomerular = st.number_input("Filtrado Glomerular", min_value=10, max_value=120, value=80)

# Botón para calcular y reiniciar
if st.button("Calcular Subgrupo"):
    paciente = {"HbA1c": hbA1c, "IMC": imc, "Edad": edad, "Filtrado_Glomerular": filtrado_glomerular}
    resultados = calcular_probabilidades_posteriori(paciente, subgrupos, priors, medias, desvios)
    st.write("### Resultados de Clasificación")
    for subgrupo, probabilidad in resultados.items():
        st.write(f"{subgrupo}: {probabilidad:.2%}")

if st.button("Borrar Datos"):
    st.rerun()
