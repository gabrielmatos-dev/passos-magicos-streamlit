
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Passos Mágicos - Predição Individual",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load("modelo_risco_defasagem.pkl")
    features = joblib.load("features_modelo.pkl")
    return modelo, features

modelo, features_modelo = carregar_modelo()

THRESHOLD = 0.4539

def classificar_faixa(prob):
    if prob < 0.30:
        return "Baixo"
    elif prob < 0.60:
        return "Médio"
    return "Alto"

def preparar_entrada(df, features_esperadas):
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    for col in features_esperadas:
        if col not in df.columns:
            df[col] = np.nan

    df = df[features_esperadas].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

st.title("🎓 Passos Mágicos — Predição Individual de Risco")
st.markdown(
    "Preencha os indicadores do aluno para estimar a **probabilidade de entrada em risco de defasagem**."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    ano = st.number_input("ANO", min_value=2020, max_value=2035, value=2024)
    ano_ingresso = st.number_input("ANO_INGRESSO", min_value=2010, max_value=2035, value=2023)
    idade = st.number_input("IDADE", min_value=5, max_value=30, value=10)
    pedra = st.number_input("PEDRA", min_value=0.0, max_value=10.0, value=3.0)

with col2:
    inde = st.number_input("INDE", min_value=0.0, max_value=10.0, value=7.50, step=0.01)
    iaa = st.number_input("IAA", min_value=0.0, max_value=10.0, value=8.00, step=0.01)
    ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=8.00, step=0.01)
    ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=8.00, step=0.01)

with col3:
    ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=7.00, step=0.01)
    ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=7.00, step=0.01)
    ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=7.00, step=0.01)
    ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=7.00, step=0.01)

with col4:
    defasagem = st.number_input("DEFASAGEM", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    defas = st.selectbox("DEFAS", options=[0, 1], index=0)

tempo_programa = ano - ano_ingresso
idade_ingresso_aprox = idade - tempo_programa

st.write(f"**Tempo no programa:** {tempo_programa}")
st.write(f"**Idade de ingresso aproximada:** {idade_ingresso_aprox}")

if st.button("Calcular risco", type="primary"):
    entrada = pd.DataFrame([{
        "ANO": ano,
        "ANO_INGRESSO": ano_ingresso,
        "IDADE": idade,
        "PEDRA": pedra,
        "INDE": inde,
        "IAA": iaa,
        "IEG": ieg,
        "IPS": ips,
        "IPP": ipp,
        "IDA": ida,
        "IPV": ipv,
        "IAN": ian,
        "DEFASAGEM": defasagem,
        "DEFAS": defas,
        "TEMPO_PROGRAMA": tempo_programa,
        "IDADE_INGRESSO_APROX": idade_ingresso_aprox
    }])

    entrada_modelo = preparar_entrada(entrada, features_modelo)

    prob = float(modelo.predict_proba(entrada_modelo)[0, 1])
    pred = int(prob >= THRESHOLD)
    faixa = classificar_faixa(prob)

    k1, k2, k3 = st.columns(3)
    k1.metric("Probabilidade de risco", f"{prob:.1%}")
    k2.metric("Classificação", "Risco" if pred == 1 else "Sem risco")
    k3.metric("Faixa", faixa)

    st.progress(max(0.0, min(prob, 1.0)))

    if pred == 1:
        st.warning("O aluno foi sinalizado como prioritário para acompanhamento preventivo.")
    else:
        st.success("O aluno não apresentou sinal crítico no limiar operacional atual.")
