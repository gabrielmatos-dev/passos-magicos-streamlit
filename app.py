import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Passos Magicos - Predicao de Risco",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    modelo   = joblib.load("modelo_risco_defasagem.pkl")
    features = joblib.load("features_modelo.pkl")
    return modelo, features

modelo, features_modelo = carregar_modelo()

THRESHOLD = 0.4539

def classificar_faixa(prob):
    if prob < 0.30:
        return "Baixo", "#2ecc71"
    elif prob < 0.60:
        return "Medio", "#f39c12"
    return "Alto", "#e74c3c"

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

def gauge_risco(prob, faixa, cor):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": cor}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": cor},
            "steps": [
                {"range": [0, 30],   "color": "#d5f5e3"},
                {"range": [30, 60],  "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": THRESHOLD * 100
            }
        },
        title={"text": "Faixa: " + faixa, "font": {"size": 16}}
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig

INDICADORES = {
    "INDE":      "Indice de Desenvolvimento Educacional - nota global do aluno.",
    "IAA":       "Indice de Autoavaliacao - como o aluno percebe seu proprio desempenho.",
    "IEG":       "Indice de Engajamento - frequencia e participacao nas atividades.",
    "IPS":       "Indice Psicossocial - aspectos emocionais e sociais avaliados pela equipe.",
    "IPP":       "Indice Psicopedagogico - avaliacao das dificuldades de aprendizagem.",
    "IDA":       "Indice de Desempenho Academico - resultado nas avaliacoes escolares.",
    "IPV":       "Indice do Ponto de Virada - marco de transformacao no programa.",
    "IAN":       "Indice de Adequacao de Nivel - se o aluno esta na fase correta para sua idade.",
    "PEDRA":     "Fase do programa: Quartzo (1), Agata (2), Ametista (3) ou Topazio (4).",
    "DEFASAGEM": "Diferenca entre a fase atual e a ideal. Valores negativos indicam defasagem.",
    "DEFAS":     "Flag binaria: 1 se o aluno esta defasado, 0 caso contrario.",
}

st.title("🎓 Passos Magicos - Predicao de Risco de Defasagem")
st.markdown("Sistema preditivo para identificar alunos em risco de defasagem escolar.")

aba1, aba2, aba3 = st.tabs(["Predicao Individual", "Predicao em Lote", "Sobre os Indicadores"])

with aba1:
    st.markdown("##### Preencha os indicadores do aluno")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ano          = st.number_input("ANO",          min_value=2020, max_value=2035, value=2024)
        ano_ingresso = st.number_input("ANO_INGRESSO", min_value=2010, max_value=2035, value=2023)
        idade        = st.number_input("IDADE",        min_value=5,    max_value=30,   value=10)
        pedra        = st.number_input("PEDRA",        min_value=0.0,  max_value=10.0, value=3.0)

    with col2:
        inde = st.number_input("INDE", min_value=0.0, max_value=10.0, value=7.50, step=0.01)
        iaa  = st.number_input("IAA",  min_value=0.0, max_value=10.0, value=8.00, step=0.01)
        ieg  = st.number_input("IEG",  min_value=0.0, max_value=10.0, value=8.00, step=0.01)
        ips  = st.number_input("IPS",  min_value=0.0, max_value=10.0, value=8.00, step=0.01)

    with col3:
        ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=7.00, step=0.01)
        ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=7.00, step=0.01)
        ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=7.00, step=0.01)
        ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=7.00, step=0.01)

    with col4:
        defasagem = st.number_input("DEFASAGEM", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
        defas     = st.selectbox("DEFAS", options=[0, 1], index=0)

    tempo_programa       = ano - ano_ingresso
    idade_ingresso_aprox = idade - tempo_programa

    st.markdown(f"**Tempo no programa:** {tempo_programa} ano(s) | **Idade de ingresso aproximada:** {idade_ingresso_aprox} anos")

    if st.button("Calcular risco", type="primary"):
        entrada = pd.DataFrame([{
            "ANO": ano, "ANO_INGRESSO": ano_ingresso, "IDADE": idade,
            "PEDRA": pedra, "INDE": inde, "IAA": iaa, "IEG": ieg, "IPS": ips,
            "IPP": ipp, "IDA": ida, "IPV": ipv, "IAN": ian,
            "DEFASAGEM": defasagem, "DEFAS": defas,
            "TEMPO_PROGRAMA": tempo_programa, "IDADE_INGRESSO_APROX": idade_ingresso_aprox
        }])

        entrada_modelo = preparar_entrada(entrada, features_modelo)
        prob  = float(modelo.predict_proba(entrada_modelo)[0, 1])
        pred  = int(prob >= THRESHOLD)
        faixa, cor = classificar_faixa(prob)

        st.divider()
        g_col, r_col = st.columns([1, 1])

        with g_col:
            st.plotly_chart(gauge_risco(prob, faixa, cor), use_container_width=True)

        with r_col:
            st.metric("Probabilidade de risco", f"{prob:.1%}")
            st.metric("Classificacao", "Risco" if pred == 1 else "Sem risco")
            st.metric("Faixa", faixa)
            if pred == 1:
                st.warning("O aluno foi sinalizado como prioritario para acompanhamento preventivo.")
            else:
                st.success("O aluno nao apresentou sinal critico no limiar operacional atual.")

with aba2:
    st.markdown("##### Faca upload de uma planilha com multiplos alunos")
    st.markdown(
        "A planilha deve conter as colunas: "
        "`ANO, ANO_INGRESSO, IDADE, PEDRA, INDE, IAA, IEG, IPS, IPP, IDA, IPV, IAN, DEFASAGEM, DEFAS`"
    )

    arquivo = st.file_uploader("Upload da planilha (.xlsx ou .csv)", type=["xlsx", "csv"])

    if arquivo:
        try:
            if arquivo.name.endswith(".csv"):
                df_lote = pd.read_csv(arquivo)
            else:
                df_lote = pd.read_excel(arquivo)

            df_lote.columns = [c.strip().upper() for c in df_lote.columns]
            entrada_lote = preparar_entrada(df_lote, features_modelo)
            probas = modelo.predict_proba(entrada_lote)[:, 1]
            preds  = (probas >= THRESHOLD).astype(int)
            faixas = [classificar_faixa(p)[0] for p in probas]

            df_resultado = df_lote.copy()
            df_resultado["PROB_RISCO"]  = (probas * 100).round(1)
            df_resultado["PRED_RISCO"]  = ["Risco" if p == 1 else "Sem risco" for p in preds]
            df_resultado["FAIXA_RISCO"] = faixas

            st.success(f"{len(df_resultado)} alunos processados com sucesso.")

            r1, r2, r3 = st.columns(3)
            r1.metric("Total de alunos", len(df_resultado))
            r2.metric("Em risco", int(preds.sum()))
            r3.metric("Sem risco", int((preds == 0).sum()))

            st.dataframe(
                df_resultado[["PROB_RISCO", "PRED_RISCO", "FAIXA_RISCO"] + features_modelo],
                use_container_width=True
            )

            csv = df_resultado.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar resultado (.csv)",
                data=csv,
                file_name="resultado_risco.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

with aba3:
    st.markdown("##### O que significa cada indicador?")
    st.markdown(
        "Os indicadores abaixo fazem parte do sistema de avaliacao da "
        "Associacao Passos Magicos, utilizado para acompanhar o desenvolvimento "
        "educacional e socioemocional dos alunos."
    )
    for sigla, descricao in INDICADORES.items():
        st.markdown(f"**{sigla}** — {descricao}")

st.divider()
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem;'>"
    "Datathon - Data Analytics - FIAP - 2026"
    "</p>",
    unsafe_allow_html=True
)