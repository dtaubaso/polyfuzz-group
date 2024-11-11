import streamlit as st
import time, os, re
import pandas as pd
from polyfuzz import PolyFuzz
#from polyfuzz.models import Embeddings
#from flair.embeddings import WordEmbeddings
import plotly.express as px
#os.environ["FLAIR_CACHE_ROOT"] = "modelos_flair" 
#fasttext = WordEmbeddings('es')
#fasttext_matcher = Embeddings(fasttext)
model = PolyFuzz("TF-IDF")

st.set_page_config(page_title="Agrupar Top Query", page_icon=":penguin:")

# Configuración de la app Streamlit
st.title('Agrupa querys con Polyfuzz')
# Cargador de archivos CSV
uploaded_file = st.file_uploader("Suba un archivo CSV", type=["csv"])

def get_top_query(df, col_queries, col_num, brand):
    df = df.groupby(col_queries).sum(numeric_only=True).reset_index()
    pattern = '|'.join(rf'\b{palabra}\b' for palabra in brand)
    #df = df[~df[col_queries].isin(brand)]
    df = df[~df[col_queries].str.contains(pattern, case=False, na=False)]
    query = df[col_queries].tolist()
    model.match(query)
    model.group()
    df_first = model.get_matches()
    df_first = df_first[['From', 'Group']]
    df_first.columns = [col_queries, 'group']
    df_first.fillna("n/a", inplace=True)
    model.match(df_first['group'].unique().tolist())
    model.group()
    df_second = model.get_matches()
    df_second = df_second[['From', 'Group']]
    df_second.columns = ['group', 'group_group']
    df_second.fillna("n/a", inplace=True)
    df_merge = df.merge(df_first, on=col_queries, how='left')
    df_merge = df_merge.merge(df_second, on='group', how='left')
    df_merge.dropna(inplace=True)
    df_merge = df_merge.sort_values(['group_group', col_num], ascending=[True, False])
    df_merge['query_top'] = df_merge.groupby('group_group')[col_queries].transform('first')
    df_merge.drop(['group', 'group_group'], axis=1, inplace=True)
    return df_merge


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.dropna()
    cols = data.columns.tolist()
    col_queries = st.radio("Elija la columna que tiene las queries",
                           cols, horizontal=True)

    col_num = st.radio("Elija la columna que tiene la métrica",
                           cols, horizontal=True)
    brand = st.text_input("Ingrese separados por comas los términos que no quiere analizar")
    brand = list(map(str.strip, brand.split(',')))
    if st.button("Generar"):
        with st.spinner('Generando Gráfico...'):
            df_top_query = get_top_query(data, col_queries, col_num, brand)
        # Crear el treemap
        fig = px.treemap(df_top_query, path=['query_top'], values=col_num, 
                        color=col_num, color_continuous_scale='blues')
        fig.update_coloraxes(showscale=False)  # Desactivar la escala de colores

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig)

        # Convertir el DataFrame a CSV en bytes
        csv = df_top_query.to_csv(index=False).encode('utf-8')

        # Crear un botón de descarga
        st.download_button(
            label="Descargar como CSV",
            data=csv,
            file_name=f'top_query_agrupado_{int(time.time())}.csv',
            mime='text/csv'
        )