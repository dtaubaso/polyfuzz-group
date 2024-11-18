import streamlit as st
import time, os, re, psutil
import pandas as pd
from polyfuzz import PolyFuzz
#from polyfuzz.models import Embeddings
#from flair.embeddings import WordEmbeddings
import plotly.express as px
import plotly.graph_objects as go
#os.environ["FLAIR_CACHE_ROOT"] = "modelos_flair" 
#fasttext = WordEmbeddings('es')
#fasttext_matcher = Embeddings(fasttext)
model = PolyFuzz.load('rapidfuzz_matcher')

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert to MB

def get_top_query(df, col_queries, col_num, brand):
    if brand:
        pattern = '|'.join(rf'\b{palabra}\b' for palabra in brand)
        df = df[~df[col_queries].str.contains(pattern, case=False, na=False)]
    df = df[[col_queries, col_num]]
    df = df.groupby(col_queries).sum(numeric_only=True).sort_values(col_num, ascending=False).reset_index().head(20000)
    query = df[col_queries].tolist()
    model.match(query)
    model.group(link_min_similarity=0.5)
    print(f"Memory usage: {get_memory_usage()}")
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
    model.match(df_second['group_group'].unique().tolist())
    model.group()
    df_third = model.get_matches()
    df_third = df_third[['From', 'Group']]
    df_third.columns = ['group_group', 'group_group_group']
    df_third.fillna("n/a", inplace=True)
    df_merge = df.merge(df_first, on=col_queries, how='left')
    df_merge = df_merge.merge(df_second, on='group', how='left')
    df_merge = df_merge.merge(df_third, on='group_group', how='left')
    df_merge.dropna(inplace=True)
    df_merge = df_merge.sort_values(['group_group_group', col_num], ascending=[True, False])
    df_merge['query_top'] = df_merge.groupby('group_group')[col_queries].transform('first')
    df_merge.drop(['group', 'group_group', 'group_group_group'], axis=1, inplace=True)
    return df_merge

def generate_treemap(df, col_num, col_queries):
    # Calcular la suma de clics por grupo
    group_clicks_sum = df.groupby('query_top')[col_num].sum().reset_index()

    # Seleccionar los 10 grupos con la mayor suma de clics
    top_10_groups = group_clicks_sum.nlargest(10, col_num)['query_top']

    # Filtrar el DataFrame original para mantener solo los grupos seleccionados
    df_filtered = df[df['query_top'].isin(top_10_groups)]

    # Dentro de cada uno de estos 10 grupos, seleccionar las 10 consultas con más clics
    df_filtered = df_filtered.sort_values(['query_top', col_num], ascending=[True, False]) \
                         .groupby('query_top').head(10).reset_index(drop=True)

    fig = px.treemap(
    df_filtered,
    path=["query_top", col_queries],  # Define la jerarquía (nivel 1: grupo, nivel 2: consulta)
    values=col_num,              # El tamaño de cada cuadro se basa en el valor de 'clicks'
    title="Top 10 Query tops y Top 10 Consultas",
    maxdepth=2,
    color_continuous_scale='blues'
)
    

# Configurar para que solo muestre el nivel de "grupo" inicialmente
    #fig.update_traces(maxdepth=1, root_color="lightgrey")
    #fig.update_coloraxes(showscale=False)  # Desactivar la escala de colores
    return fig
st.set_page_config(page_title="Agrupar Top Query", page_icon=":penguin:")

# Configuración de la app Streamlit
st.title('Agrupa querys con Polyfuzz')
# Cargador de archivos CSV
uploaded_file = st.file_uploader("Suba un archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.dropna()
    cols = data.columns.tolist()
    col_queries = st.radio("Elija la columna que tiene las queries",
                           cols, horizontal=True)

    col_num = st.radio("Elija la columna que tiene la métrica",
                           cols, horizontal=True)
    brand = st.text_input("Ingrese separados por comas los términos que no quiere analizar")
    if brand:
        brand = list(map(str.strip, brand.split(',')))
    if st.button("Generar"):
        with st.spinner('Generando Gráfico...'):
            df_top_query = get_top_query(data, col_queries, col_num, brand)
            
        memory_usage = get_memory_usage()
        st.write(memory_usage)
        fig = generate_treemap(df_top_query, col_num, col_queries)
        

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Convertir el DataFrame a CSV en bytes
        csv = df_top_query.to_csv(index=False).encode('utf-8')

        # Crear un botón de descarga
        st.download_button(
            label="Descargar como CSV",
            data=csv,
            file_name=f'top_query_agrupado_{int(time.time())}.csv',
            mime='text/csv'
        )