import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle
import sklearn
import numpy as np
# Cargar los datos
wage = pd.read_csv('salario1.csv')

st.title("Análisis de Datos - Salario")

with open('wage.pickle', 'rb') as m:
    model_loaded = pickle.load(m)

variables =['educ','exper',"female","married","numdep"]
# Crear pestañas
tab1, tab2, tab3 = st.tabs(["Análisis univariado", "Análisis bivariado","Modelo"])

# Análisis univariado
with tab1:
    st.header("Análisis Univariado")
    
    # 1. Histograma de Salario
    fig_salario = px.histogram(wage, x="salario", nbins=30, title="Distribución de Salario", marginal="box")
    st.plotly_chart(fig_salario)
    
    # 2. Histograma de Educación
    fig_educ = px.histogram(wage, x="educacion", nbins=15, title="Distribución de Educación", marginal="box")
    st.plotly_chart(fig_educ)

    # 3. Histograma de Experiencia
    fig_exper = px.histogram(wage, x="experiencia", nbins=20, title="Distribución de Experiencia", marginal="box")
    st.plotly_chart(fig_exper)

    # 4. Conteo de Sexo
    fig_sexo = px.bar(wage['sexo'].value_counts().reset_index(), x='sexo', y='count',
                      labels={'index': 'Sexo', 'sexo': 'Frecuencia'},
                      title="Distribución por Sexo")
    st.plotly_chart(fig_sexo)

    # 5. Conteo de Estado Civil
    fig_casado = px.bar(wage['casado'].value_counts().reset_index(), x='casado', y='count',
                        labels={'index': 'Estado Civil', 'casado': 'Frecuencia'},
                        title="Distribución por Estado Civil")
    st.plotly_chart(fig_casado)

    # 6. Histograma de Dependientes
    fig_dep = px.histogram(wage, x="dependientes", nbins=wage['dependientes'].nunique(),
                           title="Distribución de Dependientes", marginal="rug")
    st.plotly_chart(fig_dep)

    st.subheader("Resumen estadístico de las variables")
    st.dataframe(wage.describe())


# Análisis bivariado (aquí puedes agregar el análisis bivariado más tarde)
with tab2:
    st.header("Análisis Bivariado")
    st.write("En esta sección puedes realizar el análisis bivariado entre las variables.")
    # Ejemplo de gráfico bivariado:
    fig_biv = px.scatter(wage, x="educacion", y="salario", color="sexo", title="Salario vs Educación por Sexo")
    st.plotly_chart(fig_biv)

        # 2. Salario vs Experiencia
    fig_biv_exper = px.scatter(wage, x="experiencia", y="salario", color="sexo", title="Salario vs Experiencia por Sexo")
    st.plotly_chart(fig_biv_exper)

    # 3. Salario vs Dependientes
    fig_biv_dep = px.scatter(wage, x="dependientes", y="salario", color="sexo", title="Salario vs Dependientes por Sexo")
    st.plotly_chart(fig_biv_dep)
      
    st.header("Análisis Bivariado - Boxplots")

    # Boxplot: Salario por Género
    fig_box_sexo = px.box(wage, x="sexo", y="salario", title="Distribución del Salario por Género")
    st.plotly_chart(fig_box_sexo)

    # Boxplot: Salario por Estado Civil
    fig_box_civil = px.box(wage, x="casado", y="salario", title="Distribución del Salario por Estado Civil")
    st.plotly_chart(fig_box_civil)

    # Boxplot: Salario por Número de Dependientes (como categoría)
    wage['dependientes_cat'] = wage['dependientes'].astype(str)  # Convertir a categórica para x
    fig_box_dep = px.box(wage, x="dependientes_cat", y="salario",
                         title="Distribución del Salario por Número de Dependientes")
    st.plotly_chart(fig_box_dep)
with tab3:
    
    st.title("Ejemplo")
    educ = st.slider("Educación", 0, 99)
    exper = st.slider("Experiencia laboral", 1, 77)
    num_deps = st.slider("Número de dependientes",0,30)
    sexo = st.selectbox("Género",["Femenino","Masculino"])
    estado_civil = st.selectbox("Estado civil",["Casado","Soltero"])
    # Transformar las variables categóricas en valores numéricos
    if sexo == "Femenino":
        sexo_num = 1
    else:
        sexo_num = 0

    if estado_civil == "Casado":
        estado_civil_num = 1
    else:
        estado_civil_num = 0

    # Mostrar los valores seleccionados y transformados
    st.write(f"Educación: {educ}")
    st.write(f"Experiencia laboral: {exper}")
    st.write(f"Número de dependientes: {num_deps}")
    st.write(f"Género: {sexo} (Transformado a: {sexo_num})")
    st.write(f"Estado civil: {estado_civil} (Transformado a: {estado_civil_num})")  
    if st.button("predecir"):
        pred = model_loaded.predict(np.array([[educ,exper,sexo_num,estado_civil_num,num_deps]]))
        st.write(pred[0])