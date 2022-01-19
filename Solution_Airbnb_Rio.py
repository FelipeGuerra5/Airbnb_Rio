#!/usr/bin/env python
# coding: utf-8


# Import
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Data set consolidation
meses = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}

caminho_base = pathlib.Path("dataset")

base_airbnb = pd.DataFrame()

for arquivo in caminho_base.iterdir():
    mes = meses[arquivo.name[:3]]
    ano = int(arquivo.name[-8:-4])
    
    df = pd.read_csv(caminho_base / arquivo.name)
    df["ano"] = ano
    df["mes"] = mes
    base_airbnb = base_airbnb.append(df)

# display(base_airbnb)

# Cleaning unwanted columns
print(list(base_airbnb.columns))

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','host_identity_verified','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
# display(base_airbnb)

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
print(list(base_airbnb.columns))
# display(base_airbnb)

# Treating NaN Values
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum())

base_airbnb = base_airbnb.dropna()
print(base_airbnb.shape)

# Verifying data type in all columns
print(base_airbnb.dtypes)
print("-" * 80)
print(base_airbnb.iloc[0])

# Price is a object
base_airbnb["price"] = base_airbnb["price"].str.replace("$", "")
base_airbnb["price"] = base_airbnb["price"].str.replace(",", "")
base_airbnb["price"] = base_airbnb["price"].astype(np.float32, copy=False)

# Extra_people is a object
base_airbnb["extra_people"] = base_airbnb["extra_people"].str.replace("$", "")
base_airbnb["extra_people"] = base_airbnb["extra_people"].str.replace(",", "")
base_airbnb["extra_people"] = base_airbnb["extra_people"].astype(np.float32, copy=False)

# Verifying types again.
print(base_airbnb.dtypes)

# Treating outliers
plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap="Greens")

# Functions to help treat outliers


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 -  1.5 * amplitude, q3 + 1.5 * amplitude


def excluir_outliers(df, nome_coluna):
    qtd_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtd_linhas - df.shape[0]
    return df, linhas_removidas


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)


def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna)


def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


diagrama_caixa(base_airbnb["price"])
histograma(base_airbnb["price"])

# Treating each outlier
# Price
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print("Foram removidas {} linhas".format(linhas_removidas))

histograma(base_airbnb["price"])

# Extra_People
diagrama_caixa(base_airbnb["extra_people"])
histograma(base_airbnb["extra_people"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print("Foram removidas {} linhas".format(linhas_removidas))

base_airbnb.shape

# Hosting listing count
diagrama_caixa(base_airbnb["host_listings_count"])
grafico_barra(base_airbnb["host_listings_count"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print("Foram removidas {} linhas".format(linhas_removidas))


# Accommodates
diagrama_caixa(base_airbnb["accommodates"])
grafico_barra(base_airbnb["accommodates"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print("Foram removidas {} linhas".format(linhas_removidas))


# Bathrooms
diagrama_caixa(base_airbnb["bathrooms"])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb["bathrooms"].value_counts().index, y=base_airbnb["bathrooms"].value_counts())

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print("Foram removidas {} linhas".format(linhas_removidas))


# Bedrooms
diagrama_caixa(base_airbnb["bedrooms"])
grafico_barra(base_airbnb["bedrooms"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print("Foram removidas {} linhas".format(linhas_removidas))


# Beds
diagrama_caixa(base_airbnb["beds"])
grafico_barra(base_airbnb["beds"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print("Foram removidas {} linhas".format(linhas_removidas))

# Guests_included
print(limites(base_airbnb["guests_included"]))
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb["guests_included"].value_counts().index, y=base_airbnb["guests_included"].value_counts())

base_airbnb = base_airbnb.drop("guests_included", axis=1)

# Minimum_nights
diagrama_caixa(base_airbnb["minimum_nights"])
grafico_barra(base_airbnb["minimum_nights"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print("Foram removidas {} linhas".format(linhas_removidas))

# Maximum_nights
diagrama_caixa(base_airbnb["maximum_nights"])
grafico_barra(base_airbnb["maximum_nights"])

base_airbnb = base_airbnb.drop("maximum_nights", axis=1)
base_airbnb.shape

# Number_of_reviews
diagrama_caixa(base_airbnb["number_of_reviews"])
grafico_barra(base_airbnb["number_of_reviews"])

base_airbnb = base_airbnb.drop("number_of_reviews", axis=1)



# Treating columns with text

# Property_type
print(base_airbnb["property_type"].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot("property_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)

tabela_tipos_casa = base_airbnb["property_type"].value_counts()
lista_tipos = base_airbnb["property_type"].value_counts().index

lista_filtro = []

for tipo in lista_tipos:
     if tabela_tipos_casa[tipo] < 2000:
            lista_filtro.append(tipo)
            
print(lista_filtro)

for tipo in lista_filtro:
    base_airbnb.loc[base_airbnb["property_type"]==tipo, "property_type"] = "Outros"

print(base_airbnb["property_type"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("property_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)


# Room type
print(base_airbnb["room_type"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("room_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)

# Bed_type
print(base_airbnb["bed_type"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("bed_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)

# Joining smaller categories of bed_type
tabela_tipos_casa = base_airbnb["bed_type"].value_counts()
lista_tipos = base_airbnb["bed_type"].value_counts().index

lista_filtro = []

for tipo in lista_tipos:
     if tabela_tipos_casa[tipo] < 10000:
            lista_filtro.append(tipo)
            
print(lista_filtro)

for tipo in lista_filtro:
    base_airbnb.loc[base_airbnb["bed_type"]==tipo, "bed_type"] = "Outros"

print(base_airbnb["bed_type"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("bed_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)


# Cancellation_policy
tabela_tipos_casa = base_airbnb["cancellation_policy"].value_counts()
lista_tipos = base_airbnb["cancellation_policy"].value_counts().index

lista_filtro = []

for tipo in lista_tipos:
     if tabela_tipos_casa[tipo] < 10000:
            lista_filtro.append(tipo)
            
print(lista_filtro)

for tipo in lista_filtro:
    base_airbnb.loc[base_airbnb["cancellation_policy"]==tipo, "cancellation_policy"] = "Strict"

print(base_airbnb["cancellation_policy"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("cancellation_policy", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)


# Amenities
print(base_airbnb["amenities"].iloc[1].split(","))
print(len(base_airbnb["amenities"].iloc[1].split(",")))

base_airbnb["n_amenities"] = base_airbnb["amenities"].str.split(",").apply(len)

base_airbnb = base_airbnb.drop("amenities", axis=1)

diagrama_caixa(base_airbnb["n_amenities"])
grafico_barra(base_airbnb["n_amenities"])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print("Foram removidas {} linhas".format(linhas_removidas))


# Visualização de Mapa dos Imóveis
amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat="latitude", lon="longitude", z="price", radius=2.5,
                        center=centro_mapa,
                        zoom=10,
                        mapbox_style="stamen-terrain")
mapa.show()


# Encoding
# Changing the text in columns to separated columns with a binary value.
colunas_tf = ["host_is_superhost", "instant_bookable", "is_business_travel_ready"]
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=="t", coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=="f", coluna] = 0
print(base_airbnb_cod.iloc[0])

colunas_categoria = ["property_type", "room_type", "bed_type", "cancellation_policy"]
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categoria)
# display(base_airbnb_cod.head())


#  Modelo de Previsão
def avaliar_modelo(nome_modelo, y_teste, previsão):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f"Modelo {nome_modelo}: \nR²:{r2:.2%}\nRSME:{RSME:.2f}"


# - Choosing the best model to use
#     1. Random Forest
#     2. Linear Regression
#     3. Extra Tree

modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {"RamdomForest": modelo_rf, 
           "LinearRegression": modelo_lr, 
           "ExtraTree": modelo_et
          }

y = base_airbnb_cod["price"]
X = base_airbnb_cod.drop("price", axis=1)


# - Splitting the data in to training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    # train
    modelo.fit(X_train, y_train)
    # test
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


for nome_modelo, modelo in modelos.items():
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))

# Ajustes e Melhorias no Melhor Modelo

importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
# display(importancia_features)

plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis="x", rotation=90)

# Final Adjusting
base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))

base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:    
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# # Deploy
# 
# - Step 1 -> Create model file (jobib)
# - Step 2 -> Choose deploy format deploy:
#     - .exe file + Tkinter
#     - Micro site with Flask
#     - Just running it easily with streamlit.
# - Step 3 -> Other py file to run it.
# - Step 4 -> Import streamlit e create code.
# - Step 5 -> Configurate button
# - Step 6 -> Deploy done.

X["price"] = y
X.to_csv("dados.csv")

get_ipython().run_cell_magic('time', '', '\nimport joblib\njoblib.dump(modelo_et, "modelo.joblib")')

