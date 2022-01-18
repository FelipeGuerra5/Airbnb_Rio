#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Nosso objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Elas estão disponíveis para download abaixo da aula (se você puxar os dados direto do Kaggle pode ser que encontre resultados diferentes dos meus, afinal as bases de dados podem ter sido atualizadas).
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importar Bibliotecas e Bases de Dados

# In[1]:


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


# ### Consolidar Base de Dados

# In[2]:


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

display(base_airbnb)


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir

# - Com o objetivo de aumentar a velocidade e ter um data frame mais claro e voltado para as analise de machine learning:
# - As colunas não relevantes para a analise serão excluidas a seguir.
# - Tipos que serão excluidos
#     1. IDs, links e informações não relevantes para o modelo
#     2. Colunas redundantes; ex: Data x ano/mes
#     3. Colunas com texto livre
#     4. Colunas em que todos, ou quase todos são iguais.
#     

# In[3]:


print(list(base_airbnb.columns))

# base_airbnb.head(1000).to_excel("Primeiros Registros.xlsx")print(base_airbnb["experiences_offered"].value_counts())
#Posso excluir.print((base_airbnb["host_listings_count"]==base_airbnb["host_total_listings_count"]).value_counts())
#diferença infima, posso ecluir uma.print(base_airbnb["square_feet"].isnull().sum())
#Posso excluir pois a maior parte é nula
# ### Depois de tratar excluir as colunas indesejáveis. fazer uma lista com o nome das colunas do DF tratado, usando a formula do excel 
# #### ="'"&celula&"'" -> para concatenar o nome com aspas, em seguida 
# #### =unirtexto(","; falso; seleção de celulas) -> para ter uma lista, que usarei no notebook Jupyter.
# 

# In[4]:


colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','host_identity_verified','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
display(base_airbnb)

colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']

base_airbnb = base_airbnb.loc[:, colunas]
print(list(base_airbnb.columns))
display(base_airbnb)


# ### Tratar Valores Faltando
# 
# - Devido ao alto numero de celulas nulas de algumas colunas uma limpeza foi feita, excluindo as colunas com mais de 300 mil linhas nulas.
# - Foram retiradas tambem as linhas nulas, tendo eem vista que não sua exclusão é desprezavel em relação a amostra.

# In[5]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
print(base_airbnb.isnull().sum())


# In[6]:


base_airbnb = base_airbnb.dropna()
print(base_airbnb.shape)


# ### Verificar Tipos de Dados em cada coluna

# In[7]:


print(base_airbnb.dtypes)
print("-" * 80)
print(base_airbnb.iloc[0])


# In[8]:


#price esta como object
base_airbnb["price"] = base_airbnb["price"].str.replace("$", "")
base_airbnb["price"] = base_airbnb["price"].str.replace(",", "")
base_airbnb["price"] = base_airbnb["price"].astype(np.float32, copy=False)
#extra_people esta como object
base_airbnb["extra_people"] = base_airbnb["extra_people"].str.replace("$", "")
base_airbnb["extra_people"] = base_airbnb["extra_people"].str.replace(",", "")
base_airbnb["extra_people"] = base_airbnb["extra_people"].astype(np.float32, copy=False)
#verificando os tipos
print(base_airbnb.dtypes)


# ### Análise Exploratória e Tratar Outliers
# 
# - Vamos basicamente olhar as features para:
# 
#     1. Ver a correlação entre as feaures e decidirse manterems todas ass eatures que temos.
#     2. Excluir outliers (usaremos como regra, valores abaixo de Q1 - 1,5 x amplitude Q3 + 1,5 x Amplitude, amplitude = Q3 - Q1)
#     3. Comfirmar se todas as features que temos faem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir.
# 
# - Vamos começar pelas colunas de preço (resultado que queremos) e de extra_people (também valor monetário. essses são os valores numéricos contínuos.
# - Depois vamos analisar as colunas de valores numéricos discrtos (accomodates, bedroom, guests_included, etc)
# - Por fim, va,os avaiar as colunas fe texto e definir quais categorias fazem sentido mantermos ou não.

# In[9]:


plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap="Greens")


# Vamos definier algumas funções para ajudar na analise de outliers nas colunas.

# In[10]:


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


# In[11]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)

def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna)
    #histplot
    
def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# # Price

# In[12]:


diagrama_caixa(base_airbnb["price"])
histograma(base_airbnb["price"])


# Como estamos construindo um modelo para imóveis comuns, acredito que os valores cima do limite superior serão papenas de apartamentos de alto padrão, o que não é o escopo deste projeto, sendo assim excluirei estes outliers.

# In[13]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print("Foram removidas {} linhas".format(linhas_removidas))


# In[14]:


histograma(base_airbnb["price"])


# ## Extra people

# In[15]:


diagrama_caixa(base_airbnb["extra_people"])
histograma(base_airbnb["extra_people"])


# In[16]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print("Foram removidas {} linhas".format(linhas_removidas))


# In[17]:


base_airbnb.shape


# In[18]:


diagrama_caixa(base_airbnb["host_listings_count"])
grafico_barra(base_airbnb["host_listings_count"])


# De acordo com o escopo do projeto (auxiliar o pequeno locador ou comprador) a precificar seu imóvel, não há necessidade de manter "hosts" com alto numero de imóveis.

# In[19]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Accommodates

# In[20]:


diagrama_caixa(base_airbnb["accommodates"])
grafico_barra(base_airbnb["accommodates"])


# In[21]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Bathrooms

# In[22]:


diagrama_caixa(base_airbnb["bathrooms"])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb["bathrooms"].value_counts().index, y=base_airbnb["bathrooms"].value_counts())


# In[23]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Bedrooms

# In[24]:


diagrama_caixa(base_airbnb["bedrooms"])
grafico_barra(base_airbnb["bedrooms"])


# In[25]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Beds

# In[26]:


diagrama_caixa(base_airbnb["beds"])
grafico_barra(base_airbnb["beds"])


# In[27]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Guests_included

# In[28]:


#diagrama_caixa(base_airbnb["guests_included"])
#grafico_barra(base_airbnb["guests_included"])
print(limites(base_airbnb["guests_included"]))
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb["guests_included"].value_counts().index, y=base_airbnb["guests_included"].value_counts())


# Tendo em vista que a coluna "Guests included" conta com sua grande maioria em apenas 1, levando-nos a entender que pode haver algum problema no preenchimento dos imóveis, é interessante desconsiderar o parâmetro por completo, assim não tendenciando o nosso projeto por este viés.

# In[29]:


base_airbnb = base_airbnb.drop("guests_included", axis=1)
base_airbnb.shape


# ## Minimum_nights

# In[30]:


diagrama_caixa(base_airbnb["minimum_nights"])
grafico_barra(base_airbnb["minimum_nights"])


# In[31]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Maximum_nights

# In[32]:


diagrama_caixa(base_airbnb["maximum_nights"])
grafico_barra(base_airbnb["maximum_nights"])


# Tendo em vista que o Maximum Nights é um elemento com altissima taxa de preenchimento igual a zero, demostra certa incoerência mante-lo neste projeto, sendo assim irei retira-lo.

# In[33]:


base_airbnb = base_airbnb.drop("maximum_nights", axis=1)
base_airbnb.shape


# ## Number_of_reviews

# In[34]:


diagrama_caixa(base_airbnb["number_of_reviews"])
grafico_barra(base_airbnb["number_of_reviews"])


# Tendo em vista que quanto maior a quntidade de "reviews" amior a probabilidade do locador já ter um tempo considerávvel no 
# AirBNB e que o projeto tem por escopo ajudar aquele que inicia nesta atividade a precificar o valor do imóvel a alugar,
# é interessante excluir esta feature por inteiro, ao invés de somente os outliers.

# In[35]:


base_airbnb = base_airbnb.drop("number_of_reviews", axis=1)
base_airbnb.shape


# # Tratamento de colunas com texto

# ## Property_type

# In[36]:


print(base_airbnb["property_type"].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot("property_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)


# In[37]:


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


# ## Room type

# In[38]:


print(base_airbnb["room_type"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("room_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)


# ## Bed_type

# In[39]:


print(base_airbnb["bed_type"].value_counts())
plt.figure(figsize=(15, 5))
grafico = sns.countplot("bed_type", data=base_airbnb)
grafico.tick_params(axis="x", rotation=90)

#agrupando categorias de bed_type
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


# ## Cancellation_policy

# In[40]:


#print(base_airbnb["cancellation_policy"].value_counts())
#plt.figure(figsize=(15, 5))
#grafico = sns.countplot("cancellation_policy", data=base_airbnb)
#grafico.tick_params(axis="x", rotation=90)

#agrupando categorias de cancellation_pollicy
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


# ## Amenities

# Tendo em vista que amenities é uma coluna demasiadamente variada com aparente preenchimento livre pelo host, o que a torna não padroizada e mesmo assim desejando mante-la na analise, a solução encontrada foi contar a quantia de amenities de cada imovel com a esperença de que host mais detalhistas nos amenities possam cobrar mais nos imóveis.

# In[41]:


print(base_airbnb["amenities"].iloc[1].split(","))
print(len(base_airbnb["amenities"].iloc[1].split(",")))

base_airbnb["n_amenities"] = base_airbnb["amenities"].str.split(",").apply(len)


# In[42]:


base_airbnb = base_airbnb.drop("amenities", axis=1)
base_airbnb.shape


# In[43]:


diagrama_caixa(base_airbnb["n_amenities"])
grafico_barra(base_airbnb["n_amenities"])


# In[44]:


base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print("Foram removidas {} linhas".format(linhas_removidas))


# ## Visualização de Mapa dos Imóveis

# In[45]:


amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat="latitude", lon="longitude", z="price", radius=2.5,
                        center=centro_mapa,
                        zoom=10,
                        mapbox_style="stamen-terrain")
mapa.show()


# ### Encoding
# 
# Precisamos ajustar as features para facilitar o trabalho do modelo de IA (features de categuria T or F, etc.)
# 
# - Fetaures de Valores True or False, vamos substituir True=1 e False=0
# - Features de Categorias (features que tem valor em texto) sera utilizado o método de encoding variavel dummies.

# In[46]:


colunas_tf = ["host_is_superhost", "instant_bookable", "is_business_travel_ready"]
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=="t", coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=="f", coluna] = 0
print(base_airbnb_cod.iloc[0])


# In[47]:


colunas_categoria = ["property_type", "room_type", "bed_type", "cancellation_policy"]
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categoria)
display(base_airbnb_cod.head())


# In[ ]:





# ### Modelo de Previsão

# In[48]:


def avaliar_modelo(nome_modelo, y_teste, previsão):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f"Modelo {nome_modelo}: \nR²:{r2:.2%}\nRSME:{RSME:.2f}"


# - Escolha dos modelos a serem testados
#     1. Random Forest
#     2. Linear Regression
#     3. Extra Tree

# In[49]:


modelo_rf = RandomForestRegressor ()
modelo_lr = LinearRegression ()
modelo_et = ExtraTreesRegressor ()

modelos = {"RamdomForest": modelo_rf, 
           "LinearRegression": modelo_lr, 
           "ExtraTree": modelo_et
          }

y = base_airbnb_cod["price"]
X = base_airbnb_cod.drop("price", axis=1)


# - Separa os dados em treinos e teste + Treino do Modelo

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ### Análise do Melhor Modelo

# In[ ]:


for nome_modelo, modelo in modelos.items():
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - Modelo escolhido  como Melhor MOdelo: ExtraTreeRegressor
# 
#     Esse foi o medelo com o maior valor de R2 e menor valor de RSM. Como não tivemos uma grande diferenca entre a velocidade do modelo de RandomForest e ExtraTreeRegressor o escolhido foi o ultimo.
#     
#     O modelo de RegressaoLinear foi excluido por ser o pior nas previsões embora bem veloz.
#     
#     métricas do modelo vencedor:<br>
#     Modelo ExtraTree:<br>
#     R²:97.50%<br>
#     RSME:41.91
#    

# ### Ajustes e Melhorias no Melhor Modelo

# In[58]:


#print(modelo_et.feature_impo#rtances_)
#print(X_train.columns)
importancia_features = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)

plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis="x", rotation=90
              )


# ## Ajustes finais no modelo
# 
# - Como a feature is_business_travel_ready parece não ter nenhum efeito no modelo iremos retira-lo da análise.

# In[ ]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))


# In[ ]:


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


# # Deploy do projeto
# 
# - Passo 1 -> Criar aaruqivo do modelo (jobib) 
# - Passo 2 -> Escolher a forma de deploy:
#     - Arquivo Executável + Tkinter
#     - Deploy em um Microsite (Flask)
#     - Deploy apenas para uso direto (streamlit)
# - Passo 3 -> Outro arquivo Python (pode se Jupyter ou PyCharm)
# - Passo 4 -> Importar streamlit e criar código
# - Passo 5 -> Atribuir ao botão o carregamento do modelo
# - Passo 6 -> Deploy feito
# 

# In[ ]:


X["price"] = y
X.to_csv("dados.csv")


# In[57]:


get_ipython().run_cell_magic('time', '', '\nimport joblib\njoblib.dump(modelo_et, "modelo.joblib")')

