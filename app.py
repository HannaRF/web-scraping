import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output 

import requests 
import plotly.graph_objects as go

from models import *
from functions import *
from comparing_models import *
from scipy.optimize import minimize
from Levenshtein import distance
from scipy.stats import poisson
from datetime import timedelta
from random import choice
from glob import glob
import pickle
import time as tm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.random as rdm
import seaborn as sns

sns.set_style("white")

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from IPython.display import clear_output

barra = "\\"

def calculate_missing_date(df, interval, column):
    beg, end = interval
    years = {'all' : [0 , 0]}
    for i in range(beg, end):
        years['all'][1] += 1
        if type(df.loc[i, column]) != str:
            # não temos a data
            years['all'][0] += 1
            if df.loc[i, 'Year'] not in years:
                years[df.loc[i, 'Year']] = [1, 1]
            else:
                years[df.loc[i, 'Year']][0] += 1
                years[df.loc[i, 'Year']][1] += 1
        else:
            # temos a data
            if df.loc[i, 'Year'] not in years:
                years[df.loc[i, 'Year']] = [0, 1]
            else:
                years[df.loc[i, 'Year']][1] += 1

    return years, years['all'][0]/years['all'][1]

"""""""""""
if 'Clean Data'+barra+'data.csv' not in glob('Clean Data/*'): 
    cleaning_data()
"""""""""""
    
df = pd.read_csv('Clean Data/data.csv')

# ----------------------------------------------------------------------------------------------

df['Date_by_Nearest'] = matplotlib.dates.date2num(pd.to_datetime(df['Date'], dayfirst = True))
df['Date_by_Nearest'].interpolate(method = 'nearest', inplace = True)
df['Date_by_Nearest'] = df['Date_by_Nearest'].apply(np.round)
df['Date_by_Nearest'] = matplotlib.dates.num2date(df['Date_by_Nearest'])
df['Date_by_Nearest'] = df['Date_by_Nearest'].dt.strftime('%d/%m/%Y')

nearest_date = list(df['Date_by_Nearest'])
for i in range(len(nearest_date)):
    if int(nearest_date[i][3:5]) > 12:
        nearest_date[i] = nearest_date[i][3:5] + '/' + nearest_date[i][:2] + '/' + nearest_date[i][-4:]
    else:
        if int(nearest_date[i][:2]) < 12:
            nearest_date[i] = nearest_date[i][3:5] + '/' + nearest_date[i][:2] + '/' + nearest_date[i][-4:]
            
df['Date_by_Nearest'] = nearest_date

# ---------------------------------------------------------------------------------------------

df['New_Date'] = df['Date']
i = 0
while df.loc[i, 'Competition'] == 'Serie A':
    i_init = i
    rd = df.loc[i, 'Round']
    nan_count = 0
    dates = []
    while df.loc[i, 'Round'] == rd:
        if type(df.loc[i, 'New_Date']) != str:
            nan_count += 1
        else:
            dates.append(df.loc[i, 'New_Date'])
            
        i += 1
        
    if nan_count <= 5:
        i = i_init
        while df.loc[i, 'Round'] == rd:
            if type(df.loc[i, 'New_Date']) != str:
                df.loc[i, 'New_Date'] = choice(dates)
                
            i += 1
        
# -----------------------------------------------------------------------------------------------
sa_end = None
sb_end = None
sc_end = None
sd_end = None
cdb_end = len(df) - 1
for i in range(1, cdb_end + 1):
    if df.loc[i, 'Competition'] != 'Serie A' and df.loc[i - 1, 'Competition'] == 'Serie A':
        sa_end = i - 1
    elif df.loc[i, 'Competition'] != 'Serie B' and df.loc[i - 1, 'Competition'] == 'Serie B':
        sb_end = i - 1
    elif df.loc[i, 'Competition'] != 'Serie C' and df.loc[i - 1, 'Competition'] == 'Serie C':
        sc_end = i - 1
    elif df.loc[i, 'Competition'] != 'Serie D' and df.loc[i - 1, 'Competition'] == 'Serie D':
        sd_end = i - 1
        
# -------------------------------------------------------------------------------------------------------
        
fix = []
count = {}
for i in range(sa_end + 1):
    if type(df.loc[i, 'New_Date']) != str:
        year = df.loc[i, 'Year']
        rd = df.loc[i, 'Round']
        fix.append([i, year, rd])
        if year in count:
            if rd in count[year]:
                count[year][rd] += 1
            else:
                count[year][rd] = 1
        
        else:
            count[year] = {}
            count[year][rd] = 1

years = [*range(2013, 2022)]
i = 0
while i < len(years):        
    if years[i] in count:
        years.remove(years[i])
    else:
        i += 1
        
if 2020 in years: years.remove(2020) # calendário afetado pela COVID-19
if 2021 in years: years.remove(2021) # calendário afetado pela COVID-19

# -----------------------------------------------------------------------------------------------
    
calendario = {}
for i in range(sa_end + 1):
    if df.loc[i, 'Year'] in years:
        year = df.loc[i, 'Year']
        rd = df.loc[i, 'Round']
        if rd not in calendario:
            calendario[df.loc[i, 'Round']] = {}
        
        if year not in calendario[rd]:
            calendario[rd][year] = [date2int(df.loc[i, 'New_Date'])]
        else:
            calendario[rd][year].append(date2int(df.loc[i, 'New_Date']))
            
for rd in calendario:
    for year in calendario[rd]:
        calendario[rd][year] = np.array(calendario[rd][year]).mean()
        
    a = int(round(np.array(list(calendario[rd].values())).mean()))
    calendario[rd] = date.fromordinal(a)
    
for i in range(1, 10):
    # para não ter uma grande pausa em junho, o que não é normal, mas aconteceu por
    # causa da Copa do Mundo, em 2014
    calendario[i] += timedelta(days = 12)
    
# -----------------------------------------------------------------------------------------------

round_days_to_fit = {}
for i in range(sa_end + 1):
    if df.loc[i, 'Year'] in count:
        year = df.loc[i, 'Year']
        rd = df.loc[i, 'Round']
        if type(df.loc[i, 'New_Date']) == str:
            if rd not in round_days_to_fit:
                round_days_to_fit[df.loc[i, 'Round']] = {}

            if year not in round_days_to_fit[rd]:
                round_days_to_fit[rd][year] = [date2int(df.loc[i, 'New_Date'])]
            else:
                round_days_to_fit[rd][year].append(date2int(df.loc[i, 'New_Date']))
            
for rd in round_days_to_fit:
    for year in round_days_to_fit[rd]:
        round_days_to_fit[rd][year] = round(np.array(round_days_to_fit[rd][year]).mean())
        round_days_to_fit[rd][year] = date.fromordinal(int(round_days_to_fit[rd][year]))

# -----------------------------------------------------------------------------------------------

# 2013
rds2013 = []
for rd in round_days_to_fit:
    if 2013 in round_days_to_fit[rd]:
        rds2013.append([rd, round_days_to_fit[rd][2013]])

rds2013.sort(key = lambda x : x[0])
k = 0
while len(rds2013) < 38 and k < 37:
    # corrigindo rodada dupla em um fim de semana
    if rds2013[k][0] == 2:
        rds2013[k][1] -= timedelta(days = 4)
    
    if rds2013[k][0] == 3:
        rds2013[k][1] += timedelta(days = 1)
    
    if rds2013[k][0] != rds2013[k + 1][0] - 1:
        rounds_to_add = rds2013[k + 1][0] - rds2013[k][0] - 1
        time = rds2013[k + 1][1] - rds2013[k][1]
        if time / (rounds_to_add + 1) >= timedelta(days = 7):
            for i in range(rounds_to_add):
                rds2013.insert(k + i + 1, [rds2013[k][0] + i + 1, rds2013[k + i][1] + timedelta(days = 7)])
        elif time / (rounds_to_add + 1) >= timedelta(days = 3, hours = 12):
            for i in range(rounds_to_add):
                if i % 2 == 0:
                    rds2013.insert(k + i + 1, [rds2013[k][0] + i + 1, rds2013[k + i][1] + timedelta(days = 4)])
                else:
                    rds2013.insert(k + i + 1, [rds2013[k][0] + i + 1, rds2013[k + i][1] + timedelta(days = 3)])

        else:
            for i in range(rounds_to_add):
                rds2013.insert(k + i + 1, [rds2013[k][0] + i + 1, rds2013[k + i][1] + timedelta(days = 3)])

    k += 1
    
# arrumando para ter folga durante a Copa das Confederações
rds2013[4] = [5, rds2013[3][1] + timedelta(days = 3)]
rds2013[5] = [6, rds2013[4][1] + timedelta(days = 4)]


# -----------------------------------------------------------------------------------------------

# 2015
rds2015 = []
for rd in round_days_to_fit:
    if 2015 in round_days_to_fit[rd]:
        rds2015.append([rd, round_days_to_fit[rd][2015]])

rds2015.sort(key = lambda x : x[0])
k = 0
while len(rds2015) < rds2015[-1][0] - rds2015[0][0] + 1 and k < rds2015[-1][0] - rds2015[0][0]:
    if rds2015[k][0] != rds2015[k + 1][0] - 1:
        rounds_to_add = rds2015[k + 1][0] - rds2015[k][0] - 1
        time = rds2015[k + 1][1] - rds2015[k][1]
        if time / (rounds_to_add + 1) >= timedelta(days = 7):
            for i in range(rounds_to_add):
                rds2015.insert(k + i + 1, [rds2015[k][0] + i + 1, rds2015[k + i][1] + timedelta(days = 7)])
        elif time / (rounds_to_add + 1) >= timedelta(days = 3, hours = 12):
            for i in range(rounds_to_add):
                if i % 2 == 0:
                    rds2015.insert(k + i + 1, [rds2015[k][0] + i + 1, rds2015[k + i][1] + timedelta(days = 4)])
                else:
                    rds2015.insert(k + i + 1, [rds2015[k][0] + i + 1, rds2015[k + i][1] + timedelta(days = 3)])

        else:
            for i in range(rounds_to_add):
                rds2015.insert(k + i + 1, [rds2015[k][0] + i + 1, rds2015[k + i][1] + timedelta(days = 3)])

    k += 1

# para as demais datas vamos encaixar os calendário antes e depois das datas que
# temos com um intervalo de 7 dias
if rds2015[0][0] != 1:
    dia_calendario = date(year, calendario[rds2015[0][0]].month, calendario[rds2015[0][0]].day)
    dt = rds2015[0][1] - dia_calendario
    for k in range(1, rds2015[0][0]):
        dia_calendario_rd_k = date(year, calendario[k + 1].month, calendario[k + 1].day)
        dia_insert = dia_calendario_rd_k + dt + timedelta(days = -7)
        rds2015.insert(k - 1, [k, dia_insert])

# -----------------------------------------------------------------------------------------------

# 2016
rds2016 = []
for rd in round_days_to_fit:
    if 2016 in round_days_to_fit[rd]:
        rds2016.append([rd, round_days_to_fit[rd][2016]])

rds2016.sort(key = lambda x : x[0])
k = 0
while len(rds2016) < rds2016[-1][0] - rds2016[0][0] + 1 and k < rds2016[-1][0] - rds2016[0][0]:
    if rds2016[k][0] != rds2016[k + 1][0] - 1:
        rounds_to_add = rds2016[k + 1][0] - rds2016[k][0] - 1
        time = rds2016[k + 1][1] - rds2016[k][1]
        if time / (rounds_to_add + 1) >= timedelta(days = 7):
            for i in range(rounds_to_add):
                rds2016.insert(k + i + 1, [rds2016[k][0] + i + 1, rds2016[k + i][1] + timedelta(days = 7)])
        elif time / (rounds_to_add + 1) >= timedelta(days = 3, hours = 12):
            for i in range(rounds_to_add):
                if i % 2 == 0:
                    rds2016.insert(k + i + 1, [rds2016[k][0] + i + 1, rds2016[k + i][1] + timedelta(days = 4)])
                else:
                    rds2016.insert(k + i + 1, [rds2016[k][0] + i + 1, rds2016[k + i][1] + timedelta(days = 3)])

        else:
            for i in range(rounds_to_add):
                rds2016.insert(k + i + 1, [rds2016[k][0] + i + 1, rds2016[k + i][1] + timedelta(days = 3)])

    k += 1

# para as demais datas vamos encaixar os calendário antes e depois das datas que
# temos com um intervalo de 7 dias
if rds2016[0][0] != 1:
    dia_calendario = date(year, calendario[rds2016[0][0]].month, calendario[rds2016[0][0]].day)
    dt = rds2016[0][1] - dia_calendario
    for k in range(1, rds2016[0][0]):
        dia_calendario_rd_k = date(year, calendario[k + 1].month, calendario[k + 1].day)
        dia_insert = dia_calendario_rd_k + dt + timedelta(days = -7)
        rds2016.insert(k - 1, [k, dia_insert])

if rds2016[-1][0] != 38:
    dia_calendario = date(year, calendario[rds2016[-1][0]].month, calendario[rds2016[-1][0]].day)
    dt = rds2016[-1][1] - dia_calendario
    for k in range(rds2016[-1][0] + 1, 39):
        dia_calendario_rd_k = date(year, calendario[k].month, calendario[k].day)
        dia_insert = dia_calendario_rd_k + dt
        rds2016.append([k, dia_insert])

# -----------------------------------------------------------------------------------------------

i = 0
while df.loc[i, 'Competition'] == 'Serie A':
    years = [2013, 2015, 2016]
    rds = [rds2013, rds2015, rds2016]
    if df.loc[i, 'Year'] in years:
        year = df.loc[i, 'Year']
        rdsyear = rds[years.index(year)]
        while df.loc[i, 'Year'] == year:
            rd = df.loc[i, 'Round']
            if type(df.loc[i, 'New_Date']) != str:
                d = rdsyear[rd - 1][1]
                df.loc[i, 'New_Date'] = str(d.day).zfill(2) + '/' + str(d.month).zfill(2) + '/' + str(year)
            
            i += 1
    else:
        i += 1

# -----------------------------------------------------------------------------------------------

if 'Clean Data'+barra+'cdb.csv' not in glob('Clean Data/*'): 
    cleaning_data(competitions1 = ['Data/Copa do Brasil'], output_file = 'Clean Data/cdb.csv')
df_cdb = pd.read_csv('Clean Data/cdb.csv')
calculate_missing_date(df_cdb, (0, len(df_cdb)), 'Date')

# -----------------------------------------------------------------------------------------------

df = df.loc[:sb_end]
df

# ------------------------------------------------------------------------------------------------

app = dash.Dash() 

server = app.server

# -------------------------------------------------------------------------------------------------

colors = {'vinho':'#A60A33',
          'azul':'#194073',
          'verde':'#7A8C3A',
          'amarelo':'#F2BB16',
          'vermelho':'#D93829'}

# ------------------------------------------------------------------------------------------------

labels = []
wins = []
draws = []
losses = []
width = 0.35
fig, ax = plt.subplots(figsize = (8, 6))
for i in range(9):
    year = 2013 + i
    labels.append(str(year))
    gols_casa = np.array([int(df.loc[i, 'Result'][0]) for i in range(len(df)) 
                          if df.loc[i, 'Year'] == year and 
                          df.loc[i, 'Competition'] == 'Serie A'])
    
    gols_fora = np.array([int(df.loc[i, 'Result'][4]) for i in range(len(df))
                          if df.loc[i, 'Year'] == year and
                          df.loc[i, 'Competition'] == 'Serie A'])
    
    n = len(gols_casa)
    wins.append(np.sum(gols_casa > gols_fora) / n)
    draws.append(np.sum(gols_casa == gols_fora) / n)
    losses.append(np.sum(gols_casa < gols_fora) / n)

wins = np.array(wins)
draws = np.array(draws)
losses = np.array(losses)
labels[-1] = '2021*'

mean_losses = losses.mean()
mean_draws = draws.mean()
mean_wins = wins.mean()

# ---------------------------------------------------------------------------------------------------------------

fig1 = go.Figure(go.Bar(x=labels, y=losses,name='Derrotas',
                        marker_color=colors["vermelho"],width = 0.5))

fig1.add_trace(go.Bar(x=labels, y=draws, name='Empates',
                      marker_color=colors["azul"],width = 0.5))

fig1.add_trace(go.Bar(x=labels, y=wins, name='Vitórias',
                      marker_color=colors["verde"],width = 0.5))

fig1.add_trace(go.Scatter(x=labels, y=[0.5]*len(labels), name='50%',
                          line=dict(color='black', width=1., dash='dot')))

fig1.add_trace(go.Scatter(x=labels, y=[mean_losses]*len(labels),
                          name=f'Média de derrotas: {100*mean_losses:.2f}%',
                          line=dict(color=colors['vermelho'], width=1., dash='dot')))

fig1.add_trace(go.Scatter(x=labels, y=[mean_draws + mean_losses]*len(labels),
                          name=f'Média de empates: {100*mean_draws:.2f}%',
                          line=dict(color=colors['azul'], width=1., dash='dot')))

fig1.add_trace(go.Scatter(x=labels, y=[mean_draws + mean_losses + mean_wins]*len(labels),
                          name=f'Média de vitórias: {100*mean_wins:.2f}%',
                          line=dict(color=colors['verde'], width=1., dash='dot')))

fig1.update_layout(barmode='stack',
                   yaxis={'categoryorder':'total descending'},
                   xaxis_title="* dados de 2021 incompletos, campeonato em andamento.",
                   yaxis_title="Jogos(%)",
                   template = "plotly_white",
                   font=dict(size=10),
                   title='Resultados dos jogos\nVisão do Mandante',
                  legend = dict(font = dict(size = 10,color = "black")))
#fig1.show()


# ------------------------------------------------------------------------------------------------------

year = 6 # 2013 = 0, 2014 = 1, etc
br = df.loc[380 * year:379 + 380 * year, ['Team 1', 'Team 2', 'Result']]
table = classification_table(br)
table

# ----------------------------------------------------------------------------------------------------

fig2 = go.Figure(go.Bar(x=list(table['points']), y=list(table.index),
                        orientation='h', marker_color=colors["azul"]))
fig2.update_layout(barmode='stack',
                   yaxis={'categoryorder':'total descending'}, # not working 
                   template = "plotly_white",
                   title='\nClassificação final do campeonato',
                   xaxis_title="Pontos",
                   font=dict(size=10),
                   yaxis_title="")
#fig2.show()



fig3 = go.Figure(go.Bar(x=list(table['goal difference']), y=list(table.index),
                        orientation='h', marker_color=colors["azul"]))

fig3.update_layout(barmode='stack',
                   yaxis={'categoryorder':'total descending'}, # not working 
                   template = "plotly_white",
                   font=dict(size=10),
                   xaxis_title="Saldo de Gols")
#fig3.update_yaxes(showticklabels=False)

#fig3.show()


# ----------------------------------------------------------------------------------------------------


plt.rcParams['figure.figsize'] = (12, 8)
rds = [*range(20, 39)]
year = 2020
model = '8 - Modelo com Esquecimento'
clubs = ['São Paulo / SP',
         'Atlético / MG',
         'Flamengo / RJ',
         'Internacional / RS',
         'Palmeiras / SP',
         'Grêmio / RS']

colors = ['lightsalmon',
          'black',
          'red',
          'firebrick',
          'forestgreen',
          'cornflowerblue']

with open('results.json', 'rb') as fp:
    results = pickle.load(fp)

    

fig4 = go.Figure()

for i in range(len(clubs)):
    club = clubs[i]
    color = colors[i]
    champ = []
    for rd in rds:
        champ.append(results[model][year][rd][club][1] / 50000)
        
    fig4.add_trace(go.Scatter(x=rds, y=champ,mode='lines',
                              name=club,line=dict(color=colors[i], width=1.)))


fig4.update_layout(template = "plotly_white",
                   title = f'Probabilidades de Título - {year}\nSegundo {model[4:]}',
                   yaxis_title='Probabilidade',
                   xaxis_title='Rodada',
                   font=dict(size=10),
                   legend = dict(font = dict(size = 10,color = "black")))

    
#fig4.show()

# ------------------------------------------------------------------------------------------

fig5 = go.Figure()

for i in range(len(clubs)):
    club = clubs[i]
    color = colors[i]
    g6 = []
    for rd in rds:
        prob_g6 = 0
        for pos in range(1, 7):
            prob_g6 += results[model][year][rd][club][pos]
            
        g6.append(prob_g6 / 50000)
        
    fig5.add_trace(go.Scatter(x=rds, y=g6,mode='lines',
                              name=club,line=dict(color=colors[i], width=1.)))
    
fig5.update_layout(template = "plotly_white",
                   title = f'Probabilidades de G6 - {year}\nSegundo {model[4:]}',
                   yaxis_title='Probabilidade',
                   xaxis_title='Rodada',
                   font=dict(size=10),
                   legend = dict(font = dict(size = 10,color = "black")))

#fig5.show()

# ----------------------------------------------------------------------------------------------------------

clubs = ['Botafogo / RJ',
         'Vasco da Gama / RJ',
         'Goiás / GO',
         'Coritiba / PR',
         'Sport / PE',
         'Fortaleza / CE']

colors = ['dimgray',
          'black',
          'forestgreen',
          'limegreen',
          'firebrick',
          'mediumorchid']

fig6 = go.Figure()

for i in range(len(clubs)):
    club = clubs[i]
    color = colors[i]
    z4 = []
    for rd in rds:
        prob_z4 = 0
        for pos in range(17, 21):
            prob_z4 += results[model][year][rd][club][pos]
            
        z4.append(prob_z4 / 50000)
        
    fig6.add_trace(go.Scatter(x=rds, y=z4,mode='lines',
                              name=club,
                              line=dict(color=colors[i], width=1.)))

fig6.update_layout(template = "plotly_white",
                   title = f'Probabilidades de Z4 - {year}\nSegundo {model[4:]}',
                   yaxis_title='Probabilidade',
                   xaxis_title='Rodada',
                   font=dict(size=10),
                   legend = dict(font = dict(size = 10,color = "black")))
#fig6.show()


# ---------------------------------------------------------------------------------------------------------------

# sem forças salvas:
models = [('1 - Modelo Ingênuo', naive_model, train_naive_model),
          ('2 - Modelo Semi-ingênuo', seminaive_model, train_seminaive_model),
          ('3 - Modelo Observador', observer_model, train_observer_model),
          ('4 - Modelo de Poisson pela média de gols', simple_poisson_neutral, train_simple_poisson_neutral),
          ('5 - Modelo de Poisson apenas com média pelo mando', simple_poisson, train_simple_poisson_non_neutral),
          ('6 - Modelo de Poisson sem mando de campo', robust_poisson, train_simple_poisson),
          ('7 - Modelo de Poisson com mando de campo', more_robust_poisson, train_complex_poisson),
          ('8 - Modelo com Esquecimento', forgetting_poisson, train_forgetting_poisson)]

# vet2force4getting(x, clubs) # len == 82
# vet2force4(x, clubs) # len == 80
# vet2force2(x, clubs) # len == 40

# com forças já salvas:
forces_file = lambda year : f'Forces/forces - {year}.json'
models = [('1 - Modelo Ingênuo', naive_model, (forces_file, 'treinado!')),
          ('2 - Modelo Semi-ingênuo', seminaive_model, (forces_file, 'treinado!')),
          ('3 - Modelo Observador', observer_model, (forces_file, 'treinado!')),
          ('4 - Modelo de Poisson pela média de gols', simple_poisson_neutral, (forces_file, 'treinado!')),
          ('5 - Modelo de Poisson apenas com média pelo mando', simple_poisson, (forces_file, 'treinado!')),
          ('6 - Modelo de Poisson sem mando de campo', robust_poisson, (forces_file, 'treinado!')),
          ('7 - Modelo de Poisson com mando de campo', more_robust_poisson, (forces_file, 'treinado!')),
          ('8 - Modelo com Esquecimento', forgetting_poisson, (forces_file, 'treinado!'))]


# ----------------------------------------------------------------------------------------------------------

year = 2020
with open('results.json', 'rb') as fp:
    results = pickle.load(fp)

final_results = final_classification(2020)
colors = ['red', 'blue', 'green', 'purple', 'yellow', 'brown', 'gray', 'lightskyblue']

fig7 = go.Figure()

for i in range(len(models)):
    model = models[i][0]
    roc = []
    for rd in rds:
        final_result, predict_result = probabilities_matrix(final_results, results, model, year, rd)
        roc.append(roc_auc_score(final_result, predict_result))
        
    fig7.add_trace(go.Scatter(x=rds, y=roc,mode='lines',name=model[4:],line=dict(color=colors[i], width=1.)))
    
fig7.update_layout(template = "plotly_white",
                   title = f'AUC ROC por rodada - {year}',
                   yaxis_title='AUC ROC Score',
                   xaxis_title='Rodada',
                   font=dict(size=10),
                   legend = dict(font = dict(size = 10,color = "black")))

#fig7.show()


app.layout = html.Div([
   html.Div(children=[dcc.Graph(id="graph1",figure=fig1,
                                style={'width': '180vh', 'height': '90vh'}),
                      dcc.Graph(id="graph2",figure=fig2,
                                style={'width': '100vh', 'height': '90vh'}),
                      dcc.Graph(id="graph3",figure=fig3,
                                style={'width': '100vh', 'height': '90vh'})],
            style={'display': 'flex'}),
   html.Div(children=[dcc.Graph(id="graph4",figure=fig4,
                                style={'width': '200vh', 'height': '60vh'})],
            style={'display': 'flex'}),
   html.Div(children=[dcc.Graph(id="graph5",figure=fig5,
                                style={'width': '200vh', 'height': '60vh'})],
            style={'display': 'flex'}),
   html.Div(children=[dcc.Graph(id="graph6",figure=fig6,
                                style={'width': '200vh', 'height': '60vh'})],
            style={'display': 'flex'}),
   html.Div(children=[dcc.Graph(id="graph7",figure=fig7,
                                style={'width': '200vh', 'height': '60vh'})],
            style={'display': 'flex'})]
)

if __name__ == '__main__':
    app.run_server(debug=True)