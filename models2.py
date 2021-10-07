import numba
import warnings
import matplotlib
import numpy as np
import pandas as pd
import numpy.random as rdm
from time import time
from numba import jit
from numba import prange
from functions import *
from scipy.stats import poisson
from scipy.optimize import minimize
from numba.core.errors import NumbaWarning
from numba.core.errors import NumbaDeprecationWarning
from numba.core.errors import NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category = NumbaWarning)
warnings.simplefilter('ignore', category = NumbaDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaPendingDeprecationWarning)

@jit
def naive_model(n_sims, *args):
    '''
    Modelo ingênuo, onde cada resultado tem probabilidade de 1/3 
    '''
    
    return list(rdm.choice(['1 x 0', '0 x 0', '0 x 1'], n_sims))

@jit
def seminaive_model(n_sims, retrospect):
    '''
    Modelo semi-ingênuo.
    
    Recebe uma lista de retrospecto (em relação ao mandante)
    e retorna os placares de acordo com esse retrospecto.
    '''
    
    return list(rdm.choice(['1 x 0', '0 x 0', '0 x 1'], n_sims, p = retrospect))

@jit(fastmath = True)
def observer_model(n_sims, home, away):
    '''
    Modelo observador.
    
    Recebe os retrospectos do mandante e do visitante, retornando
    os resultados de forma proporcional a média dos resultados,
    ou seja, se o mandandante tem probabilidade de vitória de 50%
    em casa, enquanto o visistante tem probabilidade de derrota
    de 40% fora de casa, então o mandante ganha com probabilidade
    45%.
    '''
    
    home = np.array(home)
    away = np.array(away)
    prob = (home + away) / 2
        
    return list(rdm.choice(['1 x 0', '0 x 0', '0 x 1'], n_sims, p = prob))

@jit(fastmath = True)    
def simple_poisson_neutral(n_sims, goals_mean):
    '''
    Modelo de Poisson simples, onde a média de gols de cada
    clube é metade da média de gols por partida, ou seja,
    os dois clubes tem a mesma "força".
    '''
    
    home_goals = poisson.rvs(goals_mean / 2, size = n_sims)
    away_goals = poisson.rvs(goals_mean / 2, size = n_sims)
    return [str(home_goals[i]) + ' x ' + str(away_goals[i]) for i in range(n_sims)]

@jit(fastmath = True)
def simple_poisson(n_sims, means):
    '''
    Modelo de Poisson simples, onde a média de gols de cada
    clube depende apenas do mando de campo.
    '''
    
    home_mean, away_mean = means
    home_goals = poisson.rvs(home_mean, size = n_sims)
    away_goals = poisson.rvs(away_mean, size = n_sims)
    return [str(home_goals[i]) + ' x ' + str(away_goals[i]) for i in range(n_sims)]

@jit(fastmath = True)
def robust_poisson(n_sims, home, away):
    '''
    Modelo de Poisson robusto, com cada time tendo uma força
    de ataque e uma de defesa, independentemente do mando de
    campo.
    '''
    
    home_atk, home_def = home['Ataque'], home['Defesa']
    away_atk, away_def = away['Ataque'], away['Defesa']
    home_goals = poisson.rvs(home_atk / away_def, size = n_sims)
    away_goals = poisson.rvs(away_atk / home_def, size = n_sims)
    return [str(home_goals[i]) + ' x ' + str(away_goals[i]) for i in range(n_sims)]

@jit(fastmath = True)
def more_robust_poisson(n_sims, home, away):
    '''
    Modelo de Poisson mais robusto, com cada time tendo uma
    força de ataque e uma de defesa, dependente do mando de
    campo.
    '''
    
    home_atk, home_def = home['Ataque'], home['Defesa']
    away_atk, away_def = away['Ataque'], away['Defesa']
    home_goals = poisson.rvs(home_atk / away_def, size = n_sims)
    away_goals = poisson.rvs(away_atk / home_def, size = n_sims)
    return [str(home_goals[i]) + ' x ' + str(away_goals[i]) for i in range(n_sims)]

@jit(fastmath = True)
def forgetting_poisson(n_sims, home, away):
    '''
    Modelo de Poisson com esquecimento, com cada time tendo
    uma força de ataque e uma de defesa, dependente do mando
    de campo e dos desempenhos recentes.
    '''
    
    home_atk, home_def = home['Ataque'], home['Defesa']
    away_atk, away_def = away['Ataque'], away['Defesa']
    home_goals = poisson.rvs(home_atk / away_def, size = n_sims)
    away_goals = poisson.rvs(away_atk / home_def, size = n_sims)
    return [str(home_goals[i]) + ' x ' + str(away_goals[i]) for i in range(n_sims)]

@jit
def train_naive_model(*args):
    '''
    Retorna um vetor de probabilidades para o modelo
    ingênuo (tudo igual a 1/3).
    '''
    return [1/3, 1/3, 1/3]

@jit
def train_seminaive_model(games, *args):
    '''
    Recebe um dataframe de jogos e retorna a proporção
    de jogos com cada resultado, na visão do mandante.
    '''
    
    results = [0, 0, 0]
    for i in games.index:
        result = games.loc[i, 'Result']
        if int(result[0]) > int(result[4]):
            results[0] += 1
        elif int(result[0]) == int(result[4]):
            results[1] += 1
        else:
            results[2] += 1
            
    results = np.array(results)
    return results / np.sum(results)

@jit
def train_observer_model(games, *args):
    '''
    Recebe um dataframe de jogos e retorna a proporção
    de jogos com cada resultado, por clube, separando
    os jogos em casa e fora, sempre na visão do mandante.
    '''
    
    results = {}
    for i in games.index:
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        if home not in results:
            results[home] = {'Casa' : [0, 0, 0], 'Fora' : [0, 0, 0]}
        
        if away not in results:
            results[away] = {'Casa' : [0, 0, 0], 'Fora' : [0, 0, 0]}
            
        result = games.loc[i, 'Result']
        if int(result[0]) > int(result[4]):
            results[home]['Casa'][0] += 1
            results[away]['Fora'][0] += 1
        elif int(result[0]) == int(result[4]):
            results[home]['Casa'][1] += 1
            results[away]['Fora'][1] += 1
        else:
            results[home]['Casa'][2] += 1
            results[away]['Fora'][2] += 1
            
    for club in results:
        for local in results[club]:
            results[club][local] = np.array(results[club][local])
            results[club][local] = results[club][local] / np.sum(results[club][local])
    
    return results

@jit(fastmath = True)
def train_simple_poisson_neutral(games, *args):
    '''
    Recebe um dataframe de jogos e retorna a média de
    gols por partida
    '''
    
    goals = 0
    for i in games.index:
        result = games.loc[i, 'Result']
        goals += int(result[0])
        goals += int(result[4])
        
    n_games = len(games)
    return goals / n_games

@jit(fastmath = True)
def train_simple_poisson_non_neutral(games, *args):
    '''
    Recebe um dataframe de jogos e retorna a média de
    gols por partida do mandante e do visitante
    '''
    
    home_goals = 0
    away_goals = 0
    for i in games.index:
        result = games.loc[i, 'Result']
        home_goals += int(result[0])
        away_goals += int(result[4])
        
    n_games = len(games)
    home_goals = home_goals / n_games
    away_goals = away_goals / n_games
    return [home_goals, away_goals]

@jit
def vet2force2(x, clubs):
    forces = numba.typed.Dict()
    for club in clubs:
        forces[club] = {'Ataque' : x[0], 'Defesa' : x[1]}
        x = x[2:]
        
    return forces

@jit
def force22vet(forces):
    x = numba.typed.List()
    for club in forces:
        for force in forces[club]:
            x.append(forces[club][force])
    
    return x

@jit(fastmath = True)
def likelihood_simple_poisson(x, clubs, games):
    '''
    Recebe um vetor x de forças, onde a cada duas forças
    temos um time, e retorna a log verossimilhança
    negativa dessas forças com os dados
    '''
    
    forces = vet2force2(x, clubs)
    log_ver_neg = 0
    for i in games.index:
        result = games.loc[i, 'Result']
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        
        mu = forces[home]['Ataque'] / forces[away]['Defesa']
        log_ver_neg -= poisson.logpmf(int(result[0]), mu)
        
        mu = forces[away]['Ataque'] / forces[home]['Defesa']
        log_ver_neg -= poisson.logpmf(int(result[4]), mu)
        
    return log_ver_neg

@jit
def vet2force4(x, clubs):
    forces = numba.typed.Dict()
    for club in clubs:
        forces[club] = {'Casa' : {'Ataque' : x[0], 'Defesa' : x[1]},
                        'Fora' : {'Ataque' : x[2], 'Defesa' : x[3]}}
        x = x[4:]
        
    return forces

@jit
def force42vet(forces):
    x = []
    for club in forces:
        for local in forces[club]:
            for force in forces[club][local]:
                x.append(forces[club][local][force])
    
    return x

@jit(fastmath = True)
def likelihood_complex_poisson(x, clubs, games):
    '''
    Recebe um vetor x de forças, onde a cada quatro forças
    temos um time, e retorna a log verossimilhança negativa
    dessas forças com os dados
    '''
    
    forces = vet2force4(x, clubs)
    log_ver_neg = 0
    for i in games.index:
        result = games.loc[i, 'Result']
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        
        mu = forces[home]['Casa']['Ataque'] / forces[away]['Fora']['Defesa']
        log_ver_neg -= poisson.logpmf(int(result[0]), mu)
        
        mu = forces[away]['Fora']['Ataque'] / forces[home]['Casa']['Defesa']
        log_ver_neg -= poisson.logpmf(int(result[4]), mu)
        
    return log_ver_neg

@jit
def forgetting(t, c, k):
    return k / (c * np.log(t) + k)

@jit
def vet2force4getting(x, clubs):
    forces = numba.typed.Dict()
    for club in clubs:
        forces[club] = {'Casa' : {'Ataque' : x[0], 'Defesa' : x[1]},
                        'Fora' : {'Ataque' : x[2], 'Defesa' : x[3]}}
        x = x[4:]
    
    k, c = x
    
    return forces, k, c

@jit
def force4getting2vet(forces, k, c):
    x = []
    for club in forces:
        for local in forces[club]:
            for force in forces[club][local]:
                x.append(forces[club][local][force])
    
    x.append(k)
    x.append(c)
    
    return x

@jit(fastmath = True)
def likelihood_forgetting_poisson(x, clubs, games, date):
    '''
    Recebe um vetor x de forças, onde a cada quatro forças
    temos um time, e retorna a log verossimilhança negativa
    dessas forças com os dados
    '''
    
    games['New_Date_Num'] = date - matplotlib.dates.date2num(pd.to_datetime(games['New_Date'], dayfirst = True))
    
    forces, k, c = vet2force4getting(x, clubs)
    log_ver_neg = 0
    for i in games.index:
        result = games.loc[i, 'Result']
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        date = games.loc[i, 'New_Date_Num']
        if date > 0:
            weight = forgetting(date, c, k)

            mu = forces[home]['Casa']['Ataque'] / forces[away]['Fora']['Defesa']
            log_ver_neg -= poisson.logpmf(int(result[0]), mu) - np.log(weight)

            mu = forces[away]['Fora']['Ataque'] / forces[home]['Casa']['Defesa']
            log_ver_neg -= poisson.logpmf(int(result[4]), mu) - np.log(weight)
        
    return log_ver_neg

def train_simple_poisson(games, x0 = None, *args):
    clubs = []
    for i in games.index:
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        if home not in clubs:
            clubs.append(home)
            
        if away not in clubs:
            clubs.append(away)
            
        if len(clubs) == 20:
            break
            
    if x0 == None:
        x0 = [1 for i in range(40)]
    bounds = [(0.01, np.inf) for i in range(40)] # evitando divisão por zero
    res = minimize(likelihood_simple_poisson,
                   x0,
                   args = (clubs, games),
                   method = 'SLSQP',
                   bounds = bounds)

    x = res.x
    x = x / x[0]
    forces = vet2force2(x, clubs)

    return forces

def train_complex_poisson(games, x0 = None, *args):
    clubs = []
    for i in games.index:
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        if home not in clubs:
            clubs.append(home)
            
        if away not in clubs:
            clubs.append(away)
            
        if len(clubs) == 20:
            break
    
    if x0 == None:
        x0 = [1 for i in range(80)]
    bounds = [(0.01, np.inf) for i in range(80)] # evitando divisão por zero
    res = minimize(likelihood_complex_poisson,
                   x0,
                   args = (clubs, games),
                   method = 'SLSQP',
                   bounds = bounds)

    x = res.x
    x = x / x[0]
    forces = vet2force4(x, clubs)

    return forces

def train_forgetting_poisson(games, x0 = None, date = '01/01/2021', *args):
    clubs = []
    if type(date) == str:
    	date = matplotlib.dates.date2num(pd.to_datetime(date, dayfirst = True))
    
    for i in games.index:
        home = games.loc[i, 'Team 1']
        away = games.loc[i, 'Team 2']
        if home not in clubs:
            clubs.append(home)
            
        if away not in clubs:
            clubs.append(away)
            
        if len(clubs) == 20:
            break
            
    if x0 == None:
        x0 = [1 for i in range(82)]
    bounds = [(0.01, np.inf) for i in range(82)] # evitando divisão por zero
    res = minimize(likelihood_forgetting_poisson,
                   x0,
                   args = (clubs, games, date),
                   method = 'SLSQP',
                   bounds = bounds)

    x = res.x
    x = x / x[0]
    forces, k, c = vet2force4getting(x, clubs)
    
    return forces, k, c
    
@jit(fastmath = True)
def update_results(results, year, rd, table):
    count = 1
    for club in table.index:
        if club not in results[year][rd]:
            results[year][rd][club] = {}
            for pos in range(1, 21):
                results[year][rd][club][pos] = 0

        results[year][rd][club][count] += 1
        count += 1
        
    return results

@jit(parallel = True, fastmath = True)
def classification(games_results, results, year, rd, n):
    for i in prange(n):
        clubs = {'points' : {},
                 'games' : {},
                 'wins' : {},
                 'draws' : {},
                 'defeats' : {},
                 'goals for' : {},
                 'goals against' : {},
                 'goal difference' : {}}

        for home_team in games_results:
            for away_team in games_results[home_team]:
                if year == 2016 and rd == 38 and 'Chape' in home_team:
                    pass
                else:
                    if home_team not in clubs['points']:
                        for key in clubs:
                            clubs[key][home_team] = 0

                    if away_team not in clubs['points']:
                        for key in clubs:
                            clubs[key][away_team] = 0

                    result = games_results[home_team][away_team][i]
                    sep = result.find('x')
                    home_score = int(result[:sep - 1])
                    away_score = int(result[sep + 2:])

                    clubs['games'][home_team] += 1
                    clubs['games'][away_team] += 1
                    clubs['goals for'][home_team] += home_score
                    clubs['goals for'][away_team] += away_score
                    clubs['goals against'][home_team] += away_score
                    clubs['goals against'][away_team] += home_score
                    clubs['goal difference'][home_team] += home_score - away_score
                    clubs['goal difference'][away_team] += away_score - home_score

                    if home_score > away_score:
                        clubs['points'][home_team] += 3
                        clubs['wins'][home_team] += 1
                        clubs['defeats'][away_team] += 1
                    elif home_score == away_score:
                        clubs['points'][home_team] += 1
                        clubs['points'][away_team] += 1
                        clubs['draws'][home_team] += 1
                        clubs['draws'][away_team] += 1
                    else:
                        clubs['points'][away_team] += 3
                        clubs['wins'][away_team] += 1
                        clubs['defeats'][home_team] += 1

        table = pd.DataFrame(clubs)
        table.sort_values(['points',  'wins', 'goal difference', 'goals for'],
                          axis = 0,
                          ascending = False,
                          inplace = True)

        results = update_results(results, year, rd, table)
    
    return results

def run_models(model, years, rounds, games, n_sims = 10000):
    '''
    Treina e executa os modelos dados para os anos a partir das
    rodadas dadas.
    '''
    results = {}
    exe_times = {}
    games_results = {}
    if type(model) == list:
        for i in range(len(model)):
            result, exe_time = run_models(model[i], years, rounds, games, n_sims = n_sims)
            results[model[i][0]] = result
            exe_times[model[i][0]] = exe_time
            print()
            
        return results, exe_times
    
    name, model, train = model
    print(name + ':')
    if model == forgetting_poisson:
        games['New_Date_Num'] = matplotlib.dates.date2num(pd.to_datetime(games['New_Date'],
                                                                         dayfirst = True))
        with_date = True
    else:
        with_date = False

    for year in years:
        print('  ' + str(year) + ':')
        if year not in results:
            results[year] = {}
            exe_times[year] = {}
        
        x0 = None
        for rd in rounds:
            if rd not in results[year]:
                results[year][rd] = {}
                exe_times[year][rd] = {}
            
            fit_games = pd.DataFrame()
            test_games = pd.DataFrame()
            
            print('    Treinando rodada', rd)
            train_time_i = time()
            if with_date:
                date = min(games.loc[((games['Round'] == rd) * (games['Year'] == year)), 'New_Date_Num'])
                fit_games = pd.concat([fit_games, games.loc[((games['New_Date_Num'] < date) * (games['Year'] == year))]],
                                    ignore_index = True)
                test_games = pd.concat([test_games, games.loc[((games['New_Date_Num'] >= date) * (games['Year'] == year))]],
                                    ignore_index = True)
                forces, k, c = train(fit_games, x0, date)
            else:
                fit_games = pd.concat([fit_games, games.loc[((games['Round'] <= rd) * (games['Year'] == year))]],
                                    ignore_index = True)
                test_games = pd.concat([test_games, games.loc[((games['Round'] > rd) * (games['Year'] == year))]],
                                    ignore_index = True)
                forces = train(fit_games, x0)
                
            train_time_f = time()
            exe_times[year][rd]['Treino'] = train_time_f - train_time_i
            
            print('    Simulando a partir da rodada', rd)
            sim_time_i = time()
            for game in fit_games.index:
                home_club = fit_games.loc[game, 'Team 1']
                away_club = fit_games.loc[game, 'Team 2']
                if home_club not in games_results:
                    games_results[home_club] = {}
                    
                games_results[home_club][away_club] = [fit_games.loc[game, 'Result'] for i in range(n_sims)]
            
            for game in test_games.index:
                home_club = test_games.loc[game, 'Team 1']
                away_club = test_games.loc[game, 'Team 2']
                if home_club not in games_results:
                    games_results[home_club] = {}

                if type(forces) == dict or type(forces) == numba.typed.typeddict.Dict:
                    home = forces[test_games.loc[game, 'Team 1']]
                    away = forces[test_games.loc[game, 'Team 2']]
                    if 'Casa' in home:
                        # nem todo modelo faz a separação por mando de campo
                        home = forces[test_games.loc[game, 'Team 1']]['Casa']
                        away = forces[test_games.loc[game, 'Team 2']]['Fora']

                    games_results[home_club][away_club] = model(n_sims, home, away)

                else:
                    games_results[home_club][away_club] = model(n_sims, forces)

            print('    Guardando resultados')
            results = classification(games_results, results, year, rd, n_sims)
            
            sim_time_f = time()
            exe_times[year][rd]['Simulações'] = sim_time_f - sim_time_i
        
            if model == forgetting_poisson:
                x0 = force4getting2vet(forces, k, c)
            elif model == more_robust_poisson:
                x0 = force42vet(forces)
            elif model == robust_poisson:
                x0 = force22vet(forces)
            
    return results, exe_times
