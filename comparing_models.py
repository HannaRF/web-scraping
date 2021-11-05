import requests
import numpy as np
import pandas as pd
from models import *
from Levenshtein import distance
from bs4 import BeautifulSoup as bs

def final_classification(year):
	clubs = {'Flamengo' : 'Flamengo / RJ',
			 'Criciúma' : 'Criciúma / SC',
			 'Figueirense' : 'Figueirense / SC',
			 'Chapecoense' : 'Chapecoense / SC',
			 'Avaí' : 'Avaí / SC',
			 'Joinville' : 'Joinville / SC',
			 'Internacional' : 'Internacional / RS',
			 'Atlético Mineiro' : 'Atlético / MG',
			 'América Mineiro' : 'América / MG',
			 'Cruzeiro' : 'Cruzeiro / MG',
			 'São Paulo' : 'São Paulo / SP',
			 'Vitória' : 'Vitória / BA',
			 'Fluminense' : 'Fluminense / RJ',
			 'Grêmio' : 'Grêmio / RS',
			 'Palmeiras' : 'Palmeiras / SP',
			 'Santos' : 'Santos / SP',
			 'Ceará' : 'Ceará / CE',
			 'Athletico Paranaense' : 'Athlético / PR', # tem as duas versões na wikipédia
			 'Atlético Paranaense' : 'Athlético / PR', # (antiga e atual)
			 'Corinthians' : 'Corinthians / SP',
			 'Portuguesa' : 'Portuguesa / SP',
			 'Ponte Preta' : 'Ponte Preta / SP',
			 'Red Bull Bragantino' : 'Red Bull Bragantino / SP',
			 'Atlético Goianiense' : 'Atlético / GO',
			 'Sport' : 'Sport / PE',
			 'Sport[nota 4]' : 'Sport / PE', # gambiarra por causa de 2018, onde perdeu 3 pontos
			 'Bahia' : 'Bahia / BA',
			 'Fortaleza' : 'Fortaleza / CE',
			 'Vasco da Gama' : 'Vasco da Gama / RJ',
			 'Goiás' : 'Goiás / GO',
			 'Coritiba' : 'Coritiba / PR',
			 'Paraná' : 'Paraná / PR',
			 'Náutico' : 'Náutico / PE',
			 'Santa Cruz' : 'Santa Cruz / PE',
			 'CSA' : 'Csa / AL',
			 'Botafogo' : 'Botafogo / RJ'}
			 
	url = f'https://pt.wikipedia.org/wiki/Campeonato_Brasileiro_de_Futebol_de_{year}_-_S%C3%A9rie_A'
	content = requests.get(url)
	soup = bs(content.text, 'lxml')
	tag = soup.findAll('table', {'class' : 'wikitable'})
	if year >= 2020:
		df = pd.DataFrame(pd.read_html(str(tag[3]))[0])
	else:
		df = pd.DataFrame(pd.read_html(str(tag[2]))[0])
		
	df.drop(['M', 'Classificação ou rebaixamento'], axis = 1, inplace = True)
	if year == 2018:
		# dando três pontos ao Sport (perdeu por causa da justiça, não em campo)
		df.loc[17, 'P'] += 3
		
		# agora vamos arrumar as posições
		df.loc[17, 'Pos.'] = 17
		df.loc[16, 'Pos.'] = 18
	elif year == 2013:
		# 4 pontos para Flamengo, mesmo caso acima
		df.loc[15, 'P'] = 49
		
		# 4 pontos para Portuguesa, idem
		df.loc[16, 'P'] = 48
		
		# agora vamos arrumar as posições
		df.loc[15, 'Pos.'] = 11
		df.loc[16, 'Pos.'] = 12
		for pos in range(10, 15):
			df.loc[pos, 'Pos.'] += 2
		
	df.sort_values('Pos.', axis = 0, inplace = True)
	df.set_index('Pos.', inplace = True)
	for pos in df.index:
		if df.loc[pos, 'Equipes'] in clubs:
			df.loc[pos, 'Equipes'] = clubs[df.loc[pos, 'Equipes']]
			
	return df
    
def probabilities_matrix(classification, results, model, year, rd):
    final_result = np.eye(20)
    predict_result = np.zeros((20, 20))
    for pos in range(1, 21):
        club = classification.loc[pos, 'Equipes']
        for pred_pos in range(1, 21):
            predict_result[pos - 1, pred_pos - 1] = results[model][year][rd][club][pred_pos] / 10000
            
    return final_result, predict_result
    
def calculate_prob(f_home, f_away):
    k, hp, ap = 0, [], []
    ph, pa = poisson.pmf(k, f_home), poisson.pmf(k, f_away)
    while ph > 1e-20 or pa > 1e-20:
        hp.append(ph)
        ap.append(pa)
        k += 1
        ph, pa = poisson.pmf(k, f_home), poisson.pmf(k, f_away)

    hp, ap = np.matrix(hp), np.matrix(ap)
    probs = hp.T @ ap
    probs = [np.sum(np.tril(probs, -1)), np.sum(np.diag(probs)), np.sum(np.triu(probs, 1))]
    return probs

def poisson_betting(home, away, odds):
    if type(home) == float:
        probs = calculate_prob(home, away)
        ev = [p * o for p, o in zip(probs, odds)]
        bet1 = np.argmax(ev)
        return 'HDA'[bet1], probs
        
    if 'Casa' in home:
        h_atk, h_def = home['Casa']['Ataque'], home['Casa']['Defesa']
        a_atk, a_def = away['Fora']['Ataque'], away['Fora']['Defesa']
    else:
        h_atk, h_def = home['Ataque'], home['Defesa']
        a_atk, a_def = away['Ataque'], away['Defesa']
        
    probs = calculate_prob(h_atk / a_def, a_atk / h_def)
    ev = [p * o for p, o in zip(probs, odds)]
    bet1 = np.argmax(ev)
    return 'HDA'[bet1], probs
    
def unpack_odds(df):
    all_odds = pd.read_excel('BRA.xlsx')
    a = sorted(list(all_odds['Home'].unique()))
    b = sorted(list(df['Team 1'].unique()))
    changes = {}
    for i in range(len(a)):
        if a[i] == 'Vasco':
            changes[a[i]] = 'Vasco da Gama / RJ'
        elif a[i] == 'Bragantino':
            changes[a[i]] = 'Red Bull Bragantino / SP'
        else:
            ds = []
            for j in range(len(b)):
                ds.append(distance(a[i], b[j]))

            changes[a[i]] = b[np.argmin(ds)]

    for i in range(len(all_odds)):
        if all_odds.loc[i, 'Season'] < 2013:
            all_odds.drop(index = i, inplace = True)
        else:
            all_odds.loc[i, 'Home'] = changes[all_odds.loc[i, 'Home']]
            all_odds.loc[i, 'Away'] = changes[all_odds.loc[i, 'Away']]

    all_odds.reset_index(inplace = True)
    all_odds.drop(columns = ['Country', 'League', 'Date', 'Time',
                             'HG', 'AG', 'MaxH', 'MaxD', 'MaxA',
                             'AvgH', 'AvgD', 'AvgA', 'index'],
                  inplace = True)

    return all_odds
    
def separate_games(df, year, rd):
    test_games = df.loc[((df['Round'] == rd) * (df['Year'] == year))].copy()
    test_games.reset_index(inplace = True)
    test_games.drop(columns = ['index',
                               'Competition',
                               'Result',
                               'Date',
                               'Date_by_Nearest',
                               'New_Date',
                               'New_Date_Num'],
                    inplace = True)

    return test_games
    
def find_odds(all_odds, year, home, away):
    for k in range(len(all_odds)):
        if all_odds.loc[k, 'Season'] == year:
            if all_odds.loc[k, 'Home'] == home:
                if all_odds.loc[k, 'Away'] == away:
                    odds = [all_odds.loc[k, 'PH'],
                            all_odds.loc[k, 'PD'],
                            all_odds.loc[k, 'PA']]

                    result = all_odds.loc[k, 'Res']
                    return odds, result
                    
def calculate_gain(models, rds, year, all_forces, df, p = 0.5):
    all_odds = unpack_odds(df)
    strategy1 = []
    strategy2 = []
    strategy3 = []
    strategy4 = []
    for i in range(len(models)):
        model = models[i][0]
        force = all_forces[model][year]
        gains1 = []
        gains2 = []
        gains3 = []
        gains4 = []
        for rd in rds:
            games_f_clubs = pd.concat([pd.DataFrame(), df.loc[((df['Round'] < rd) * (df['Year'] == year))]],
                                    ignore_index = True)

            clubs = find_clubs(games_f_clubs)
            gains1.append(0)
            gains2.append(0)
            gains3.append(0)
            gains4.append(0)
            test_games = separate_games(df, year, rd)
            for j in range(len(test_games)):
                home = test_games.loc[j, 'Team 1']
                away = test_games.loc[j, 'Team 2']
                odds, result = find_odds(all_odds, year, home, away)
                if i < 2:
                    probs = force[rd]
                    ev = [p * o for p, o in zip(probs, odds)]
                    b1 = 'HDA'[np.argmax(ev)]
                elif i == 2:
                    probs = force[rd][home]['Casa']
                    probs += force[rd][away]['Fora']
                    probs /= 2
                    ev = [p * o for p, o in zip(probs, odds)]
                    b1 = 'HDA'[np.argmax(ev)]
                elif i == 3:
                    b1, probs = poisson_betting(force[rd], force[rd], odds)
                elif i == 4:
                    home, away = force[rd]
                    b1, probs = poisson_betting(home, away, odds)
                elif i == 5:
                    act_force = vet2force2(force[rd], clubs)
                    b1, probs = poisson_betting(act_force[home], act_force[away], odds)
                elif i == 6:
                    act_force = vet2force4(force[rd], clubs)
                    b1, probs = poisson_betting(act_force[home], act_force[away], odds)
                else:
                    act_force, _, __ = vet2force4getting(force[rd], clubs)
                    b1, probs = poisson_betting(act_force[home], act_force[away], odds)

                b2 = 'HDA'[np.argmax(probs)]
                if b1 == result:
                    gains1[-1] += odds['HDA'.index(b1)] - 1
                else:
                	gains1[-1] -= 1

                if b2 == result:
                    gains2[-1] += odds['HDA'.index(b2)] - 1
                else:
                	gains2[-1] -= 1

                if max(probs) > p and b2 == result:
                    gains3[-1] += odds['HDA'.index(b2)] - 1
                    gains4[-1] += odds['HDA'.index(b2)] - 1
                elif max(probs) > p and b2 != result:
                	gains3[-1] -= 1
                	gains4[-1] -= 1
                elif b1 == result:
                    gains4[-1] += odds['HDA'.index(b1)] - 1
                else:
                	gains4[-1] -= 1
            
        strategy1.append(gains1)
        strategy2.append(gains2)
        strategy3.append(gains3)
        strategy4.append(gains4)
        
    return strategy1, strategy2, strategy3, strategy4
