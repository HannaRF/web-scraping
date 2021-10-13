import requests
import numpy as np
import pandas as pd
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
