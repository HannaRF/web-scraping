import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from datetime import date

def find_teams(game):
    ignore_list = ['FIFA', 'CBF', 'AB', 'CD', 'ESP', '000', 'ASS', 'MAST',
                   'MTR', 'FD', 'BAS', 'INT /', 'G /', 'F /']
    replacing = [('America / MG', 'América / MG'),
                 ('Athletico Paranaense / PR', 'Athlético / PR'),
                 ('Atletico / PR', 'Athlético / PR'),
                 ('Atlético / PR', 'Athlético / PR'),
                 ('Atlético Mineiro / MG', 'Atlético / MG'),
                 ('BOTAFOGO / RJ', 'Botafogo / RJ'),
                 ('Criciuma / SC', 'Criciúma / SC')]
    
    teams = []
    with open(game) as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            for value in row:
                try_value = True
                for ignore in ignore_list:
                    if ignore in value:
                        try_value = False

                if try_value:
                    clubs = re.findall('[\w{1,} \w{1,}]+ / [A-Z]{2}', value)
                    for club in clubs:
                        if ' X ' in club:
                            club = club[3:]

                        for replace in replacing:
                            if club == replace[0]:
                                club = replace[1]
                        
                        aux = club.strip()[0]
                        if aux != '/' and not aux.isdigit() and club not in teams:
                            teams.append(club)
                            
    return teams
    
def find_date(game):
    game_round = str(ceil(int(game[-7:-4])/10))
    game_year = game[-17:-13]
    date = []
    finded = False
    with open(game) as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            if not finded:
                for value in row:
                    date = re.findall('[0-9]{2}/[0-9]{2}/[0-9]{4}', value)
                    if date != []:
                        finded = True
                        break
                        
    date = [game_round, game_year] + date
    
    return date
    
def find_score(game):
    score = ''
    with open(game) as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            for value in row:
                possible_scores = re.findall('[0-9] x [0-9]', value.lower())
                if possible_scores != [] and score == '':
                    score = possible_scores[0]
                    
                for new_score in possible_scores:
                    if int(new_score[0]) >= int(score[0]) and int(new_score[-1]) >= int(score[-1]):
                        score = new_score
                
    if score == '':
        # only one error in Brazilian Seria A (and the score was 0 x 0)
        print('No score found at ' + game + ', returned 0 x 0')
        return '0 x 0'
    
    return score
    
def classification_table(br):
    clubs = {'points' : {},
             'games' : {},
             'wins' : {},
             'draws' : {},
             'defeats' : {},
             'goals for' : {},
             'goals against' : {},
             'goal difference' : {}}
    for i in br.index:
        if i != 1517: # chape e atlético, após o acidente da chape (WO duplo)
            if br.loc[i, 'Team 1'] not in clubs['points']:
                for key in clubs:
                    clubs[key][br.loc[i, 'Team 1']] = 0

            if br.loc[i, 'Team 2'] not in clubs['points']:
                for key in clubs:
                    clubs[key][br.loc[i, 'Team 2']] = 0
                    
            sep = br.loc[i, 'Result'].find('x')
            home_score = int(br.loc[i, 'Result'][:sep - 1])
            away_score = int(br.loc[i, 'Result'][sep + 2:])

            clubs['games'][br.loc[i, 'Team 1']] += 1
            clubs['games'][br.loc[i, 'Team 2']] += 1
            clubs['goals for'][br.loc[i, 'Team 1']] += home_score
            clubs['goals for'][br.loc[i, 'Team 2']] += away_score
            clubs['goals against'][br.loc[i, 'Team 1']] += away_score
            clubs['goals against'][br.loc[i, 'Team 2']] += home_score
            clubs['goal difference'][br.loc[i, 'Team 1']] += home_score - away_score
            clubs['goal difference'][br.loc[i, 'Team 2']] += away_score - home_score

            if home_score > away_score:
                clubs['points'][br.loc[i, 'Team 1']] += 3
                clubs['wins'][br.loc[i, 'Team 1']] += 1
                clubs['defeats'][br.loc[i, 'Team 2']] += 1
            elif home_score == away_score:
                clubs['points'][br.loc[i, 'Team 1']] += 1
                clubs['points'][br.loc[i, 'Team 2']] += 1
                clubs['draws'][br.loc[i, 'Team 1']] += 1
                clubs['draws'][br.loc[i, 'Team 2']] += 1
            else:
                clubs['points'][br.loc[i, 'Team 2']] += 3
                clubs['wins'][br.loc[i, 'Team 2']] += 1
                clubs['defeats'][br.loc[i, 'Team 1']] += 1

    table = pd.DataFrame(clubs)
    table.sort_values(['points',  'wins', 'goal difference', 'goals for'],
                      axis = 0,
                      ascending = False,
                      inplace = True)
    
    return table
    
def date2int(string):
    d = string.split('/')
    d = date(int(d[2]), int(d[1]), int(d[0]))
    return d.toordinal()
    
def plot_goals(opt, year, goals, h_goals, a_goals, colors):
    name = ''
    if opt == 'All':
        using_goals = goals
        name += 'Gols por jogo'
    elif opt == 'Home':
        using_goals = h_goals
        name += 'Gols do mandante'
    elif opt == 'Away':
        using_goals = a_goals
        name += 'Gols do visitante'

    if year == 'All':
        year = 2022
        name += ' - 2013 a 2021'
    else:
        name += ' - ' + str(year)

    using_mean = np.mean(using_goals[year - 2013])
    plt.hist(using_goals[year - 2013], bins = [*range(10)], alpha = 0.5, density = True, color = colors['azul'])
    plt.axvline(using_mean, ls = '--', lw = 0.75, c = 'black')
    plt.annotate(f'Média: {using_mean:.4f}',
                 (0,0),
                 (0, -20),
                 xycoords = 'axes fraction',
                 textcoords = 'offset points',
                 va = 'top')
    plt.title(name, loc = 'center')
    plt.show()
