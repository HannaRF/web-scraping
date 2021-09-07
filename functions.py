import csv
import re

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
    game_round = game[-7:-5]
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
