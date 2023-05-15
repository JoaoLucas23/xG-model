# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:23:11 2023

@author: jllgo
"""

import pandas as pd
import d6tflow as d6t



class wyLoadData(d6t.tasks.TaskCSVPandas):
    data_dir = d6t.Parameter()
    
    def run(self):
        players = pd.read_json(path_or_buf=self.data_dir+'\players.json')
        teams = pd.read_json(path_or_buf=self.data_dir+'\\teams.json')
        players = players.rename(columns={'wyId': 'player_id', 'shortName': 'player_name', 'currentTeamId': 'team_id'})
        players['player_name'] = players['player_name'].str.decode('unicode-escape')
        teams = teams.rename(columns={'wyId': 'team_id', 'name': 'team_name'})
        teams['team_name'] = teams['team_name'].str.decode('unicode-escape')
        players = players[['player_id', 'player_name', 'team_id']]
        teams = teams[['team_id', 'team_name']]
        wyData = players.merge(teams, on='team_id')

        self.save(wyData)

class wyLoadTimePlayed(d6t.tasks.TaskCSVPandas):
    data_dir = d6t.Parameter()

    def run(self):
        minutes_played_england = pd.read_json(path_or_buf=self.data_dir+'\minutes_played_per_game_England.json')
        minutes_played_spain = pd.read_json(path_or_buf=self.data_dir+'\minutes_played_per_game_Spain.json')
        minutes_played = pd.concat([minutes_played_england,minutes_played_spain], axis=0)
        minutes_played = minutes_played.rename(columns={'playerId': 'player_id', 'minutesPlayed':'minutes_played','matchId':'game_id','shortName':'player_name'})
        minutes_played = minutes_played[['player_id', 'minutes_played','game_id','player_name']]
        self.save(minutes_played)