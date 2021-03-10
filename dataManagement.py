import pandas
import datetime as dt

import csv
import urllib3
import requests


EMILIA_ROMAGNA = 8
LOMBARDIA = 3
csv_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"

# req = requests.get(csv_url)
# url_content = req.content
# csv_file = open('./region.csv', 'wb')
# csv_file.write(url_content)

"""Importare il file"""
df = pandas.read_csv('./region.csv', index_col=None);
# Matrice di contatto di Italia di Prem et al
contact_data = pandas.read_excel('./MUestimates_all_locations_1.xlsx','Italy');

"""Selezionare le colonne"""
columns = ['data', 'totale_positivi', 'dimessi_guariti', 'deceduti', 'tamponi', 'totale_casi']
df_regione = pandas.DataFrame(columns=columns);
for index, row in df.iterrows():

    """Scegliere la regione da voler analizzare"""
    if (row['codice_regione'] == LOMBARDIA):
        df_regione = df_regione.append(row[columns]);

#Data è array con righe i giorni a partire dal 24 Febbraio e come colonne 0: 'totale_positiv', 1:'dimessi guariti', 2:'deceduti'
data = df_regione.loc[:, ['totale_positivi', 'dimessi_guariti', 'deceduti']].astype('float').values

dates = df_regione.loc[:, ['data']].values
dateForPlot = [dt.datetime.strptime(d[0], '%Y-%m-%dT%H:%M:%S').date() for d in dates]



