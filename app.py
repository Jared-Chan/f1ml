import math
import requests
import json
import streamlit as st
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from fsplit.filesplit import Filesplit
import os
import SessionState

fs = Filesplit()

st.set_page_config(page_title='F1 Laps with ML', page_icon=None, layout='centered', initial_sidebar_state='auto')

db_dir = './db/'
#if not os.path.exists('./model_sd.pth'):
#    fs.merge(input_dir="./db/models/model_split",output_file="./model_sd.pth", cleanup=False)
if not os.path.exists('./model_sd_47.pth'):
    fs.merge(input_dir="./db/models/model_split_47",output_file="./model_sd_47.pth", cleanup=False)

def main():

    session_state = SessionState.get(user_name='', model=0, record=[], circuitName='', circuitLoc='', year='', round='', graph=None)

    model = RacePredictionModel(4051, 1200, 1200, 2, 0.2)
    if (session_state.model == 0):
        #model.load_state_dict(torch.load('./model_sd.pth',map_location=torch.device('cpu')))
        model.load_state_dict(torch.load('./model_sd_47.pth',map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('./model_sd_47.pth',map_location=torch.device('cpu')))
    model.eval()

    years = range(2001, 2022)
    st.sidebar.title("Model Input")
    year = st.sidebar.selectbox("Season", years, index=20)
    _round = st.sidebar.number_input("Round", min_value=1, step=1)
    if (year < 2021):
        qualifying = st.sidebar.checkbox("Qualifying")
        if (not qualifying):
            laps = st.sidebar.number_input("Use data up to lap", step=1, min_value=1)
        else:
            laps = 1
    else:
        laps = 1
    pred_laps = st.sidebar.number_input("Total number of laps", min_value=1, max_value=200, value=50, step=1, help='The model will predict up to this many laps.')
    randomness = st.sidebar.slider("Randomness factor", min_value=0, max_value=100, value=0, step=1)
    model_selection = st.sidebar.selectbox('Model selection', index=0,
    options=['Regular','Optimized for Pit Stop Prediction (Not available)'],
    help='Regular: Trained with data from 2001 to 2020\n\n Optimized for Pit Stop Prediction: Trained with data from 2012 to 2020 (Not available due to size limitations.)')
    if (model_selection == 'Regular'):
        session_state.model = 0
    elif (model_selection == 'Optimized for Pit Stop Prediction'):
        session_state.model = 1
    predict = st.sidebar.button("Predict")
    st.sidebar.markdown('Learn more about this web app [here](https://github.com/Jared-Chan/f1ml).')

    st.title('Formula One Race Lap-by-Lap Prediction with Machine Learning')
    st.markdown('***')

    probar = st.empty()
    pos_line_chart = {}
    if (predict):
        probar = st.progress(0)
        laps_record = []
        _input, exp = get_times(year, _round, laps)
        probar.progress(0.1)
        if (len(_input) == 0):
            st.info('Sorry, there is no data for this race.')
            return
        states = model.zero_states()

        _in = torch.from_numpy(_input[0])
        if (len(exp) != 0):
            for i in range(0, len(_input)):
                _in = torch.from_numpy(_input[i])
                _out = torch.from_numpy(exp[i])
                circuit_name, circuit_loc, circuit_country, d, pos_line_chart = position_analysis(_in, _out, pred_laps, pos_line_chart)
                laps_record.append(d)
                with torch.no_grad():
                    out, states = model(_in.unsqueeze(0).unsqueeze(0).float(), states)
        probar.progress(0.2)
        for i in range(0, pred_laps):
            if (not (len(exp) == 0 and i == 0)):
                _in = out_to_in(_in, out.squeeze().squeeze(), True, pred_laps, randomness)
            with torch.no_grad():
                out, states = model(_in.unsqueeze(0).unsqueeze(0).float(), states)
            out = out.squeeze().squeeze()
            circuit_name, circuit_loc, circuit_country, d, pos_line_chart = position_analysis(_in, out, pred_laps, pos_line_chart)
            laps_record.append(d)
            probar.progress(0.2 + 0.8 * (i+1)/pred_laps)
        session_state.record = laps_record
        session_state.circuitName = circuit_name
        session_state.circuitLoc = circuit_loc
        session_state.year = year
        session_state.round = _round
        graph = pd.DataFrame.from_dict(pos_line_chart)
        session_state.graph = graph
        probar.progress(1.0)
        probar.progress(0)
        probar = st.empty()

    if (session_state.year):
        st.subheader(f'{session_state.year} Round {session_state.round}')
    else:
        st.subheader('Please configure model input from the sidebar.')
    if (len(session_state.circuitName) > 0):
        st.subheader(f'{session_state.circuitName}, {session_state.circuitLoc}')
    lap_num = st.slider("Lap Number", min_value=1, max_value=pred_laps, step=1, value=1)
    if (lap_num <= laps):
        st.text("From database")
    else:
        st.text("Prediction")

    if (len(session_state.record) > 0):
        if (lap_num >= len(session_state.record)):
            st.table(session_state.record[-1])
        else:
            st.table(session_state.record[lap_num-1])
        st.line_chart(session_state.graph)




@st.cache
def time_to_int(t):
  if (t == float):
    return t
  t2 = str(t)
  ts = t2.rsplit(':')
  if ('\\N' in t2):
    return None
  if (not '.' in t2):
    return None
  if (len(ts) > 1):
    return int(ts[0]) * 60 + float(ts[1])
  else:
    return float(ts[0])

races = pd.read_csv(db_dir + 'races.csv')
circuits = pd.read_csv(db_dir + 'circuits.csv')
drivers = pd.read_csv(db_dir + 'drivers.csv')
constructor = pd.read_csv(db_dir + 'constructors.csv')
status = pd.read_csv(db_dir + 'status.csv')

@st.cache
def race_info(raceId):
  _r = races.query(f'raceId  == {raceId}')
  if (_r.empty):
    return None, None, None
  _year = _r['year'].item()
  _round = _r['round'].item()
  _circuitId = _r['circuitId'].item()
  return _year, _round, _circuitId

@st.cache
def circuit_info(circuitId):
  _c = circuits.query(f'circuitId  == {circuitId}')
  if (_c.empty):
    return None, None, None
  _name = _c['name'].item()
  _location = _c['location'].item()
  _country = _c['country'].item()
  return _name, _location, _country

@st.cache
def driver_info(id):
  _d = drivers.query(f'driverId  == {id}')
  if (_d.empty):
    return None, None, None, None, None, None
  _number = _d['number'].item()
  _code = _d['code'].item()
  _forename = _d['forename'].item()
  _surname = _d['surname'].item()
  _dob = _d['dob'].item()
  _nationality = _d['nationality'].item()
  return _number, _code, _forename, _surname, _dob, _nationality

@st.cache
def constructor_info(id):
  _c = constructor.query(f'constructorId  == {id}')
  if (_c.empty):
    return None, None
  _name = _d['name'].item()
  _nationality = _d['nationality'].item()
  return _name, _nationality

@st.cache
def status_info(id):
  _s = status.query(f'statusId == {id}')
  if (_s.empty):
    return None
  _sstr = _s['status'].item()
  return _sstr

stat_emb = [
  [4.0, 3.0, 130.0], # Accident/Collision
  [22.0, 5.0, 10.0, 23.0, 44.0, 47.0, 30.0, 32.0, 8.0, 38.0, 43.0, 85.0, 9.0, 86.0, 6.0, 2.0, 7.0, 87.0, 71.0, 41.0, 46.0, 37.0, 65.0, 78.0, 25.0, 74.0, 75.0, 26.0, 51.0, 40.0, 79.0, 36.0, 83.0, 80.0, 21.0, 69.0, 72.0, 70.0, 27.0, 60.0, 63.0, 29.0, 64.0, 66.0, 56.0, 59.0, 61.0, 42.0, 39.0, 48.0, 49.0, 34.0, 35.0, 28.0, 24.0, 33.0, 129.0, 76.0, 91.0, 131.0, 101.0, 132.0, 135.0,  84.0,  136.0,  105.0,  137.0,  138.0,  139.0], # Car issues
  [11.0,  13.0,  12.0,  14.0,  17.0,  15.0,  16.0, 18.0,  55.0,  58.0,  45.0, 88.0], # Lapped
  [0.0], # No problem
  [77.0, 73.0, 82.0, 81.0, 62.0, 54.0, 31.0, 96.0], # Other
  [20.0] #'Spun off'
]
@st.cache
def stat_embed(id):
  _emb = np.zeros(6)
  for i in range(6):
    if id in stat_emb[i]:
      _emb[i] = 1
      return _emb
  _emb[4] = 1
  return _emb # Other

@st.cache
def stat_unbed(array, retired=False):
  _a = np.copy(array)
  if (retired):
    _a[3] = 0
  _i = np.argmax(_a)
  if (_i == 0):
    return 'Accident/Collision'
  elif (_i == 1):
    return 'Car Issues'
  elif (_i == 2):
    return 'Lapped'
  elif (_i == 3):
    return 'No Problem'
  elif (_i == 4):
    return 'Other'
  elif (_i == 5):
    return 'Spun off'
  else:
    return 'something is wrong'

@st.cache
def lt_embed(laptime):
  # laptime should be a float with 3 decimal places
  _lt = math.floor(laptime * 10)
  _lt_emb = []
  _ret = []
  for i in range(4):
    _lt_emb.append(int(_lt % 10))
    _lt = math.floor(_lt / 10)
  _ret = np.zeros(2)
  if (_lt_emb[-1] == 1):
    _ret[0] = 1
  elif (_lt_emb[-1] == 2):
    _ret[1] = 1
  elif (_lt_emb[-1] > 2):
    _ret[0] = 1
    _ret[1] = 1
  for i in range(3):
    _t = np.zeros(10)
    _t[_lt_emb[2 - i]] = 1
    _ret = np.append(_ret, _t)
  return _ret

@st.cache
def lt_unbed(l_array):
  _ret = 0
  if (l_array[0] >= 0.5 and l_array[1] >= 0.5):
    _ret += 300
  elif (l_array[0] >= 0.5):
    _ret += 100
  elif (l_array[1] >= 0.5):
    _ret += 200
  _ret += np.argmax(l_array[2:12]) * 10
  _ret += np.argmax(l_array[12:22]) * 1
  _ret += np.argmax(l_array[22:32]) * 0.1
  return _ret

drivers_short = pd.read_csv(db_dir + 'drivers_short.csv')
# from driverId to our id
@st.cache
def driver_embed_idx(driverId):
  row = drivers_short.query(f'driverId == {driverId}').index
  if (row.empty):
    return 0
  return row.item() + 1

# from our id to driverId
@st.cache
def driver_unbed_idx(idx):
  row = drivers_short.iloc[idx-1]
  return row['driverId']

# from our id to array
@st.cache
def driver_embed(idx):
  _e = np.zeros(130)
  _e[idx-1] = 1
  return _e

# from array to our id
@st.cache
def driver_unbed(d_array):
  return np.argmax(d_array) + 1

@st.cache(suppress_st_warning=True)
def get_times(year, _round, lap):
  if (year <= 2020):
    race = np.load(db_dir + f'/races_npy/{year}/{_round-1}_in.npy')
    race_out = np.load(db_dir + f'/races_npy/{year}/{_round-1}_exp.npy')
    if (lap >= len(race)):
        return race, race_out
    race = race[:lap]
    race_out = race_out[:lap]
    return race, race_out
  else:

    if not os.path.exists(db_dir + f'/cache/{year}_{_round}_q.json'):
        quali = requests.get(f'http://ergast.com/api/f1/{year}/{_round}/qualifying.json')
        if (quali.status_code < 200):
          return [], []
        j = quali.json()

        if (_round - 1 < 1):
          d_s = requests.get(f'http://ergast.com/api/f1/{year-1}/driverStandings.json')
          c_s = requests.get(f'http://ergast.com/api/f1/{year-1}/constructorStandings.json')
        else:
          d_s = requests.get(f'http://ergast.com/api/f1/{year}/{_round-1}/driverStandings.json')
          c_s = requests.get(f'http://ergast.com/api/f1/{year}/{_round-1}/constructorStandings.json')
        if (d_s.status_code < 200):
          ds_ok = False
        else:
          ds_ok = True
          try:
              d_s = d_s.json()
          except:
              st.error('Something went wrong while getting race information.')
              return [], []
        if (c_s.status_code < 200):
          cs_ok = False
        else:
          cs_ok = True
          try:
              c_s = c_s.json()
          except:
              st.error(f'{c_s.text}')
              st.error('Something went wrong while getting race information.')
              return [], []

        if (len(j['MRData']['RaceTable']['Races']) != 0):
            with open(db_dir + f'cache/{year}_{_round}_q.json', 'w') as f:
                json.dump(j, f)
            with open(db_dir + f'cache/{year}_{_round}_ds.json', 'w') as f:
                json.dump(d_s, f)
            with open(db_dir + f'cache/{year}_{_round}_cs.json', 'w') as f:
                json.dump(c_s, f)
    else:
        with open(db_dir + f'cache/{year}_{_round}_q.json', 'r') as f:
            j = json.load(f)
        with open(db_dir + f'cache/{year}_{_round}_ds.json', 'r') as f:
            d_s = json.load(f)
            ds_ok = True
        with open(db_dir + f'cache/{year}_{_round}_cs.json', 'r') as f:
            c_s = json.load(f)
            cs_ok = True


    if (len(j['MRData']['RaceTable']['Races']) == 0):
        return [], []
    circuitRef = j['MRData']['RaceTable']['Races'][0]['Circuit']['circuitId']
    circuitId = circuits.query(f'circuitRef == \'{circuitRef}\'')['circuitId'].item()

    ret = np.zeros(130)
    ret[circuitId] = 1
    ret = np.append(ret, np.zeros(1)) # lap number/ total number of laps

    for i in range(20):
      if (i < len(j['MRData']['RaceTable']['Races'][0]['QualifyingResults'])):
        driverRef = j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]['Driver']['driverId']
        did = drivers.query(f'driverRef == \'{driverRef}\'')['driverId'].item()
        our_did = driver_embed_idx(did)
        ret = np.append(ret, driver_embed(our_did))

        cref = j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]['Constructor']['constructorId']

        ds = np.zeros(3)
        if (not ds_ok):
          ret = np.append(ret, ds)
        else:
          for k in range(20):
            if (k < len(d_s['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings'])):
              if (d_s['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings'][k]['Driver']['driverId'] == driverRef):
                if (k <= 1):
                  ds[0] = 1
                if (k <=3):
                  ds[1] = 1
                if (k <=10):
                  ds[2] = 1
                ret = np.append(ret, ds)
                break
            if (k == 19): # if there is no standing for this driver
              ret = np.append(ret, ds)

        cs = np.zeros(2)
        if (not cs_ok):
          ret = np.append(ret, cs)
        else:
          for k in range(len(c_s['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings'])):
            if (c_s['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings'][k]['Constructor']['constructorId'] == cref):
              if (k <= 1):
                cs[0] = 1
              if (k <= 3):
                cs[1] = 1
              ret = np.append(ret, cs)
              break
            if (k == len(c_s['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings'])-1): # if there is no standing for this constructor
              ret = np.append(ret, cs)

        pos = np.zeros(21)
        _p = i
        pos[_p] = 1
        ret = np.append(ret, pos)

        pit = np.zeros(1)
        ret = np.append(ret, pit)

        stat = stat_embed(0)
        ret = np.append(ret, stat)

        if ('Q3' in j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]):
          t = j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]['Q3']
        elif ('Q2' in j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]):
          t = j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]['Q2']
        elif ('Q1' in j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]):
          t = j['MRData']['RaceTable']['Races'][0]['QualifyingResults'][i]['Q1']
        else:
          t = 0.0
        t = time_to_int(t)
        if (t == None):
            t = 0
        laptime = lt_embed(float(t))
        ret = np.append(ret, laptime)

        rando = np.zeros(1)
        ret = np.append(ret, rando)

      else:
        ret = np.append(ret, np.zeros(3920 - (20-i) * 196))
        break

    return np.expand_dims(ret, 0), []

@st.cache(allow_output_mutation=True)
def position_analysis(lap_in, out, num_of_laps=1, line_chart={}):
  df = pd.DataFrame(columns=['code', 'driver', 'position', 'laps till pitting', 'status', 'laptime'])
  _lap = lap_in.detach().clone().numpy()
  _o = out.detach().clone().numpy()
  _name, _loc, _country = circuit_info(np.argmax(_lap[:130]))
  for i in range(20):
    _d_idx = driver_unbed_idx(driver_unbed(_lap[131 + i * 196 : 131 + i * 196 + 130]))
    _num, _code, _fn, _ln, _, _ = driver_info(_d_idx)
    _pos = np.argmax(_o[i*60 : i*60 + 21]) + 1
    if (_code == '\\N'):
        _tempcode = _ln[:3]
    else:
        _tempcode = _code
    if (_tempcode in line_chart):
        line_chart[_tempcode].append(21-_pos)
    else:
        line_chart[_tempcode] = [21-_pos]
    _pitting = _o[i*60+21] * num_of_laps
    if (_pitting == 0):
      _pitting = 'NA'
    _retired = False
    if (_pos == 21):
      _retired = True
    _status = stat_unbed(_o[i*60 + 22: i*60 + 28], _retired)
    if (_d_idx == 853):
        _status = 'Spun off'
    if (_retired):
        _time = 'NA'
    else:
        _time = lt_unbed(_o[i*60 + 28:])
        _time = f'{_time:.1f}'
    df = df.append({
        'code': f'{_code}',
        'driver': f'{_fn} {_ln}',
        'position': int(_pos),
        'laps till pitting': _pitting,
        'status': _status,
        'laptime': _time
    }, ignore_index=True)

  df = df.sort_values(by=['position', 'laptime'], ascending=[True, False])
  return _name, _loc, _country, df, line_chart

# Returns a tensor with the size of in but content of out
@st.cache
def out_to_in(in_, out_, random=False, num_of_laps=50, randomness=20):
  _ret = in_.detach().clone().numpy()
  _o = out_.detach().clone().numpy()
  _o = _o.reshape([1200])
  _ret = _ret.reshape([4051])
  j = 0
  for i in range(0, 20):
    _ret[131 + i*196 + 135: i*196 + 131 + 195] = _o[j*60: (j+1) * 60]
    if (round(_o[j*60 + 21] * num_of_laps) <= 1):
      _ret[131 + i*196 + 130 + 26] = 1
    else:
      _ret[131 + i*196 + 130 + 26] = 0
    if (random):
      _ret[i * 196 + 131 + 195] = np.random.uniform(0, randomness, [1])
    j += 1

  return torch.from_numpy(_ret).float()

class RacePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, lstm_hids, lstm_layers, dropout):
        super(RacePredictionModel, self).__init__()

        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.lstm_hids = lstm_hids

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hids, num_layers=lstm_layers, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(lstm_hids, output_size)

    def zero_states(self, batchsize=1):
        hidden_state = torch.zeros(self.lstm_layers, batchsize, self.lstm_hids)
        cell_state = torch.zeros(self.lstm_layers, batchsize, self.lstm_hids)
        return (hidden_state, cell_state)

    def forward(self, ins, prev_states=None):
        lstm_outs, next_states = self.lstm(ins, prev_states)
        outs = self.fc(lstm_outs)
        return outs, next_states

if __name__ == '__main__':
    main()
