"""
Streamlit Scheduler (protótip)
Fitxer: streamlit_scheduler.py

Com funciona:
- Inicia l'app amb: streamlit run streamlit_scheduler.py
- Formulari per afegir serveis (origen, destí, data/hora inici, durada hores, km, pass, maletes, revenue)
- Llista de serveis a la sessió (es pot esborrar)
- Paràmetres d'optimització: pesos i descans mínim
- Botó "Generar horari" que executa el solver (PuLP) i mostra assignacions i resum
- Botó per descarregar CSV

Dependències: pip install streamlit pulp pandas python-dateutil
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import pulp
from dateutil import parser
import io

st.set_page_config(page_title='Scheduler Taxi - Protótip', layout='wide')
st.title('Scheduler Taxi - Protótip (Streamlit)')

# Inicialitzar session state
if 'services' not in st.session_state:
    st.session_state['services'] = []
if 'drivers' not in st.session_state:
    st.session_state['drivers'] = [
        {'id': 'D1', 'type': 'car'},
        {'id': 'D2', 'type': 'car'},
        {'id': 'D3', 'type': 'car'},
        {'id': 'D4', 'type': 'car'},
        {'id': 'D5', 'type': 'car'},
        {'id': 'D6', 'type': 'car'},
        {'id': 'M1', 'type': 'minibus'},
        {'id': 'M2', 'type': 'minibus'},
    ]
if 'vehicles' not in st.session_state:
    st.session_state['vehicles'] = {
        'a': {'seats': 6, 'trunk': 'normal', 'type': 'van_normal'},
        'b': {'seats': 6, 'trunk': 'normal', 'type': 'van_normal'},
        'c': {'seats': 6, 'trunk': 'large', 'type': 'van_gran'},
        'd': {'seats': 7, 'trunk': 'large', 'type': 'van_gran'},
        'e': {'seats': 3, 'trunk': 'normal', 'type': 'luxury'},
        'mb1': {'seats': 18, 'trunk': 'large', 'type': 'minibus'},
        'mb2': {'seats': 18, 'trunk': 'large', 'type': 'minibus'},
    }

st.sidebar.header('Afegir servei')
with st.sidebar.form('add_service'):
    sid = st.text_input('ID servei', value=f'S{len(st.session_state.services)+1}')
    start_str = st.text_input('Data i hora inici (YYYY-MM-DD HH:MM)', value=datetime.now().strftime('%Y-%m-%d 08:00'))
    dur = st.number_input('Durada (hores)', min_value=0.5, max_value=24.0, value=2.0, step=0.5)
    km = st.number_input('Quilòmetres', min_value=1, value=50)
    passengers = st.number_input('Passatgers', min_value=0, value=4)
    luggage = st.number_input('Maletes', min_value=0, value=1)
    revenue = st.number_input('Ingressos (€)', min_value=0.0, value=100.0)
    start_loc = st.text_input('Origen', value='Girona')
    end_loc = st.text_input('Destí', value='Barcelona')
    add = st.form_submit_button('Afegir')
    if add:
        try:
            start_dt = parser.parse(start_str)
        except Exception as e:
            st.error('Format data incorrecte. Exemple: 2025-11-17 08:00')
            st.stop()
        s = {'id': sid,
             'start': start_dt,
             'end': start_dt + timedelta(hours=float(dur)),
             'km': float(km),
             'pass': int(passengers),
             'luggage': int(luggage),
             'revenue': float(revenue),
             'start_loc': start_loc,
             'end_loc': end_loc}
        st.session_state['services'].append(s)
        st.experimental_rerun()

st.sidebar.markdown('---')
st.sidebar.header('Paràmetres d\'optimització')
W_BALANCE_HOURS = st.sidebar.slider('Pes equilibrar hores', 0.0, 10.0, 1.0)
W_BALANCE_KM = st.sidebar.slider('Pes equilibrar quilòmetres', 0.0, 10.0, 1.0)
W_REVENUE = st.sidebar.slider('Pes maximitzar ingrés net', 0.0, 10.0, 5.0)
MIN_REST_HOURS = st.sidebar.number_input('Descans mínim entre serveis (hores)', min_value=0.0, max_value=48.0, value=12.0)

st.sidebar.markdown('---')
if st.sidebar.button('Netejar serveis'):
    st.session_state['services'] = []
    st.experimental_rerun()

# Mostrar serveis
st.subheader('Serveis pendents')
if len(st.session_state['services']) == 0:
    st.info('No hi ha serveis. A la barra lateral pots afegir-ne.')
else:
    df_services = pd.DataFrame([{
        'id': s['id'],
        'start': s['start'],
        'end': s['end'],
        'km': s['km'],
        'pass': s['pass'],
        'luggage': s['luggage'],
        'revenue': s['revenue'],
        'start_loc': s['start_loc'],
        'end_loc': s['end_loc']
    } for s in st.session_state['services']])
    st.dataframe(df_services)
    # Esborrar selecció
    to_del = st.multiselect('Selecciona serveis per esborrar', df_services['id'].tolist())
    if st.button('Esborrar serveis seleccionats'):
        st.session_state['services'] = [s for s in st.session_state['services'] if s['id'] not in to_del]
        st.experimental_rerun()

# Funció de preferència tipus vehicle
def preferred_vehicle_type_for_service(s):
    if s['pass'] > 6:
        return 'minibus'
    elif s['pass'] > 3:
        return 'van_gran'
    else:
        return 'luxury'

# Solució (solver)
st.header('Generar horari')
if st.button('Generar horari amb IA'):
    services = st.session_state['services']
    if len(services) == 0:
        st.warning('No hi ha serveis per assignar')
        st.stop()

    service_ids = [s['id'] for s in services]
    driver_ids = [d['id'] for d in st.session_state['drivers']]

    # Estimar benefici net per servei segons tipus preferent
    FUEL_COST_PER_KM = {
        'van_normal': 0.12,
        'van_gran': 0.14,
        'luxury': 0.18,
        'minibus': 0.20
    }
    service_net_profit = {}
    service_pref_type = {}
    for s in services:
        pref = preferred_vehicle_type_for_service(s)
        service_pref_type[s['id']] = pref
        avg_fuel_cost = 0.14
        if pref == 'minibus': avg_fuel_cost = FUEL_COST_PER_KM['minibus']
        elif pref == 'van_gran': avg_fuel_cost = FUEL_COST_PER_KM['van_gran']
        elif pref == 'van_normal': avg_fuel_cost = FUEL_COST_PER_KM['van_normal']
        elif pref == 'luxury': avg_fuel_cost = FUEL_COST_PER_KM['luxury']
        net = s['revenue'] - avg_fuel_cost * s['km']
        service_net_profit[s['id']] = net

    # Model MIP
    prob = pulp.LpProblem('taxi_scheduler', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('assign', (service_ids, driver_ids), 0, 1, cat='Binary')

    # Cada servei a exactament 1 driver
    for sid in service_ids:
        prob += pulp.lpSum([x[sid][d] for d in driver_ids]) == 1

    # No solapament i descans mínim
    for i, s1 in enumerate(services):
        for j, s2 in enumerate(services):
            if i == j:
                continue
            # si s1 acaba abans s2 i descans < MIN_REST_HOURS
            if s1['end'] <= s2['start']:
                rest = (s2['start'] - s1['end']).total_seconds() / 3600.0
                if rest < MIN_REST_HOURS:
                    for d in driver_ids:
                        prob += x[s1['id']][d] + x[s2['id']][d] <= 1
            # si s2 acaba abans s1
            if s2['end'] <= s1['start']:
                rest = (s1['start'] - s2['end']).total_seconds() / 3600.0
                if rest < MIN_REST_HOURS:
                    for d in driver_ids:
                        prob += x[s1['id']][d] + x[s2['id']][d] <= 1

    # Evitar solapaments
    for i, s1 in enumerate(services):
        for j, s2 in enumerate(services):
            if i >= j:
                continue
            overlap = not (s1['end'] <= s2['start'] or s2['end'] <= s1['start'])
            if overlap:
                for d in driver_ids:
                    prob += x[s1['id']][d] + x[s2['id']][d] <= 1

    # Enforçar que serveis que necessiten minibus assignats a drivers minibus
    for s in services:
        pref = service_pref_type[s['id']]
        for d in st.session_state['drivers']:
            if pref == 'minibus' and d['type'] != 'minibus':
                prob += x[s['id']][d['id']] == 0

    # Variables auxiliars hores i km
    hours_driver = pulp.LpVariable.dicts('hours', driver_ids, 0, None, cat='Continuous')
    km_driver = pulp.LpVariable.dicts('kms', driver_ids, 0, None, cat='Continuous')
    for d in driver_ids:
        prob += hours_driver[d] == pulp.lpSum([((s['end'] - s['start']).total_seconds()/3600.0) * x[s['id']][d] for s in services])
        prob += km_driver[d] == pulp.lpSum([s['km'] * x[s['id']][d] for s in services])

    avg_hours = (sum([(s['end'] - s['start']).total_seconds()/3600.0 for s in services]) / len(driver_ids))
    avg_km = (sum([s['km'] for s in services]) / len(driver_ids))
    dev_hours = pulp.LpVariable.dicts('dev_h', driver_ids, 0, None, cat='Continuous')
    dev_km = pulp.LpVariable.dicts('dev_km', driver_ids, 0, None, cat='Continuous')
    for d in driver_ids:
        prob += hours_driver[d] - avg_hours <= dev_hours[d]
        prob += -(hours_driver[d] - avg_hours) <= dev_hours[d]
        prob += km_driver[d] - avg_km <= dev_km[d]
        prob += -(km_driver[d] - avg_km) <= dev_km[d]

    balance_penalty = pulp.lpSum([dev_hours[d] for d in driver_ids]) * W_BALANCE_HOURS + pulp.lpSum([dev_km[d] for d in driver_ids]) * W_BALANCE_KM
    revenue_term = pulp.lpSum([service_net_profit[sid] * x[sid][d] for sid in service_ids for d in driver_ids])
    prob += W_REVENUE * revenue_term - balance_penalty

    # Resoldre
    with st.spinner('Resoldre... (pot trigar uns segons)'):
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extreure assignacions
    assignments = []
    for sid in service_ids:
        for d in driver_ids:
            val = pulp.value(x[sid][d])
            if val is not None and val > 0.5:
                sv = next(filter(lambda z: z['id']==sid, services))
                assignments.append({'service': sid, 'driver': d, 'start': sv['start'], 'end': sv['end'], 'km': sv['km'], 'revenue': sv['revenue'], 'net': service_net_profit[sid]})

    if len(assignments) == 0:
        st.error('No s\'ha trobat cap assignació factible amb les restriccions actuals')
        st.stop()

    df = pd.DataFrame(assignments)
    st.subheader('Assignacions')
    st.dataframe(df)

    # Resum per driver
    summary = df.groupby('driver').agg({'service':'count','km':'sum','net':'sum'})
    summary = summary.rename(columns={'service':'num_services'})
    st.subheader('Resum per conductor')
    st.dataframe(summary)

    # CSV per descarregar
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    st.download_button('Descarregar horari (CSV)', data=csv_bytes, file_name='schedule_output.csv', mime='text/csv')

st.markdown('---')
st.caption('Aquest és un prototip. Si vols que el desplegui com a app multiusuari o el connecti a una base de dades, et puc preparar la següent fase.')
