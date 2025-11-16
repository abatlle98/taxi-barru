"""
streamlit_app.py
Versió completa per desplegar a Streamlit Cloud

- Càrrega CSV:
    - serveis.csv (service_id, start, end, km, passengers, luggage, location_start, location_end, revenue)
    - conductors.csv (driver_id, name, type, home_base)
    - vehicles.csv (vehicle_id, type, seats, luggage, driver_type)

- Algorisme d'assignació (PuLP)
    - Restriccions: solapaments, descans mínim, tipus vehicle
    - Objectiu mixt (recomanat): 40% equitat, 40% benefici net, 20% minimitzar kms buits (heurístic)

- Exportació a CSV i visualització resum

Instruccions d'ús local:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

"""

import streamlit as st
import pandas as pd
import pulp
from datetime import datetime, timedelta
from dateutil import parser
import io

st.set_page_config(page_title='Taxi Scheduler (Completa)', layout='wide')
st.title('Taxi Scheduler — Versió completa (per desplegar)')

st.markdown("""
Aquesta aplicació permet pujar fitxers CSV amb serveis, conductors i vehicles, i genera horaris optimitzats.
Formatos CSV recomanats (mostra a la README):
- serveis.csv: service_id,start,end,km,passengers,luggage,location_start,location_end,revenue
- conductors.csv: driver_id,name,type,home_base
- vehicles.csv: vehicle_id,type,seats,luggage,driver_type
""")

# -----------------------------
# Sidebar: càrrega arxius i paràmetres
# -----------------------------
st.sidebar.header('1) Carrega fitxers CSV (o usa mostres)')
services_file = st.sidebar.file_uploader('Serveis (CSV)', type=['csv'])
drivers_file = st.sidebar.file_uploader('Conductors (CSV)', type=['csv'])
vehicles_file = st.sidebar.file_uploader('Vehicles (CSV)', type=['csv'])

use_example = st.sidebar.checkbox('Usar dades d\'exemple si no hi ha arxius', value=True)

st.sidebar.markdown('---')
st.sidebar.header('2) Paràmetres d\'optimització (Mix recomanat)')
# Default mix: 40% equitat, 40% benefici, 20% minimitzar kms buits
WEIGHT_EQUITY = st.sidebar.slider('Pes equitat (0-1)', 0.0, 1.0, 0.4)
WEIGHT_BENEFIT = st.sidebar.slider('Pes benefici (0-1)', 0.0, 1.0, 0.4)
WEIGHT_EMPTY = st.sidebar.slider('Pes kms buits (0-1)', 0.0, 1.0, 0.2)
# Normalize to sum 1
w_sum = WEIGHT_EQUITY + WEIGHT_BENEFIT + WEIGHT_EMPTY
if w_sum == 0:
    WEIGHT_EQUITY, WEIGHT_BENEFIT, WEIGHT_EMPTY = 0.4,0.4,0.2
else:
    WEIGHT_EQUITY /= w_sum; WEIGHT_BENEFIT /= w_sum; WEIGHT_EMPTY /= w_sum

MIN_REST_HOURS = st.sidebar.number_input('Descans mínim (hores)', min_value=0.0, max_value=48.0, value=12.0)
MAX_HOURS_PER_DAY = st.sidebar.number_input('Hores màximes ideals / dia', min_value=1.0, max_value=24.0, value=12.0)

st.sidebar.markdown('---')

# -----------------------------
# Funcions auxiliars
# -----------------------------

def load_example_data():
    base = datetime(2025,11,17,8,0)
    services = [
        {'service_id':'S1','start':'2025-11-17 08:00','end':'2025-11-17 10:00','km':120,'passengers':5,'luggage':2,'location_start':'Girona','location_end':'Barcelona','revenue':150},
        {'service_id':'S2','start':'2025-11-17 11:00','end':'2025-11-17 13:00','km':80,'passengers':4,'luggage':1,'location_start':'Barcelona','location_end':'Tarragona','revenue':90},
        {'service_id':'S3','start':'2025-11-18 09:00','end':'2025-11-18 12:00','km':200,'passengers':6,'luggage':4,'location_start':'Girona','location_end':'Girona','revenue':240},
        {'service_id':'S4','start':'2025-11-18 14:00','end':'2025-11-18 16:00','km':60,'passengers':2,'luggage':0,'location_start':'Girona','location_end':'Barcelona','revenue':70},
        {'service_id':'S5','start':'2025-11-19 16:00','end':'2025-11-19 21:00','km':300,'passengers':10,'luggage':6,'location_start':'Barcelona','location_end':'Valencia','revenue':400},
        {'service_id':'S6','start':'2025-11-19 23:00','end':'2025-11-20 01:00','km':70,'passengers':3,'luggage':1,'location_start':'Valencia','location_end':'Tarragona','revenue':85},
        {'service_id':'S7','start':'2025-11-20 17:00','end':'2025-11-20 19:00','km':90,'passengers':4,'luggage':2,'location_start':'Tarragona','location_end':'Barcelona','revenue':120},
    ]
    drivers = [
        {'driver_id':'D1','name':'Jordi','type':'car','home_base':'Girona'},
        {'driver_id':'D2','name':'Maria','type':'car','home_base':'Girona'},
        {'driver_id':'D3','name':'Pau','type':'car','home_base':'Figueres'},
        {'driver_id':'D4','name':'Laia','type':'car','home_base':'Banyoles'},
        {'driver_id':'D5','name':'Omar','type':'car','home_base':'Lloret'},
        {'driver_id':'D6','name':'Sara','type':'car','home_base':'Girona'},
        {'driver_id':'M1','name':'Joel','type':'minibus','home_base':'Girona'},
        {'driver_id':'M2','name':'Anna','type':'minibus','home_base':'Girona'},
    ]
    vehicles = [
        {'vehicle_id':'a','type':'van','seats':6,'luggage':'normal','driver_type':'car'},
        {'vehicle_id':'b','type':'van','seats':6,'luggage':'normal','driver_type':'car'},
        {'vehicle_id':'c','type':'van','seats':6,'luggage':'large','driver_type':'car'},
        {'vehicle_id':'d','type':'van','seats':7,'luggage':'large','driver_type':'car'},
        {'vehicle_id':'e','type':'luxury','seats':3,'luggage':'normal','driver_type':'car'},
        {'vehicle_id':'mb1','type':'minibus','seats':18,'luggage':'large','driver_type':'minibus'},
        {'vehicle_id':'mb2','type':'minibus','seats':18,'luggage':'large','driver_type':'minibus'},
    ]
    return pd.DataFrame(services), pd.DataFrame(drivers), pd.DataFrame(vehicles)


def parse_services_df(df):
    # Expect columns: service_id,start,end,km,passengers,luggage,location_start,location_end,revenue
    df = df.copy()
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(df['start'])
    if 'end' in df.columns:
        df['end'] = pd.to_datetime(df['end'])
    return df

# -----------------------------
# Carregar dades
# -----------------------------
if services_file is not None:
    services_df = pd.read_csv(services_file)
    services_df = parse_services_df(services_df)
elif use_example:
    s_example, d_example, v_example = load_example_data()
    services_df, drivers_df, vehicles_df = s_example, pd.DataFrame(d_example), pd.DataFrame(v_example)
else:
    st.error('Puja els fitxers CSV de serveis (o activa "usar dades d\'exemple").')
    st.stop()

if drivers_file is not None:
    drivers_df = pd.read_csv(drivers_file)
else:
    if 'drivers_df' not in globals():
        st.warning('No s\'ha pujat conductors: s\'usarà l\'exemple si està activat.')
        if not use_example:
            st.stop()

if vehicles_file is not None:
    vehicles_df = pd.read_csv(vehicles_file)
else:
    if 'vehicles_df' not in globals():
        st.warning('No s\'ha pujat vehicles: s\'usarà l\'exemple si està activat.')
        if not use_example:
            st.stop()

# Mostrar previsualització
st.subheader('Previsualització: serveis')
st.dataframe(services_df.head(20))
st.subheader('Previsualització: conductors')
st.dataframe(drivers_df.head(50))
st.subheader('Previsualització: vehicles')
st.dataframe(vehicles_df.head(50))

# -----------------------------
# Funció que construeix i resol el model
# -----------------------------
def solve_scheduler(services_df, drivers_df, vehicles_df,
                    weight_equity=0.4, weight_benefit=0.4, weight_empty=0.2,
                    min_rest_hours=12.0):
    # Preparar llistes
    services = services_df.to_dict('records')
    drivers = drivers_df.to_dict('records')
    vehicles = vehicles_df.to_dict('records')
    service_ids = [s['service_id'] for s in services]
    driver_ids = [d['driver_id'] for d in drivers]

    # Tipus vehicle preferent per servei (simple)
    def pref_type(s):
        if s.get('passengers',0) > 6:
            return 'minibus'
        elif s.get('passengers',0) > 3:
            return 'van'
        else:
            return 'luxury'

    # Estimar cost combustible per km per tipus
    FUEL = {'van':0.14,'luxury':0.18,'minibus':0.20}

    service_net = {}
    service_pref = {}
    for s in services:
        pref = pref_type(s)
        service_pref[s['service_id']] = pref
        fuel_cost = FUEL.get(pref,0.14)
        net = float(s.get('revenue',0.0)) - fuel_cost * float(s.get('km',0.0))
        service_net[s['service_id']] = net

    # Model
    prob = pulp.LpProblem('taxi_scheduler', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('assign',(service_ids, driver_ids),0,1,cat='Binary')

    # Cada servei assignat a 1 driver
    for sid in service_ids:
        prob += pulp.lpSum([x[sid][d] for d in driver_ids]) == 1

    # Restringir per tipus driver/vehicle (si servei necessita minibus, només drivers type minibus)
    driver_type_map = {d['driver_id']: d.get('type','car') for d in drivers}

    for s in services:
        sid = s['service_id']
        pref = service_pref[sid]
        if pref == 'minibus':
            for d in driver_ids:
                if driver_type_map.get(d) != 'minibus':
                    prob += x[sid][d] == 0

    # No solapaments ni descans mínim per conductor
    for i, s1 in enumerate(services):
        for j, s2 in enumerate(services):
            if i == j: continue
            # si s1 acaba abans s2
            if s1['end'] <= s2['start']:
                rest = (pd.to_datetime(s2['start']) - pd.to_datetime(s1['end'])).total_seconds()/3600.0
                if rest < min_rest_hours:
                    for d in driver_ids:
                        prob += x[s1['service_id']][d] + x[s2['service_id']][d] <= 1
            # solapament (encavalc)
            overlap = not (s1['end'] <= s2['start'] or s2['end'] <= s1['start'])
            if overlap:
                for d in driver_ids:
                    prob += x[s1['service_id']][d] + x[s2['service_id']][d] <= 1

    # Variables auxiliars per driver
    hours_driver = pulp.LpVariable.dicts('hours', driver_ids, 0, None, cat='Continuous')
    km_driver = pulp.LpVariable.dicts('kms', driver_ids, 0, None, cat='Continuous')

    for d in driver_ids:
        prob += hours_driver[d] == pulp.lpSum([((pd.to_datetime(s['end']) - pd.to_datetime(s['start'])).total_seconds()/3600.0) * x[s['service_id']][d] for s in services])
        prob += km_driver[d] == pulp.lpSum([float(s.get('km',0.0)) * x[s['service_id']][d] for s in services])

    # Equitat: penalitzem desviacio L1 respecte la mitjana per hores i kms
    avg_hours = sum([((pd.to_datetime(s['end']) - pd.to_datetime(s['start'])).total_seconds()/3600.0) for s in services]) / max(1,len(driver_ids))
    avg_km = sum([float(s.get('km',0.0)) for s in services]) / max(1,len(driver_ids))

    dev_h = pulp.LpVariable.dicts('dev_h', driver_ids, 0, None, cat='Continuous')
    dev_k = pulp.LpVariable.dicts('dev_k', driver_ids, 0, None, cat='Continuous')
    for d in driver_ids:
        prob += hours_driver[d] - avg_hours <= dev_h[d]
        prob += -(hours_driver[d] - avg_hours) <= dev_h[d]
        prob += km_driver[d] - avg_km <= dev_k[d]
        prob += -(km_driver[d] - avg_km) <= dev_k[d]

    equity_penalty = pulp.lpSum([dev_h[d] + dev_k[d] for d in driver_ids])

    # Benefici total
    benefit = pulp.lpSum([service_net[sid] * x[sid][d] for sid in service_ids for d in driver_ids])

    # Kms buits heurístic: premiar assignacions que connectin localitzacions iguals (minimitzar trasllats buits)
    # Creem una mesura de 'congruència' entre serveis consecutius assignats al mateix conductor
    # Simplificació: per cada parella s1->s2 on s1.end == s2.start_loc, sum x[s1][d]*x[s2][d]
    # Implementació lineal: introduïm variables y_{s1,s2,d} <= x[s1][d], <= x[s2][d] i max - (x+x-1)
    congr_vars = []
    y = {}
    for i, s1 in enumerate(services):
        for j, s2 in enumerate(services):
            if s1['service_id'] == s2['service_id']: continue
            # considerem com a consecutiu si s1.end <= s2.start (no solapament) i la localitzacio final coincideix
            if pd.to_datetime(s1['end']) <= pd.to_datetime(s2['start']):
                if str(s1.get('location_end','')).strip().lower() == str(s2.get('location_start','')).strip().lower():
                    for d in driver_ids:
                        var_name = f"y_{s1['service_id']}_{s2['service_id']}_{d}"
                        y[(s1['service_id'],s2['service_id'],d)] = pulp.LpVariable(var_name,0,1,cat='Binary')
                        prob += y[(s1['service_id'],s2['service_id'],d)] <= x[s1['service_id']][d]
                        prob += y[(s1['service_id'],s2['service_id'],d)] <= x[s2['service_id']][d]
                        prob += y[(s1['service_id'],s2['service_id'],d)] >= x[s1['service_id']][d] + x[s2['service_id']][d] - 1

    # Sum of congruent pairs (more is better)
    congruent_sum = pulp.lpSum([y[k] for k in y]) if len(y)>0 else 0

    # Objective: combine (we convert equity_penalty to a maximizable term by negating)
    # Normalize terms roughly by estimated scales
    # benefit ~ hundreds, equity_penalty ~ tens, congruent_sum ~ small integer
    obj = WEIGHT_BENEFIT * benefit - WEIGHT_EQUITY * equity_penalty + WEIGHT_EMPTY * congruent_sum
    prob += obj

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract assignments
    assigns = []
    for sid in service_ids:
        for d in driver_ids:
            val = pulp.value(x[sid][d])
            if val is not None and val > 0.5:
                s = next(filter(lambda z: z['service_id']==sid, services))
                assigns.append({'service_id':sid,'driver_id':d,'start':s['start'],'end':s['end'],'km':s.get('km',0),'revenue':s.get('revenue',0),'net':service_net[sid]})

    return pd.DataFrame(assigns) if len(assigns)>0 else pd.DataFrame()

# -----------------------------
# Botó per generar
# -----------------------------
st.header('Genera horari')
if st.button('Generar horari (IA)'):
    with st.spinner('Resoldre model...'):
        df_assign = solve_scheduler(services_df, drivers_df, vehicles_df,
                                    weight_equity=WEIGHT_EQUITY,
                                    weight_benefit=WEIGHT_BENEFIT,
                                    weight_empty=WEIGHT_EMPTY,
                                    min_rest_hours=MIN_REST_HOURS)
    if df_assign.empty:
        st.error('No s\'ha trobat una solució factible amb les restriccions actuals.')
    else:
        st.subheader('Assignacions')
        st.dataframe(df_assign)
        st.subheader('Resum per driver')
        try:
            summary = df_assign.groupby('driver_id').agg({'service_id':'count','km':'sum','net':'sum'})
            summary = summary.rename(columns={'service_id':'num_services'})
            st.dataframe(summary)
        except Exception:
            pass
        # CSV
        buf = io.StringIO()
        df_assign.to_csv(buf,index=False)
        st.download_button('Descarregar horari (CSV)', data=buf.getvalue().encode('utf-8'), file_name='schedule_output.csv', mime='text/csv')

st.markdown('---')
st.caption('Aquest prototip és una base per producció. Es poden afegir més restriccions laborals, rotacions 6/2, i integració amb BD.')
