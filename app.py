import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize
from streamlit.dataframe_util import OptionSequence

st.set_page_config(layout="wide")

################################################# Déclaration des variables globales #################################################
yearly_load = 130400  # MWh/an

diesel_already_installed = 16  # MW

unserved_energy_variable_cost = 10000  # €/MWh
diesel_variable_cost = 150  # €/MWh

init_cost_solar = 800000  # €/MW
init_cost_wind = 1700000  # €/MW
init_cost_diesel = 600000  # €/MW
init_cost_storage_power = 250000  # €/MW
init_cost_storage_energy = 250000  # €/MWh

onm_solar = 15000  # €/MW.an
onm_wind = 45000  # €/MW.an
onm_diesel = 30000  # €/MW.an

npv = 12.46

diesel_emissions = 1  # tCO2/MWh

tax_carbon = 265  # €/tCO2

df_base = pd.read_csv("data_base.csv", sep=";")
# Clean empty columns
df_base = df_base.dropna(axis=1, how="all")
################################################# Définition des fonctions de calcul des bases de données #################################################


def calculate_storage_columns(df, charging_power, energy_storage, P_diesel):
    # Cette fonction permet le calcul des 3 colonnes interdépendantes : storage charging, storage releasing et storage stock
    # Elle sera utilisée dans la fonction 'simulation'
    n = len(df)
    storage_charging = [None] * n
    storage_releasing = [None] * n
    storage_stock = [None] * n
    initial_storage_stock = (
        energy_storage  # MWh, stockage en ligne 4 du excel, au temps t0-dt
    )

    for i in range(n):
        # lecture de la i-ème ligne pour les 3 colonnes (car elles sont interdépendantes)
        net_load = df["Net Load"].iloc[i]
        missing_capacity = df["Missing capacity"].iloc[i]
        storage_stock_prev = storage_stock[i - 1] if i > 0 else initial_storage_stock

        # Somme des 'Missing capacities' des 5 prochains jours
        sum_missing = df["Missing capacity"].iloc[i + 1 : min(i + 121, len(df))].sum()

        # Calcul du storage charging
        # Lecture de la formule excel :
        # - si la net load est négative = surplus d'énergie, on charge
        # - Si la somme des besoins des 5 prochains jours est plus grande que le stock actuel, on charge pour ne pas être limité par la puissance du diesel quand il n'y aura pas les énergies renouvelables
        # - Sinon on ne charge pas (si on chargeait, on va utiliser du diésel en plus alors que ce n'est pas nécessaire)
        # Quand on charge, on ne peut pas dépasser la puissance max de charge ni la capacité de stockage
        # On peut charger au diésel uniquement dans le cas où on a un besoin futur (attention à ne pas dépasser la limite de puissance du diesel)

        if net_load < 0:
            storage_charging[i] = min(
                charging_power, min(-net_load, energy_storage - storage_stock_prev)
            )
        elif sum_missing > storage_stock_prev:
            storage_charging[i] = max(
                min(
                    sum_missing,
                    P_diesel - net_load,
                    energy_storage - storage_stock_prev,
                ),
                0,
            )  # on charge autant que possible sans dépasser la puissance du diesel ni la capacité de stockage
        else:
            storage_charging[i] = 0

        storage_charging[i] = min(
            charging_power, storage_charging[i]
        )  # on ne peut pas charger plus que la puissance de charge

        # Calcul du storage releasing
        if missing_capacity > 0:
            storage_releasing[i] = min(
                charging_power, min(missing_capacity, storage_stock_prev)
            )
        elif sum_missing > 0:
            storage_releasing[i] = 0
        else:
            storage_releasing[i] = min(max(net_load, 0), storage_stock_prev)

        storage_releasing[i] = min(charging_power, storage_releasing[i])

        # Calcul du storage stock
        storage_stock[i] = (
            storage_stock_prev + storage_charging[i] - storage_releasing[i]
        )

    return storage_charging, storage_releasing, storage_stock


def simulation(P_solar, P_wind, P_diesel, charging_power, energy_storage):
    """
    Prend en entrée les paramètres du système :
     P_solar, P_wind, P_diesel, charging_power, energy_storage
     et retourne le dataframe avec les calculs effectués.
    """
    df = df_base.copy()
    # Calculate derived columns
    df["Solar"] = df["SOLAR_FC"] * P_solar

    df["Wind"] = df["WIND_FC"] * P_wind
    df["Net Load"] = df["Load"] - df["Solar"] - df["Wind"]
    df["Missing capacity"] = (df["Net Load"] - P_diesel).clip(lower=0)

    # Application de la fonction définie plus haut pour les colonnes interdépendantes
    storage_charging_col, storage_releasing_col, storage_stock_col = (
        calculate_storage_columns(df, charging_power, energy_storage, P_diesel)
    )

    df["storage charging"] = storage_charging_col
    df["storage releasing"] = storage_releasing_col
    df["storage stock"] = storage_stock_col

    df["Diesel"] = (
        df["Net Load"] + df["storage charging"] - df["storage releasing"]
    ).clip(lower=0, upper=P_diesel)
    df["Unserved energy"] = (
        df["Net Load"] + df["storage charging"] - df["storage releasing"] - df["Diesel"]
    ).clip(lower=0)
    df["Unused energy"] = (0 - df["Net Load"] - df["storage charging"]).clip(lower=0)

    # Je réordonne les colonnes dans l'ordre du tableau excel
    # Reorder columns in the specified order
    column_order = [
        "Time (UTC)",
        "Load",
        "Solar",
        "Wind",
        "Net Load",
        "Missing capacity",
        "storage charging",
        "storage releasing",
        "storage stock",
        "Diesel",
        "Unserved energy",
        "Unused energy",
        "SOLAR_FC",
        "WIND_FC",
    ]
    df = df[column_order]
    return df


####################################################### Fonction pour le calcul des KPI résultats #######################################################
def results(df, P_solar, P_wind, P_diesel, charging_power, energy_storage):
    # Prend en argument un df de cas et les variables d'entrée
    # Renvoie une liste :
    # Liste contenant les 7 éléments du tableau résultat sur excel
    # cf l'affichage du tableau de résultats pour plus de précisions
    total_initial_cost = (
        init_cost_solar * P_solar
        + init_cost_wind * P_wind
        + init_cost_diesel * max(0, (P_diesel - diesel_already_installed))
        + init_cost_storage_power * charging_power
        + init_cost_storage_energy * energy_storage
    )
    total_cost_OnM = onm_solar * P_solar + onm_wind * P_wind + onm_diesel * P_diesel
    total_cost_fuel = diesel_variable_cost * df["Diesel"].sum()
    total_cost_unserved = unserved_energy_variable_cost * df["Unserved energy"].sum()
    total_present_costs = total_initial_cost + npv * (
        total_cost_OnM + total_cost_fuel + total_cost_unserved
    )
    total_CO2_emissions = diesel_emissions * df["Diesel"].sum()
    system_LCOE = total_present_costs / (yearly_load * npv)
    total_present_costs_with_tax = total_present_costs + npv * (tax_carbon * total_CO2_emissions)

    return [
        total_initial_cost,
        total_cost_OnM,
        total_cost_fuel,
        total_cost_unserved,
        total_present_costs,
        total_CO2_emissions,
        system_LCOE,
        total_present_costs_with_tax,
    ]


####################################################### Affichage #################################################
"# ECO 52064_project_isolated-system"

#################### Affichage des variables globales
"## Constantes choisies"
# <style>
# p { margin: 0px; line-height: 1.2; }
# </style>
st.markdown(
    """

**Yearly load** :
- """
    + str(yearly_load)
    + """ MWh/an

**Already installed capacity** :
- Diesel: """
    + str(diesel_already_installed)
    + """ MW

**Variable costs** :
- Unserved energy : """
    + str(unserved_energy_variable_cost)
    + """ €/MWh
- Diesel : """
    + str(diesel_variable_cost)
    + """ €/MWh


**Initial cost of new capacities** :
- Solar : """
    + str(init_cost_solar)
    + """ €/MW
- Wind : """
    + str(init_cost_wind)
    + """ €/MW
- Diesel : """
    + str(init_cost_diesel)
    + """ €/MW
- Storage power : """
    + str(init_cost_storage_power)
    + """ €/MW
- Storage energy : """
    + str(init_cost_storage_energy)
    + """ €/MWh

**O&M** :
- Solar : """
    + str(onm_solar)
    + """ €/MW.an
- Wind : """
    + str(onm_wind)
    + """ €/MW.an
- Diesel : """
    + str(onm_diesel)
    + """ €/MW.an

**NPV over 20 years at 5% discount rate** : """
    + str(npv)
    + """

**Diesel emissions** : """
    + str(diesel_emissions)
    + """ tCO2/MWh
""",
    unsafe_allow_html=True,
)


#################### Pour les 3 cas : affichage des variables d'ajustement et calcul des dataframes

"# Variables d'ajustement des 3 cas"
# refcase
"## Reference case"
P_solar = 0
P_wind = 0
P_diesel = 24
charging_power = 0
energy_storage = 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("P_solar (MW)", value=P_solar)
with col2:
    st.metric("P_wind (MW)", value=P_wind)
with col3:
    st.metric("P_diesel (MW)", value=P_diesel)
with col4:
    st.metric("charging_power (MW)", value=charging_power)
with col5:
    st.metric("energy_storage (MWh)", value=energy_storage)

df_refcase = simulation(P_solar, P_wind, P_diesel, charging_power, energy_storage)
results_refcase = results(
    df_refcase, P_solar, P_wind, P_diesel, charging_power, energy_storage
)

# Case 1
"## Case 1"
P_solar = 27.0898352
P_wind = 29.6019129
P_diesel = 19.14803567
charging_power = 4.9394435
energy_storage = 28.72101096

# attendu : 212024336.85141787

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    P_solar = st.number_input("P_solar (MW)", value=P_solar, format="%.7f", key="case1_solar")
with col2:
    P_wind = st.number_input("P_wind (MW)", value=P_wind, format="%.7f", key="case1_wind")
with col3:
    P_diesel = st.number_input("P_diesel (MW)", value=P_diesel, format="%.7f", key="case1_diesel")
with col4:
    charging_power = st.number_input(
        "charging_power (MW)", value=charging_power, format="%.7f", key="case1_charging"
    )
with col5:
    energy_storage = st.number_input(
        "energy_storage (MWh)", value=energy_storage, format="%.7f", key="case1_storage"
    )

df_cas_1 = simulation(P_solar, P_wind, P_diesel, charging_power, energy_storage)

results_case1 = results(
    df_cas_1, P_solar, P_wind, P_diesel, charging_power, energy_storage
)


################################
# Recherche des paramètres optimaux pour le cas 1 : minimisation du coût total (Total present costs) : recherche avec scipy.optimize.minimize, qui est uniquement un optimiseur local, donc on a eu des solutions de coût supérieur au cas référence
def fonction_optimisation(x): # param_a_optimiser = 7 pour minimiser en prenant en compte la taxe CO2
    # prend en entrée un vecteur x contenant tous les paramètres à optimiser, du type :
    # x = [P_solar, P_wind, P_diesel, charging_power, energy_storage]
    # renvoie la valeur du coût total "Total present costs" (float)
    P_solar, P_wind, P_diesel, charging_power, energy_storage = x
    df = simulation(P_solar, P_wind, P_diesel, charging_power, energy_storage)
    return results(df, P_solar, P_wind, P_diesel, charging_power, energy_storage)[
        7
    ]  # results()[4] corresponds to Total present costs

# Solution d'optimisation globale
def optimiser_couts_global():
    # Définition des bornes (min, max) pour chaque variable
    bounds = [(0, 50), (0, 50), (16, 24), (0, 50), (0, 50)]

    result = differential_evolution(
        fonction_optimisation,
        bounds=bounds,
        strategy="best1bin",
        popsize=15,
        tol=0.01,
    )
    return result.x, result.fun


# optimal_params, optimal_cost = optimiser_couts_global()
optimal_params, optimal_cost = ([27.0898352, 29.6019129, 19.14803567, 4.9394435, 28.72101096], 212024336.85141787)
print(optimal_params)
print(optimal_cost)


# st.subheader("Optimal Parameters for Case 1")
# col1, col2, col3, col4, col5 = st.columns(5)
# with col1:
#     st.metric("P_solar (MW)", value=optimal_params[0])
# with col2:
#     st.metric("P_wind (MW)", value=optimal_params[1])
# with col3:
#     st.metric("P_diesel (MW)", value=optimal_params[2])
# with col4:
#     st.metric("charging_power (MW)", value=optimal_params[3])
# with col5:
#     st.metric("energy_storage (MWh)", value=optimal_params[4])
# st.metric("Minimized Total Present Costs (€)", value=f"{optimal_cost:.2f}")


# Case 2
"## Case 2 : ajout d'une taxe carbone"
P_solar = 44.67212544
P_wind = 49.19442023
P_diesel = 18.35416079
charging_power = 9.54329771
energy_storage = 49.2845442

# 335078755.7470432


col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    P_solar = st.number_input("P_solar (MW)", value=P_solar, format="%.7f", key="case2_solar")
with col2:
    P_wind = st.number_input("P_wind (MW)", value=P_wind, format="%.7f", key="case2_wind")
with col3:
    P_diesel = st.number_input("P_diesel (MW)", value=P_diesel, format="%.7f", key="case2_diesel")
with col4:
    charging_power = st.number_input(
        "charging_power (MW)", value=charging_power, format="%.7f", key="case2_charging"
    )
with col5:
    energy_storage = st.number_input(
        "energy_storage (MWh)", value=energy_storage, format="%.7f", key="case2_storage"
    )

df_cas_2 = simulation(P_solar, P_wind, P_diesel, charging_power, energy_storage)
results_case2 = results(
    df_cas_2, P_solar, P_wind, P_diesel, charging_power, energy_storage
)

# optimal_params, optimal_cost = optimiser_couts_global()
optimal_params, optimal_cost = ([27.0898352, 29.6019129, 19.14803567, 4.9394435, 28.72101096], 212024336.85141787)
print(optimal_params)
print(optimal_cost)

#################### Affichage des résulatas via KPI
"# Résultats"

KPI_matrix = pd.DataFrame(
    {
        "Reference case": results_refcase,
        "Case 1": results_case1,
        "Case 2": results_case2,
    },
    index=[
        "Total C_I (€)",
        "Total C_O&M (€/an)",
        "Total C_Fuel (€/an)",
        "Total C_LOLE (€/an)",
        "Total present costs (€/an)",
        "CO2_Emissions (tCO2/an)",
        "System LCOE (€_2026/MWh)",
        "Total present costs with tax (€/an)",
    ],
)

st.table(KPI_matrix)

"# Comparison with reference case "
comparison_cas_1 = [
    results_case1[4] - results_refcase[4],
    (results_refcase[5] - results_case1[5]) * npv,
]
comparison_cas_1.append(comparison_cas_1[0] / comparison_cas_1[1])
comparison_cas_1.append(results_case1[7] - results_refcase[7])
comparison_cas_2 = [
    results_case2[4] - results_refcase[4],
    (results_refcase[5] - results_case2[5]) * npv,
]
comparison_cas_2.append(comparison_cas_2[0] / comparison_cas_2[1])
comparison_cas_2.append(results_case2[7] - results_refcase[7])
comparison_matrix = pd.DataFrame(
    {
        "Case 1": comparison_cas_1,
        "Case 2": comparison_cas_2,
    },
    index=[
        "overcost if positive / gain if negative, €_2026",
        "CO2 emissions actualised in tCO2_2026",
        "€/tCO2",
        "(With carbon tax) overcost if positive / gain if negative, €_2026",
    ],
)
st.table(comparison_matrix)

#################### Affichage des df
"# Affichage des bases de données"
"## Reference case data"
st.write(df_refcase)
"## Case 1 data"
st.write(df_cas_1)
"## Case 2 data"
st.write(df_cas_2)
