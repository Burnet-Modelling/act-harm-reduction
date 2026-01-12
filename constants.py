"""
This script initialises constants and names.
"""
import numpy as np
import datetime

ESTIMATES = ('best', 'lower', 'upper')
SCENARIO_YEARS = [2026, 2027, 2028, 2029, 2030]
SCALEUP_STEP = [1, 1, 1, 1, 1]
SCALEUP_LINEAR = np.linspace(start=0, stop=1, num=len(SCENARIO_YEARS) + 1)[1:]

ANNUAL_DISCOUNTING = 1 / 1.05
COST_REFERENCE_YEAR = 2026
SETUP_COST_YEAR = SCENARIO_YEARS[0]

TODAY_STR = datetime.datetime.now().strftime('%Y%m%d')
TIME_STR = datetime.datetime.now().strftime('%H%M%S')
DEFAULT_POPSIZE = 1

PLOT_NODES = [
    'overdose_inj_opioid', 'emergency_response_inj_opioid', 'death_inj_opioid',
    'overdose_inj_nonopioid', 'emergency_response_inj_nonopioid', 'death_inj_nonopioid',

    'overdose_noninj_opioid', 'emergency_response_noninj_opioid', 'death_noninj_opioid',
    'overdose_noninj_nonopioid', 'emergency_response_noninj_nonopioid', 'death_noninj_nonopioid',

    'irid', 'irid_hospital',
    'bbv_inci', 'bbv_treat',
]

COST_CAT_LABELS = {'Intervention': 'Intervention',
                   'Social': 'Societal cost of years of life lost',
                   'Emergency': 'Emergency response',
                   'IRID': 'Hospitalised treatment of IRI',
                   'Hepatitis C': 'Hepatitis C treatment'}

OUTCOME_CAT_LABELS = {'Opioid overdose': 'Opioid overdose',
                      'Emergency response': 'Emergency response',
                      'Death': 'Overdose-related death',
                      'Non-opioid overdose': 'Non-opioid overdose',
                      'IRID incidence': 'IRI incidence',
                      'Hepatitis C incidence': 'Hepatitis C incidence'}
