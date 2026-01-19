"""
This script provides most of the functionality for the model.
"""
import graphlib
import itertools

from pgmpy.inference import VariableElimination, ApproxInference
from pgmpy.models import DiscreteBayesianNetwork

import logging
logging.disable(logging.CRITICAL)

import concurrent.futures as cf
import warnings
from pathlib import Path
from copy import deepcopy as dcp
from joblib import hash as jl_hash
import cloudpickle

from constants import *
from utils import *

pool = None  # We keep the pool around because it is time consuming to spin up new processes and import everything

class Project(object):
    """Project

    Main class to store everything.
    """

    def __init__(self, name='default'):
        """Initialize project object."""
        self.nodes = None
        self.arcs = None
        self.name = name
        self.dags = dict()
        self.costs = dict()
        self.results = None
        self.baseline_results = dict()
        self.info = {'created': datetime.datetime.now()}
        self.scenarios = None
        self.scenario_names = []
        self.node_info = None
        self.intervention_nodes = None
        # self.maxcovs = None

    def save(self, folder, filename=None):
        """Save project to file."""
        if filename is None:
            filename = f"{self.name}_{datetime.datetime.now().strftime('%Y%m%d')}"
        if not filename.endswith('.prj'):
            filename = filename + '.prj'
        Path(folder).mkdir(parents=True, exist_ok=True)
        outfilepath = sc.save(filename=filename, folder=folder, obj=self)
        print(f"Saved Project file to: {outfilepath}")

    def load_model(self, filename, folder=""):
        """Load model from spreadsheet."""
        model_path = os.path.join(folder, filename)
        self.info['model_path'] = model_path
        self.info['model_loaded'] = datetime.datetime.now()

        df_all_nodes = read_excel_safely(model_path, sheet_name='nodes')
        df_all_arcs = read_excel_safely(model_path, sheet_name='arcs')
        # df_interventions = read_excel_safely(model_path, sheet_name='interventions')

        df_all_nodes['levels'] = [tuple([word.strip() for word in string.split(',')]) for string in
                                  df_all_nodes['levels'].tolist()]
        df_all_nodes['parents'] = [list() for _ in range(len(df_all_nodes))]
        df_all_nodes['children'] = [list() for _ in range(len(df_all_nodes))]
        df_all_nodes['nodekey'] = df_all_nodes['short']

        df_node_dup = df_all_nodes.loc[df_all_nodes.duplicated(['population', 'DAG', 'short']), ['population', 'DAG', 'short']].drop_duplicates().reset_index(drop=True)
        for i in range(len(df_node_dup)):
            print("Duplicate node removed: [{}: {}]".format(df_node_dup['DAG'][i], df_node_dup['short'][i]))
        df_all_nodes = df_all_nodes.drop_duplicates(['population', 'DAG', 'short'])

        df_arc_dup = df_all_arcs.loc[df_all_arcs.duplicated(['population', 'DAG', 'parent', 'child']), ['population', 'DAG', 'parent', 'child']].drop_duplicates().reset_index(drop=True)
        for i in range(len(df_arc_dup)):
            print("Duplicate arc removed: [{}: {} -> {}]".format(df_arc_dup['DAG'][i], df_arc_dup['parent'][i], df_arc_dup['child'][i]))
        df_all_arcs = df_all_arcs.drop_duplicates(['population', 'DAG', 'parent', 'child'])
        ###

        pop_names_node_sheet = sorted(df_all_nodes['population'].unique())
        pop_names_arc_sheet = sorted(df_all_arcs['population'].unique())
        dag_names_node_sheet = sorted(df_all_nodes['DAG'].unique())
        dag_names_arc_sheet = sorted(df_all_arcs['DAG'].unique())
        if pop_names_node_sheet != pop_names_arc_sheet:
            print("\nWARNING: Check model_info.xlsx for population name consistency between tabs:")
            print(f"Populations in 'nodes' sheet = {pop_names_node_sheet}")
            print(f"Populations in 'arcs' sheet = {pop_names_arc_sheet}\n")
        if dag_names_node_sheet != dag_names_arc_sheet:
            print("\nWARNING: Check model_info.xlsx for DAG name consistency between tabs")
            print(f"DAG names in 'nodes' sheet = {dag_names_node_sheet}")
            print(f"DAG names in 'arcs' sheet = {dag_names_arc_sheet}\n")

        all_nodes = df_all_nodes.to_dict('records')
        all_arcs = df_all_arcs.to_dict('records')

        all_nodes_info = dict(zip(zip(df_all_nodes['DAG'].values, df_all_nodes['short'].values), df_all_nodes.to_dict('records')))

        dag_ids = df_all_nodes[['population', 'DAG']].drop_duplicates().values.tolist()

        # Create each DAG
        nodes = []
        for pop, dag_name in dag_ids:

            # Filter only relevant parts of the tables for this dag
            df_node = df_all_nodes[(df_all_nodes['population'] == pop) & (df_all_nodes['DAG'] == dag_name)]
            df_arc = df_all_arcs[(df_all_arcs['population'] == pop) & (df_all_arcs['DAG'] == dag_name)]

            nodes_dict = dict(zip(df_node['short'].values, df_node.to_dict('records')))
            arcs = df_arc.to_dict('records')

            graph = {k: set() for k in nodes_dict.keys()}
            for arc in arcs:
                parent = arc['parent']
                child = arc['child']

                if child in graph:
                    graph[child].add(parent)
                else:
                    raise Exception(f"Reference to undefined node [{child}] in DAG [{pop}: {dag_name}]. This node must be defined in the 'nodes' tab of the databook")

                nodes_dict[parent]['children'].append(child)
                nodes_dict[child]['parents'].append(parent)

            nodes_dag = []
            for nodekey, node_dict in nodes_dict.items():
                parents = node_dict['parents']
                parents_levels = {parent: nodes_dict[parent]['levels'] for parent in parents}
                node = Node(nodekey=nodekey, long=node_dict['long'], levels=node_dict['levels'], parents=parents, children=node_dict['children'], parents_levels=parents_levels, population=pop, dag_name=dag_name, cost_cat=node_dict['cost_cat'], outcome_cat=node_dict['outcome_cat'])
                nodes_dag.append(node)
            nodes.extend(nodes_dag)

            self.dags[dag_name] = DAG(graph=graph, nodes=nodes_dag, arcs=arcs, interventions=None, population=pop, dag_name=dag_name)
            self.dags[dag_name].node_keys = list(self.dags[dag_name].static_order())
            self.dags[dag_name].intervention_nodes = tuple([nodekey for nodekey in self.dags[dag_name].node_keys if 'int' in nodekey])

        self.nodes = nodes
        self.node_info = all_nodes_info
        self.arcs = all_arcs
        self.intervention_nodes = np.unique([node['short'] for node in all_nodes if 'int' in node['short']])

        print(f"Loaded model from {model_path}")
        print("Initialised {} DAGs: {}".format(len(dag_ids), [dag_id[1] for dag_id in dag_ids]))

    def generate_databook_sheets(self):
        """Generate databook sheets from model."""
        l_probs = []
        l_evidence = []
        l_maxcov = []
        l_popsize = []
        l_costs = []
        for dag_name, dag in self.dags.items():
            pop = dag.population
            probset = dcp(dag.baseline_probset)
            for node in dag.nodes:
                nodekey = node.nodekey

                # Probabilities
                cpt = probset[nodekey]
                cpt_df = cpt.df.reset_index()
                if cpt.default_outcome:
                    cpt_df = cpt_df.loc[cpt_df[nodekey] != cpt.default_outcome]

                # Optional evidence for intermediate nodes
                if len(cpt.parents):
                    row_evs = [[pop, dag_name, nodekey, level, None] for level in node.levels if level != cpt.default_outcome]
                    l_evidence.extend(row_evs)

                # Max coverage table
                if 'int' in nodekey:
                    row_maxcov = [[pop, dag_name, nodekey, level, None] for level in node.levels if level != 'none']
                    l_maxcov.extend(row_maxcov)

                for cond_prob_dict in cpt_df.reset_index(drop=True).to_dict('records'):
                    outcome_val = cond_prob_dict.pop(nodekey)
                    [best, lower, upper] = [cond_prob_dict.pop(k) for k in ESTIMATES]
                    row = [pop, dag_name, nodekey, outcome_val, best, lower, upper]
                    row.extend([value for parent_and_condition in reversed(cond_prob_dict.items()) for value in parent_and_condition]) # remaining entries are the conditions
                    l_probs.append(row)

                # costs
                row_costs = [[pop, dag_name, nodekey, level, float()] for level in node.levels]
                l_costs.extend(row_costs)

            row_popsize = [pop] + [''] * len(SCENARIO_YEARS)
            l_popsize.append(row_popsize)

        # create pd.DataFrame for each table
        df_probs = pd.DataFrame(l_probs).drop_duplicates().reset_index(drop=True)
        l_probs_cols = ['population', 'DAG', 'node', 'outcome', 'best', 'lower', 'upper', 'condition1', 'value1', 'condition2', 'value2', 'condition3', 'value3', 'condition4', 'value4', 'condition5', 'value5', 'condition6', 'value6']
        df_probs.columns = l_probs_cols[:len(df_probs.columns)]

        df_evs = pd.DataFrame(l_evidence).drop_duplicates().reset_index(drop=True)
        df_evs.columns = ['population', 'DAG', 'node', 'outcome', 'baseline']

        df_maxcov = pd.DataFrame(l_maxcov).drop_duplicates().reset_index(drop=True)
        df_maxcov.columns = ['population', 'DAG', 'node', 'outcome', 'max_coverage']

        df_popsize = pd.DataFrame(l_popsize).drop_duplicates().reset_index(drop=True)
        df_popsize.columns = ['population_size'] + [str(year) for year in SCENARIO_YEARS]

        df_costs = pd.DataFrame(l_costs).drop_duplicates().reset_index(drop=True)
        df_costs.columns = ['population', 'DAG', 'node', 'outcome', 'unit_cost']

        databook_sheets = {
            'probabilities': df_probs,
            'evidence': df_evs,
            'max_coverage': df_maxcov,
            'population_size': df_popsize,
            'costs': df_costs,
                           }

        return databook_sheets

    def create_databook(self, filename, folder="", existing_filename="", existing_folder=""):
        """Export probabilities to spreadsheet."""
        output_file = os.path.join(folder, filename)
        existing_file = os.path.join(existing_folder, existing_filename)

        if existing_file == "":
            databook_sheets = self.generate_databook_sheets()
        else:
            databook_sheets = self.update_databook(existing_file=existing_file)

        df_probs = databook_sheets['probabilities']
        df_evs = databook_sheets['evidence']
        df_maxcov = databook_sheets['max_coverage']
        df_popsize = databook_sheets['population_size']
        df_costs = databook_sheets['costs']

        # Initialise Excel workbook
        wb = openpyxl.Workbook()

        # Probabilities
        ws_probs = wb.create_sheet('probabilities')
        df_to_excel(df_probs, ws_probs, grouping=None)
        probs_styles = {'node': {'border': {'left': DASHED_BORDER},
                                 'font': styles.Font(bold=True)},
                        'best': {'border': {'left': DASHED_BORDER}},
                        'condition1': {'border': {'left': DASHED_BORDER},
                                       'font': styles.Font(bold=True)},
                        'condition2': {'border': {'left': DASHED_BORDER},
                                       'font': styles.Font(bold=True)},
                        'condition3': {'border': {'left': DASHED_BORDER},
                                       'font': styles.Font(bold=True)},
                        'condition4': {'border': {'left': DASHED_BORDER},
                                       'font': styles.Font(bold=True)},
                        'condition5': {'border': {'left': DASHED_BORDER},
                                       'font': styles.Font(bold=True)},
                        'value4': {'border': {'right': THIN_BORDER}},
                        'right': {'border': {'right': THIN_BORDER}}
                        }
        excel_styles(ws_probs, probs_styles, bottom_border={'bottom': THIN_BORDER})

        # Evidence
        ws_evs = wb.create_sheet('evidence')
        df_to_excel(df_evs, ws_evs)
        evs_styles = {'node': {'border': {'left': THIN_BORDER}},
                      'outcome': {'border': {'right': THIN_BORDER}},
                      'baseline': {'border': {'right': THIN_BORDER},
                                   'number_format': '0.0%'},
                      'right': {'border': {'right': THIN_BORDER}}
                      }
        excel_styles(ws_evs, evs_styles, bottom_border={'bottom': THIN_BORDER})

        # Max coverage
        ws_maxcov = wb.create_sheet('max_coverage')
        df_to_excel(df_maxcov, ws_maxcov)
        maxcov_styles = {'max_coverage': {'border': {'right': THIN_BORDER}},
                         'right': {'border': {'right': THIN_BORDER}}}
        excel_styles(ws_maxcov, maxcov_styles, bottom_border={'bottom': THIN_BORDER})

        # Population size
        ws_popsize = wb.create_sheet('population_size')
        df_to_excel(df_popsize, ws_popsize)
        popsize_styles = {'population_size': {'border': {'right': THIN_BORDER}},
                          'right': {'border': {'right': THIN_BORDER}}}
        excel_styles(ws_popsize, popsize_styles, bottom_border={'bottom': THIN_BORDER})

        # Costs
        ws_costs = wb.create_sheet('costs')
        df_to_excel(df_costs, ws_costs)
        costs_styles = {'node': {'border': {'left': THIN_BORDER}},
                        'outcome': {'border': {'right': THIN_BORDER}},
                        'unit_cost': {'border': {'right': THIN_BORDER},
                                      'number_format': '0.00'}
                      }
        excel_styles(ws_costs, costs_styles, bottom_border={'bottom': THIN_BORDER})

        # Write file
        del wb['Sheet']
        Path(folder).mkdir(parents=True, exist_ok=True)
        wb.save(output_file)

        print("Exported databook to: {}".format(output_file))
        
    def update_databook(self, existing_file):
        """Export probabilities to spreadsheet."""

        xl = pd.ExcelFile(existing_file)
        sheet_names = xl.sheet_names

        databook_sheets_new = self.generate_databook_sheets()
        df_probs = databook_sheets_new['probabilities']
        df_evs = databook_sheets_new['evidence']
        df_maxcov = databook_sheets_new['max_coverage']
        df_popsize = databook_sheets_new['population_size']
        df_costs = databook_sheets_new['costs']

        # Update with old databook entries where they exist
        if 'probabilities' in sheet_names:
            df_probs = update_dataframe(df_existing=xl.parse('probabilities'), df_new=df_probs,
                                        id_cols=['population', 'DAG', 'node', 'outcome', 'condition1', 'value1', 'condition2', 'value2', 'condition3', 'value3', 'condition4', 'value4', 'condition5', 'value5'])
        if 'evidence' in sheet_names:
            df_evs = update_dataframe(df_existing=xl.parse('evidence'), df_new=df_evs,
                                      id_cols=['population', 'DAG', 'node', 'outcome'])
        if 'max_coverage' in sheet_names:
            df_maxcov = update_dataframe(df_existing=xl.parse('max_coverage'), df_new=df_maxcov,
                                         id_cols=['population', 'DAG', 'node', 'outcome'])
        if 'population_size' in sheet_names:
            df_popsize = update_dataframe(df_existing=xl.parse('population_size'), df_new=df_popsize,
                                          id_cols=['population_size'])
        if 'costs' in sheet_names:
            df_costs = update_dataframe(df_existing=xl.parse('costs'), df_new=df_costs,
                                        id_cols=['population', 'DAG', 'node', 'outcome'])

        databook_sheets_updated = {
            'probabilities': df_probs,
            'evidence': df_evs,
            'max_coverage': df_maxcov,
            'population_size': df_popsize,
            'costs': df_costs,
                                   }

        return databook_sheets_updated

    def import_databook(self, filename, folder=""):
        """Import probabilities from spreadsheet."""
        databook_path = os.path.join(folder, filename)
        self.info['databook_path'] = databook_path
        self.info['databook_imported'] = datetime.datetime.now()

        df_probs = read_excel_safely(databook_path, sheet_name='probabilities')
        df_evs = read_excel_safely(databook_path, sheet_name='evidence').dropna(subset='baseline').reset_index(drop=True)
        df_maxcov = read_excel_safely(databook_path, sheet_name='max_coverage').dropna(subset='max_coverage').reset_index(drop=True)
        df_popsize = read_excel_safely(databook_path, sheet_name='population_size', index_col=0)
        df_costs = read_excel_safely(databook_path, sheet_name='costs')

        df_popsize.columns = [str(x) for x in df_popsize.columns]

        dict_probs_all = df_probs.to_dict('records')
        dict_evs_all = df_evs.to_dict('records')
        dict_maxcov_all = df_maxcov.to_dict('records')
        dict_popsize_all = df_popsize.to_dict('index')
        dict_costs_all = df_costs.to_dict('records')

        n_conditions = sum(['condition' in col for col in df_probs.columns])

        for dag_name in self.dags.keys():
            dag = self.dags[dag_name]
            pop = dag.population
            popsize = dict_popsize_all[pop]
            if np.all(pd.isna(list(popsize.values()))):
                print(f"WARNING: Missing all popsizes for DAG [{dag_name}]. Using default popsize of 1000.")
                dag.popsize = {year: DEFAULT_POPSIZE for year in popsize.keys()}
            elif np.any(pd.isna(list(popsize.values()))):
                print(f"WARNING: Missing some popsizes for DAG [{dag_name}]. Interpolating.")
                popsize_num = {int(year): val for year, val in popsize.items() if not pd.isna(val)}
                nans = [pd.isna(x) for x in popsize.values()]
                x_nans = [int(year) for year, val in popsize.items() if pd.isna(val)]
                y_nans = np.interp(x=list(x_nans), xp=list(popsize_num.keys()), fp=list(popsize_num.values()))
                nans_popsize = dict(zip(x_nans, y_nans))
                for year, val in nans_popsize.items():
                    popsize[str(year)] = val
                dag.popsize = popsize
            else:
                dag.popsize = popsize
            dag.popsize = {k: round(v) for k, v in dag.popsize.items()}

            dict_probs = [row for row in dict_probs_all if row['population'] == pop and row['DAG'] == dag_name and row['node'] in dag.node_keys]
            dict_evs = [row for row in dict_evs_all if row['population'] == pop and row['DAG'] == dag_name and row['node'] in dag.node_keys]
            dict_maxcovs = [row for row in dict_maxcov_all if row['population'] == pop and row['DAG'] == dag_name and row['node'] in dag.node_keys]
            dict_costs = [row for row in dict_costs_all if row['population'] == pop and row['DAG'] == dag_name and row['node'] in dag.node_keys]

            baseline_probset = self.dags[dag_name].baseline_probset

            for row in dict_probs:
                nodekey = row['node']
                outcome = row['outcome']

                # Create dictionary with all conditions including outcome
                conditions = {row['condition{}'.format(i)]: row['value{}'.format(i)] for i in range(1, n_conditions+1)}
                conditions = {k: v for k, v in conditions.items() if not pd.isna(k)}
                all_conds = dict(conditions)
                all_conds[nodekey] = outcome

                # Set conditional probability if node in DAG
                if set(baseline_probset[nodekey].parents) == set(conditions.keys()):
                    baseline_probset[nodekey].set(all_conds,
                                                  estimates=ESTIMATES,
                                                  values=[row[v] for v in ESTIMATES])

                ## uncomment this to add intervention nodes as evidence. Note this does not guarantee they will be kept
                ## the same after running the Bayesian inference
                # if nodekey in self.intervention_nodes:
                #     if nodekey not in dag.evidence:
                #         dag.evidence[nodekey] = dict()
                #     dag.evidence[nodekey][outcome] = row['best']

            for nodekey, cpt in baseline_probset.items():
                cpt.make_unity_probs()
                cpt.set_default_bounds()

            for row in dict_evs:
                if row['node'] not in dag.evidence:
                    dag.evidence[row['node']] = dict()
                dag.evidence[row['node']][row['outcome']] = row['baseline']
            dag.normalise_evidence()

            dag.maxcovs = {
                nodekey: {row['outcome']: row['max_coverage'] for row in dict_maxcovs if row['node'] == nodekey} for
                nodekey in df_maxcov['node'].unique()}

            costs = {'unit_costs': {},
                     'setup_costs': {},
                     'cost_weights': {},
                     }
            for row in dict_costs:
                if row['node'] not in costs['unit_costs']:
                    costs['unit_costs'][row['node']] = dict()
                if row['node'] not in costs['setup_costs']:
                    costs['setup_costs'][row['node']] = dict()
                if row['node'] not in costs['cost_weights']:
                    costs['cost_weights'][row['node']] = dict()
                costs['unit_costs'][row['node']][row['outcome']] = row['unit_cost']
                costs['setup_costs'][row['node']][row['outcome']] = row['setup_cost']
                costs['cost_weights'][row['node']][row['outcome']] = row['cost_weight']
                dag.costs = costs

            self.costs[dag_name] = dag.costs

        print("Imported databook: {}".format(databook_path))

    def create_default_scenarios_old(self, individual_scaleups=False):
        """Create default scenarios and save them to self.scenarios."""
        print('Creating default scenarios')
        intv_nodes = [node for node in self.nodes if node.nodekey in self.intervention_nodes]

        # Baseline scenario - no changes
        scen_baseline = Scenario(name='baseline', long='Baseline', years=SCENARIO_YEARS)

        # No coverage scenario - set all interventions to 'none' from 2025
        intvs_nocov = []
        for node in intv_nodes:
            if any([node.nodekey == intv.nodekey for intv in intvs_nocov]): continue  # avoid duplicates
            levels = node.levels
            intv_dict = {level: 1 if level == 'none' else 0 for level in levels}
            intvs_nocov.append(Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_STEP, values_target=intv_dict))
        scen_nocov = Scenario(name='no_coverage', long='No Coverage', years=SCENARIO_YEARS, interventions=intvs_nocov)

        # Individual scaleup - scale up each individual intervention (by outcome) seperately
        scens_individual = []
        for node in intv_nodes:
            if any([node.nodekey == scen.interventions[0].nodekey for scen in scens_individual]): continue  # avoid duplicates
            nodekey = node.nodekey
            levels = node.levels

            for outcome in [level for level in node.levels if level != 'none']:
                intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome=outcome)
                scen_name = f'maxcov-{nodekey}-{outcome}' if len(levels) > 2 else f'maxcov-{nodekey}'
                scen_long = f'Scaleup: {node.long}: {outcome}' if len(levels) > 2 else f'Scaleup {node.long}'
                scen = Scenario(name=scen_name, long=scen_long, years=SCENARIO_YEARS, interventions=[intv])
                scens_individual.append(scen)

        # Full scaleup medicalised - scale up all interventions to max coverage
        intvs_full_med = []
        for node in intv_nodes:
            if any([node.nodekey == intv.nodekey for intv in intvs_full_med]): continue  # avoid duplicates
            if node.nodekey == 'int_check': continue  # no drug checking
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome='medicalised')
            intvs_full_med.append(intv)
        scen_full_med = Scenario(name='full_package_med', long='Full Scale-up (medicalised DCR)', years=SCENARIO_YEARS, interventions=intvs_full_med)

        # Full scaleup nurse - scale up all interventions to max coverage
        intvs_full_nurse = []
        for node in intv_nodes:
            if any([node.nodekey == intv.nodekey for intv in intvs_full_nurse]): continue  # avoid duplicates
            if node.nodekey == 'int_check': continue  # no drug checking
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome='peerled')
            intvs_full_nurse.append(intv)
        scen_full_nurse = Scenario(name='full_package_nurse', long='Full Scale-up (nurse-led DCR)', years=SCENARIO_YEARS, interventions=intvs_full_nurse)

        # Full scaleup nurse + drug checking - scale up all interventions to max coverage
        intvs_full_nurse_check = []
        for node in intv_nodes:
            if any([node.nodekey == intv.nodekey for intv in intvs_full_nurse_check]): continue  # avoid duplicates
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome='peerled')
            intvs_full_nurse_check.append(intv)
        scen_full_nurse_checking = Scenario(name='full_package_nurse_check', long='Full Scale-up (nurse-led DCR) with drug checking', years=SCENARIO_YEARS, interventions=intvs_full_nurse_check)

        # Add scenarios
        if individual_scaleups:
            scenarios_list = [scen_baseline, scen_nocov, scen_full_med, scen_full_nurse, scen_full_nurse_checking] + scens_individual
        else:
            scenarios_list = [scen_baseline, scen_nocov, scen_full_med, scen_full_nurse, scen_full_nurse_checking]
        scenarios = {scen.name: scen for scen in scenarios_list}
        self.scenarios = scenarios


    def create_default_scenarios(self, individual_scaleups=False):
        """Create default scenarios and save them to self.scenarios."""
        print('Creating default scenarios')
        # TODO: fix this hacky method of removing duplicate nodes
        intv_nodes = [node for node in self.nodes if node.nodekey in self.intervention_nodes]
        intv_nodes_unique = []
        for intv_node in intv_nodes:
            if np.all([node.nodekey != intv_node.nodekey for node in intv_nodes_unique]):
                intv_nodes_unique.append(intv_node)
        intv_nodes = intv_nodes_unique

        # Baseline scenario - no changes
        scen_baseline = Scenario(name='baseline', long='Baseline', years=SCENARIO_YEARS)

        # No coverage scenario - set all interventions to 'none' from 2025
        intvs_nocov = []
        for node in intv_nodes:
            levels = node.levels
            intv_dict = {level: 1 if level == 'none' else 0 for level in levels}
            intvs_nocov.append(Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_STEP, values_target=intv_dict))
        scen_nocov = Scenario(name='no_coverage', long='No Coverage', years=SCENARIO_YEARS, interventions=intvs_nocov)

        # Individual scaleup - scale up each individual intervention (by outcome) seperately
        scens_individual = []
        for node in intv_nodes:
            nodekey = node.nodekey
            levels = node.levels

            for outcome in [level for level in node.levels if level != 'none']:
                intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome=outcome)
                scen_name = f'maxcov-{nodekey}-{outcome}' if len(levels) > 2 else f'maxcov-{nodekey}'
                scen_long = f'Scale up {node.long} - {outcome}' if len(levels) > 2 else f'Scale up {node.long}'
                scen = Scenario(name=scen_name, long=scen_long, years=SCENARIO_YEARS, interventions=[intv])
                scens_individual.append(scen)
        scens_individual.sort(key=lambda x: x.long)

        # Package A
        intvs_package_a = []
        for node in intv_nodes:
            nodekey = node.nodekey
            if nodekey in ['int_check', 'int_techno', 'int_safesupply']: continue
            target_outcome = 'medicalised' if nodekey == 'int_dcr' else None
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome=target_outcome)
            intvs_package_a.append(intv)
        scen_package_a = Scenario(name='package_a', long='Package A', years=SCENARIO_YEARS, interventions=intvs_package_a)

        # Package B
        intvs_package_b = []
        for node in intv_nodes:
            nodekey = node.nodekey
            if nodekey in ['int_check', 'int_techno', 'int_safesupply']: continue
            target_outcome = 'peerled' if nodekey == 'int_dcr' else None
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome=target_outcome)
            intvs_package_b.append(intv)
        scen_package_b = Scenario(name='package_b', long='Package B', years=SCENARIO_YEARS, interventions=intvs_package_b)

        # Package C
        intvs_package_c = []
        for node in intv_nodes:
            nodekey = node.nodekey
            if nodekey in ['int_safesupply']: continue
            target_outcome = 'peerled' if nodekey == 'int_dcr' else None
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome=target_outcome)
            intvs_package_c.append(intv)
        scen_package_c = Scenario(name='package_c', long='Package C', years=SCENARIO_YEARS, interventions=intvs_package_c)

        # Package D
        intvs_package_d = []
        for node in intv_nodes:
            nodekey = node.nodekey
            if nodekey in ['int_oat']: continue
            target_outcome = 'peerled' if nodekey == 'int_dcr' else None
            intv = Intervention(node=node, years=SCENARIO_YEARS, scaleup=SCALEUP_LINEAR, target_outcome=target_outcome)
            intvs_package_d.append(intv)
        scen_package_d = Scenario(name='package_d', long='Package D', years=SCENARIO_YEARS, interventions=intvs_package_d)

        # Add scenarios
        if individual_scaleups:
            scenarios_list = [scen_baseline, scen_nocov, scen_package_a, scen_package_b, scen_package_c, scen_package_d] + scens_individual
        else:
            scenarios_list = [scen_baseline, scen_nocov, scen_package_a, scen_package_b, scen_package_c, scen_package_d]
        scenarios = {scen.name: scen for scen in scenarios_list}
        self.scenarios = scenarios

    def initialise_scenarios(self, n_samples=None, sample_method='best', seed=None):
        """Initialise scenarios, including setting up sampled CPDs and calculating coverages."""
        print(f"Initialising scenarios (including setting up {n_samples} samples)")
        for dag_name in self.dags.keys():
            dag = self.dags[dag_name]
            dag.process_scenarios(self.scenarios, n_samples=n_samples, sample_method=sample_method, seed=seed)

    def run_baseline(self):
        """Run model for each DAG."""
        start = sc.tic()

        # Run baseline model, using Bayesian Inference with the evidence provided
        results = dict()
        for dag_name in self.dags.keys():
            dag = self.dags[dag_name]
            dag.initialise_bayes_net()
            result = dag.run_inference(use_baseline_evidence=True)
            results[dag_name] = result
        self.baseline_results = results

        # Bake inferences into the CPTs of the DAGs
        self.bake_baseline_inferences()

        # Rerun baseline with newly inferred probabilities
        self.baseline_results = {dag_name: dag.run_inference(use_baseline_evidence=False) for dag_name, dag in self.dags.items()}

        sc.toc(start=start, label="Finished running baseline model")

    def run_model_direct(self, scenario='baseline', sample_method='best', seed=None):
        """Calculates all the resulting outcomes directly, based on the CPTs.
        Results are saved in Project.results.
        """
        results = {dag_name: {} for dag_name in self.dags.keys()}
        for dag_name in results.keys():
            result = results[dag_name]
            dag = self.dags[dag_name]
            for nodekey in dag.node_keys:
                node = dag.nodes[nodekey]
                cpt = dag.baseline_probset[nodekey]
                cond_dict = {k: result[k] for k in node['parents']}
                result[nodekey] = cpt.sample_outcome(cond_dict=cond_dict, sample_method=sample_method, seed=seed)
        self.results['{}_direct'.format(scenario)] = results

    def run_scenarios(self, sample=False, seed=None, sample_individuals=False):
        """Run scenarios in self.scenarios."""
        start = sc.tic()
        print(f'Running scenarios sample={sample}')

        if self.results is None:
            result_set = ResultSet(costs=self.costs)
        else:
            result_set = self.results

        if not sample:
            for dag_name, dag in self.dags.items():
                for scen_name, scen in dag.scenarios.items():
                    for year_idx, year_num in enumerate(scen.years):
                        year = str(year_num)
                        result = dag.run_inference(scen_name=scen_name, year=year, use_baseline_evidence=False)
                        result_set.add_result(result)

            sc.toc(start=start, label="Finished running best estimates")

        else:
            iterkwargs = []
            for dag_name, dag in self.dags.items():
                cloudpickled_sampled_bns = cloudpickle.dumps(dag.sampled_bns)

                for scen_name, scen in dag.scenarios.items():
                    iterkwargs.append(dict(cloudpickled_sampled_bns=cloudpickled_sampled_bns,
                                           this_scen_evidence_cpds=dag.scen_evidence_cpds[scen_name],
                                           scen_name=scen_name, years=scen.years,
                                           dag_name=dag.dag_name, node_keys=dag.node_keys,
                                           node_levels=dag.node_levels, popsize=dag.popsize,
                                           seed=seed, sample_individuals=sample_individuals,
                                           parallel=False))

            global pool
            if pool is None: pool = cf.ProcessPoolExecutor(max_workers=sc.cpu_count() - 1)

            all_sens_results = sc.parallelize(func=run_inference_sens_func, iterkwargs=iterkwargs, parallelizer=pool.map)

            for sens_results in all_sens_results: # all_sens_results is a list of lists of sensitivity results
                for sens_result in sens_results:
                    result_set.add_result(sens_result.to_result())

            sc.toc(start=start, label="Finished running uncertainty")

        result_set.scenarios = self.scenarios
        result_set.nodes = self.nodes
        result_set.node_info = self.node_info
        result_set.df = result_set.make_dataframe()
        result_set.df_diff = result_set.diff_from_baseline()
        if sample:
            result_set.df_sens = result_set.summarise_df(result_set.df, result_set.df_diff)

        self.results = result_set

    def bake_baseline_inferences(self):
        self.info['inference_run'] = datetime.datetime.now()

        for dag_name in self.dags.keys():
            dag = self.dags[dag_name]
            baseline_result = self.baseline_results[dag_name]
            dag.baseline_result = baseline_result
            updated_cpt_dict = dag.bake_inferences(baseline_result)

            # TODO: uncomment / fix. Currently the Bayesian inference with virtual evidence is producing weird results
            # # Create new Bayesian Network with inferred probabilities
            # for nodekey, cpt in updated_cpt_dict.items():
            #     dag.baseline_probset[nodekey].df = cpt
            # dag.initialise_bayes_net()


class DAG(graphlib.TopologicalSorter):
    """Directed Acyclic Graph

    The base mathematical structure of the model.
    """
    def __init__(self, graph, nodes, arcs, interventions, population='default_pop', dag_name='default_dag'):
        """Initialize DAG."""
        graphlib.TopologicalSorter.__init__(self, graph)
        self.graph = graph
        self.nodes = nodes
        self.arcs = arcs
        self.population = population
        self.dag_name = dag_name
        self.interventions = interventions
        self.intervention_nodes = None
        self.evidence = dict()
        self.maxcovs = dict()
        self.costs = None
        self.node_levels = {node.nodekey: node.levels for node in nodes}
        self.state_map = {nodekey: {x: i for i, x in enumerate(levels)} for nodekey, levels in self.node_levels.items()}
        self.node_keys = None
        self.baseline_probset = self.create_empty_probset()
        self.baseline_result = None
        self.scenarios = dict()
        self.scen_evidence_cpds = dict()
        self.bayes_net = None
        self.sampled_bns = dict()
        self.inference = None
        self.result = None
        self.popsize = None

    def create_empty_probset(self):
        """Create empty probset with correct structure.
        Also initialise CPTs in Node objects.
        """
        probset = dict()
        for node in self.nodes:
            nodekey = node.nodekey
            cpt = CPT(nodekey=nodekey, dag_name=self.dag_name, levels=node.levels, parents=node.parents, parents_levels=node.parents_levels)
            probset[nodekey] = cpt
            node.cpt = cpt
        return probset

    def normalise_evidence(self):
        """Ensure all probabilities add up to 100%.
        Give errors where this condition isn't met.
        """
        for nodekey in list(self.evidence.keys()):
            evidence = self.evidence[nodekey]
            for level in self.node_levels[nodekey]:
                if level not in evidence:
                    evidence[level] = np.nan
            n_levels = len(evidence)
            n_ev = np.sum(~pd.isna(list(evidence.values())))
            if n_ev == 0:  # no evidence
                del self.evidence[nodekey]
            elif n_ev == n_levels:
                sum_ev = np.sum(list(evidence.values()))
                if not sum_ev == 1:
                    print("Normalising evidence for [{}: {}]".format(self.dag_name, nodekey))
                    self.evidence[nodekey] = {k: v / sum_ev for k, v in evidence.items()}
            elif n_levels - n_ev == 1:
                sum_ev = np.nansum(list(evidence.values()))
                which_nan = [k for k, v in evidence.items() if pd.isna(v)][0]
                new_ev = 1 - sum_ev
                if new_ev < 0:
                    print("Existing evidence adds up to >100% so normalising for [{}: {}]".format(self.dag_name, nodekey))
                    evidence[which_nan] = 0
                    self.evidence[nodekey] = {k: v / sum_ev for k, v in evidence.items()}
                else:
                    evidence[which_nan] = new_ev
            else:
                print("Insufficient evidence provided for [{}: {}]".format(self.dag_name, nodekey))
                del self.evidence[nodekey]
        return

    def initialise_bayes_net(self):
        """Creates pgmpy.BayesianNetwork."""
        arc_list = [(arc['parent'], arc['child']) for arc in self.arcs]
        self.bayes_net = DiscreteBayesianNetwork(arc_list)
        cpds = []
        for node in self.nodes:
            cpd = node.to_cpd()  # Note not sampled (sample_method = 'best' so no seed needed)
            cpds.append(cpd)

        self.bayes_net.add_cpds(*cpds)

    def get_baseline_evidence_cpds(self):
        """Convert baseline evidence to TabularCPD for pgmpy."""
        evidence_cpds = []
        for node in self.nodes:
            nodekey = node.nodekey
            if nodekey not in self.evidence:
                continue
            node_evidence = self.evidence[nodekey]
            levels = node.levels
            parents = node.parents
            if len(parents) == 0 and nodekey not in self.intervention_nodes:
                print("Node [{}] has evidence but zero parents. Delete this from evidence".format(nodekey))
            state_names = {nodekey: levels} | {parent: self.node_levels[parent] for parent in parents}
            values = [[node_evidence[level]] for level in levels]
            imp_cpd = TabularCPD(variable=nodekey,
                                 variable_card=len(levels),
                                 state_names=state_names,
                                 values=values)
            evidence_cpds.append(imp_cpd)

        return evidence_cpds

    def bake_inferences(self, result):
        """Create new CPDs with evidence given reference result."""
        result_cpt_dict = result.inferred_cpt_dict
        updated_cpt_dict = {}
        for nodekey in result_cpt_dict.keys():
            result_cpt = result_cpt_dict[nodekey]
            merged_cpt = dcp(self.baseline_probset[nodekey].df)

            replace_cpt = pd.DataFrame()
            replace_cpt['best'] = result_cpt
            replace_cpt['lower'] = result_cpt
            replace_cpt['upper'] = result_cpt
            replace_cpt.index = merged_cpt.index

            which_update = (~pd.isnull(result_cpt)).tolist()
            merged_cpt[which_update] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                merged_cpt.update(replace_cpt, overwrite=False)

            updated_cpt_dict[nodekey] = merged_cpt

        return updated_cpt_dict

    def process_scenarios(self, scenarios, n_samples=None, sample_method='best', seed=None):
        """Imports scenarios and saves them to DAG object"""
        dag_scenarios = dcp(scenarios)
        for scen_name in dag_scenarios.keys():
            scen = dag_scenarios[scen_name]
            scen.interventions = [intv for intv in scen.interventions if
                                  intv.nodekey in self.intervention_nodes
                                  # and intv.dag_name == self.dag_name
                                  # and intv.population == self.population
                                  ]

            # Add in specific max coverages for each scenario
            for intv in scen.interventions:
                nodekey = intv.nodekey
                levels = intv.levels
                intv.dag_name = self.dag_name
                intv.population = self.population

                if intv.values_target is not None:
                    continue

                intv_dict = {level: 0 for level in levels}
                target_outcome = intv.target_outcome
                maxcov = self.maxcovs[nodekey][intv.target_outcome]
                intv_dict[target_outcome] = maxcov
                intv_dict['none'] = 1 - np.sum(v for k, v in intv_dict.items() if k != 'none')
                intv.values_target = intv_dict

        self.scenarios = dag_scenarios
        self.initialise_sampled_probset(n_samples=n_samples, sample_method=sample_method, seed=seed)
        self.create_scenario_evidence_cpds()

    def initialise_sampled_probset(self, n_samples=1, sample_method='best', seed=None):
        if seed is None and sample_method != 'best': raise Exception('Provide a seed (was None)')

        arc_list = [(arc['parent'], arc['child']) for arc in self.arcs]
        bn_base = DiscreteBayesianNetwork(arc_list)

        cpds_sampled = [node.to_cpds_sample(n_samples=n_samples, sample_method='best' if 'int' in node.nodekey else sample_method, seed=seed) for node in self.nodes]

        self.sampled_bns = []
        for i_sample in range(n_samples):
            bn = dcp(bn_base)
            bn.add_cpds(*([cpds[i_sample] for cpds in cpds_sampled]))
            self.sampled_bns.append(bn)


    def create_scenario_evidence_cpds(self):
        for scen_name, scenario in self.scenarios.items():
            self.scen_evidence_cpds[scen_name] = dict()
            for year_num in scenario.years:
                year = str(year_num)
                scen_cpds = self.get_scenario_cpds(scen_name=scen_name, year=year, baseline_outcomes_dict=self.baseline_result.result_dict['best'])
                self.scen_evidence_cpds[scen_name][year] = scen_cpds

    def get_scenario_cpds(self, scen_name=None, year=None, baseline_outcomes_dict=None):
        """Get CPDs of DAG based on scenario and year specified"""
        if scen_name is not None and year is not None and baseline_outcomes_dict is not None:
            scen = self.scenarios[scen_name]

            impact_cpds = [impact.outcome_dict_to_cpd(year=year, outcome_dict=baseline_outcomes_dict[impact.nodekey])
                                   for impact in scen.interventions]

            return impact_cpds
        else:
            return []

    def get_scenario_cpds_old(self, scen_name, year):
        if scen_name is not None and year is not None:
            impact_list = self.scenarios[scen_name][year]
        else:
            return []
        cpt_dicts = {}
        impact_cpds = []

        for impact in impact_list:
            nodekey = impact.nodekey
            if nodekey not in cpt_dicts:
                new_nodekey = True
                cpt_dicts[nodekey] = self.baseline_probset[nodekey].df['best'].to_dict()
            else:
                new_nodekey = False
            cpt_dict = cpt_dicts[nodekey]

            outcome_from = impact.outcome_from
            outcome_to = impact.outcome_to
            value = impact.value

            if outcome_from is None:
                if outcome_to in cpt_dict.keys():
                    for outcome in cpt_dict.keys():
                        cpt_dict[outcome] = 1 if outcome == outcome_to else 0
                else:
                    print("[{}: {}] node does not contain [{}] outcome referenced in [{}] scenario".
                          format(self.dag_name, nodekey, outcome_to, scen_name))
            else:
                keys_absent = np.setdiff1d([x for x in [outcome_from, outcome_to]], list(cpt_dict.keys()))
                if len(keys_absent) != 0:
                    print("[{}: {}] node does not contain outcome(s) referenced in [{}] scenario: {}".
                          format(self.dag_name, nodekey, scen_name, keys_absent))
                    if new_nodekey:
                        del cpt_dicts[nodekey]
                    continue

                check_impact = impact.check_impact(cpt_dict)
                if check_impact:
                    cpt_dict[outcome_from] -= value
                    cpt_dict[outcome_to] += value
                else:
                    print("{}: impact of [{}] intervention on [{}: {}] in year {} causes infeasible probabilities".
                          format(impact.scen_long, impact.name, self.dag_name, nodekey, year))
                    # raise ValueError
                    if new_nodekey:
                        del cpt_dicts[nodekey]
                    continue

        for nodekey, cpt_dict in cpt_dicts.items():
            node = self.nodes[nodekey]
            levels = node['levels']
            parents = list(node['parents'])
            if len(parents) != 0:
                print("Node [{}] has parents but is impacted. Delete this from evidence".format(nodekey))
                continue

            state_names = {nodekey: levels}
            values = [[cpt_dict[level]] for level in levels]
            imp_cpd = TabularCPD(variable=nodekey,
                                 variable_card=len(levels),
                                 state_names=state_names,
                                 values=values)
            impact_cpds.append(imp_cpd)

        return impact_cpds

    def run_inference(self, use_baseline_evidence=False, scen_name=None, year=None):
        """Run inference given CPTs and evidence."""

        if use_baseline_evidence:
            evidence_list = self.get_baseline_evidence_cpds()
        else:
            evidence_list = []

        if scen_name is not None and year is not None:
            updated_cpt_list = self.scen_evidence_cpds[scen_name][year]
            n_pop = round(self.popsize[year])
        else:
            updated_cpt_list = []
            n_pop = 1

        bn = dcp(self.bayes_net)
        bn.add_cpds(*updated_cpt_list)

        inference = VariableElimination(bn)

        # TODO: virtual evidence is not working as expected
        output_joint = inference.query(variables=self.node_keys, virtual_evidence=evidence_list, joint=True)
        output_marginal = joint2marginal(variables=self.node_keys, output_joint=output_joint)
        result_dict = {nodekey: dict(zip(self.node_levels[nodekey], discrete_factor.values * n_pop)) for nodekey, discrete_factor in output_marginal.items()}

        inferred_cpt_dict = {}
        for node in self.nodes:
            nodekey = node.nodekey
            parents = list(node.parents)
            if nodekey not in self.intervention_nodes:  # this ensures consistency in the intervention nodes
                levels = {parent: self.node_levels[parent] for parent in parents} | {nodekey: node.levels}
                inferred_cpt_dict[nodekey] = get_updated_cpt(inference=inference, evidence=evidence_list, levels=levels)

        result = Result(scen_name=scen_name, year=year, dag_name=self.dag_name, inference=inference, node_levels=self.node_levels, result_dict={'best': result_dict}, output_joint=output_joint, inferred_cpt_dict=inferred_cpt_dict)

        return result


    def run_inference_sens(self, scen_name, year, sample_individuals=True, seed=None, parallel=False):
        """Run inference given CPTs and evidence."""

        return run_inference_sens_func(
                        cloudpickle.dumps(self.sampled_bns), self.scen_evidence_cpds[scen_name], self.dag_name, self.node_keys, self.node_levels, self.popsize,
                        scen_name, years=[year],
                        sample_individuals=sample_individuals, seed=seed, parallel=parallel)[0] # 0 because run_inference_sens_func returns list of results


def run_inference_sens_func(cloudpickled_sampled_bns, this_scen_evidence_cpds, dag_name, node_keys, node_levels, popsize,
                        scen_name, years,
                        sample_individuals=True, seed=None, parallel=False):
    """Run inference given CPTs and evidence.
    years is list of years to loop over
    Returns: a list of sens_result, 1 for each year
    """
    if sample_individuals and seed is None: raise Exception('seed=None, need to provide a seed because sample_individuals=True')

    sens_results = []

    year_seeds = np.random.default_rng(seed).integers(0, np.iinfo(np.int32).max, len(years))

    for year, year_seed in zip(years, year_seeds):
        start = sc.tic()

        year = str(year)

        sampled_bns = cloudpickle.loads(cloudpickled_sampled_bns)  # We unpickle a new copy each time as it gets overwritten in the inside func

        if scen_name is not None and year is not None:
            updated_cpt_list = this_scen_evidence_cpds[year]
            n_pop = round(popsize[year])
        else:
            updated_cpt_list = []
            n_pop = 1

        sens_result = SensResult(scen_name=scen_name, year=year, dag_name=dag_name, node_levels=node_levels)

        # Now we loop over the n_samples given by sampled_bns, with a new seed for each sample / year combo

        seeds = np.random.default_rng(year_seed).integers(0, np.iinfo(np.int32).max, len(sampled_bns))  # NOTE: different seed for each simulation, as well as different sampled probabilities

        iterkwargs = dict(bn_scen=sampled_bns, this_seed=seeds)
        kwargs = dict(updated_cpt_list=updated_cpt_list, n_pop=n_pop, sample_individuals=sample_individuals,
                        node_keys=node_keys, node_levels=node_levels)

        global pool
        if pool is None: pool = cf.ProcessPoolExecutor(max_workers=sc.cpu_count() - 1)

        result_dicts = sc.parallelize(func=run_inference_single_sens, iterkwargs=iterkwargs, kwargs=kwargs, parallelizer=pool.map,
                                      serial=(not parallel))  # Currently serial = True is faster, parallelize the call to this function instead
        for result in result_dicts: sens_result.add_result(result)
        sens_results.append(sens_result)

        sc.toc(start=start, label=f'Run sampled scenario: {dag_name} {scen_name} {year}')

    return sens_results


def run_inference_single_sens(node_keys, node_levels, bn_scen, this_seed, updated_cpt_list, n_pop, sample_individuals):
    bn_scen.add_cpds(*updated_cpt_list)
    if sample_individuals:
        inference = ApproxInference(bn_scen)
        output_marginal = inference.query(variables=node_keys, state_names=node_levels, joint=False,
                                          n_samples=n_pop, show_progress=False, seed=this_seed)
    else:
        inference = VariableElimination(bn_scen)
        output_marginal = inference.query(variables=node_keys, joint=False)
    result_dict = {nodekey: dict(zip(node_levels[nodekey], discrete_factor.values * n_pop)) for nodekey, discrete_factor in output_marginal.items()}
    return result_dict


class Node(object):
    """Node.

    Stores node information and create CPT, either samples or best estimate.
    """

    def __init__(self, nodekey, long, levels, parents, children, parents_levels, population=None, dag_name=None, cost_cat=None, outcome_cat=None):
        self.nodekey = nodekey
        self.short = nodekey
        self.long = long
        self.levels = levels
        self.parents = parents
        self.children = children
        self.parents_levels = parents_levels
        self.dag_name = dag_name
        self.population = population
        self.cost_cat = cost_cat
        self.outcome_cat = outcome_cat
        self.cpt = self.initialise_cpt()
        self.state_names = {nodekey: levels} | self.parents_levels

    def initialise_cpt(self):
        """Create empty CPT."""
        cpt = CPT(nodekey=self.nodekey, dag_name=self.dag_name, levels=self.levels, parents=self.parents, parents_levels=self.parents_levels)
        return cpt

    def to_cpd(self, sample_method='best', seed=None):
        probs = self.cpt.sample_probs(sample_method=sample_method, seed=seed)

        if len(self.parents) == 0:
            values = [[x] for x in probs]
            cpd = TabularCPD(variable=self.nodekey,
                             variable_card=len(self.levels),
                             state_names=self.state_names,
                             values=values)
        else:
            values = probs
            for parent in self.parents:
                values = unstack_keep_order(values, parent)
            cpd = TabularCPD(variable=self.nodekey,
                             variable_card=len(self.levels),
                             state_names=self.state_names,
                             values=values,
                             evidence=self.parents,
                             evidence_card=[len(parent_levels) for parent_levels in self.parents_levels.values()])

        return cpd

    def to_cpds_sample(self, sample_method='best', n_samples=1, seed=None):
        if seed is None: raise Exception(f'Please pass a seed (was None) (sample_method = {sample_method})')
        seeds = np.random.default_rng(seed=seed).integers(0, np.iinfo(np.int32).max, n_samples)

        global pool
        if pool is None:
            pool = cf.ProcessPoolExecutor(max_workers=sc.cpu_count()-1)

        # cpd_list = sc.parallelize(func=self.to_cpd, iterkwargs={'seed':seeds}, sample_method=sample_method, parallelizer=pool.map)
        cpd_list = [self.to_cpd(sample_method=sample_method, seed=this_seed) for this_seed in seeds]
        return cpd_list

    def sample_outcomes(self,  n_samples=1, cond_dict=None, sample_method=None, seed=None):
        if seed is None: raise Exception(f'Please pass a seed (was None) (sample_method = {sample_method})')
        seeds = np.random.default_rng(seed=seed).integers(0, np.iinfo(np.int32).max, n_samples)
        return [self.cpt.sample_outcome(cond_dict=cond_dict, sample_method=sample_method, seed=this_seed) for this_seed in seeds]


class CPT(object):
    """Conditional Probability Table.

    Stores the conditional probability table of a node as a pd.DataFrame
    """

    def __init__(self, nodekey, dag_name, levels, parents, parents_levels):
        """Initialize the CPT."""
        self.nodekey = nodekey
        self.levels = levels
        self.parents = parents_levels
        self.parents_levels = parents_levels
        self.dag_name = dag_name
        self.df = self.initialise_df()
        self.default_outcome = self.get_default_outcome()

    def get_default_outcome(self):
        levels = self.levels
        if len(levels) == 2 or True:  # only consider nodes with binary outcomes
            if 'no' in levels:
                return 'no'
            elif 'none' in levels:
                return 'none'
        return False

    def get(self, cond_dict, estimates=ESTIMATES):
        """Retrieve the probabilities of a dict of conditions."""
        row_selectors = [self.df.index.get_level_values(k) == v for k, v in cond_dict.items()]
        if len(row_selectors):
            row_index = np.prod(np.vstack(row_selectors), axis=0, dtype=bool)
        else:
            row_index = self.df.index
        return self.df.loc[row_index, estimates]

    def set(self, cond_dict, values, estimates=ESTIMATES):
        """Set the probabilities of a dict of conditions."""
        row_selectors = [self.df.index.get_level_values(k) == v for k, v in cond_dict.items()]
        row_index = np.prod(np.vstack(row_selectors), axis=0, dtype=bool)
        self.df.loc[row_index, estimates] = values

        lower, best, upper = values[estimates.index('lower')], values[estimates.index('best')], values[estimates.index('upper')]
        if np.isfinite(lower) and lower > best:
            raise Exception(f'Invalid lower bound (lower={lower} > best={best}) for probability: dag={self.dag_name}, nodekey={self.nodekey}, conditions={cond_dict}')
        if np.isfinite(upper) and upper < best:
            raise Exception(f'Invalid upper bound (upper={upper} < best={best}) for probability: dag={self.dag_name}, nodekey={self.nodekey}, conditions={cond_dict}')

    def sample_outcome(self, cond_dict=None, sample_method='best', seed=None):
        cpt = self.sample_probs(sample_method=sample_method, seed=seed)

        if cond_dict is None:
            assert len(self.parents) == 0
        else:
            for nodekey, probs in cond_dict.items():
                cpt = cpt.mul(pd.Series(probs), level=nodekey)
                cpt.index = cpt.index.droplevel(nodekey)

        outcome = cpt.groupby([self.nodekey], sort=False, dropna=False).sum().to_dict()

        return outcome

    def sample_probs(self, sample_method='best', seed=None):
        """Sample probabilities within uncertainty ranges to produce a sample set of 'real' probabilities.
        If sample_method=='best', simply return the best estimate.
        """
        if seed is None and sample_method != 'best':
            raise Exception(f'Please pass a seed (was None) (sample_method = {sample_method})')

        thisseed = int(jl_hash((self.nodekey, self.dag_name, seed)), 16) % np.iinfo(np.int32).max
        rng = np.random.default_rng(thisseed)

        if sample_method is None or sample_method == 'best':
            cpt = dcp(self.df['best'])
        elif sample_method == 'uniform':
            low  = np.min((self.df['lower'], self.df['upper']), axis=0).astype(float)
            high = np.max((self.df['lower'], self.df['upper']), axis=0).astype(float)

            cpt = pd.Series(rng.uniform(low=low, high=high), index=self.df.index)

        elif sample_method in ['triangle', 'tri', 'triangular']:
            low = np.min((self.df['lower'], self.df['upper']), axis=0).astype(float)
            high = np.max((self.df['lower'], self.df['upper']), axis=0).astype(float)
            best = dcp(self.df['best'].values.astype(float))

            sample_index = low < high  # need to manually select divergent elements because rng.triangular can't handle low==high
            cpt = pd.Series(best, index=self.df.index)
            cpt[sample_index] = rng.triangular(left=low[sample_index], right=high[sample_index], mode=best[sample_index])

        elif sample_method in ['modified_triangular', 'mod_tri', 'modified_tri']:
            low  = np.min((self.df['lower'], self.df['upper']), axis=0).astype(float)
            high = np.max((self.df['lower'], self.df['upper']), axis=0).astype(float)
            best = dcp(self.df['best'].values.astype(float))

            sample_index = low < high  # need to manually select divergent elements because rng.triangular can't handle low==high
            cpt = pd.Series(best, index=self.df.index)
            cpt[sample_index] = modified_triangle_distribution(low=low[sample_index], high=high[sample_index], best=best[sample_index], rng=rng)

        elif sample_method in ['beta', 'b', 'modified_beta']:
            low  = np.min((self.df['lower'], self.df['upper']), axis=0).astype(float)
            high = np.max((self.df['lower'], self.df['upper']), axis=0).astype(float)
            best = dcp(self.df['best'].values.astype(float))

            sample_index = low < high  # manually select divergent elements for speed
            cpt = pd.Series(best, index=self.df.index)
            cpt[sample_index] = transformed_beta(low=low[sample_index], high=high[sample_index], mean=best[sample_index], rng=rng)

        else:
            raise NotImplementedError(f'Not implemented sample_method="{sample_method}"')

        cpt_df = cpt.to_frame(name='value')
        cpt_wide = cpt_df.pivot_table(index=list(self.parents.keys()), columns=self.nodekey, values='value', sort=False)

        if np.all(cpt_wide.sum(axis=1) == 1) and False:
            return cpt_df['value']
        elif 'no' in cpt_wide.columns:
            fill_colname = 'no'
        elif 'none' in cpt_wide.columns:
            fill_colname = 'none'
        else:
            cpt_wide = cpt_wide.div(cpt_wide.sum(axis=1), axis=0)
            return cpt_wide.stack(future_stack=True)

        # remove no/none value
        cpt_wide[fill_colname] = 0

        # normalise
        row_sums = cpt_wide.sum(axis=1)
        row_sums[row_sums < 1] = 1
        cpt_wide = cpt_wide.div(row_sums, axis=0)

        # fill no/none value
        cpt_wide[fill_colname] = 1 - cpt_wide.sum(axis=1)

        # reshape
        probs = cpt_wide.stack(future_stack=True)
        if None in probs.index.names:
            probs.index = probs.index.droplevel([None])

        return probs

    def initialise_df(self):
        """Initialise the CPT as a pd.DataFrame"""
        conds = dcp(self.parents_levels)
        conds[self.nodekey] = self.levels

        df = pd.DataFrame(list(itertools.product(*conds.values())), columns=list(conds.keys()))
        df[list(ESTIMATES)] = None
        df = df.set_index(list(conds.keys()))
        return df

    def set_default_bounds(self):
        """Set lower and upper values equal to best estimate if no values provided."""
        df = self.df
        with pd.option_context('future.no_silent_downcasting', True):
            for est in ESTIMATES:
                df[est] = df[est].fillna(df['best'])

    def make_unity_probs(self):
        """Ensure all CPT probabilities add up to 1."""
        df = self.df
        nodekey = self.nodekey
        if self.default_outcome:
            default_outcome = self.default_outcome

            index_yes = df.index.get_level_values(nodekey) != default_outcome
            index_no = df.index.get_level_values(nodekey) == default_outcome

            missing_count = np.sum(df.loc[index_yes, 'best'].isna())
            if missing_count:
                print("WARNING: Missing {} probabilities for [{}: {}]. Filling with random values... (this is not seeded)".format(missing_count, self.dag_name, nodekey))
                df.loc[index_yes, 'best'] = df.loc[index_yes, 'best'].apply(lambda x: x if not pd.isna(x) else np.random.uniform())

            # Make the probabilities add to 1
            if len(self.parents):
                df.loc[index_no, 'best'] = 1 - df.loc[index_yes, 'best'].groupby(list(self.parents.keys()), sort=False).sum().values

                group_sums = df.loc[:, 'best'].groupby(list(self.parents.keys()), sort=False).sum().values
                assert np.all(group_sums == 1.)
            else:
                df.loc[index_no, 'best'] = 1 - df.loc[index_yes, 'best'].sum(axis=0)

        else:
            missing_count = np.sum(df['best'].isna())
            if missing_count:
                print("WARNING: Missing {} probabilities for [{}: {}]. Filling with random values... (this is not seeded)".format(missing_count, self.dag_name, nodekey))
                df['best'] = df['best'].apply(lambda x: x if not pd.isna(x) else np.random.uniform())

            if len(self.parents):
                if not np.all(df['best'].groupby(list(self.parents.keys())).sum() == 1.):
                    if not missing_count:
                        print("WARNING: Non-unity probabilities for [{}: {}]. Normalising...".format(self.dag_name, nodekey))
                    df['best'] = df['best'].groupby(list(self.parents.keys())).transform(lambda x: x/sum(x))
            else:
                if not np.sum(df['best']) == 1:
                    if not missing_count:
                        print("WARNING: Non-unity probabilities for [{}: {}]. Normalising...".format(self.dag_name, nodekey))
                    df['best'] = df['best'] / np.sum(df['best'])


class Intervention(object):
    """Intervention.

    Stores information about each intervention
    """

    def __init__(self, node=None, name=None, nodekey=None, levels=None, dag_name=None, population=None, years=None, scaleup=None, values_baseline=None, values_target=None, target_outcome=None):
        self.name = name
        if node is not None:
            self.nodekey = node.nodekey
            self.levels = node.levels
            self.dag_name = node.dag_name
            self.population = node.population
        else:
            self.nodekey = nodekey
            self.levels = levels
            self.dag_name = dag_name
            self.population = population
        self.years = years
        self.scaleup = scaleup
        self.values_target = values_target

        non_trivial_outcomes = [level for level in node.levels if level != 'none']
        if target_outcome is not None:
            self.target_outcome = target_outcome
        elif len(non_trivial_outcomes) == 1:
            self.target_outcome = non_trivial_outcomes[0]
        elif self.values_target is not None:
            self.target_outcome = None
        else:
            print(f"WARNING: Need to explicitly set target outcome for {self.nodekey} in create_default_scenarios()")
            self.target_outcome = None

    def get_cpt_dict(self, year, outcome_dict):
        """Extract CPT dictionary for a given year."""
        year = float(year)
        scale = np.interp(year, self.years, self.scaleup)
        try:
            cpt_dict = {outcome: (1-scale)*outcome_dict[outcome] + scale*self.values_target[outcome] for outcome in self.levels}
        except:
            raise ValueError
        return cpt_dict

    def outcome_dict_to_cpd(self, year, outcome_dict):
        """Create TabularCPD for a given year."""
        nodekey = self.nodekey
        cpt_dict = self.get_cpt_dict(year=year, outcome_dict=outcome_dict)
        levels = tuple(cpt_dict.keys())
        values = [[cpt_dict[level]] for level in levels]
        imp_cpd = TabularCPD(variable=nodekey,
                             variable_card=len(levels),
                             state_names={nodekey: levels},
                             values=values)
        return imp_cpd


class Scenario(object):
    """Scenario

     Stores scenario information, including interventions.
     """
    def __init__(self, name, years=None, long=None, interventions=None):
        self.name = name
        self.years = years
        self.long = long if long is not None else name
        if interventions is None:
            self.interventions = []
        elif isinstance(interventions, Intervention):
            self.interventions = [interventions]
        else:
            self.interventions = interventions

    def check_scenario(self):
        if self.years is None:
            all_years = [intervention.years for intervention in self.interventions]
            self.years = sorted(list(set().union(*all_years)))


class ResultSet(object):
    """ResultSet

    Stores multiple Results.
    Stored as {scen_name: {dag_name: {year: {estimate: {nodekey: outcome}}}}}
    """

    def __init__(self, costs=None):
        self.results = dict()
        self.scen_names = []
        self.scenarios = []
        self.years = []
        self.dag_names = []
        self.est_keys = []
        self.nodekeys = {}
        self.node_levels = {}
        self.nodes = None
        self.node_info = None
        self.outcome_sheets_dict = None
        self.costs_sheets_dict = None
        self.df = None
        self.df_diff = None
        self.df_sens = None
        self.costs = costs

    def add_result(self, result):
        """Add Result, maintaining consistent structure of ResultSet"""

        scen_name = result.scen_name
        dag_name = result.dag_name
        year = result.year
        node_levels = result.node_levels

        if scen_name not in self.results:
            self.results[scen_name] = {}
            self.scen_names.append(scen_name)

        if dag_name not in self.results[scen_name]:
            self.results[scen_name][dag_name] = {}
            if dag_name not in self.dag_names:
                self.dag_names.append(dag_name)
                self.nodekeys[dag_name] = list(node_levels.keys())
                self.node_levels[dag_name] = node_levels

        if year not in self.results[scen_name][dag_name]:
            self.results[scen_name][dag_name][year] = {}
            if year not in self.years:
                self.years.append(year)

        for est_key in result.est_keys:
            if est_key not in self.est_keys:
                self.est_keys.append(est_key)

        if isinstance(result, Result):
            self.results[scen_name][dag_name][year].update(result.result_dict)
        else:
            print("Don't know how to add result to ResultSet")

    def save_outcomes(self, filename=None, folder="", filter_dict=None, which_df=None):
        """Save results to an Excel file."""
        if which_df is None: which_df = self.df
        if filename is None:
            filename = 'results_outcomes_{}'.format(TODAY_STR)
        results_path = sc.makefilepath(filename=filename, folder=folder, ext='xlsx', makedirs=True)

        self.outcome_sheets_dict = self.make_outcome_sheets_dict(filter_dict=filter_dict, which_df=which_df)

        # Save as Excel workbook
        wb = sheets_dict_to_wb(self.outcome_sheets_dict)
        results_path = save_safely(wb, results_path)
        # wb.save(results_path)

        print("Saved results to: {}".format(results_path))

    def save_costs(self, filename=None, folder="", filter_dict=None, cost_cat_summarised=False):
        """Save results to an Excel file."""
        if filename is None:
            filename = 'results_costs_{}'.format(TODAY_STR)
        if not filename.endswith('.xlsx'):
            filename = filename + '.xlsx'
        results_path = os.path.join(folder, filename)
        Path(folder).mkdir(parents=True, exist_ok=True)

        self.costs_sheets_dict = self.make_costs_sheets_dict(filter_dict=filter_dict, cost_cat_summarised=cost_cat_summarised)

        # Save as Excel workbook
        wb = sheets_dict_to_wb(self.costs_sheets_dict)
        wb.save(results_path)

        print("Saved results to: {}".format(results_path))

    def make_outcome_sheets_dict(self, filter_dict=None, which_df=None):
        """Create dictionary of outputs."""
        if which_df is None: which_df = which_df
        sheets = {}
        for scen_name in self.scen_names:
            df_scen = which_df.loc[which_df['scen_name'] == scen_name]
            if filter_dict is not None:
                for col, vals in filter_dict.items():
                    df_scen = df_scen.loc[df_scen[col].isin(vals)].reset_index(drop=True)
            df_sheet = df_scen.pivot_table(values='value', index=['dag_name', 'nodekey', 'outcome'], columns=['year', 'estimate'], sort=False)
            df_sheet['Total'] = df_sheet.sum(axis=1)
            headers = np.array([x for x in df_sheet.columns.values])
            rows = [['', '', ''] + list(headers[:,0])]
            rows += [['DAG', 'node', 'outcome'] + list(headers[:,1])]
            rows += df_sheet.to_records(index=True).tolist()
            sheets[scen_name] = rows

        return sheets

    def make_costs_sheets_dict(self, filter_dict=None, cost_cat_summarised=False, which_df=None):
        """Create dictionary of outputs."""
        if which_df is None: which_df = self.df
        years_estimates = list(zip(*list(itertools.product(self.years, self.est_keys))))
        if cost_cat_summarised:
            header_rows = [['', ''] + list(years_estimates[0]) + ['']]
            header_rows += [['DAG', 'cost_cat'] + list(years_estimates[1]) + ['Total']]
        else:
            header_rows = [['', '', ''] + list(years_estimates[0]) + ['']]
            header_rows += [['DAG', 'node', 'cost_cat'] + list(years_estimates[1]) + ['Total']]

        sheets = {}
        for scen_name in self.scen_names:
            df_scen = dcp(which_df.loc[which_df['scen_name'] == scen_name])
            df_scen = df_scen.drop(['scen_name'], axis=1)
            if filter_dict is not None:
                for col, vals in filter_dict.items():
                    df_scen = df_scen.loc[df_scen[col].isin(vals)].reset_index(drop=True)
            df_scen = df_scen.drop(['outcome', 'value'], axis=1)
            df_scen = df_scen.groupby(['dag_name', 'nodekey', 'cost_cat', 'year', 'estimate'], sort=False, dropna=False).sum()
            df_sheet = df_scen.pivot_table(values='cost', index=['dag_name', 'nodekey', 'cost_cat'], columns=['year', 'estimate'], sort=False)
            if cost_cat_summarised:
                df_sheet = df_sheet.droplevel(['nodekey']).groupby(level=['dag_name', 'cost_cat'], sort=False, dropna=False).sum()
            df_sheet['Total'] = df_sheet.sum(axis=1)

            headers = np.array([x for x in df_sheet.columns.values])
            if cost_cat_summarised:
                rows = [['', ''] + list(headers[:, 0])]
                rows += [['DAG', 'cost_cat'] + list(headers[:,1])]
            else:
                rows = [['', '', ''] + list(headers[:, 0])]
                rows += [['DAG', 'node', 'cost_cat'] + list(headers[:, 1])]

            rows += df_sheet.to_records(index=True).tolist()
            sheets[scen_name] = rows

        return sheets

    def make_dataframe(self):
        """Convert results to long dataframe."""
        col_names = ['scen_name', 'dag_name', 'nodekey', 'outcome', 'year', 'estimate', 'value']
        rows = []
        for scen_name in self.scen_names:
            for dag_name in self.dag_names:
                for year in self.years:
                    for est_key, estimate in self.results[scen_name][dag_name][year].items():
                        for nodekey, outcomes in self.node_levels[dag_name].items():
                            for outcome in outcomes:
                                value = estimate[nodekey][outcome]
                                row = [scen_name, dag_name, nodekey, outcome, year, est_key, value]
                                rows.append(row)

        df = pd.DataFrame(data=rows, columns=col_names)
        df = self.apply_costs(df)

        return df

    def apply_costs(self, data, discounting=True, use_setup_costs=True, use_weights=True, add_totals=True):
        """Calculate costs due to outcomes and setup costs."""

        df = dcp(data)

        if 'discounting' not in df.columns:
            df['discounting'] = ANNUAL_DISCOUNTING ** (pd.to_numeric(df['year']) - COST_REFERENCE_YEAR)
        if 'is_setup_year' not in df.columns:
            df['is_setup_year'] = (pd.to_numeric(df['year']) == SETUP_COST_YEAR)

        df['unit_cost'] = df.apply(lambda row: self.costs[row.dag_name]['unit_costs'][row.nodekey][row.outcome] , axis=1)
        df['cost_weight'] = df.apply(lambda row: self.costs[row.dag_name]['cost_weights'][row.nodekey][row.outcome] , axis=1)
        df['setup_cost'] = df.apply(lambda row: self.costs[row.dag_name]['setup_costs'][row.nodekey][row.outcome] , axis=1)

        if not discounting:
            df['discounting'] = 1
        if not use_setup_costs:
            df['setup_cost'] = 0
        if not use_weights:
            df['cost_weight'] = 1

        # Calculate annual costs
        df['cost'] = df['value'] * df['unit_cost']  * df['discounting'] * df['cost_weight']

        # Calculate setup costs
        df_setup = dcp(df)
        df_setup['cost'] = df_setup['setup_cost'] * df_setup['is_setup_year'] * (df_setup['value'] > 0)
        df_setup['cost'] = df_setup['cost'] * df_setup['discounting'] * df_setup['cost_weight']
        df_setup = df_setup[df_setup['is_setup_year']].reset_index(drop=True)
        df_setup['year'] = '2026_setup'
        df_setup['value'] = 0

        df_all = pd.concat([df_setup, df])

        drop_cols = ['unit_cost', 'cost_weight', 'setup_cost', 'cost_oneoff', 'cost_annual']
        df_all = df_all.drop([col for col in drop_cols if col in df_all.columns], axis=1)

        df_all['cost_cat'] = df_all.apply(lambda row: self.node_info[(row.dag_name, row.nodekey)]['cost_cat'], axis=1)
        df_all['outcome_cat'] = df_all.apply(lambda row: self.node_info[(row.dag_name, row.nodekey)]['outcome_cat'], axis=1)

        if add_totals: df_all = self.add_total_costs(df_all)
        if add_totals: df_all = self.add_total_outcomes(df_all)

        return df_all

    def add_total_costs(self, df):
        """Add total costs to dataframe."""
        df_costs = df.loc[df['cost_cat'] != 'none']

        total_keyword = '!TOTAL-COST'
        fill_cols = {'nodekey':total_keyword, 'outcome':total_keyword, 'cost_cat':'none', 'outcome_cat':'none'}
        val_columns = ['value', 'cost']

        aggregate_over = ['nodekey', 'outcome', 'outcome_cat']
        keep_columns = [column for column in df_costs.columns if column not in (val_columns + aggregate_over)] # aggregates the val_columns, summing over the different aggregate_over
        df_total = df_costs.groupby(keep_columns, sort=False, dropna=False)[val_columns].sum().reset_index()
        for col in aggregate_over: df_total[col] = fill_cols[col]
        # they have different cost_cat, so move that cost_cat to the 'nodekey'
        df_total['nodekey'] = total_keyword + '-' + df_total['cost_cat']
        df_total['cost_cat'] = fill_cols['cost_cat']

        aggregate_over = ['nodekey', 'outcome', 'outcome_cat', 'cost_cat']
        keep_columns = [column for column in df_costs.columns if column not in (val_columns + aggregate_over)] # aggregates the val_columns, summing over the different aggregate_over
        df_total_all_cats = df_costs.groupby(keep_columns, sort=False, dropna=False)[val_columns].sum().reset_index()
        for col in aggregate_over: df_total_all_cats[col] = fill_cols[col]

        df_total_non_intervention = df_costs.loc[df_costs['cost_cat'] != 'Intervention']
        aggregate_over = ['nodekey', 'outcome', 'outcome_cat', 'cost_cat']
        keep_columns = [column for column in df_total_non_intervention.columns if column not in (val_columns + aggregate_over)] # aggregates the val_columns, summing over the different aggregate_over
        df_total_non_intervention = df_total_non_intervention.groupby(keep_columns, sort=False, dropna=False)[val_columns].sum().reset_index()
        for col in aggregate_over: df_total_non_intervention[col] = fill_cols[col]
        df_total_non_intervention['nodekey'] = total_keyword + '-' + 'NONIntervention'

        df_all_so_far = pd.concat((df_total, df_total_all_cats, df_total_non_intervention), copy=False)

        aggregate_over = ['dag_name']
        keep_columns = [column for column in df_all_so_far.columns if column not in (val_columns + aggregate_over)] # aggregates the val_columns, summing over the different aggregate_over
        df_total_DAGS = df_all_so_far.groupby(keep_columns, sort=False, dropna=False)[val_columns].sum().reset_index()
        for col in aggregate_over: df_total_DAGS[col] = '!DAGS-TOTAL'

        df_all_totals = pd.concat((df_all_so_far, df_total_DAGS), copy=False)
        df_all_totals['value'] = 0  ## Set all values to zero, only need the total costs

        df = pd.concat((df, df_all_totals), copy=False)

        return df

    def add_total_outcomes(self, df):
        """Add total outcomes to dataframe.
        We include "yes" outcomes that are in an outcome_cat, getting a total for each outcome_cat, and then also summing that across DAGS
        """
        df_outcomes = df.loc[df['outcome_cat'] != 'none']
        df_outcomes = df_outcomes.loc[df_outcomes['outcome'] == 'yes']

        total_keyword = '!TOTAL-OUTCOME'
        fill_cols = {'nodekey':total_keyword, 'outcome':total_keyword, 'cost_cat':'none', 'outcome_cat':'none'}
        val_columns = ['value', 'cost']

        aggregate_over = ['nodekey', 'outcome', 'cost_cat']
        keep_columns = [column for column in df_outcomes.columns if column not in (val_columns + aggregate_over)] # aggregates the val_columns, summing over the different aggregate_over
        df_total = df_outcomes.groupby(keep_columns, sort=False, dropna=False)[val_columns].sum().reset_index()
        for col in aggregate_over: df_total[col] = fill_cols[col]
        # they have different outcome_cat, so move that outcome_cat to the 'nodekey'
        df_total['nodekey'] = total_keyword + '-' + df_total['outcome_cat']
        df_total['outcome_cat'] = fill_cols['outcome_cat']

        aggregate_over = ['dag_name']
        keep_columns = [column for column in df_total.columns if column not in (val_columns + aggregate_over)] # aggregates the val_columns, summing over the different aggregate_over
        df_total_DAGS = df_total.groupby(keep_columns, sort=False, dropna=False)[val_columns].sum().reset_index()
        for col in aggregate_over: df_total_DAGS[col] = '!DAGS-TOTAL'

        df_all_totals = pd.concat((df_total, df_total_DAGS), copy=False)
        df_all_totals['cost'] = 0  ## Set all costs to zero, only need the total values

        df = pd.concat((df, df_all_totals), copy=False)

        return df

    def diff_from_baseline(self, baseline_scen_name='baseline'):
        """Calculate difference from baseline, to extract additional/averted costs/outcomes"""
        df = dcp(self.df)
        if baseline_scen_name not in df['scen_name'].unique():
            print("Need to run baseline estimate in order to compare sensitivity results to baselines")
            return None

        df_wide = df.pivot_table(values=['value','cost'], columns='scen_name', index=[col for col in df.columns if col not in ['value', 'cost', 'scen_name']], sort=False)

        df_wide['value'] = df_wide['value'].sub(df_wide[('value',baseline_scen_name)], axis=0)
        df_wide['cost'] = df_wide['cost'].sub(df_wide[('cost', baseline_scen_name)], axis=0)

        df_diff_long = df_wide.stack(future_stack=True).reset_index()

        return df_diff_long

    def summarise_df(self, data, diff_data, add_benefit_cost_ratios=True):
        """Calculate summary statistics of results."""
        print("Doing uncertainty summarisation")
        remove_cols = ['discounting', 'is_setup_year']

        df = dcp(data)
        df_diff = dcp(diff_data)

        df = df.drop([col for col in remove_cols if col in df.columns], axis=1)  # remove unwanted columns
        df_diff = df_diff.drop([col for col in remove_cols if col in df_diff.columns], axis=1)  # remove unwanted columns

        df_diff = df_diff.loc[df_diff['scen_name'] != 'baseline']  # Remove baseline: Should be all zero since it a difference from baseline
        df_diff['scen_name'] = df_diff['scen_name'] + ' - baseline'

        # df_diff_best = df_diff.loc[df_diff['estimate'] == 'best'].drop(columns='estimate')  # Split out best

        # Now add in df_diff into df
        df_with_diff = pd.concat((df, df_diff), copy=False)

        # Sum over years - gives both value and cost
        val_columns = ['value', 'cost']
        df_total = df_with_diff.groupby(['scen_name', 'dag_name', 'nodekey', 'outcome', 'estimate', 'cost_cat', 'outcome_cat'], sort=False, dropna=False)[val_columns].sum().reset_index()

        if add_benefit_cost_ratios: # Ok now that we have the totals from 2026-2030 we can calculate these
            for scen_name in df_total['scen_name'].unique():
                if ' - baseline' not in scen_name: continue

                select_vals = dict(scen_name=scen_name,
                                   dag_name='!DAGS-TOTAL',
                                   nodekey='!TOTAL-COST-Intervention',
                                   outcome='!TOTAL-COST',
                                   outcome_cat='none',
                                   cost_cat='none',
                                   )
                intervention_rows = df_total.loc[df_total[list(select_vals.keys())].eq(pd.Series(select_vals)).all(axis=1)]
                select_vals['nodekey'] = '!TOTAL-COST-NONIntervention'
                nonintervention_rows = df_total.loc[df_total[list(select_vals.keys())].eq(pd.Series(select_vals)).all(axis=1)]

                # we want: increase in spending, decrease in nonintervention cost to be positive
                # so do: -1 * delta nonintervention / delta spending
                # note: decrease in spending, increase in nonintervention cost will also be positive

                intervention_rows.loc[:,'cost'] = -1 * nonintervention_rows['cost'].values / intervention_rows['cost'].values
                intervention_rows.loc[:,'value'] = 0.0
                intervention_rows.loc[:,'nodekey'] = '!BENEFIT-COST-RATIO'

                df_total = pd.concat((df_total, intervention_rows), copy=False)

        df_best_total = df_total.loc[df_total['estimate'] == 'best'].drop(columns='estimate', inplace=False)  # Split out best
        df_total = df_total.loc[df_total['estimate'] != 'best']

        aggregate_over = ['estimate']
        keep_columns = [column for column in df_total.columns if column not in (val_columns + aggregate_over)]
        # df_summary = df_total.groupby(keep_columns, sort=False, dropna=False)[val_columns].agg(['mean', 'median', 'count', 'std']).reset_index()
        list_aggregators = [list_vals, list_vals_sorted, 'mean', 'count', 'std', ci95_low, ci95_high, ci95_low_quantile, ci95_high_quantile, Q1, 'median', Q3, outlier_cutoff_low, outlier_cutoff_high, list_outliers, min_without_outliers, max_without_outliers, 'min','max']
        df_summary = df_total.groupby(keep_columns, sort=False, dropna=False)[val_columns].agg(list_aggregators).reset_index()

        df_with_diff_best = df_with_diff.loc[df_with_diff['estimate'] == 'best'].drop(columns='estimate')

        def mapper(column, _val_columns=val_columns):
            if column in _val_columns: return (column, 'best')
            return (column, '')

        merge_columns = [(col,'') for col in df_best_total.columns if col not in val_columns]

        df_best_total.columns = pd.MultiIndex.from_tuples([mapper(column) for column in list(df_best_total.columns)]) # Turn basic columns into MultiIndex so can merge with MultiIndex of df_summary

        # Merge in the ('value','best') and ('cost','best') from df_best_total, matching the rows based on merge_columns
        df_summary = df_summary.merge(df_best_total, how='inner', on=merge_columns)

        # # Re-order manually to put 'best's in the right spots
        best_columns = [(column, 'best') for column in val_columns]

        columns = list(df_summary.columns)
        for column in best_columns:
            ind = columns.index((column[0],'list_vals'))
            columns.remove(column)
            columns.insert(ind, column)

        df_summary = df_summary[columns]

        def mapper2(column, _val_columns=val_columns):
            new_value_columns = {'value':'outcome_total', 'cost':'cost_total'}
            if column[0] in _val_columns: return (new_value_columns[column[0]], column[1])
            return column

        df_summary.columns = pd.MultiIndex.from_tuples([mapper2(column) for column in list(df_summary.columns)])

        df_best_value = df_with_diff_best.drop(columns='cost').reset_index()
        df_best_cost  = df_with_diff_best.drop(columns='value').reset_index()

        id_cols = [col for col in df_with_diff_best.columns if col not in val_columns + ['year']]

        df_best_value = pd.pivot_table(df_best_value, values='value', columns='year', index=id_cols, sort=False)
        df_best_cost  = pd.pivot_table(df_best_cost,  values='cost', columns='year', index=id_cols, sort=False)

        df_best_value.columns = pd.MultiIndex.from_tuples([('outcome_best',col) for col in df_best_value.columns])
        df_best_value.reset_index(inplace=True)

        df_best_cost.columns = pd.MultiIndex.from_tuples([('cost_best', col) for col in df_best_cost.columns])
        df_best_cost.reset_index(inplace=True)

        df_summary = df_summary.merge(df_best_value, how='left', on=merge_columns)
        df_summary = df_summary.merge(df_best_cost, how='left', on=merge_columns)

        return df_summary


class Result(object):
    """Result.

    Stores result, including metadata and outputs.
    """
    def __init__(self, scen_name, year, dag_name, node_levels, result_dict, inference=None, output_joint=None, inferred_cpt_dict=None):
        self.scen_name = scen_name
        self.year = year
        self.dag_name = dag_name
        self.inference = inference
        self.node_levels = node_levels
        self.result_dict = result_dict
        self.est_keys = list(result_dict.keys())
        self.output_joint = output_joint
        self.inferred_cpt_dict = inferred_cpt_dict


class SensResult(object):
    """SensResult.

    Stores sensitivity result, including metadata and outputs.
    """
    def __init__(self, scen_name, year, dag_name, node_levels, inference=None):
        self.scen_name = scen_name
        self.year = year
        self.dag_name = dag_name
        self.node_levels = node_levels
        self.result_list = list()
        self.results_df = None
        self.results_df_summarised = None
        self.inference = inference

    def add_result(self, rd):
        self.result_list.append(rd)

    def make_df(self):
        result_rows = [] # [node, outcome, value, sample]
        for i, result_dict in enumerate(self.result_list):
            row = [[nodekey, outcome, value, i] for nodekey, nodevalue in result_dict.items() for outcome, value in nodevalue.items()]
            result_rows.extend(row)
        self.results_df = pd.DataFrame(result_rows, columns = ['node', 'outcome', 'value', 'sample'])

    def summarise(self, ci=None):
        if self.results_df is None:
            self.make_df()
        if ci is None:
            ci = {'lower': 0.025, 'upper': 0.975}
        df = self.results_df.drop('sample', axis=1)
        df = df.groupby(['node', 'outcome']).quantile(list(ci.values())).unstack(-1)
        df.columns = ci.keys()
        return df

    def to_dict(self, ci=None, summarise=False):
        if self.results_df is None:
            self.make_df()

        if summarise:
            self.results_df_summarised = self.summarise(ci)
            df_dict = self.results_df_summarised.to_dict('dict')
            result_dict = {est_key: {nodekey: {outcome: (df_dict[est_key][(nodekey, outcome)])
                                               for outcome in outcomes}
                                     for nodekey, outcomes in self.node_levels.items()}
                           for est_key in df_dict.keys()}
        else:
            df = self.results_df
            result_dict = df_to_nested_dict(df, indices=['sample', 'node', 'outcome'])
            result_dict = {est_key: {nodekey: {outcome: val['value'] for outcome, val in node.items()}
                                     for nodekey, node in est.items()}
                           for est_key, est in result_dict.items()}
        return result_dict

    def to_result(self, ci=None):
        return Result(scen_name=self.scen_name,
                      year=self.year,
                      dag_name=self.dag_name,
                      node_levels=self.node_levels,
                      result_dict=self.to_dict(ci=ci),
                      inference=self.inference)

