"""
This script runs the Bayesian Network model.
"""
from graph import Project
from utils import get_onedrive_data_folder, get_desktop_folder, TODAY_STR, save_safely, TIME_STR
from plotting import main_plotting
import warnings
import sciris as sc
import os
import shutil
warnings.filterwarnings("ignore", message="Probability values don't exactly sum to 1")

onedrive_data_folder = get_onedrive_data_folder()
desktop_folder = get_desktop_folder()

n_samples = 100
seed = 0

model_file = "model_info_*.xlsx"            # model_info_20241206.xlsx
databook_file = "Databook/databook_*standard*cost.xlsx"           # standard.xlsx
# databook_file = "Databook/databook_* market change *.xlsx"  # market change.xlsx

if __name__ == '__main__':
    this_folder = os.path.join(desktop_folder, f'Results_{TODAY_STR}_{TIME_STR}')
    run_info_folder = os.path.join(this_folder, 'Run info')
    plots_folder = os.path.join(this_folder, f'Plots_{TIME_STR}')

    P = Project()  # Create project

    latest_model_file    = sc.glob(folder=onedrive_data_folder, pattern=model_file)[-1]  # -1 is last file sorted by name
    latest_databook_file = sc.glob(folder=onedrive_data_folder, pattern=databook_file)[-1]  # -1 is last file sorted by name

    # Save run info
    run_info = sc.gitinfo()
    run_info.update({'model_file': latest_model_file, 'databook_file': latest_databook_file})
    sc.savetext(string=str(run_info), filename=sc.makefilepath(f'run_info_{TODAY_STR}.txt', folder=run_info_folder, makedirs=True))

    # Copy the model and databook that we are using to the `run_info_folder`
    shutil.copy(src=latest_model_file, dst=os.path.join(run_info_folder, os.path.basename(latest_model_file)))
    shutil.copy(src=latest_databook_file, dst=os.path.join(run_info_folder, os.path.basename(latest_databook_file)))

    # Load model and data
    P.load_model(filename=latest_model_file)
    P.import_databook(filename=latest_databook_file)

    # Run baseline scenario with no interventions
    P.run_baseline()

    # Run default intervention scenarios and save results
    P.create_default_scenarios(individual_scaleups=True)
    P.initialise_scenarios(n_samples=n_samples, sample_method='beta', seed=seed)  # This is responsible for sampling the probabilities for sampled runs
    P.run_scenarios()  # Run best estimate
    P.run_scenarios(sample=True, seed=seed, sample_individuals=True)  # Run sampled trial

    P.save(folder=this_folder, filename=f'default_{TODAY_STR}_{TIME_STR}.prj')

    # Save results
    P.results.save_outcomes(filename=f'results_best_outcomes_{TODAY_STR}_{TIME_STR}', folder=this_folder, filter_dict={'estimate': ['best']})
    P.results.save_costs(filename=f'results_best_costs_{TODAY_STR}_{TIME_STR}', folder=this_folder, filter_dict={'estimate': ['best']})
    P.results.save_costs(filename=f'results_best_costs_summarised_{TODAY_STR}_{TIME_STR}', folder=this_folder, filter_dict={'estimate': ['best']}, cost_cat_summarised=True)
    save_safely(P.results.df_sens, f'results_sensitivity_summary_{TODAY_STR}_{TIME_STR}.xlsx', folder=this_folder, doprint=True)

    # Save plots
    main_plotting(P, save_folder=plots_folder)