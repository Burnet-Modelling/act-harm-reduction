"""
This script creates a databook, filling with pre-existing values if an existing databook has been provided
"""
from graph import Project
from utils import get_desktop_folder
from constants import *
import os
import sciris as sc

data_folder = os.path.dirname(os.path.realpath(__file__))
desktop_folder = get_desktop_folder()

model_file = "model_info_*.xlsx"
databook_file = "databook_*standard*.xlsx"

if __name__ == '__main__':

    P = Project(name='default')  # create project

    latest_model_file    = sc.glob(folder=data_folder, pattern=model_file)[-1]  # -1 is last file sorted by name
    latest_databook_file = sc.glob(folder=data_folder, pattern=databook_file)[-1]  # -1 is last file sorted by name

    P.load_model(filename=latest_model_file)  # load model from Excel
    P.create_databook(filename=f"databook_blank_{TODAY_STR}.xlsx", folder=desktop_folder)
    P.create_databook(filename=f"databook_merged_{TODAY_STR}.xlsx", folder=desktop_folder,
                      existing_filename=latest_databook_file,
                      )
