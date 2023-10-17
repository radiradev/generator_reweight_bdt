# create a dataclass for the reweighting object
from dataclasses import dataclass
from pydantic import BaseModel, Field, root_validator
from glob import glob
from typing import List, Optional, Tuple, Union
from pathlib import Path
import os 
import yaml
import re

@dataclass
class ReweightVariable:
    """A class to represent a variable to be reweighted.
    Attributes
    ----------
    name : str
        the name of the variable
    n_bins : int
        the number of bins
    hist_range : tuple
        the range of the histogram
    observer : bool
        whether the variable is an observer (only used during plotting and not for reweighting)
    """

    name: str
    n_bins: int 
    hist_range: List[float]
    observer: bool = False

class ReweightConfig(BaseModel):
    """A class to represent the reweighting configuration.
    Attributes
    ----------
    nominal_name : str
        the name of the nominal generator
    nominal_files : List[Path]
        the list of paths to the nominal files
    plotting_nominal: str
        a wildcard pattern which matches to the files used for plotting
    target_name : str
        the name of the target generator
    target_files : List[Path]
        the list of paths to the target files
    plotting_target:  str
        a wildcard pattern which matches to the files used for plotting
    reweight_variables : List[ReweightVariable]   
        the list of variables to be reweighted
    observer_variables : List[ReweightVariable]
        the list of variables to be observed
    bdt_ckpt_path : Optional[Path]
        the path to the BDT checkpoint
    number_of_train_events : Optional[int]
        the number of events to use for training the BDT
    """
    
    nominal_name: str
    nominal_files: Union[Path, List[Path]]
    plotting_nominal_pattern: str
    plotting_nominal: List[Path]
    plotting_nominal_oscillated_pattern: str
    plotting_nominal_oscillated: List[Path]

    target_name: str
    target_files: Union[Path, List[Path]]

    plotting_target_pattern: str
    plotting_target: List[Path]
    plotting_target_oscillated_pattern: str
    plotting_target_oscillated: List[Path]

    variables_plotting_info: Optional[Path] = 'config/plots.yaml'
    reweight_variables_names: List[str]

    observer_variables: Optional[List[ReweightVariable]] = []

    bdt_ckpt_path : Optional[Path] = None
    number_of_train_events: Optional[int] = None
    plots_path : Optional[Path] = None
    
    @property
    def reweight_variables(self) -> List[ReweightVariable]:
        with open(self.variables_plotting_info, 'r') as file:
            variable_details = yaml.safe_load(file)
        
        variables = []
        for name in self.reweight_variables_names:
            details = variable_details.get(name)
            if details:
                variables.append(ReweightVariable(name=name, **details))
            else:
                print(f"Details for {name} not found in {self.variables_plotting_info}")
        return variables

    
    @staticmethod
    def match_string(filename_pattern: str) -> List[Path]:
        filenames = glob(filename_pattern)
        return [Path(path) for path in filenames]


    @root_validator(pre=True)
    def populate_plotting_fields(cls, values):
        values['plotting_nominal'] = cls.match_string(values['plotting_nominal_pattern'])
        values['plotting_target'] = cls.match_string(values['plotting_target_pattern'])
        values['plotting_nominal_oscillated'] = cls.match_string(values['plotting_nominal_oscillated_pattern'])
        values['plotting_target_oscillated'] = cls.match_string(values['plotting_target_oscillated_pattern'])
        return values
    
    def get_variable_by_name(self, variable_name: str) -> Optional[ReweightVariable]:
        """Get a ReweightVariable object by its name.
        
        Parameters
        ----------
        variable_name : str
            The name of the variable to fetch.

        Returns
        -------
        Optional[ReweightVariable]
            The ReweightVariable object if found, otherwise None.
        """
        for variable in self.reweight_variables:
            if variable.name == variable_name:
                return variable
        return None



