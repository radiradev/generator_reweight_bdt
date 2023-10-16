# create a dataclass for the reweighting object
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Union
from pathlib import Path
import yaml

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
    target_name : str
        the name of the target generator
    target_files : List[Path]
        the list of paths to the target files
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
    
    target_name: str
    target_files: Union[Path, List[Path]]

    variables_plotting_info: Optional[Path] = '/data/rradev/reweighting/generator_reweight_bdt/config/plots.yaml'
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