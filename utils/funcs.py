import awkward as ak
import numpy as np
import uproot
import os
import yaml
from pathlib import Path
from config.config import ReweightConfig
from config.archive.plots import direct_test_vars
import pandas as pd

def load_config(path):
    # Define the path to your YAML file
    yaml_file_path = Path(path)

    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_config = yaml.safe_load(file)
    # Create an instance of ReweightConfig
    config = ReweightConfig(**yaml_config)
    return config

def get_variables_in(test_data=False):
    variables_in = [
            "cc",
            "PDGnu",
            "Enu_true",
            "ELep",
            "Eav",
            "CosLep",
            "Q2",
            "W",
            "x",
            "y",
            "pdg",
            "E",
            "px",
            "py",
            "pz",
            "Weight"]

    variables_in += list(direct_test_vars().keys())

def get_variables_out():
    return  [
            "isNu",
            "isNue",
            "isNumu",
            "isNutau",
            "cc",
            "Enu_true",
            "ELep",
            "EavAlt",
            "CosLep",
            "Q2",
            "W",
            "x",
            "y",
            "nP",
            "nN",
            "nipip",
            "nipim",
            "nipi0",
            "niem",
            "eP",
            "eN",
            "ePip",
            "ePim",
            "ePi0"]

def get_mass():
        return {
            "P": 0.93827,
            "N": 0.93957,
            "piC": 0.13957,
            "pi0": 0.13498,
            "kC": 0.49368,
            "k0": 0.49764,
        }


def calculate_leading_momentum(pdg, p_abs):
    """
    Calculate the leading momentum of each particle type in the final state.
    Parameters
    ----------
    pdg : awkward array
        The pdg codes of the particles in the final state.
    p_abs : awkward array
        The absolute momentum of the particles in the final state.
    Returns
    -------
    leading_momentum : a dict of numpy arrays
        The leading momentum of each particle type in the final state.
        When a particle type is not present in the final state, the value is NaN.
    """
    particles = {
    'proton': 2212,
    'neutron': 2112,
    'piplus': 211,
    'piminus': -211,
    'pizero': 111
    }
    leading_momenta = []
    for particle in particles:
        mask = pdg == particles[particle]
        leading_momentum = ak.max(p_abs[mask], axis=1)
        leading_momentum = leading_momentum.to_numpy().filled(np.nan)
        leading_momenta.append(leading_momentum)

    return np.transpose(np.vstack(leading_momenta))

def compute_Erec(treeArr):
    Elep = treeArr['ELep']
    muon_mass = 0.105658 # GeV
    treeArr['pLep'] = np.sqrt(Elep**2 - muon_mass**2)
    treeArr['Erec'] = Elep + treeArr['EavAlt']
    treeArr['Erec_bias_abs'] = treeArr['Erec'] - treeArr['Enu_true']
    treeArr['Erec_bias_rel'] = treeArr['Erec_bias_abs'] / Elep
    return treeArr


# TODO File loading is a huge mess and needs to be refactored completely
def rootfile_to_array(filename, variables_out, return_weights=False, compute_leading_momentum=False, return_dataframe=False):
        variables_in, m = get_variables_in(), get_mass()
        
        with uproot.open(filename)["FlatTree_VARS"] as tree:
            print(f"Reading {os.path.basename(filename)}")
            treeArr = tree.arrays(variables_in)


            Lepmask = ((treeArr["pdg"] == 11) + (treeArr["pdg"] == -11) +
                       (treeArr["pdg"] == 13) + (treeArr["pdg"] == -13) +
                       (treeArr["pdg"] == 15) + (treeArr["pdg"] == -15))
            Numask = ((treeArr["pdg"] == 12) + (treeArr["pdg"] == -12) +
                      (treeArr["pdg"] == 14) + (treeArr["pdg"] == -14) +
                      (treeArr["pdg"] == 16) + (treeArr["pdg"] == -16))
            Pmask = treeArr["pdg"] == 2212
            Nmask = treeArr["pdg"] == 2112
            Pipmask = treeArr["pdg"] == 211
            Pimmask = treeArr["pdg"] == -211
            Pi0mask = treeArr["pdg"] == 111
            Kpmask = treeArr["pdg"] == 321
            Kmmask = treeArr["pdg"] == -321
            K0mask = ((treeArr["pdg"] == 311) + (treeArr["pdg"] == -311) +
                      (treeArr["pdg"] == 130) + (treeArr["pdg"] == 310))
            EMmask = treeArr["pdg"] == 22

            othermask = (Numask + Lepmask + Pmask + Nmask + Pipmask + Pimmask +
                         Pi0mask + Kpmask + Kmmask + K0mask + EMmask) == False

            # Count particles based on PDG type
            treeArr["nP"] = ak.count_nonzero(Pmask, axis=1)
            treeArr["nN"] = ak.count_nonzero(Nmask, axis=1)
            treeArr["nipip"] = ak.count_nonzero(Pipmask, axis=1)
            treeArr["nipim"] = ak.count_nonzero(Pimmask, axis=1)
            treeArr["nipi0"] = ak.count_nonzero(Pi0mask, axis=1)
            treeArr["nikp"] = ak.count_nonzero(Kpmask, axis=1)
            treeArr["nikm"] = ak.count_nonzero(Kmmask, axis=1)
            treeArr["nik0"] = ak.count_nonzero(K0mask, axis=1)
            treeArr["niem"] = ak.count_nonzero(EMmask, axis=1)
            
            
            px, py, pz = treeArr['px'], treeArr['py'], treeArr['pz']
            p = np.sqrt(pz**2 + px**2 + py**2)
            
            
            # Energy per particle type
            treeArr["eP"] = ak.sum(treeArr["E"][Pmask],
                                   axis=1) - treeArr["nP"] * m["P"]
            treeArr["eN"] = ak.sum(treeArr["E"][Nmask],
                                   axis=1) - treeArr["nN"] * m["N"]
            treeArr["ePip"] = (ak.sum(treeArr["E"][Pipmask], axis=1) -
                               treeArr["nipip"] * m["piC"])
            treeArr["ePim"] = (ak.sum(treeArr["E"][Pimmask], axis=1) -
                               treeArr["nipim"] * m["piC"])
            treeArr["ePi0"] = (ak.sum(treeArr["E"][Pi0mask], axis=1) -
                               treeArr["nipi0"] * m["pi0"])

            treeArr["eOther"] = ak.sum(
                treeArr["E"][othermask] -
                (treeArr["E"][othermask]**2 - treeArr["px"][othermask]**2 -
                 treeArr["py"][othermask]**2 - treeArr["pz"][othermask]**2)**
                0.5,
                axis=1,
            )
            # One hot encoding of nutype
            treeArr["isNu"] = treeArr["PDGnu"] > 0
            treeArr["isNue"] = abs(treeArr["PDGnu"]) == 12
            treeArr["isNumu"] = abs(treeArr["PDGnu"]) == 14
            treeArr["isNutau"] = abs(treeArr["PDGnu"]) == 16

            # Get Weights
            weights = ak.to_numpy(treeArr['Weight'])
            pdg = treeArr['pdg']
            # Convert to float32

            treeArr = compute_Erec(treeArr)
            treeArr = ak.values_astype(treeArr[variables_out], np.float32)

            data = ak.to_numpy(treeArr)
            data = data.view(np.float32).reshape(
                (len(data), len(variables_out)))

            if compute_leading_momentum:
                # This also has to be refactored : changing the config wont affect this
                leading_momenta = calculate_leading_momentum(pdg, p)
                data = np.concatenate((data, leading_momenta), axis=1)
                
            if return_weights:
                return data, np.expand_dims(weights, axis=1)
            
            if return_dataframe:
                return pd.DataFrame(data, columns=variables_out)


            return data

def load_files(filenames, reweight_variables, return_dataframe=False):
    if return_dataframe:
        return rootfile_to_array(filenames, reweight_variables, return_dataframe=return_dataframe)
    else: 
        raise NotImplementedError
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_histogram(x, bins=100, bin_range=(0,50), density=True, weights=None):
    import torch
    if weights is None:
        weights = torch.ones_like(x)
    return torch.histogram(x, bins=bins, range=bin_range, density=density, weight=weights.cpu())
    
def predict_histogram_weights(nominal, target):
    """
    Args:
        nominal (torch.Tensor): A tensor of nominal events 
        target (torch.Tensor): A tensor of target events (We are trying to reweight: nominal -> target)
        
    Returns:
        weights (torch.Tensor): A tensor containing weights for each event in the nominal tensor
    """
    import torch
    nominal_counts, nominal_edges = compute_histogram(nominal)
    target_counts, _ = compute_histogram(target)
    weights = torch.ones_like(target)
    ratio = target_counts/nominal_counts
    for idx in range(len(nominal_counts)):
        weights = torch.where(torch.logical_and(nominal > nominal_edges[idx], nominal < nominal_edges[idx + 1]), ratio[idx], weights)
    return weights


