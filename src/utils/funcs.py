import awkward as ak
import numpy as np
import torch
import uproot

def make_cut(pdg):
    """
    Applying a cut on the final state topology.
    Only apply to argon 14 files.
    """
    proton_muon = ak.any(pdg == 2212, axis=1) & ak.any(pdg == 13, axis=1)
    pions = ak.any(pdg == -211, axis=1) + ak.any(pdg == 111, axis=1) + ak.any(pdg == 211, axis=1)
    multiplicity_cut = (ak.num(pdg, axis=1) < 4)
    cut = proton_muon & multiplicity_cut & pions
    return cut


def rootfile_to_array(filename, return_weights=False, apply_cut=False):
        variables_in, variables_out, m = get_constants()

        with uproot.open(filename + ":FlatTree_VARS") as tree:
            print("Reading {0}".format(filename))
            treeArr = tree.arrays(variables_in)

            if apply_cut:
                print('Applying cut on FS')
                cut = make_cut(treeArr['pdg'])
                treeArr = treeArr[cut]

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
            # Convert to float32
            treeArr = ak.values_astype(treeArr[variables_out], np.float32)

            data = ak.to_numpy(treeArr)
            data = data.view(np.float32).reshape(
                (len(data), len(variables_out)))
            if return_weights:
                data, np.expand_dims(weights, axis=1)
                
            return data

def get_vars_meta(manyBins):
    vars_meta = np.array(
        [
            ["isNu", 2, 0, 1, r"$\nu / \bar{\nu}$ flag"],
            ["isNue", 2, 0, 1, r"$\nu_{e}$ flag"],
            ["isNumu", 2, 0, 1, r"$\nu_{\mu}$ flag"],
            ["isNutau", 2, 0, 1, r"$\nu_{\tau}$ flag"],
            ["cc", 2, 0, 1, "CC flag"],
            ["Enu_true", manyBins, 0, 10, "Neutrino energy [GeV]"],
            ["ELep", manyBins, 0, 5, "Lepton energy [GeV]"],
            ["CosLep", manyBins, -1, 1, r"cos$\theta_{\ell}$"],
            ["Q2", manyBins, 0, 10, r"Q^2"],
            ["W", manyBins, 0, 5, r"W [GeV/$c^{2}$]"],
            ["x", manyBins, 0, 1, "x"],
            ["y", manyBins, 0, 1, "y"],
            ["nP", 15, 0, 15, "Number of protons"],
            ["nN", 15, 0, 15, "Number of neutrons"],
            ["nipip", 10, 0, 10, r"Number of $\pi^{+}$"],
            ["nipim", 10, 0, 10, r"Number of $\pi^{-}$"],
            ["nipi0", 10, 0, 10, r"Number of $\pi^{0}$"],
            ["niem", 10, 0, 10, r"Number of EM objects"],
            ["eP", manyBins - 1, 1.0 / manyBins, 5, "Total proton kinetic energy"],
            ["eN", manyBins - 1, 1.0 / manyBins, 5, "Total neutron kinetic energy"],
            ["ePip", manyBins - 1, 1.0 / manyBins, 5, r"Total $\pi^{+}$ kinetic energy"],
            ["ePim", manyBins - 1, 1.0 / manyBins, 5, r"Total $\pi^{-}$ kinetic energy"],
            ["ePi0", manyBins - 1, 1.0 / manyBins, 5, r"Total $\pi^{0}$ kinetic energy"],
        ]
    ).transpose()
    return vars_meta

def get_constants():
        variables_in = [
            "cc",
            "PDGnu",
            "Enu_true",
            "ELep",
            "CosLep",
            "Q2",
            "W",
            "x",
            "y",
            "nfsp",
            "pdg",
            "E",
            "px",
            "py",
            "pz",
            "Weight",
        ]

        variables_out = [
            "isNu",
            "isNue",
            "isNumu",
            "isNutau",
            "cc",
            "Enu_true",
            "ELep",
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
            "ePi0",
            "pP",
            "pN",
            "pPip",
            "pPim",
            "pPi0"
        ]
        m = {
            "P": 0.93827,
            "N": 0.93957,
            "piC": 0.13957,
            "pi0": 0.13498,
            "kC": 0.49368,
            "k0": 0.49764,
        }

        return variables_in, variables_out, m

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_weights(logits, weight_cap=None, nominal_is_zero=True):
    weights = np.exp(logits)
    probas = sigmoid(logits)
    if not nominal_is_zero:
        print('Nominal generator is not with zero label')
        weights = probas / (1 - probas)
    if weight_cap is not None:
        weights = np.clip(weights, 0, weight_cap)
    return weights, probas

def compute_histogram(x, bins=100, bin_range=(0,50), density=True, weights=None):
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
    nominal_counts, nominal_edges = compute_histogram(nominal)
    target_counts, _ = compute_histogram(target)
    weights = torch.ones_like(target)
    ratio = target_counts/nominal_counts
    for idx in range(len(nominal_counts)):
        weights = torch.where(torch.logical_and(nominal > nominal_edges[idx], nominal < nominal_edges[idx + 1]), ratio[idx], weights)
    return weights


def replace_nan_values(array):
    """
    Replaces nan values within an array
    """
    X = array.copy()
    nan_index = np.isnan(X)
    X[nan_index] = np.random.randn(len(X[nan_index]))
    return X

def get_pdg_codes():
    lepton = [11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16]
    proton = [2212]
    neutron = [2112]
    piplus = [211]
    piminus = [-211]
    pizero = [111]
    kaon = [321, -321, 311, 130, 31]
    return lepton + proton + neutron + piplus + piminus + pizero
