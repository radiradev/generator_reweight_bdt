from dataclasses import dataclass

@dataclass
class TrainVariable:
    n_bins: int 
    hist_range: tuple
    reweight_variable: bool = True

def train_var(n_bins, hist_range, train=True):
    return TrainVariable(n_bins, hist_range, train)

def get_train_vars(n_bins):
    return {
        "isNu": train_var(2, (0.0, 1.0)),
        "isNue": train_var(2, (0.0, 1.0)),
        "isNumu": train_var(2, (0.0, 1.0)),
        "isNutau": train_var(2, (0.0, 1.0)),
        "cc": train_var(2, (0.0, 1.0)),
        "Enu_true": train_var(n_bins, (0.0, 10.0)),
        "ELep": train_var(n_bins, (0.0, 5.0)),
        "Eav":  train_var(n_bins, (0.0, 10.0)),
        "CosLep": train_var(n_bins, (-1.0, 1.0)),
        "Q2": train_var(n_bins, (0.0, 10.0)),
        "W": train_var(n_bins, (0.0, 5.0)),
        "x": train_var(n_bins, (0.0, 1.0)),
        "y": train_var(n_bins, (0.0, 1.0)),
        "nP": train_var(15, (0.0, 15.0)),
        "nN": train_var(15, (0.0, 15.0)),
        "nipip": train_var(10, (0.0, 10.0)),
        "nipim": train_var(10, (0.0, 10.0)),
        "nipi0": train_var(10, (0.0, 10.0)),
        "niem": train_var(10, (0.0, 10.0)),
        "eP": train_var(n_bins - 1, (1.0 / n_bins, 5.0)),
        "eN": train_var(n_bins - 1, (1.0 / n_bins, 5.0)),
        "ePip": train_var(n_bins - 1, (1.0 / n_bins, 5.0)),
        "ePim": train_var(n_bins - 1, (1.0 / n_bins, 5.0)),
        "ePi0": train_var(n_bins - 1, (1.0 / n_bins, 5.0)),
    }


def direct_test_vars(n_bins=100):
    return {
    "q0": train_var(n_bins, (0.0, 20.00), False),
    "q3": train_var(n_bins, (0.0, 20.00), False),
    "dpt": train_var(int(n_bins/2), (0.0, 2000.0), False),
    "dalphat": train_var(n_bins, (0.0, 3.0), False),
    "dphit" : train_var(n_bins, (0.0, 3.0), False),
    "pnreco_C": train_var(n_bins, (0.0, 2.0), False),
}

def leading_momenta(n_bins=100):
    return {
        "LeadPP": train_var(n_bins - 1, (1.0 / n_bins, 5.0), False),
        "LeadPN": train_var(n_bins - 1, (1.0 / n_bins, 5.0),  False),
        "LeadPPip": train_var(n_bins - 1, (1.0 / n_bins, 5.0), False),
        "LeadPPim": train_var(n_bins - 1, (1.0 / n_bins, 5.0), False),
        "LeadPPi0": train_var(n_bins - 1, (1.0 / n_bins, 5.0), False),
    }


def get_test_vars(n_bins=100):
    test_vars = {
        "pLep": train_var(n_bins - 1, (1.0 / n_bins, 5.0), False),
        "Erec_bias_rel": train_var(n_bins, (-1.0, 0.0), False),
        "Erec_bias_abs": train_var(n_bins, (-5.0, 0.0), False),
        "Erec": train_var(n_bins, (0.0, 10.0), False),
    }
    test_vars.update(direct_test_vars(n_bins))
    test_vars.update(leading_momenta(n_bins))
    return test_vars

def plots_meta(n_bins=100):
    train_vars = get_train_vars(n_bins)
    test_vars = get_test_vars(n_bins)
    
    # add train_vars
    train_vars.update(test_vars)
    return train_vars



@dataclass
class PairVars:
    feature_names: list
    n_bins: list
    hist_range: list


def pair_vars(feature_names, n_bins, hist_range):
    return PairVars(feature_names, n_bins, hist_range)


def get_pair_variables(pair_bins=30):
    return {
        "dpt_dalphat": pair_vars(["dpt", "dalphat"], [pair_bins, pair_bins], [(0.0, 2000.0), (0.5, 3.0)]),
        "q0_q3": pair_vars(["q0", "q3"], [pair_bins, pair_bins], [(0.0, 30.0), (0.0, 30.0)]),
        "pLep_CosLep" : pair_vars(["pLep", "CosLep"], [pair_bins, pair_bins], [(0.0, 5.0), (0.75, 1.0)]),
        "Q2_W" : pair_vars(["Q2", "W"], [pair_bins, pair_bins], [(1.0, 10.0), (1.0, 5.0)]),
        "x_y" : pair_vars(["x", "y"], [pair_bins, pair_bins], [(0.1, 1.0), (0.1, 1.0)]),
        "Enu_true_Erec" : pair_vars(["Enu_true", "Erec"], [pair_bins, pair_bins], [(0.0, 10.0), (0.0, 5.0)]),
    }




