import pickle
import tl2cgen
import fire

from treelite import Model
from pathlib import Path


def load_model_from_pkl(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def pkl_to_treelite(model):
    path = 'placeholder.txt'
    model = model.booster_.save_model(path)
    model = Model.load(path, model_format='lightgbm')
    # remove the placeholder file
    Path(path).unlink()
    return model

def pkl_to_C(model_path, output_dir=None):
    model = load_model_from_pkl(model_path)
    model = pkl_to_treelite(model)
    if output_dir is None:
        # use the same directory as the model
        output_dir = Path(model_path).parent

    tl2cgen.generate_c_code(model, dirpath=output_dir, params={})  



if __name__ == "__main__":
    fire.Fire(pkl_to_C)