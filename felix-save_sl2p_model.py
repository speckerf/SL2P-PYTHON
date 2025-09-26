import json
import os
import pickle
import zipfile

import numpy as np
from loguru import logger

from felix_model import LeafToolbox_MLPRegressor
from tools import SL2P


def load_snap_test_data():
    def read_csv_like(z, name, sep=","):
        with z.open(name) as f:
            txt = f.read().decode("utf-8").strip()
        rows = [
            list(map(float, line.strip().split(sep)))
            for line in txt.splitlines()
            if line.strip()
        ]
        return np.array(rows, dtype=np.float32)

    # read test set from here:
    path = os.path.join("..", "test_s2lp", "parametrisation_ann.zip")  # here in the

    with zipfile.ZipFile(path, "r") as z:
        norm = read_csv_like(z, "LAI_Normalisation")
        denorm = read_csv_like(z, "LAI_Denormalisation")
        l1_b = read_csv_like(z, "LAI_Weights_Layer1_Bias").ravel()
        l1_w = read_csv_like(z, "LAI_Weights_Layer1_Neurons").T
        l2_b = read_csv_like(z, "LAI_Weights_Layer2_Bias").ravel()
        l2_w = read_csv_like(z, "LAI_Weights_Layer2_Neurons").T
        tests = read_csv_like(z, "LAI_TestCases")

    s2_inp, lai_label = (
        np.moveaxis(tests[:, 0:11].reshape(10, 10, 11), -1, 0),
        tests[:, 11].reshape(10, 10),
    )
    return s2_inp, lai_label


def leaf_toolbox_model_extraction():
    """
    Extracts SL2P Leaf Toolbox models, serializes them, and validates consistency.

    Workflow:
    1. Runs the default SL2P Leaf Toolbox implementation for each vegetation variable
       ("LAI", "fAPAR", "fCOVER") using Sentinel-2 input reflectance and geometry bands.
    2. Saves the model estimation and uncertainty parameters to JSON files.
    3. Loads the parameters into a custom LeafToolbox_MLPRegressor to replicate the model.
    4. Compares predictions from the JSON-based models against the original Leaf Toolbox
       outputs to ensure numerical equivalence.
    5. Serializes both the estimation and uncertainty models into a pickle file for reuse.
    6. Reloads the models from pickle and re-validates that predictions remain consistent.

    Assertions:
    - Ensures per-pixel estimates and uncertainties from the JSON/pickle models match
      the Leaf Toolbox outputs within a tolerance (1e-5).

    Side effects:
    - Writes JSON parameter files and pickle model files to the specified folder.
    - Logs progress and validation success.

    This function is primarily intended for model translation, reproducibility checks,
    and archiving of SL2P Leaf Toolbox models outside the SNAP/GEE environment.
    """

    for variableName in ["LAI", "fAPAR", "fCOVER"]:
        s2_inp, true_label = load_snap_test_data()

        # band order test data
        band_order_test_data = [
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "cosVZA",
            "cosSZA",
            "cosRAA",
        ]

        band_order_sl2p = [
            "cosVZA",
            "cosSZA",
            "cosRAA",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
        ]

        # reorder input data to fit SL2P band order
        s2_inp_reordered = s2_inp[
            [band_order_test_data.index(b) for b in band_order_sl2p], :, :
        ]

        ## test forward with default LEAF toolbox implementation
        imageCollectionName = "S2_SR"
        varmap = SL2P.SL2P(s2_inp_reordered, variableName, imageCollectionName)

        folder = "felix-params"
        estimate_json = (
            f"{folder}/{variableName}-estimation_SL2P_Corrected_LeafToolBox.json"
        )
        uncertainty_json = (
            f"{folder}/{variableName}-uncertainty_SL2P_Corrected_LeafToolBox.json"
        )

        with open(estimate_json) as f:
            estimate_params = json.load(f)

        with open(uncertainty_json) as f:
            uncertainty_params = json.load(f)

        model_estimate = LeafToolbox_MLPRegressor(estimate_params)
        model_uncertainty = LeafToolbox_MLPRegressor(uncertainty_params)

        ## compare with MLPRegressor from json
        var_est = model_estimate.predict(
            np.moveaxis(s2_inp_reordered, 0, -1).reshape(-1, s2_inp_reordered.shape[0])
        ).reshape(s2_inp_reordered.shape[1], s2_inp_reordered.shape[2])

        var_unc = model_uncertainty.predict(
            np.moveaxis(s2_inp_reordered, 0, -1).reshape(-1, s2_inp_reordered.shape[0])
        ).reshape(s2_inp_reordered.shape[1], s2_inp_reordered.shape[2])

        ## assert that results are similar
        assert np.allclose(varmap[variableName], var_est, atol=1e-5)
        assert np.allclose(varmap[f"{variableName}_uncertainty"], var_unc, atol=1e-5)

        logger.info(
            "Test from json passed successfully! Both models give the same results."
        )

        # save to SL2P-MODELS folder / save as pickle
        model_folder = "SL2P_MODELS"
        model_filename = f"{model_folder}/SL2P_Corrected_LeafToolBox_{variableName}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(
                {
                    "estimate_model": model_estimate,
                    "uncertainty_model": model_uncertainty,
                },
                f,
            )
        logger.info(f"Saved models to {model_filename}")

        # load from pickle and test again
        with open(model_filename, "rb") as f:
            models = pickle.load(f)
        model_estimate_loaded = models["estimate_model"]
        model_uncertainty_loaded = models["uncertainty_model"]
        var_est_loaded = model_estimate_loaded.predict(
            np.moveaxis(s2_inp_reordered, 0, -1).reshape(-1, s2_inp_reordered.shape[0])
        ).reshape(s2_inp_reordered.shape[1], s2_inp_reordered.shape[2])
        var_unc_loaded = model_uncertainty_loaded.predict(
            np.moveaxis(s2_inp_reordered, 0, -1).reshape(-1, s2_inp_reordered.shape[0])
        ).reshape(s2_inp_reordered.shape[1], s2_inp_reordered.shape[2])
        assert np.allclose(varmap[variableName], var_est_loaded, atol=1e-5)
        assert np.allclose(
            varmap[f"{variableName}_uncertainty"], var_unc_loaded, atol=1e-5
        )
        logger.info(
            "Test from pickle passed successfully! Both models give the same results."
        )


if __name__ == "__main__":
    leaf_toolbox_model_extraction()
