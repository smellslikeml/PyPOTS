"""
Test cases for MDTIM imputation model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation import MDTIM
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse, calc_quantile_crps
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestMDTIM(unittest.TestCase):
    logger.info("Running tests for an imputation model MDTIM...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "MDTIM")
    model_save_name = "saved_mdtim_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize an MDTIM model
    mdtim = MDTIM(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        n_bins=40,
        n_layers=1,
        d_model=32,
        n_heads=4,
        d_conditioning=16,
        d_time_embedding=32,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="imputation-mdtim")
    def test_0_fit(self):
        self.mdtim.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-mdtim")
    def test_1_impute(self):
        imputed_X = self.mdtim.predict(TEST_SET, n_sampling_times=2)["imputation"]
        test_CRPS = calc_quantile_crps(imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        imputed_X = imputed_X.mean(axis=1)  # mean over sampling times
        assert not np.isnan(imputed_X).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        logger.info(f"MDTIM test_MSE: {test_MSE}, test_CRPS: {test_CRPS}")

    @pytest.mark.xdist_group(name="imputation-mdtim")
    def test_2_parameters(self):
        assert hasattr(self.mdtim, "model") and self.mdtim.model is not None

        assert hasattr(self.mdtim, "optimizer") and self.mdtim.optimizer is not None

        assert hasattr(self.mdtim, "best_loss")
        self.assertNotEqual(self.mdtim.best_loss, float("inf"))

        assert hasattr(self.mdtim, "best_model_dict") and self.mdtim.best_model_dict is not None

    @pytest.mark.xdist_group(name="imputation-mdtim")
    def test_3_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.mdtim)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.mdtim.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.mdtim.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-mdtim")
    def test_4_lazy_loading(self):
        self.mdtim.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        imputation_results = self.mdtim.predict(GENERAL_H5_TEST_SET_PATH)
        imputed_X = imputation_results["imputation"]
        test_CRPS = calc_quantile_crps(imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        imputed_X = imputed_X.mean(axis=1)  # mean over sampling times
        assert not np.isnan(imputed_X).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        logger.info(f"Lazy-loading MDTIM test_MSE: {test_MSE}, test_CRPS: {test_CRPS}")


if __name__ == "__main__":
    unittest.main()
