"""
Test cases for CoIFNet forecasting model.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import os.path
import unittest

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pypots.forecasting import CoIFNet
from pypots.forecasting.coifnet.data import DatasetForCoIFNet
from pypots.nn.functional import calc_mse
from pypots.optim import Adam
from pypots.utils.logging import logger
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    N_PRED_STEPS,
    FORECASTING_TRAIN_SET,
    FORECASTING_VAL_SET,
    FORECASTING_TEST_SET,
    FORECASTING_H5_TRAIN_SET_PATH,
    FORECASTING_H5_VAL_SET_PATH,
    FORECASTING_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_FORECASTING,
    check_tb_and_model_checkpoints_existence,
)


class TestCoIFNet(unittest.TestCase):
    logger.info("Running tests for a forecasting model CoIFNet...")

    # set the log and model saving path
    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_FORECASTING, "CoIFNet")
    model_save_name = "saved_coifnet_model.pypots"

    # initialize an Adam optimizer
    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    # initialize a CoIFNet model
    coifnet = CoIFNet(
        n_steps=DATA["n_steps"] - N_PRED_STEPS,
        n_features=DATA["n_features"],
        n_pred_steps=N_PRED_STEPS,
        n_pred_features=DATA["n_features"],
        use_reconstruct=True,
        use_mask=True,
        d_model=32,
        dropout=0.1,
        use_reversible_norm=True,
        use_revon=True,
        loss_lambda=0.1,
        epochs=EPOCHS,
        saving_path=saving_path,
        optimizer=optimizer,
        device=DEVICE,
    )

    @pytest.mark.xdist_group(name="forecasting-coifnet")
    def test_0_fit(self):
        self.coifnet.fit(FORECASTING_TRAIN_SET, FORECASTING_VAL_SET)

    @pytest.mark.xdist_group(name="forecasting-coifnet")
    def test_1_forecasting(self):
        forecasting_X = self.coifnet.predict(FORECASTING_TEST_SET)["forecasting"]
        assert not np.isnan(forecasting_X).any(), (
            "Output has missing values in the forecasting results that should not be."
        )
        # the forecasting results must have shape [n_samples, n_pred_steps, n_features]
        assert forecasting_X.shape == FORECASTING_TEST_SET["X_pred"].shape, (
            f"Forecasting results have shape {forecasting_X.shape}, "
            f"but should have shape {FORECASTING_TEST_SET['X_pred'].shape}."
        )
        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"CoIFNet test_MSE: {test_MSE}")

        # CoIFNet is a joint imputation-forecasting model, hence its reconstruction of the
        # observed window should be returned as well and have shape [n_samples, n_steps, n_features]
        reconstruction_X = self.coifnet.predict(FORECASTING_TEST_SET)["reconstruction"]
        assert reconstruction_X.shape == FORECASTING_TEST_SET["X"].shape, (
            f"Reconstruction results have shape {reconstruction_X.shape}, "
            f"but should have shape {FORECASTING_TEST_SET['X'].shape}."
        )
        assert not np.isnan(reconstruction_X).any(), (
            "Output has missing values in the reconstruction results that should not be."
        )

        # the sklearn-style forecast() API should return the same results as predict()["forecasting"]
        forecasting_X_api = self.coifnet.forecast(FORECASTING_TEST_SET)
        assert np.allclose(forecasting_X, forecasting_X_api), (
            "forecast() and predict()['forecasting'] should return the same results."
        )

    @pytest.mark.xdist_group(name="forecasting-coifnet")
    def test_2_training_loss_decreases(self):
        # optimize the model core on a fixed batch for a few steps and assert the joint
        # imputation-forecasting objective decreases, i.e. the ported model actually learns
        data_loader = DataLoader(
            DatasetForCoIFNet(FORECASTING_TRAIN_SET),
            batch_size=128,
            shuffle=False,
        )
        batch = next(iter(data_loader))
        inputs = self.coifnet._assemble_input_for_training(batch)

        # the training data must actually be partially observed, otherwise this test
        # would not exercise the missingness-mask path at all
        missing_rate = 1 - inputs["missing_mask"].mean().item()
        assert 0 < missing_rate < 1, f"the test data should be partially observed, got rate {missing_rate}"

        optimizer = torch.optim.Adam(self.coifnet.model.parameters(), lr=0.001)
        self.coifnet.model.train()
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            results = self.coifnet.model(inputs, calc_criterion=True)
            results["loss"].backward()
            optimizer.step()
            losses.append(results["loss"].item())

        logger.info(f"CoIFNet joint loss on the fixed batch: {losses[0]:.6f} -> {losses[-1]:.6f}")
        assert np.mean(losses[-3:]) < np.mean(losses[:3]), (
            f"training loss should decrease, but went from {np.mean(losses[:3]):.6f} "
            f"to {np.mean(losses[-3:]):.6f} instead."
        )

    @pytest.mark.xdist_group(name="forecasting-coifnet")
    def test_3_parameters(self):
        assert hasattr(self.coifnet, "model") and self.coifnet.model is not None

        assert hasattr(self.coifnet, "optimizer") and self.coifnet.optimizer is not None

        assert hasattr(self.coifnet, "best_loss")
        self.assertNotEqual(self.coifnet.best_loss, float("inf"))

        assert hasattr(self.coifnet, "best_model_dict") and self.coifnet.best_model_dict is not None

    @pytest.mark.xdist_group(name="forecasting-coifnet")
    def test_4_saving_path(self):
        # whether the root saving dir exists, which should be created by save_log_into_tb_file
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"

        # check if the tensorboard file and model checkpoints exist
        check_tb_and_model_checkpoints_existence(self.coifnet)

        # save the trained model into file, and check if the path exists
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.coifnet.save(saved_model_path)

        # test loading the saved model, not necessary, but need to test
        self.coifnet.load(saved_model_path)

    @pytest.mark.xdist_group(name="forecasting-coifnet")
    def test_5_lazy_loading(self):
        self.coifnet.fit(FORECASTING_H5_TRAIN_SET_PATH, FORECASTING_H5_VAL_SET_PATH)
        forecasting_results = self.coifnet.predict(FORECASTING_H5_TEST_SET_PATH)
        forecasting_X = forecasting_results["forecasting"]
        assert not np.isnan(forecasting_X).any(), (
            "Output has missing values in the forecasting results that should not be."
        )

        test_MSE = calc_mse(
            forecasting_X,
            FORECASTING_TEST_SET["X_pred"],
            ~np.isnan(FORECASTING_TEST_SET["X_pred"]),
        )
        logger.info(f"Lazy-loading CoIFNet test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
