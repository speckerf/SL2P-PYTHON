import numpy as np
from loguru import logger


class LeafToolbox_MLPRegressor:
    def __init__(self, net):
        # load params from json dict
        self.inp_slope = np.array(net["inp_slope"])
        self.inp_offset = np.array(net["inp_offset"])
        self.h1wt = np.array(net["h1wt"]).reshape(
            len(net["h1bi"]), len(self.inp_offset)
        )
        self.h1bi = np.array(net["h1bi"])
        self.h2wt = np.array(net["h2wt"]).reshape(1, len(self.h1bi))
        self.h2bi = np.array(net["h2bi"])
        self.out_slope = np.array(net["out_slope"])
        self.out_bias = np.array(net["out_bias"])
        self.bandorder = net["bandorder"]

        logger.warning(
            f"Make sure that the input data is ordered as in bandorder: {self.bandorder}"
        )

    def _tansig(self, x):
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

    def predict(self, X):
        """
        X: (n_samples, n_features)
        returns: (n_samples,)
        """
        # input scaling
        Xs = X * self.inp_slope + self.inp_offset

        # hidden layer
        h1 = self._tansig(np.dot(Xs, self.h1wt.T) + self.h1bi)

        # linear output layer
        h2 = np.dot(h1, self.h2wt.T) + self.h2bi

        # output scaling
        y = (h2 - self.out_bias) / self.out_slope
        return y.ravel()
