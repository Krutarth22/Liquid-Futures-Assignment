import numpy as np
from sklearn import svm


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):

    def predict(momentum, CLOSE, lookback, gap, dimension):
        X = np.concatenate([momentum[i:i + dimension] for i in range(lookback - gap - dimension)], axis=1).T
        y = np.sign((CLOSE[dimension+gap:] - CLOSE[dimension+gap-1:-1]).T[0])
        y[y==0] = 1

        clf = svm.SVC()
        clf.fit(X, y)

        return clf.predict(momentum[-dimension:].T)

    nMarkets = len(settings['markets'])
    lookback = settings['lookback']
    dimension = settings['dimension']
    gap = settings['gap']

    pos = np.zeros((1, nMarkets), dtype=np.float)

    momentum = (CLOSE[gap:, :] - CLOSE[:-gap, :]) / CLOSE[:-gap, :]

    for market in range(nMarkets):
        try:
            pos[0, market] = predict(momentum[:, market].reshape(-1, 1),
                                     CLOSE[:, market].reshape(-1, 1),
                                     lookback,
                                     gap,
                                     dimension)
        except ValueError:
            pos[0, market] = .0
    return pos, settings


def mySettings():
    """ Define your trading system settings here """

    settings = {}

    # Futures Contracts
    settings['markets'] = ['F_US']

    settings['lookback'] = 126
    settings['budget'] = 10 ** 6
    settings['slippage'] = 0.05

    settings['gap'] = 20
    settings['dimension'] = 5

    return settings


if __name__ == '__main__':
    from quantiacsToolbox.quantiacsToolbox import runts

    results = runts(__file__)