import sys
import math

# Set no traceback — only printing the AssertionError during exceptions
setattr(sys, 'tracebacklimit', 0)


class Portfolio:
    def __init__(self, returns: list[float], weights: list[float] = None, fp: int = 2):
        # Initialize returns then set
        self._returns = None
        self.returns = returns

        # Get number of securities 
        self.count = len(returns)

        # Initialize weights then set
        self._weights = None
        self.weights = weights

        # Initialize floating precision then set
        self._fp = None
        self.fp = fp

        # Determine max lengths for string formatting
        self.max_return_length = max(len(f'{100 * r:.{fp}f}') for r in self.returns)
        self.max_weight_length = max(len(f'{100 * abs(w):.{fp}f}') for w in self.weights)

    @property
    def returns(self):
        return self._returns

    @returns.setter
    def returns(self, returns_):
        assert isinstance(returns_, list), 'Returns must be a list.'
        for return_ in returns_:
            assert isinstance(return_, float) and 0 <= return_ <= 1, 'Each return must be between 0 and 1 inclusive.'
        self._returns = returns_

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights_):
        if weights_ is None:
            self._weights = self.count * [1/self.count]
        else:
            assert isinstance(weights_, list), 'Weights must be a list.'
            for weight_ in weights_:
                assert isinstance(weight_, float), 'Each weight must be a float.'
            assert math.isclose(sum(weights_), 1, rel_tol=1e-9), 'Sum of weights must be exactly 1.'
            self._weights = weights_

    @property
    def fp(self):
        return self._fp

    @fp.setter
    def fp(self, fp_):
        assert isinstance(fp_, int) and fp_ >= 0, 'Returns must be nonnegative integer.'
        self._fp = fp_

    def __repr__(self):
        return f'Portfolio(returns={self.returns}, weights={self.weights})'

    def __str__(self):
        output = f"Portfolio consists of the following {self.count} securities:\n"
        for idx in range(self.count):
            abs_weight_str = f"{100 * abs(self.weights[idx]):.{self.fp}f}".rjust(self.max_weight_length)
            sign = "-" if self.weights[idx] < 0 else " "
            return_str = f"{100 * self.returns[idx]:.{self.fp}f}".rjust(self.max_return_length)
            shorting = " (shorted)" if self.weights[idx] < 0 else ""
            output += f"[{idx + 1}] {sign}{abs_weight_str}% with expected return of {return_str}%{shorting}"
            if idx < self.count - 1:
                output += "\n"
        return output

    def calculate_portfolio_return(self, verbose=True):
        portfolio_return = sum(self.weights[idx] * self.returns[idx] for idx in range(self.count))
        if verbose:
            portfolio_return_str = f"{100 * portfolio_return:.{self.fp}f}"
            print(f"• Portfolio has expected return of {portfolio_return_str}%")
        return portfolio_return


if __name__ == '__main__':
    # Example Usage:
    security_returns = [0.05, 0.11]
    # security_weights = None  # balance weights between each security
    security_weights = [0.6, 0.4]

    # Instantiate portfolio of securities
    portfolio = Portfolio(security_returns, security_weights)

    # Inspect portfolio
    print(portfolio)

    # Begin portfolio assessment
    print('\n\033[1mAssessment:\033[0m')

    # Calculate expected return of portfolio
    portfolio.calculate_portfolio_return()
