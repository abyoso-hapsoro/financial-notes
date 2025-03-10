import sys
import math

# Set no traceback — only printing the AssertionError during exceptions
setattr(sys, 'tracebacklimit', 0)


class Portfolio:
    def __init__(
        self,
        returns: list[float],
        volatilities: list[float] = None,
        weights: list[float] = None,
        correlations: dict[tuple[int, int], float] = None,
        fp: int = 2
    ):
        # Initialize returns then set
        self._returns = None
        self.returns = returns

        # Get number of securities 
        self.count = len(returns)
        
        # Initialize volatilities then set
        self._volatilities = None
        self.volatilities = volatilities

        # Initialize weights then set
        self._weights = None
        self.weights = weights

        # Initialize correlations then set
        self._correlations = None
        self.correlations = correlations

        # Initialize floating precision then set
        self._fp = None
        self.fp = fp

        # Determine max lengths for string formatting
        self.max_return_length = max(len(f'{100 * r:.{fp}f}') for r in self.returns)
        self.max_volatility_length = max(len(f'{100 * v:.{fp}f}') for v in self.volatilities)
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
    def volatilities(self):
        return self._volatilities

    @volatilities.setter
    def volatilities(self, volatilities_):
        if volatilities_ is None:
            self._volatilities = self.count * [0]
        else:
            assert isinstance(volatilities_, list), 'Volatilities must be a list.'
            for volatility_ in volatilities_:
                assert isinstance(volatility_, float), 'Each volatility must be float.'
            self._volatilities = volatilities_

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
                assert isinstance(weight_, float), 'Each weight must be float.'
            assert math.isclose(sum(weights_), 1, rel_tol=1e-9), 'Sum of weights must be exactly 1.'
            self._weights = weights_
    
    @property
    def correlations(self):
        return self._correlations

    @correlations.setter
    def correlations(self, correlations_):
        if correlations_ is None:
            correlations_ = {}
            for idx_i in range(1, self.count):
                for idx_j in range(idx_i + 1, self.count + 1):
                    correlations_[(idx_i, idx_j)] = 0.
            self._correlations = correlations_
        else:
            assert isinstance(correlations_, dict), 'Correlations must be a dictionary.'
            for correlation_ in correlations_:
                assert isinstance(correlation_, tuple), 'Each correlation key must be tuple.'
                assert len(correlation_) == 2, 'Each correlation key tuple must have exactly 2 elements.'
                for idx in range(2):
                    assert isinstance(correlation_[idx], int), 'Each correlation key tuple element must be integer.'
                    assert 1 <= correlation_[idx] <= self.count, \
                        f'Each correlation key tuple element must be positive integer (maximum of {self.count}).'
                assert correlation_[0] != correlation_[1], 'Correlation key tuple elements must be distinct.'
                assert isinstance(correlations_[correlation_], float), 'Each correlation value must be float.'
                assert -1 <= correlations_[correlation_] <= 1, 'Each correlation value must be between -1 and 1 inclusive.'
            self._correlations = correlations_

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
            volatility_str = f"{100 * self.volatilities[idx]:.{self.fp}f}".rjust(self.max_volatility_length)
            shorting = " (shorted)" if self.weights[idx] < 0 else ""
            output += f"    [{idx + 1}] {sign}{abs_weight_str}%"
            output += f" with expected return of {return_str}%"
            output += f" and volatility of {volatility_str}%"
            output += f"{shorting}\n"
        output += f"Correlation between securities:\n"
        for idx_i in range(1, self.count):
            for idx_j in range(idx_i + 1, self.count + 1):
                correlation_str = f"{self.correlations[(idx_i, idx_j)]:.{self.fp}f}".rjust(self.fp + 3)
                output += f"• ρ([{idx_i}], [{idx_j}]) = {correlation_str}"
                if idx_i < self.count - 1:
                    output += "\n"
        return output

    def calculate_return(self, verbose=True) -> float:
        portfolio_return = sum(self.weights[idx] * self.returns[idx] for idx in range(self.count))
        if verbose:
            portfolio_return_str = f"{100 * portfolio_return:.{self.fp}f}"
            print(f"• Portfolio has expected return of {portfolio_return_str}%")
        return portfolio_return

    def calculate_volatility(self, verbose=True) -> float:
        portfolio_variance = sum([(self.weights[idx] * self._volatilities[idx])**2 for idx in range(self.count)])
        for idx_i in range(1, self.count):
            for idx_j in range(idx_i + 1, self.count + 1):
                next_term = 2 * self.correlations[(idx_i, idx_j)]
                next_term *= self.weights[idx_i - 1] * self.weights[idx_j - 1]
                next_term *= self.volatilities[idx_i - 1] * self.volatilities[idx_j - 1]
                portfolio_variance += next_term
        portfolio_volatility = portfolio_variance ** 0.5
        if verbose:
            portfolio_volatility_str = f"{100 * portfolio_volatility:.{self.fp}f}"
            print(f"• Portfolio has volatility of {portfolio_volatility_str}%")
        return portfolio_volatility

    def calculate_coefficient_of_variation(self, idx=0, verbose=True) -> float:
        if idx == 0:
            portfolio_return = self.calculate_return(verbose=False)
            portfolio_volatility = self.calculate_volatility(verbose=False)
            portfolio_cv = portfolio_volatility / portfolio_return
            if verbose:
                portfolio_cv_str = f"{portfolio_cv:.{self.fp}f}"
                print(f"• Portfolio has coefficient of variation of {portfolio_cv_str}")
            return portfolio_cv
        else:
            security_return = self.returns[idx - 1]
            security_volatility = self.volatilities[idx - 1]
            security_cv = security_volatility / security_return
            if verbose:
                security_cv_str = f"{security_cv:.{self.fp}f}"
                print(f"  Security [{idx}] has coefficient of variation of {security_cv_str}")
            return security_cv
    
    def calculate_sharpe_ratio(self, rf, idx=0, verbose=True) -> float:
        assert rf >= 0, 'Risk-Free Rate of Return must be nonnegative.'
        rf_str = f"(w/ risk-free rate {100 * rf:.{self.fp}f}%)"
        if idx == 0:
            portfolio_return = self.calculate_return(verbose=False)
            portfolio_volatility = self.calculate_volatility(verbose=False)
            portfolio_sr = (portfolio_return - rf) / portfolio_volatility
            if verbose:
                portfolio_sr_str = f"{portfolio_sr:.{self.fp}f}"
                print(f"• Portfolio has sharpe ratio of {portfolio_sr_str}    {rf_str}")
            return portfolio_sr
        else:
            security_return = self.returns[idx - 1]
            security_volatility = self.volatilities[idx - 1]
            security_sr = (security_return - rf) / security_volatility
            if verbose:
                security_sr_str = f"{security_sr:.{self.fp}f}"
                print(f"  Security [{idx}] has sharpe ratio of {security_sr_str} {rf_str}")
            return security_sr


if __name__ == '__main__':
    # Example Usage:
    security_returns = [0.05, 0.06, 0.03]
    security_volatilities = [0.2, 0.1, 0.3]
    # security_weights = None  # balance weights between each security
    security_weights = [0.3, 0.3, 0.4]
    security_correlations = {
        (1, 2): 0.3,  # security 1 and 2 have correlation of 0.3
        (1, 3): 0.5,  # security 1 and 3 have correlation of 0.5
        (2, 3): 0.2,  # security 2 and 3 have correlation of 0.2
    }
    floating_precision = 3

    # Instantiate portfolio of securities
    portfolio = Portfolio(
        security_returns,
        security_volatilities,
        security_weights,
        security_correlations,
        floating_precision
    )

    # Inspect portfolio
    print(portfolio)

    # Begin portfolio assessment
    print('\n\033[1mAssessment:\033[0m')

    # Assessment 1: Calculate portfolio expected return
    portfolio.calculate_return()

    # Assessment 2: Calculate portfolio volatility (portfolio standard deviation)
    portfolio.calculate_volatility()

    # Assessment 3: Calculate portfolio and each security's coefficient of variation
    for security in range(portfolio.count + 1):
        portfolio.calculate_coefficient_of_variation(security)

    # Assessment 4: Calculate portfolio and each security's sharpe ratio
    risk_free_rate = 0.01
    for security in range(portfolio.count + 1):
        portfolio.calculate_sharpe_ratio(risk_free_rate, security)
