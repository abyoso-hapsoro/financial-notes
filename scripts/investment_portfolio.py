import sys
import math
from termcolor import colored

# Set no traceback — only printing the AssertionError during exceptions
setattr(sys, 'tracebacklimit', 0)


class Portfolio:
    """
    A financial portfolio of securities, e.g., assets and stocks.

    Attributes:
        returns (list[float]):
            Expected returns for each security.
        volatilities (list[float]):
            Volatilities (standard deviations) for each security.
        weights (list[float]):
            Portfolio allocation weights for each security.
        correlations (dict[tuple[int, int], float]):
            Pairwise correlation coefficients between securities.
        fp (int):
            Floating precision for output formatting.
        count (int):
            Number of securities in the portfolio.
        max_return_length (int):
            Maximum formatted string length for returns (for display alignment).
        max_volatility_length (int):
            Maximum formatted string length for volatilities (for display alignment).
        max_weight_length (int):
            Maximum formatted string length for weights (for display alignment).
    """

    def __init__(
        self,
        returns: list[float],
        volatilities: list[float] = None,
        weights: list[float] = None,
        correlations: dict[tuple[int, int], float] = None,
        fp: int = 2
    ):
        """
        Initializes the Portfolio with given financial data.

        Args:
            returns (list[float]):
                Expected returns for each security.
            volatilities (list[float], optional):
                Volatilities (standard deviations) for each security.
                Defaults to None, which sets each security volatility to zero.
            weights (list[float], optional):
                Portfolio allocation weights for each security.
                Defaults to None, which sets each security weight to same value which sums to one.
            correlations (dict[tuple[int, int], float], optional):
                Pairwise correlation coefficients between securities.
                Defaults to None, which sets each correlation between securities to zero.
            fp (int, optional):
                Floating precision for output formatting.
                Defaults to 2.
        """
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
        """
        Validate 'returns' argument passed to Portfolio.
        """
        assert isinstance(returns_, list), 'Returns must be a list.'
        assert len(returns_) > 0, 'Returns must be non-empty.'
        for return_ in returns_:
            assert isinstance(return_, float) and 0 <= return_ <= 1, 'Each return must be between 0 and 1 inclusive.'
        self._returns = returns_

    @property
    def volatilities(self):
        return self._volatilities

    @volatilities.setter
    def volatilities(self, volatilities_):
        """
        Validate 'volatilities' argument passed to Portfolio.
        """
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
        """
        Validate 'weights' argument passed to Portfolio.
        """
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
        """
        Validate 'correlations' argument passed to Portfolio.
        """
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
        """
        Validate 'fp' argument passed to Portfolio.
        """
        assert isinstance(fp_, int) and fp_ >= 0, 'Floating precision must be nonnegative integer.'
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

    def calculate_return(self, verbose: bool = True) -> float:
        """
        Calculate expected return of the portfolio.

        Args:
            verbose (bool, optional):
                Whether to print the result. Defaults to True.
        
        Returns:
            float: The portfolio expected return.
        """
        portfolio_return = sum(self.weights[idx] * self.returns[idx] for idx in range(self.count))
        if verbose:
            portfolio_return_str = f"{100 * portfolio_return:.{self.fp}f}%"
            print(f"• Portfolio has expected return of {portfolio_return_str}")
        return portfolio_return

    def calculate_volatility(self, verbose: bool = True) -> float:
        """
        Calculate volatility (standard deviation) of the portfolio.

        Args:
            verbose (bool, optional):
                Whether to print the result. Defaults to True.
        
        Returns:
            float: The portfolio volatility (standard deviation).
        """
        portfolio_variance = sum([(self.weights[idx] * self._volatilities[idx])**2 for idx in range(self.count)])
        for idx_i in range(1, self.count):
            for idx_j in range(idx_i + 1, self.count + 1):
                next_term = 2 * self.correlations[(idx_i, idx_j)]
                next_term *= self.weights[idx_i - 1] * self.weights[idx_j - 1]
                next_term *= self.volatilities[idx_i - 1] * self.volatilities[idx_j - 1]
                portfolio_variance += next_term
        portfolio_volatility = portfolio_variance ** 0.5
        if verbose:
            portfolio_volatility_str = f"{100 * portfolio_volatility:.{self.fp}f}%"
            print(f"• Portfolio has volatility of {portfolio_volatility_str}")
        return portfolio_volatility

    def calculate_variance_reduction(self, verbose: bool = True) -> float:
        """
        Calculate variance reduction (quantitative diversification benefit) of the portfolio.

        Args:
            verbose (bool, optional):
                Whether to print the result. Defaults to True.
        
        Returns:
            float: The portfolio variance reduction (diversification benefit).
        """
        portfolio_variance_max = sum([(self.weights[idx] * self._volatilities[idx])**2 for idx in range(self.count)])
        for idx_i in range(1, self.count):
            for idx_j in range(idx_i + 1, self.count + 1):
                next_term = 2
                next_term *= self.weights[idx_i - 1] * self.weights[idx_j - 1]
                next_term *= self.volatilities[idx_i - 1] * self.volatilities[idx_j - 1]
                portfolio_variance_max += next_term
        portfolio_volatility_max = portfolio_variance_max ** 0.5
        portfolio_volatility = self.calculate_volatility(verbose=False)
        portfolio_vr = portfolio_volatility_max - portfolio_volatility
        if verbose:
            portfolio_vr_str = colored(f"{100 * portfolio_vr:.{self.fp}f}%", 'green')
            print(f"• Portfolio has variance reduction (diversification benefit) of {portfolio_vr_str}")
        return portfolio_vr

    def calculate_coefficient_of_variation(self, idx: int = 0, verbose: bool = True) -> float:
        """
        Calculate coefficient of variation of the portfolio or specific security.

        Args:
            idx (int, optional):
                If 0, calculate for the portfolio. Otherwise, calculate for specific security. Defaults to 0.
            verbose (bool, optional):
                Whether to print the result. Defaults to True.
        
        Returns:
            float: The portfolio or security coefficient of variation.
        """
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
    
    def calculate_sharpe_ratio(self, rf: float, idx: int = 0, verbose: bool = True) -> float:
        """
        Calculate sharpe ratio of the portfolio or specific security.

        Args:
            rf (float):
                The risk-free rate of return.
            idx (int, optional):
                If 0, calculate for the portfolio. Otherwise, calculate for specific security. Defaults to 0.
            verbose (bool, optional):
                Whether to print the result. Defaults to True.
        
        Returns:
            float: The portfolio or security sharpe ratio.
        """
        assert rf >= 0, 'Risk-Free Rate of Return must be nonnegative.'
        rf_str = colored(f"(w/ risk-free rate {100 * rf:.{self.fp}f}%)", 'grey')
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
        (2, 3): 0.2   # security 2 and 3 have correlation of 0.2
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

    # Assessment 3: Calculate portfolio variance reduction (portfolio diversification benefit)
    portfolio.calculate_variance_reduction()

    # Assessment 4: Calculate portfolio and each security's coefficient of variation
    for security in range(portfolio.count + 1):
        portfolio.calculate_coefficient_of_variation(security)

    # Assessment 5: Calculate portfolio and each security's sharpe ratio
    risk_free_rate = 0.01
    for security in range(portfolio.count + 1):
        portfolio.calculate_sharpe_ratio(risk_free_rate, security)
