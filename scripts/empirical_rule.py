import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Set random seed
np.random.seed(123)


def generate_samples(
    mu: float,
    sigma: float,
    n: int = 1000,
    random: bool = True
) -> np.ndarray:
    """
    Generate n sample data points from the distribution N(mu, sigma**2).

    Args:
        mu (float):
            Mean of the normal distribution.
        sigma (float):
            Standard deviation of the normal distribution.
        n (int, optional):
            Number of samples to generate. Defaults to 1000.
        random (bool, optional):
            Whether generated samples are random or evenly spaced. Defaults to True.
    
    Returns:
        np.ndarray: Array of generated sample data.
    """

    if random:
        # Generate n random sample data points
        return np.sort(np.random.normal(mu, sigma, n))
    else:
        # Generate n fixed sample data points
        return np.linspace(mu - 4*sigma, mu + 4*sigma, n)


def plot_empirical_rule(
    X: np.ndarray,
    mu: float,
    sigma: float,
    save_path: str = None,
    verbose: bool = False
) -> None:
    """
    Plot the Empirical Rule for the distribution N(mu, sigma**2).

    Args:
        X (np.ndarray):
            Array of sample data.
        mu (float):
            Mean of the normal distribution.
        sigma (float):
            Standard deviation of the normal distribution.
        save_path (str, optional):
            Path to save the image. Defaults to None.
        verbose (bool, optional):
            Whether to print information. Defaults to False.
    """

    # Calculate the PDF (Probability Density Function)
    y = norm.pdf(X, mu, sigma)
    
    # Print details of the distribution
    if verbose:
        print(f'X = {len(X)} samples, X ~ N({mu}, {sigma**2})')

    # Plot the normal distribution curve
    plt.figure(figsize=(10, 6))
    plt.plot(X, y, label=f'N({mu}, {sigma**2})', color='red')
    plt.axvline(mu, color='black', linestyle='--', linewidth=1, label=f'μ = {mu}')
    
    # Highlight observed samples which fall under 68%, 95% and 99.7% of the overall samples
    regions = [1, 2, 3]
    labels = ['1σ (68.0%)', '2σ (95.0%)', '3σ (99.7%)']
    colors = ['blue', 'cyan', 'lightgreen']
    for idx, region in enumerate(regions):
        left_border, right_border = mu - region*sigma, mu + region*sigma
        region_area = (X >= left_border) & (X <= right_border)
        X_fill, y_fill = X[region_area], y[region_area]
        plt.fill_between(X_fill, y_fill, label=labels[idx], color=colors[idx], alpha=1 - idx*0.38)
        if verbose:
            # Calculate observed samples which fall under the distribution curve within the region
            auc_of_region = -norm.cdf(left_border, mu, sigma) + norm.cdf(right_border, mu, sigma)
            auc_of_region_pct = round(100 * auc_of_region, 1)
            print(f'Observed sample ({region}σ): {auc_of_region_pct}%')

    # Add labels, legend, and grid
    plt.title('Empirical Rule / 68-95-99.7 Rule', fontsize=14)
    plt.xlabel('Standard Deviations from the Mean', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)

    # Save plot to path if specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        if verbose:
            print(f"Plot saved to '{save_path}.'")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Example Usage:
    mean = 0
    std_dev = 1
    result_path = './images/empirical_rule.png'

    # Generate samples
    data = generate_samples(mean, std_dev, random=False)

    # Plot the empirical rule on the samples
    plot_empirical_rule(data, mean, std_dev, save_path=result_path, verbose=True)
