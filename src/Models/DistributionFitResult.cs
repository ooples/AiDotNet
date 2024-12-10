namespace AiDotNet.Models;

public class DistributionFitResult
{
    /// <summary>
    /// Gets or sets the type of distribution that best fits the data.
    /// </summary>
    public DistributionType BestFitDistribution { get; set; }

    /// <summary>
    /// Gets or sets the goodness of fit measure. Lower values indicate better fit.
    /// </summary>
    public double GoodnessOfFit { get; set; }

    /// <summary>
    /// Gets or sets the parameters of the fitted distribution.
    /// The keys are parameter names, and the values are the corresponding parameter values.
    /// </summary>
    public Dictionary<string, double> Parameters { get; set; }

    public DistributionFitResult()
    {
        Parameters = [];
    }
}