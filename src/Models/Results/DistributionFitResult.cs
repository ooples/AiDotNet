namespace AiDotNet.Models.Results;

public class DistributionFitResult<T>
{
    private readonly INumericOperations<T> _ops;

    /// <summary>
    /// Gets or sets the type of distribution that best fits the data.
    /// </summary>
    public DistributionType DistributionType { get; set; }

    /// <summary>
    /// Gets or sets the goodness of fit measure. Lower values indicate better fit.
    /// </summary>
    public T GoodnessOfFit { get; set; }

    /// <summary>
    /// Gets or sets the parameters of the fitted distribution.
    /// The keys are parameter names, and the values are the corresponding parameter values.
    /// </summary>
    public Dictionary<string, T> Parameters { get; set; }

    public DistributionFitResult(INumericOperations<T>? ops = null)
    {
        _ops = ops ?? MathHelper.GetNumericOperations<T>();
        GoodnessOfFit = _ops.Zero;
        Parameters = [];
    }
}