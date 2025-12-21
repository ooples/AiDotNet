namespace AiDotNet.Models;

/// <summary>
/// Defines the search space for hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A search space defines all possible values for each hyperparameter.
/// For example, learning rate might be between 0.001 and 0.1, and batch size might be 16, 32, or 64.
/// </remarks>
public class HyperparameterSearchSpace
{
    /// <summary>
    /// Gets or sets the parameter distributions/ranges.
    /// </summary>
    public Dictionary<string, ParameterDistribution> Parameters { get; set; }

    /// <summary>
    /// Initializes a new instance of the HyperparameterSearchSpace class.
    /// </summary>
    public HyperparameterSearchSpace()
    {
        Parameters = new Dictionary<string, ParameterDistribution>();
    }

    /// <summary>
    /// Adds a continuous (real-valued) parameter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this for decimal numbers like learning rate (0.001) or dropout (0.5).
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when min is greater than or equal to max, or when logScale is true and min is not positive.</exception>
    public void AddContinuous(string name, double min, double max, bool logScale = false)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Parameter name cannot be null or empty.", nameof(name));

        if (min >= max)
            throw new ArgumentException($"Min ({min}) must be less than max ({max}).", nameof(min));

        if (logScale && min <= 0)
            throw new ArgumentException($"Min ({min}) must be positive when using log scale.", nameof(min));

        Parameters[name] = new ContinuousDistribution
        {
            Min = min,
            Max = max,
            LogScale = logScale
        };
    }

    /// <summary>
    /// Adds an integer parameter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this for whole numbers like batch size (32) or number of layers (5).
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when min is greater than or equal to max, or when step is not positive.</exception>
    public void AddInteger(string name, int min, int max, int step = 1)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Parameter name cannot be null or empty.", nameof(name));

        if (min >= max)
            throw new ArgumentException($"Min ({min}) must be less than max ({max}).", nameof(min));

        if (step <= 0)
            throw new ArgumentException($"Step ({step}) must be positive.", nameof(step));

        Parameters[name] = new IntegerDistribution
        {
            Min = min,
            Max = max,
            Step = step
        };
    }

    /// <summary>
    /// Adds a categorical parameter (discrete choices).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Use this when you have specific choices like optimizer type ("adam", "sgd")
    /// or activation function ("relu", "tanh").
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when choices is null or empty.</exception>
    public void AddCategorical(string name, params object[] choices)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Parameter name cannot be null or empty.", nameof(name));

        if (choices == null || choices.Length == 0)
            throw new ArgumentException("At least one choice must be provided.", nameof(choices));

        Parameters[name] = new CategoricalDistribution
        {
            Choices = choices.ToList()
        };
    }

    /// <summary>
    /// Adds a boolean parameter.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when name is null or empty.</exception>
    public void AddBoolean(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Parameter name cannot be null or empty.", nameof(name));

        Parameters[name] = new CategoricalDistribution
        {
            Choices = new List<object> { true, false }
        };
    }
}

/// <summary>
/// Base class for parameter distributions.
/// </summary>
public abstract class ParameterDistribution
{
    /// <summary>
    /// Gets the type of distribution.
    /// </summary>
    public abstract string DistributionType { get; }

    /// <summary>
    /// Samples a value from this distribution.
    /// </summary>
    public abstract object Sample(Random random);
}

/// <summary>
/// Represents a continuous (real-valued) parameter distribution.
/// </summary>
public class ContinuousDistribution : ParameterDistribution
{
    /// <summary>
    /// Gets or sets the minimum value.
    /// </summary>
    public double Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum value.
    /// </summary>
    public double Max { get; set; }

    /// <summary>
    /// Gets or sets whether to use log scale for sampling.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Log scale is useful for parameters that span several orders of magnitude,
    /// like learning rate (0.0001 to 0.1). It ensures you sample evenly across the range.
    /// </remarks>
    public bool LogScale { get; set; }

    /// <inheritdoc/>
    public override string DistributionType => "continuous";

    /// <inheritdoc/>
    public override object Sample(Random random)
    {
        var sample = random.NextDouble();

        if (LogScale)
        {
            var logMin = Math.Log(Min);
            var logMax = Math.Log(Max);
            return Math.Exp(logMin + sample * (logMax - logMin));
        }

        return Min + sample * (Max - Min);
    }
}

/// <summary>
/// Represents an integer parameter distribution.
/// </summary>
public class IntegerDistribution : ParameterDistribution
{
    /// <summary>
    /// Gets or sets the minimum value.
    /// </summary>
    public int Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum value.
    /// </summary>
    public int Max { get; set; }

    /// <summary>
    /// Gets or sets the step size.
    /// </summary>
    public int Step { get; set; } = 1;

    /// <inheritdoc/>
    public override string DistributionType => "integer";

    /// <inheritdoc/>
    /// <remarks>
    /// Samples an integer value from Min to Min + k*Step where k*Step does not exceed (Max - Min).
    /// Note: If (Max - Min) is not evenly divisible by Step, Max itself may not be sampled.
    /// For example, with Min=0, Max=10, Step=3, possible values are 0, 3, 6, 9 (not 10).
    /// To ensure Max is included, choose Step such that (Max - Min) % Step == 0.
    /// </remarks>
    public override object Sample(Random random)
    {
        var numSteps = (Max - Min) / Step;
        var stepIndex = random.Next(0, numSteps + 1);
        return Min + stepIndex * Step;
    }
}

/// <summary>
/// Represents a categorical (discrete choice) parameter distribution.
/// </summary>
public class CategoricalDistribution : ParameterDistribution
{
    /// <summary>
    /// Gets or sets the list of possible choices.
    /// </summary>
    public List<object> Choices { get; set; } = new();

    /// <inheritdoc/>
    public override string DistributionType => "categorical";

    /// <inheritdoc/>
    public override object Sample(Random random)
    {
        var index = random.Next(0, Choices.Count);
        return Choices[index];
    }
}
