namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Quantile Regression, a technique that enables prediction of specific
/// quantiles of the conditional distribution rather than just the conditional mean.
/// </summary>
/// <remarks>
/// <para>
/// Quantile Regression extends traditional regression methods by estimating conditional quantiles
/// of the response variable. While standard regression estimates the conditional mean E(Y|X),
/// Quantile Regression can estimate any conditional quantile Q(α|X) for α ∈ (0,1), including
/// medians (α = 0.5) and other percentiles. This technique provides a more comprehensive view of the
/// relationship between variables, allowing for the analysis of the full conditional distribution.
/// It is particularly valuable when the conditional distribution is non-Gaussian, skewed, or when
/// outliers are present. Quantile Regression is also robust to heteroscedasticity (non-constant variance)
/// and can reveal how different parts of the distribution respond differently to predictor variables.
/// </para>
/// <para><b>For Beginners:</b> Quantile Regression helps predict specific percentiles of possible outcomes, not just the average outcome.
/// 
/// Think about salary predictions:
/// - Regular regression might tell you "the average salary for this job is $75,000"
/// - But Quantile Regression could tell you:
///   - "10% of people in this job earn less than $50,000" (10th percentile)
///   - "Half of people in this job earn less than $70,000" (median or 50th percentile)
///   - "90% of people in this job earn less than $120,000" (90th percentile)
/// 
/// What this technique does:
/// - It focuses on specific slices of the data distribution
/// - Instead of minimizing squared errors (as in mean regression)
/// - It minimizes a different loss function that depends on which quantile you want
/// - This gives you insight into different parts of the outcome distribution
/// 
/// This is especially useful when:
/// - The outcomes aren't evenly distributed around the average
/// - You're interested in extreme cases (very high or low values)
/// - Different factors might affect different parts of the distribution differently
/// - You want to understand risk or uncertainty better
/// 
/// For example, in healthcare, knowing that a treatment reduces the risk of severe complications
/// (the high quantile) is different information than knowing it reduces the average symptom severity.
///
/// This class lets you configure how the quantile regression algorithm operates.
/// </para>
/// </remarks>
public class QuantileRegressionOptions<T> : RegressionOptions<T>
{
    private double _quantile = 0.5;
    private long _maximumDenseLinearProgramEntries = 50_000_000;
    private SimplexSolverOptions _solverOptions = new();

    /// <summary>
    /// Initializes quantile-regression options with production-safe defaults.
    /// </summary>
    public QuantileRegressionOptions()
    {
    }

    /// <summary>
    /// Initializes an independent copy of another option set.
    /// </summary>
    /// <param name="other">The options to copy.</param>
    public QuantileRegressionOptions(QuantileRegressionOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        DecompositionMethod = other.DecompositionMethod;
        UseIntercept = other.UseIntercept;
        Quantile = other.Quantile;
        SolverOptions = new SimplexSolverOptions(other.SolverOptions);
        MaximumDenseLinearProgramEntries = other.MaximumDenseLinearProgramEntries;
    }

    /// <summary>
    /// Gets or sets the quantile to be estimated by the regression model.
    /// </summary>
    /// <value>The quantile value between 0 and 1, defaulting to 0.5 (median regression).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines which quantile of the conditional distribution the model will estimate.
    /// The quantile must be a value between 0 and 1, where 0.5 represents the median (50th percentile),
    /// 0.9 represents the 90th percentile, 0.1 represents the 10th percentile, and so on. Setting this
    /// value to 0.5 (the default) results in median regression, which is more robust to outliers than
    /// mean regression. Lower values (e.g., 0.1) focus on the lower tail of the distribution, while
    /// higher values (e.g., 0.9) focus on the upper tail. Different quantiles can reveal how the
    /// relationship between predictors and the response variable changes across the distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls which percentile of the data you want to predict.
    /// 
    /// The default value of 0.5 means:
    /// - You're trying to predict the median (middle value)
    /// - Half of outcomes will likely be above your prediction
    /// - Half of outcomes will likely be below your prediction
    /// 
    /// Think of it like height predictions:
    /// - Quantile 0.5 (median): The height where half of people are taller and half are shorter
    /// - Quantile 0.9 (90th percentile): The height where only 10% of people are taller
    /// - Quantile 0.1 (10th percentile): The height where 90% of people are taller
    /// 
    /// You might choose different quantiles for different purposes:
    /// - 0.5 for a typical or central prediction (median)
    /// - 0.9 or higher when you're concerned about upper limits or worst-case scenarios
    /// - 0.1 or lower when you're concerned about lower limits or best-case scenarios
    /// - Multiple quantiles (running the model multiple times) to get a full picture of possibilities
    /// 
    /// For example, in flood risk modeling, the 0.99 quantile might tell you the water level that
    /// has only a 1% chance of being exceeded - critical information for safety planning.
    /// </para>
    /// </remarks>
    public double Quantile
    {
        get => _quantile;
        set => _quantile = value > 0 && value < 1 && !double.IsNaN(value)
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Quantile must be finite and strictly between 0 and 1.");
    }

    /// <summary>
    /// Gets or sets the simplex solver configuration used by the exact linear-program formulation.
    /// </summary>
    /// <value>An independent simplex configuration, defaulting to the standard solver settings.</value>
    /// <remarks>
    /// <para>
    /// Quantile regression is solved exactly as a linear program. These settings therefore describe
    /// simplex pivots and numerical tolerance directly instead of exposing a misleading gradient-descent
    /// learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> The defaults are appropriate for most data. Advanced users can
    /// change how long the exact solver may work and how close to zero a number must be before it
    /// is treated as numerical noise.</para>
    /// </remarks>
    public SimplexSolverOptions SolverOptions
    {
        get => _solverOptions;
        set => _solverOptions = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>
    /// Gets or sets the maximum number of scalar entries allowed in the dense linear-program matrix.
    /// </summary>
    /// <value>The entry budget, defaulting to 50,000,000 (about 400 MB for <c>double</c>).</value>
    /// <remarks>
    /// <para>
    /// The exact Koenker-Bassett formulation has <c>n</c> constraints and approximately <c>2n</c>
    /// variables, so a dense representation grows quadratically with the row count. This explicit budget
    /// prevents an accidental multi-gigabyte allocation and fails before memory pressure destabilizes the
    /// process. Increase it only when the host has enough memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is a memory safety rail. If a dataset is too large for the
    /// exact dense solver, training explains the required size and stops cleanly instead of risking
    /// an out-of-memory crash.</para>
    /// </remarks>
    public long MaximumDenseLinearProgramEntries
    {
        get => _maximumDenseLinearProgramEntries;
        set => _maximumDenseLinearProgramEntries = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "The dense linear-program entry budget must be positive.");
    }
}
