namespace AiDotNet.Regression.MixedEffects;

/// <summary>
/// Represents a random effect specification in a mixed-effects model.
/// </summary>
/// <remarks>
/// <para>
/// Random effects model group-level variation around population-level (fixed) effects.
/// They capture correlation within groups and account for unobserved heterogeneity.
/// </para>
/// <para>
/// <b>For Beginners:</b> Random effects are like "adjustments" for each group in your data.
///
/// For example, if studying student test scores across different schools:
/// - Fixed effect: Overall relationship between study time and scores
/// - Random effect: Each school might have its own baseline score level
///
/// Random effects model this group-level variation properly, accounting for the fact that
/// students in the same school are more similar to each other than to students from other schools.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class RandomEffect<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the name of this random effect.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the grouping variable column index.
    /// </summary>
    /// <remarks>
    /// This is the column in the data that identifies which group each observation belongs to.
    /// </remarks>
    public int GroupColumnIndex { get; set; }

    /// <summary>
    /// Gets or sets whether this is a random intercept.
    /// </summary>
    /// <value>Default is true (random intercept model).</value>
    public bool IsRandomIntercept { get; set; } = true;

    /// <summary>
    /// Gets or sets the feature column indices for random slopes.
    /// </summary>
    /// <remarks>
    /// If specified, these features will have random slopes varying by group.
    /// </remarks>
    public int[]? RandomSlopeColumns { get; set; }

    /// <summary>
    /// Gets or sets the estimated random effect coefficients for each group.
    /// </summary>
    /// <remarks>
    /// Keys are group identifiers (as doubles), values are the random effect vector for that group.
    /// </remarks>
    public Dictionary<double, Vector<T>>? GroupCoefficients { get; set; }

    /// <summary>
    /// Gets or sets the variance-covariance matrix of random effects.
    /// </summary>
    /// <remarks>
    /// This matrix captures the variance of random effects and their correlations.
    /// For random intercept only, this is a scalar variance.
    /// For random slopes, this includes covariances between intercept and slopes.
    /// </remarks>
    public Matrix<T>? CovarianceMatrix { get; set; }

    /// <summary>
    /// Gets the number of unique groups in this random effect.
    /// </summary>
    public int NumberOfGroups => GroupCoefficients?.Count ?? 0;

    /// <summary>
    /// Gets the dimension of the random effect (1 for intercept only, more with slopes).
    /// </summary>
    public int Dimension => (IsRandomIntercept ? 1 : 0) + (RandomSlopeColumns?.Length ?? 0);

    /// <summary>
    /// Initializes a new instance of RandomEffect for a random intercept.
    /// </summary>
    /// <param name="name">Name identifying this random effect.</param>
    /// <param name="groupColumnIndex">Column index of the grouping variable.</param>
    public RandomEffect(string name, int groupColumnIndex)
    {
        Name = name;
        GroupColumnIndex = groupColumnIndex;
        IsRandomIntercept = true;
        GroupCoefficients = [];
    }

    /// <summary>
    /// Initializes a new instance of RandomEffect with random slopes.
    /// </summary>
    /// <param name="name">Name identifying this random effect.</param>
    /// <param name="groupColumnIndex">Column index of the grouping variable.</param>
    /// <param name="randomSlopeColumns">Column indices for random slopes.</param>
    /// <param name="includeIntercept">Whether to include random intercept.</param>
    public RandomEffect(string name, int groupColumnIndex, int[] randomSlopeColumns, bool includeIntercept = true)
    {
        Name = name;
        GroupColumnIndex = groupColumnIndex;
        IsRandomIntercept = includeIntercept;
        RandomSlopeColumns = randomSlopeColumns;
        GroupCoefficients = [];
    }

    /// <summary>
    /// Gets the random effect vector for a specific group.
    /// </summary>
    /// <param name="groupId">The group identifier.</param>
    /// <returns>The random effect coefficients for the group, or zero vector if not found.</returns>
    public Vector<T> GetGroupEffect(double groupId)
    {
        if (GroupCoefficients != null && GroupCoefficients.TryGetValue(groupId, out var coeffs))
        {
            return coeffs;
        }
        return new Vector<T>(Dimension);
    }

    /// <summary>
    /// Sets the random effect vector for a specific group.
    /// </summary>
    /// <param name="groupId">The group identifier.</param>
    /// <param name="coefficients">The random effect coefficients.</param>
    public void SetGroupEffect(double groupId, Vector<T> coefficients)
    {
        GroupCoefficients ??= [];
        GroupCoefficients[groupId] = coefficients;
    }
}
