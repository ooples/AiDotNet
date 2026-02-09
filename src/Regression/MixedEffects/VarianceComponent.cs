namespace AiDotNet.Regression.MixedEffects;

/// <summary>
/// Represents variance components in a mixed-effects model.
/// </summary>
/// <remarks>
/// <para>
/// Variance components partition the total variance in the response into parts
/// attributable to different sources (fixed effects, random effects, residual).
/// </para>
/// <para>
/// <b>For Beginners:</b> Variance components tell you "how much variation comes from where".
///
/// For students in schools:
/// - Random effect variance: How much do schools differ on average?
/// - Residual variance: How much do individual students vary within schools?
///
/// This decomposition is important for:
/// - Understanding your data structure
/// - Computing Intraclass Correlation (ICC)
/// - Assessing if random effects are needed
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VarianceComponent<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the name of this variance component.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the estimated variance for this component.
    /// </summary>
    public T Variance { get; set; } = default!;

    /// <summary>
    /// Gets or sets the standard error of the variance estimate.
    /// </summary>
    public T StandardError { get; set; } = default!;

    /// <summary>
    /// Gets or sets the lower bound of the confidence interval.
    /// </summary>
    public T ConfidenceIntervalLower { get; set; } = default!;

    /// <summary>
    /// Gets or sets the upper bound of the confidence interval.
    /// </summary>
    public T ConfidenceIntervalUpper { get; set; } = default!;

    /// <summary>
    /// Gets or sets the covariance matrix for this variance component.
    /// </summary>
    /// <remarks>
    /// For random slopes, this contains the full variance-covariance matrix.
    /// For random intercept only, this is a 1x1 matrix containing the variance.
    /// </remarks>
    public Matrix<T>? CovarianceMatrix { get; set; }

    /// <summary>
    /// Gets or sets the correlation matrix for this variance component.
    /// </summary>
    /// <remarks>
    /// For random slopes, this contains correlations between random effects.
    /// </remarks>
    public Matrix<T>? CorrelationMatrix { get; set; }

    /// <summary>
    /// Gets the standard deviation (square root of variance).
    /// </summary>
    public T StandardDeviation => NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(Variance)));

    /// <summary>
    /// Initializes a new variance component.
    /// </summary>
    /// <param name="name">Name identifying this variance component.</param>
    public VarianceComponent(string name)
    {
        Name = name;
        Variance = NumOps.Zero;
        StandardError = NumOps.Zero;
        ConfidenceIntervalLower = NumOps.Zero;
        ConfidenceIntervalUpper = NumOps.Zero;
    }

    /// <summary>
    /// Computes the proportion of total variance explained by this component.
    /// </summary>
    /// <param name="totalVariance">The total variance across all components.</param>
    /// <returns>Proportion between 0 and 1.</returns>
    public T GetVarianceProportion(T totalVariance)
    {
        if (NumOps.ToDouble(totalVariance) <= 0)
        {
            return NumOps.Zero;
        }
        return NumOps.Divide(Variance, totalVariance);
    }
}

/// <summary>
/// Contains the full variance decomposition results from a mixed model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VarianceDecomposition<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the residual (within-group) variance.
    /// </summary>
    public VarianceComponent<T> ResidualVariance { get; set; }

    /// <summary>
    /// Gets or sets the random effect variance components.
    /// </summary>
    public List<VarianceComponent<T>> RandomEffectVariances { get; set; }

    /// <summary>
    /// Gets the total variance (sum of all components).
    /// </summary>
    public T TotalVariance
    {
        get
        {
            T total = ResidualVariance.Variance;
            foreach (var re in RandomEffectVariances)
            {
                total = NumOps.Add(total, re.Variance);
            }
            return total;
        }
    }

    /// <summary>
    /// Computes the Intraclass Correlation Coefficient (ICC) for a specific random effect.
    /// </summary>
    /// <param name="randomEffectIndex">Index of the random effect.</param>
    /// <returns>ICC value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// ICC measures the proportion of variance attributable to a grouping factor.
    /// It represents the correlation between observations in the same group.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ICC tells you how similar observations are within groups.
    /// - ICC near 0: Observations in the same group are not more similar than others
    /// - ICC near 1: Observations in the same group are very similar
    ///
    /// For students in schools, ICC answers: "How correlated are test scores for
    /// students in the same school?"
    /// </para>
    /// </remarks>
    public T ComputeICC(int randomEffectIndex = 0)
    {
        if (randomEffectIndex < 0 || randomEffectIndex >= RandomEffectVariances.Count)
        {
            return NumOps.Zero;
        }

        return RandomEffectVariances[randomEffectIndex].GetVarianceProportion(TotalVariance);
    }

    /// <summary>
    /// Gets variance proportions for all components.
    /// </summary>
    /// <returns>Dictionary mapping component names to their variance proportions.</returns>
    public Dictionary<string, T> GetVarianceProportions()
    {
        var proportions = new Dictionary<string, T>();
        T total = TotalVariance;

        proportions["Residual"] = ResidualVariance.GetVarianceProportion(total);

        foreach (var re in RandomEffectVariances)
        {
            proportions[re.Name] = re.GetVarianceProportion(total);
        }

        return proportions;
    }

    /// <summary>
    /// Initializes a new variance decomposition.
    /// </summary>
    public VarianceDecomposition()
    {
        ResidualVariance = new VarianceComponent<T>("Residual");
        RandomEffectVariances = [];
    }
}
