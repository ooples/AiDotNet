namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Beta Regression models.
/// </summary>
/// <remarks>
/// <para>
/// Beta Regression is used when your target variable is a proportion or rate bounded
/// between 0 and 1 (exclusive). Examples include percentages, rates, and proportions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular regression can predict any number, but what if you're
/// predicting something that must be between 0 and 1, like:
/// - Percentage of students passing an exam
/// - Proportion of defective products
/// - Probability estimates
/// - Market share percentages
///
/// Beta Regression handles this naturally by modeling the data as following a Beta
/// distribution, which is naturally bounded between 0 and 1.
///
/// Key benefits:
/// - Predictions are always in valid range (0,1)
/// - Can model varying precision (some predictions more certain than others)
/// - Handles skewed proportions well
/// </para>
/// </remarks>
public class BetaRegressionOptions
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for optimization.
    /// </summary>
    /// <value>Default is 100.</value>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets whether to model the precision parameter as a function of features.
    /// </summary>
    /// <value>Default is false (constant precision).</value>
    /// <remarks>
    /// <para>
    /// When true, the precision (inverse variance) can vary with the features,
    /// allowing for heteroskedastic models. When false, a single precision
    /// parameter is estimated for all observations.
    /// </para>
    /// <para><b>For Beginners:</b> If you believe some predictions should be more
    /// certain than others based on the input features, set this to true.
    /// Otherwise, keep it false for simpler models.
    /// </para>
    /// </remarks>
    public bool ModelVariablePrecision { get; set; }

    /// <summary>
    /// Gets or sets the link function for the mean model.
    /// </summary>
    /// <value>Default is Logit.</value>
    /// <remarks>
    /// The link function transforms the bounded (0,1) response to the unbounded
    /// real line for linear modeling.
    /// </remarks>
    public BetaLinkFunction LinkFunction { get; set; } = BetaLinkFunction.Logit;

    /// <summary>
    /// Gets or sets the learning rate for optimization.
    /// </summary>
    /// <value>Default is 0.5.</value>
    public double LearningRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use regularization.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseRegularization { get; set; } = true;

    /// <summary>
    /// Gets or sets the regularization strength.
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double RegularizationStrength { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// When true, target values of exactly 0 or 1 are compressed into the open interval (0,1) using the
    /// Smithson and Verkuilen (2006) transformation rather than being rejected. Defaults to false.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Beta density is undefined at 0 and 1, so R's <c>betareg</c> and <c>statsmodels.BetaModel</c> both
    /// reject a target that touches either boundary. Smithson, M. and Verkuilen, J. (2006), "A better lemon
    /// squeezer? Maximum-likelihood regression with beta-distributed dependent variables", Psychological
    /// Methods 11(1), 54-71, give the standard remedy: y' = (y(n-1) + 0.5)/n, which shrinks the sample toward
    /// 0.5 by half an observation so both boundaries are cleared.
    /// </para>
    /// <para>
    /// It is opt-in rather than automatic because it modifies the caller's data. Enabling it silently would be
    /// the same class of hidden substitution that made this model fit least squares behind the caller's back.
    /// </para>
    /// <para><b>For Beginners:</b> Beta regression needs proportions strictly between 0 and 1. If your data
    /// contains exact 0% or 100% observations, switch this on to apply a standard published adjustment that
    /// nudges every value very slightly inward, instead of having to drop those rows.
    /// </para>
    /// </remarks>
    public bool CompressBoundaryValues { get; set; }
}

/// <summary>
/// Link functions for Beta Regression mean model.
/// </summary>
public enum BetaLinkFunction
{
    /// <summary>
    /// Logit link: η = log(μ / (1 - μ)).
    /// Most commonly used, interprets coefficients as log-odds ratios.
    /// </summary>
    Logit,

    /// <summary>
    /// Probit link: η = Φ⁻¹(μ).
    /// Uses the inverse standard normal CDF.
    /// </summary>
    Probit,

    /// <summary>
    /// Complementary log-log link: η = log(-log(1 - μ)).
    /// Useful when the response is skewed toward 1.
    /// </summary>
    CLogLog,

    /// <summary>
    /// Log link: η = log(μ).
    /// Coefficients interpreted as multiplicative effects.
    /// </summary>
    Log
}
