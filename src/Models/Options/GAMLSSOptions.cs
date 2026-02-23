using AiDotNet.Regression;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for GAMLSS (Generalized Additive Models for Location, Scale, and Shape).
/// </summary>
/// <remarks>
/// <para>
/// GAMLSS extends traditional regression by allowing you to model all parameters of a
/// probability distribution as functions of explanatory variables, not just the mean.
/// </para>
/// <para>
/// <b>For Beginners:</b> In traditional regression, you predict the average (mean) value
/// of your target variable. But what if the spread (variance) of predictions also depends
/// on the input features? GAMLSS solves this!
///
/// For example, when predicting income:
/// - Traditional regression: "The predicted income is $60,000"
/// - GAMLSS: "The predicted income is $60,000 ± $10,000 for young workers, but
///   $60,000 ± $30,000 for self-employed people (higher uncertainty)"
///
/// This is useful when:
/// - Prediction uncertainty varies based on features (heteroskedasticity)
/// - You need to model skewness or other distributional properties
/// - You want probabilistic predictions with feature-dependent confidence intervals
/// </para>
/// </remarks>
public class GAMLSSOptions
{
    /// <summary>
    /// Gets or sets the maximum number of outer iterations for fitting all parameters.
    /// </summary>
    /// <value>Default is 50.</value>
    public int MaxOuterIterations { get; set; } = 50;

    /// <summary>
    /// Gets or sets the maximum number of inner iterations for fitting each parameter.
    /// </summary>
    /// <value>Default is 20.</value>
    public int MaxInnerIterations { get; set; } = 20;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the learning rate for parameter updates.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the type of distribution family to use.
    /// </summary>
    /// <value>Default is Normal (Gaussian).</value>
    public GAMLSSDistributionFamily DistributionFamily { get; set; } = GAMLSSDistributionFamily.Normal;

    /// <summary>
    /// Gets or sets the type of model to use for the location (mean) parameter.
    /// </summary>
    /// <value>Default is Linear.</value>
    public GAMLSSModelType LocationModelType { get; set; } = GAMLSSModelType.Linear;

    /// <summary>
    /// Gets or sets the type of model to use for the scale parameter.
    /// </summary>
    /// <value>Default is Linear.</value>
    public GAMLSSModelType ScaleModelType { get; set; } = GAMLSSModelType.Linear;

    /// <summary>
    /// Gets or sets the type of model to use for the shape parameters (if applicable).
    /// </summary>
    /// <value>Default is Constant (not modeled as function of features).</value>
    public GAMLSSModelType ShapeModelType { get; set; } = GAMLSSModelType.Constant;

    /// <summary>
    /// Gets or sets whether to use regularization for parameter estimation.
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
}

/// <summary>
/// Distribution families supported by GAMLSS.
/// </summary>
public enum GAMLSSDistributionFamily
{
    /// <summary>
    /// Normal (Gaussian) distribution with location (μ) and scale (σ) parameters.
    /// </summary>
    Normal,

    /// <summary>
    /// Laplace distribution with location (μ) and scale (b) parameters.
    /// Good for robust regression with heavy-tailed errors.
    /// </summary>
    Laplace,

    /// <summary>
    /// Student-t distribution with location (μ), scale (σ), and degrees of freedom (ν).
    /// Good for heavy-tailed data.
    /// </summary>
    StudentT,

    /// <summary>
    /// Gamma distribution with shape (α) and rate (β) parameters.
    /// For positive continuous data.
    /// </summary>
    Gamma,

    /// <summary>
    /// Log-Normal distribution with location (μ) and scale (σ) parameters.
    /// For positive, right-skewed data.
    /// </summary>
    LogNormal,

    /// <summary>
    /// Poisson distribution with rate (λ) parameter.
    /// For count data with mean equal to variance.
    /// </summary>
    Poisson,

    /// <summary>
    /// Negative Binomial distribution with mean and dispersion parameters.
    /// For overdispersed count data.
    /// </summary>
    NegativeBinomial
}

/// <summary>
/// Types of sub-models available for modeling distribution parameters.
/// </summary>
public enum GAMLSSModelType
{
    /// <summary>
    /// Parameter is a constant (not modeled as function of features).
    /// </summary>
    Constant,

    /// <summary>
    /// Linear model: parameter = β₀ + β₁x₁ + β₂x₂ + ...
    /// </summary>
    Linear,

    /// <summary>
    /// Additive model using smoothing splines.
    /// </summary>
    Additive,

    /// <summary>
    /// Gradient boosting for flexible non-linear modeling.
    /// </summary>
    GradientBoosting
}
