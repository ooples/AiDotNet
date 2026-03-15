namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Ordinal Regression (Proportional Odds Model), a classification method
/// for predicting ordered categorical outcomes.
/// </summary>
/// <remarks>
/// <para>
/// Ordinal Regression is used when the target variable has naturally ordered categories, such as
/// survey responses (strongly disagree to strongly agree), star ratings (1-5 stars), or disease
/// severity (none, mild, moderate, severe). Unlike regular classification which ignores the order,
/// ordinal regression models the cumulative probabilities using ordered thresholds.
/// </para>
/// <para>
/// The model uses the proportional odds (cumulative logit) assumption:
/// P(Y ≤ k) = sigmoid(α_k - β^T × x)
/// where α_k are ordered thresholds (cutpoints) and β are the feature coefficients.
/// </para>
/// <para><b>For Beginners:</b> Ordinal Regression is the right choice when:
///
/// - Your categories have a natural order (1 &lt; 2 &lt; 3 &lt; 4 &lt; 5)
/// - The distances between categories may not be equal
/// - You want predictions that respect the ordering
///
/// Examples:
/// - Customer satisfaction: Very Unsatisfied, Unsatisfied, Neutral, Satisfied, Very Satisfied
/// - Movie ratings: 1 star, 2 stars, 3 stars, 4 stars, 5 stars
/// - Pain levels: None, Mild, Moderate, Severe
/// - Education level: High School, Some College, Bachelor's, Master's, PhD
///
/// The model learns "thresholds" that separate the ordered categories. Each feature can
/// push predictions up or down the ordinal scale.
/// </para>
/// </remarks>
public class OrdinalRegressionOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization algorithm.
    /// </summary>
    /// <value>The maximum iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// The ordinal regression model is fitted using iterative optimization (gradient descent or IRLS).
    /// This controls how many iterations are allowed before stopping.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how many refinement steps the algorithm takes.
    /// The default of 100 works well for most datasets. Increase it for complex data or if
    /// you get convergence warnings.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence threshold, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the change in log-likelihood or parameters between iterations
    /// is less than this value, indicating convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how precise the solution needs to be.
    /// The default of 0.000001 is suitable for most applications.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the learning rate for gradient descent optimization.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// Controls the step size during optimization. Smaller values lead to more stable but
    /// slower convergence; larger values may overshoot the optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate controls how big the steps are when
    /// the algorithm adjusts the model parameters. If training is unstable (loss jumps around),
    /// try a smaller value. If training is too slow, try a larger value.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the link function type for the ordinal regression.
    /// </summary>
    /// <value>The link function type, defaulting to Logit.</value>
    /// <remarks>
    /// <para>
    /// The link function relates the cumulative probability to the linear predictor:
    /// - Logit: log(P(Y≤k)/(1-P(Y≤k))) = α_k - β^T × x (most common, proportional odds model)
    /// - Probit: Φ^(-1)(P(Y≤k)) = α_k - β^T × x (normal CDF inverse)
    /// - Complementary Log-Log: log(-log(1-P(Y≤k))) = α_k - β^T × x (asymmetric)
    /// </para>
    /// <para><b>For Beginners:</b> The link function determines how probabilities are computed:
    /// - Logit (default): Standard choice, symmetric probability curves
    /// - Probit: Similar to logit but based on normal distribution
    /// - Complementary Log-Log: Better when categories have asymmetric probabilities
    ///
    /// Stick with Logit unless you have a specific reason to change it.
    /// </para>
    /// </remarks>
    public OrdinalLinkFunction LinkFunction { get; set; } = OrdinalLinkFunction.Logit;

    /// <summary>
    /// Gets or sets the regularization strength (L2 penalty).
    /// </summary>
    /// <value>The regularization strength, defaulting to 0 (no regularization).</value>
    /// <remarks>
    /// <para>
    /// Adds L2 regularization to prevent overfitting by penalizing large coefficient values.
    /// Higher values increase regularization strength.
    /// </para>
    /// <para><b>For Beginners:</b> Regularization helps prevent overfitting:
    /// - 0: No regularization (default)
    /// - Small values (0.001-0.1): Mild regularization
    /// - Larger values (1-10): Strong regularization
    ///
    /// Use if your model is overfitting (good training score, poor validation score).
    /// </para>
    /// </remarks>
    public double RegularizationStrength { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to fit an intercept term.
    /// </summary>
    /// <value>True to fit intercept terms (thresholds), defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// In ordinal regression, the "intercepts" are the threshold parameters that separate
    /// adjacent categories. This should almost always be true.
    /// </para>
    /// <para><b>For Beginners:</b> Keep this as true. The thresholds are essential for
    /// ordinal regression to work properly.
    /// </para>
    /// </remarks>
    public bool FitIntercept { get; set; } = true;
}

/// <summary>
/// Link functions for ordinal regression.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The link function transforms cumulative probabilities to a scale
/// where linear modeling can be applied. Different links make different assumptions about
/// how the probability changes across the ordinal scale.
/// </para>
/// </remarks>
public enum OrdinalLinkFunction
{
    /// <summary>
    /// Logit link: log(P/(1-P)). The most common choice, giving the proportional odds model.
    /// Results in symmetric S-shaped probability curves.
    /// </summary>
    Logit,

    /// <summary>
    /// Probit link: Φ^(-1)(P). Uses the inverse normal CDF.
    /// Similar to logit but with slightly thinner tails.
    /// </summary>
    Probit,

    /// <summary>
    /// Complementary log-log link: log(-log(1-P)).
    /// Asymmetric, useful when one direction of the scale is more likely.
    /// </summary>
    ComplementaryLogLog
}
