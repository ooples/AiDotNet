using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Generalized Linear Mixed-Effects Models (GLMM).
/// </summary>
/// <remarks>
/// <para>
/// GLMMs combine generalized linear models with random effects for hierarchical data
/// with non-Gaussian responses (binary, count, etc.).
/// </para>
/// <para>
/// <b>For Beginners:</b> GLMMs let you model grouped/nested data when your outcome isn't
/// continuous and normally distributed.
///
/// Common scenarios:
/// - Binary outcome (pass/fail): Use Binomial family with Logit link
/// - Count data (# of events): Use Poisson family with Log link
/// - Overdispersed counts: Use NegativeBinomial family
/// - Continuous positive: Use Gamma family with Log link
///
/// Key settings:
/// - Family: The distribution of your response variable
/// - LinkFunction: How to connect predictors to the mean response
/// - EstimationMethod: PQL is faster, Laplace is more accurate
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GLMMOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the response distribution family.
    /// </summary>
    /// <value>Default is Binomial (for binary classification).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The family describes what type of data your outcome is:</para>
    /// <list type="bullet">
    /// <item><description>Binomial: Binary (0/1, yes/no, success/failure)</description></item>
    /// <item><description>Poisson: Counts (0, 1, 2, 3, ...)</description></item>
    /// <item><description>Gaussian: Continuous normal data</description></item>
    /// <item><description>Gamma: Positive continuous (times, costs)</description></item>
    /// </list>
    /// </remarks>
    public GLMMFamily Family { get; set; } = GLMMFamily.Binomial;

    /// <summary>
    /// Gets or sets the link function.
    /// </summary>
    /// <value>Default is Logit (canonical link for Binomial).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The link function transforms predicted values:</para>
    /// <list type="bullet">
    /// <item><description>Logit: For binary outcomes - converts to log-odds</description></item>
    /// <item><description>Log: For counts - ensures predictions are positive</description></item>
    /// <item><description>Identity: For normal outcomes - no transformation</description></item>
    /// </list>
    /// <para>Usually, use the default (canonical) link for your family.</para>
    /// </remarks>
    public GLMMLinkFunction LinkFunction { get; set; } = GLMMLinkFunction.Logit;

    /// <summary>
    /// Gets or sets the estimation method.
    /// </summary>
    /// <value>Default is PQL (Penalized Quasi-Likelihood).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>PQL: Faster, works well for most problems</description></item>
    /// <item><description>Laplace: More accurate for sparse data or small clusters</description></item>
    /// <item><description>AdaptiveGaussHermite: Most accurate but slowest</description></item>
    /// </list>
    /// </remarks>
    public GLMMEstimationMethod EstimationMethod { get; set; } = GLMMEstimationMethod.PQL;

    /// <summary>
    /// Gets or sets the maximum number of iterations.
    /// </summary>
    /// <value>Default is 100.</value>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets whether to bound variance components to be non-negative.
    /// </summary>
    /// <value>Default is true.</value>
    public bool BoundVarianceComponents { get; set; } = true;

    /// <summary>
    /// Gets or sets the theta parameter for Negative Binomial family.
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// Larger values mean less overdispersion (Poisson as theta -> infinity).
    /// </remarks>
    public T NegBinomialTheta { get; set; } = default!;

    /// <summary>
    /// Gets or sets whether to compute confidence intervals for fixed effects.
    /// </summary>
    /// <value>Default is true.</value>
    public bool ComputeConfidenceIntervals { get; set; } = true;

    /// <summary>
    /// Gets or sets the confidence level.
    /// </summary>
    /// <value>Default is 0.95.</value>
    public double ConfidenceLevel { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets whether to print verbose output.
    /// </summary>
    public bool Verbose { get; set; }

    /// <summary>
    /// Initializes a new instance of GLMMOptions with sensible defaults.
    /// </summary>
    public GLMMOptions()
    {
        NegBinomialTheta = NumOps.One;
    }

    /// <summary>
    /// Creates options for logistic mixed-effects model (binary outcomes).
    /// </summary>
    /// <returns>Options configured for logistic GLMM.</returns>
    public static GLMMOptions<T> ForBinaryClassification()
    {
        return new GLMMOptions<T>
        {
            Family = GLMMFamily.Binomial,
            LinkFunction = GLMMLinkFunction.Logit
        };
    }

    /// <summary>
    /// Creates options for Poisson mixed-effects model (count data).
    /// </summary>
    /// <returns>Options configured for Poisson GLMM.</returns>
    public static GLMMOptions<T> ForCountData()
    {
        return new GLMMOptions<T>
        {
            Family = GLMMFamily.Poisson,
            LinkFunction = GLMMLinkFunction.Log
        };
    }

    /// <summary>
    /// Creates options for Gamma mixed-effects model (positive continuous).
    /// </summary>
    /// <returns>Options configured for Gamma GLMM.</returns>
    public static GLMMOptions<T> ForPositiveContinuous()
    {
        return new GLMMOptions<T>
        {
            Family = GLMMFamily.Gamma,
            LinkFunction = GLMMLinkFunction.Log
        };
    }
}

/// <summary>
/// Response distribution families for GLMM.
/// </summary>
public enum GLMMFamily
{
    /// <summary>
    /// Gaussian (normal) family for continuous outcomes.
    /// </summary>
    Gaussian,

    /// <summary>
    /// Binomial family for binary outcomes (0/1).
    /// </summary>
    Binomial,

    /// <summary>
    /// Poisson family for count data.
    /// </summary>
    Poisson,

    /// <summary>
    /// Gamma family for positive continuous data.
    /// </summary>
    Gamma,

    /// <summary>
    /// Inverse Gaussian family.
    /// </summary>
    InverseGaussian,

    /// <summary>
    /// Negative Binomial family for overdispersed counts.
    /// </summary>
    NegativeBinomial
}

/// <summary>
/// Link functions for GLMM.
/// </summary>
public enum GLMMLinkFunction
{
    /// <summary>
    /// Identity link: g(mu) = mu. Default for Gaussian.
    /// </summary>
    Identity,

    /// <summary>
    /// Logit link: g(mu) = log(mu/(1-mu)). Default for Binomial.
    /// </summary>
    Logit,

    /// <summary>
    /// Log link: g(mu) = log(mu). Default for Poisson and used for Gamma in ForPositiveContinuous.
    /// </summary>
    Log,

    /// <summary>
    /// Probit link: g(mu) = Phi^-1(mu). Alternative for Binomial.
    /// </summary>
    Probit,

    /// <summary>
    /// Complementary log-log link: g(mu) = log(-log(1-mu)).
    /// </summary>
    CLogLog,

    /// <summary>
    /// Inverse link: g(mu) = 1/mu.
    /// </summary>
    Inverse,

    /// <summary>
    /// Square root link: g(mu) = sqrt(mu).
    /// </summary>
    Sqrt
}

/// <summary>
/// Estimation methods for GLMM.
/// </summary>
public enum GLMMEstimationMethod
{
    /// <summary>
    /// Penalized Quasi-Likelihood - fast but approximate.
    /// </summary>
    PQL,

    /// <summary>
    /// Laplace approximation - more accurate than PQL.
    /// </summary>
    Laplace,

    /// <summary>
    /// Adaptive Gauss-Hermite quadrature - most accurate but slowest.
    /// </summary>
    AdaptiveGaussHermite
}
