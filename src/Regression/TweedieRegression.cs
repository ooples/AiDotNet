using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Implements Tweedie regression, a flexible generalized linear model that encompasses several distributions
/// (Poisson, Gamma, Inverse Gaussian) as special cases based on the power parameter.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Tweedie regression is a powerful family of distributions where variance is proportional to a power
/// of the mean: Var(Y) = φ × μ^p. The power parameter p determines which distribution family is used:
/// - p = 0: Normal/Gaussian (variance independent of mean)
/// - p = 1: Poisson (variance = mean)
/// - 1 &lt; p &lt; 2: Compound Poisson-Gamma (handles both zeros and positive continuous values)
/// - p = 2: Gamma (variance = mean²)
/// - p = 3: Inverse Gaussian (variance = mean³)
/// </para>
/// <para>
/// The compound Poisson-Gamma case (1 &lt; p &lt; 2) is particularly important for insurance modeling,
/// where data often has many exact zeros (no claim) mixed with positive continuous values (claim amounts).
/// </para>
/// <para>
/// The model is fitted using iteratively reweighted least squares (IRLS), a form of maximum likelihood estimation.
/// </para>
/// <para>
/// For Beginners:
/// Tweedie regression is like having a "dial" that lets you choose how the variability in your data
/// relates to the average. It's especially powerful because:
///
/// - Insurance claims: Many policies have zero claims, others have positive amounts
/// - Rainfall data: Many dry days (zero) plus positive rainfall amounts
/// - Healthcare costs: Some patients have zero costs, others have positive costs
/// - Sales data: Some products have zero sales, others have positive sales
///
/// With p between 1 and 2, Tweedie can naturally handle data that has both exact zeros and positive
/// continuous values - something that neither Poisson (counts only) nor Gamma (positive only) can do alone.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Tweedie regression for data with zeros and positive values
/// var options = new TweedieRegressionOptions&lt;double&gt;();
///
/// // Prepare training data: 5 samples with 2 features, mixed zero/positive targets
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 0.0, 3.5, 0.0, 12.4, 25.1 });
///
/// // Train with power-family variance function
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new TweedieRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 11, 12 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Generalized Linear Models", "https://doi.org/10.1007/978-1-4899-3242-6")]
public partial class TweedieRegression<T> : RegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Tweedie regression model.
    /// </summary>
    private readonly TweedieRegressionOptions<T> _options;

    /// <summary>
    /// The estimated dispersion parameter φ.
    /// </summary>
    private T _dispersion;

    /// <summary>
    /// Gets the estimated dispersion parameter.
    /// </summary>
    /// <value>
    /// The dispersion parameter φ, which scales the variance: Var(Y) = φ × μ^p.
    /// </value>
    /// <remarks>
    /// <para>
    /// The dispersion parameter is estimated after model fitting using the Pearson residuals.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The dispersion parameter tells you how spread out your data is relative to the
    /// mean raised to the power p. A smaller dispersion means more precise predictions.
    /// </para>
    /// </remarks>
    public T Dispersion => _dispersion;

    /// <summary>
    /// Gets the power parameter used by this model.
    /// </summary>
    /// <value>The power parameter p that defines the variance-mean relationship.</value>
    /// <remarks>
    /// <para>
    /// This is the power parameter specified in the options during construction.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The power parameter determines which "type" of Tweedie distribution is being used.
    /// Common values are 1.5 for insurance data or 2 for always-positive data.
    /// </para>
    /// </remarks>
    public double PowerParameter => _options.PowerParameter;

    /// <summary>
    /// Initializes a new instance of the TweedieRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the Tweedie regression model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with either the provided options or default settings.
    /// It validates that the power parameter is in a valid range.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This constructor sets up the Tweedie regression model with your specified settings or uses
    /// default settings if none are provided. The most important setting is the power parameter,
    /// which should be chosen based on your data characteristics.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the power parameter is in the invalid range (0, 1).</exception>
    public TweedieRegression(TweedieRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new TweedieRegressionOptions<T>();
        _options.Validate();
        _dispersion = NumOps.FromDouble(_options.InitialDispersion);
    }

    /// <summary>
    /// Trains the Tweedie regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <remarks>
    /// <para>
    /// This method implements the iteratively reweighted least squares (IRLS) algorithm to fit the model.
    /// The variance function used is V(μ) = μ^p where p is the power parameter.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Training is the process where the model learns from your data. The algorithm iteratively
    /// improves the coefficients until they converge to the best values.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when target values don't match the power parameter requirements.</exception>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);
        ValidateTweedieData(y);
        TrainingFeatureCount = x.Columns;

        // This method previously fitted ORDINARY LEAST SQUARES and returned immediately. The
        // guard that followed it was written as a condition but is always true for any real
        // problem, so it acted as an unconditional return and left the real estimation below
        // unreachable: callers received a plain linear least-squares fit from a model named for a
        // different algorithm. The real estimation now runs.

        int numFeatures = x.Columns;
        Coefficients = new Vector<T>(numFeatures);
        Intercept = NumOps.Zero;

        // Calculate mean of positive values for initialization
        var yTensor = y.AsTensor();
        var isPositive = Engine.TensorGreaterThan(yTensor, NumOps.Zero);
        int countPositive = Engine.TensorCountNonzero(isPositive);
        T sumY = countPositive > 0
            ? Engine.TensorSum(Engine.TensorWhere(isPositive, yTensor, Engine.TensorZerosLike(yTensor)))
            : NumOps.Zero;
        T meanPositive = countPositive > 0 ? NumOps.Divide(sumY, NumOps.FromDouble(countPositive)) : NumOps.One;
        double meanPositiveDouble = NumOps.ToDouble(meanPositive);

        // Set initial intercept based on link function (link functions stay double - special math)
        Intercept = _options.LinkFunction switch
        {
            TweedieLinkFunction.Log => NumOps.FromDouble(Math.Log(meanPositiveDouble)),
            TweedieLinkFunction.Power => NumOps.FromDouble(ApplyPowerLink(meanPositiveDouble)),
            TweedieLinkFunction.Identity => meanPositive,
            _ => NumOps.FromDouble(Math.Log(meanPositiveDouble))
        };

        Matrix<T> xWithIntercept = x.AddColumn(Vector<T>.CreateDefault(x.Rows, NumOps.One));

        // Ridge regularization added to the diagonal for numerical stability (loop invariant)
        var minRegularization = 1e-10;
        var userStrength = Regularization?.GetOptions().Strength ?? 0.0;
        var effectiveStrength = NumOps.FromDouble(Math.Max(minRegularization, userStrength));

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            Vector<T> currentCoefficients = new([.. Coefficients, Intercept]);
            Vector<T> eta = xWithIntercept.Multiply(currentCoefficients);
            Vector<T> mu = ApplyInverseLink(eta);

            // Ensure mu values are positive and not too small
            mu = ClampMu(mu);

            Matrix<T> w = ComputeWeights(mu);
            Vector<T> z = ComputeWorkingResponse(eta, y, mu);

            Matrix<T> xTw = xWithIntercept.Transpose().Multiply(w);
            Matrix<T> xTwx = xTw.Multiply(xWithIntercept);
            Vector<T> xTwz = xTw.Multiply(z);

            // Add ridge regularization to ensure numerical stability
            xTwx = RegressionEngineOps2.AddDiagonal(xTwx, effectiveStrength);

            Vector<T> newCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTwx, xTwz, _options.DecompositionType);

            // Apply regularization to the coefficients
            if (Regularization != null)
            {
                newCoefficients = Regularization.Regularize(newCoefficients);
            }

            // Commit the update BEFORE testing convergence. Breaking first discarded the very
            // iteration that converged, so the model kept the previous, worse coefficients — and
            // when the first step was already small enough to satisfy the tolerance, it kept the
            // all-zero initialization, leaving only the intercept and a negative R-squared.
            bool converged = HasConverged(currentCoefficients, newCoefficients);

            Coefficients = newCoefficients.GetSubVector(0, numFeatures);
            Intercept = newCoefficients[numFeatures];

            if (converged)
            {
                break;
            }
        }

        // Estimate dispersion parameter using Pearson residuals
        EstimateDispersion(x, y);
    }

    /// <summary>
    /// Validates that target values are appropriate for the specified power parameter.
    /// </summary>
    /// <param name="y">The target values vector to validate.</param>
    /// <exception cref="ArgumentException">Thrown when values don't match power parameter requirements.</exception>
    /// <remarks>
    /// <para>
    /// For p > 0 and p ≠ 1, values must be non-negative. For p = 2 or p = 3, values must be strictly positive.
    /// For p = 1 (Poisson), non-negative integers are expected (though continuous values may work).
    /// </para>
    /// <para>
    /// For Beginners:
    /// The allowed values depend on the power parameter:
    /// - p = 0: Any real values are allowed (normal distribution)
    /// - 1 ≤ p &lt; 2: Non-negative values (zeros allowed)
    /// - p ≥ 2: Strictly positive values (no zeros)
    /// </para>
    /// </remarks>
    private void ValidateTweedieData(Vector<T> y)
    {
        double p = _options.PowerParameter;
        if (p <= 0 || y.Length == 0)
        {
            return;
        }

        // The first offending sample decides the message, exactly as a front-to-back scan would:
        // p >= 2 rejects y <= 0, any other p > 0 rejects y < 0.
        var values = y.AsTensor();
        var offending = p >= 2
            ? Engine.TensorLessEqual(values, Engine.TensorZerosLike(values))
            : Engine.TensorLessThan(values, NumOps.Zero);
        int i = RegressionEngineOps2.FirstNonzero(offending);
        if (i < 0)
        {
            return;
        }

        T value = y[i];

        // For p >= 2 (Gamma, Inverse Gaussian, etc.), values must be strictly positive
        if (p >= 2)
        {
            throw new ArgumentException(
                $"Tweedie regression with power parameter p ≥ 2 requires strictly positive target values. " +
                $"Found value {value} at index {i}. Use p in range (1, 2) if you have zeros in your data.");
        }

        // For 1 <= p < 2 (Poisson or compound Poisson-Gamma), values must be non-negative
        if (p >= 1)
        {
            throw new ArgumentException(
                $"Tweedie regression with power parameter 1 ≤ p < 2 requires non-negative target values. " +
                $"Found value {value} at index {i}.");
        }

        // For p > 0 and p != integer, values must be non-negative
        throw new ArgumentException(
            $"Tweedie regression with power parameter p > 0 requires non-negative target values. " +
            $"Found value {value} at index {i}.");
    }

    /// <summary>
    /// Applies the power link function: η = μ^(1-p).
    /// </summary>
    /// <param name="mu">The mean value.</param>
    /// <returns>The link-transformed value.</returns>
    /// <remarks>
    /// <para>
    /// The canonical power link for Tweedie is μ^(1-p). For p=1, this is log(μ).
    /// For p=2, this is 1/μ. For p=0, this is μ itself.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The power link is the "natural" link for Tweedie distributions, adapting to the power parameter.
    /// </para>
    /// </remarks>
    private double ApplyPowerLink(double mu)
    {
        double p = _options.PowerParameter;
        double oneMinusP = 1.0 - p;

        if (Math.Abs(oneMinusP) < 1e-10)
        {
            // When p ≈ 1, power link becomes log
            return Math.Log(Math.Max(mu, 1e-10));
        }

        return Math.Pow(Math.Max(mu, 1e-10), oneMinusP);
    }

    /// <summary>
    /// Applies the inverse link function to convert the linear predictor to the mean.
    /// </summary>
    /// <param name="eta">The linear predictor (X × β).</param>
    /// <returns>The predicted mean values μ.</returns>
    /// <remarks>
    /// <para>
    /// The inverse link function converts from the linear scale to the response scale:
    /// - Log link: μ = exp(η)
    /// - Power link: μ = η^(1/(1-p)) when p ≠ 1, or exp(η) when p = 1
    /// - Identity link: μ = η
    /// </para>
    /// <para>
    /// For Beginners:
    /// The inverse link function transforms the linear prediction back to the scale of the actual values.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyInverseLink(Vector<T> eta)
    {
        double p = _options.PowerParameter;

        return _options.LinkFunction switch
        {
            TweedieLinkFunction.Log => Engine.Exp(eta),
            TweedieLinkFunction.Power => ApplyInversePowerLink(eta, p),
            TweedieLinkFunction.Identity => eta.Clone(),
            _ => Engine.Exp(eta)
        };
    }

    /// <summary>
    /// Inverse of the power link, μ = η^(1/(1-p)), element-wise with the same guards for
    /// non-positive η as the scalar definition and a floor of 1e-10.
    /// </summary>
    private Vector<T> ApplyInversePowerLink(Vector<T> eta, double p)
    {
        double oneMinusP = 1.0 - p;
        if (Math.Abs(oneMinusP) < 1e-10)
        {
            // When p ≈ 1, inverse is exp
            return Engine.Exp(eta);
        }

        // μ = η^(1/(1-p))
        // Need to handle sign and ensure positive result
        double invPower = 1.0 / oneMinusP;
        double roundingGap = Math.Abs(invPower - Math.Round(invPower));
        T floor = NumOps.FromDouble(1e-10);
        var etaTensor = eta.AsTensor();
        var result = Engine.TensorPower(Engine.TensorAbs(etaTensor), NumOps.FromDouble(invPower));

        if (roundingGap > 1e-10)
        {
            // Non-integer power of non-positive number
            result = Engine.TensorWhere(
                Engine.TensorLessEqual(etaTensor, Engine.TensorZerosLike(etaTensor)),
                Engine.Fill(eta.Length, floor).AsTensor(),
                result);
        }
        else if (roundingGap < 1e-10)
        {
            // Integer power, preserve sign
            result = Engine.TensorWhere(
                Engine.TensorLessThan(etaTensor, NumOps.Zero), Engine.TensorNegate(result), result);
        }

        return Engine.TensorMax(result, floor).ToVector();
    }

    /// <summary>
    /// Clamps the mean values to ensure they're positive and numerically stable.
    /// </summary>
    /// <param name="mu">The mean values to clamp.</param>
    /// <returns>The clamped mean values.</returns>
    /// <remarks>
    /// <para>
    /// This prevents numerical issues when mu gets too close to zero or becomes negative.
    /// </para>
    /// <para>
    /// For Beginners:
    /// During fitting, predicted values might become very small or negative due to numerical errors.
    /// This method ensures they stay positive and reasonably sized.
    /// </para>
    /// </remarks>
    private Vector<T> ClampMu(Vector<T> mu)
    {
        T minValue = NumOps.FromDouble(1e-10);
        T maxValue = NumOps.FromDouble(1e10);
        return Engine.Clamp(mu, minValue, maxValue);
    }

    /// <summary>
    /// Computes the weights matrix for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="mu">The vector of predicted mean values.</param>
    /// <returns>A diagonal matrix of weights.</returns>
    /// <remarks>
    /// <para>
    /// For Tweedie regression, the variance function is V(μ) = μ^p.
    /// The weight formula is: W = 1 / (V(μ) × (g'(μ))²)
    ///
    /// - Log link: g'(μ) = 1/μ, so W = 1 / (μ^p × (1/μ)²) = μ^(2-p)
    /// - Power link: g'(μ) = (1-p)μ^(-p), so W = 1 / (μ^p × (1-p)²μ^(-2p)) = μ^p/(1-p)²
    /// - Identity link: g'(μ) = 1, so W = 1/μ^p
    /// </para>
    /// <para>
    /// For Beginners:
    /// The weights help the algorithm balance the influence of different observations.
    /// </para>
    /// </remarks>
    private Matrix<T> ComputeWeights(Vector<T> mu)
    {
        double p = _options.PowerParameter;
        var muVal = RegressionEngineOps2.ClampMin(mu, NumOps.FromDouble(1e-10));
        var weights = _options.LinkFunction switch
        {
            // W = 1 / (V(μ) × (g'(μ))²) = 1 / (μ^p × (1/μ)²) = μ^(2-p)
            TweedieLinkFunction.Log => Engine.Power(muVal, NumOps.FromDouble(2.0 - p)),
            // W = 1 / (V(μ) × (g'(μ))²) = μ^p / (1-p)²
            TweedieLinkFunction.Power => Engine.Divide(
                Engine.Power(muVal, NumOps.FromDouble(p)),
                NumOps.FromDouble(Math.Max((1.0 - p) * (1.0 - p), 1e-10))),
            // W = 1 / (V(μ) × 1) = 1/μ^p
            TweedieLinkFunction.Identity => Engine.Reciprocal(Engine.Power(muVal, NumOps.FromDouble(p))),
            _ => Engine.Power(muVal, NumOps.FromDouble(2.0 - p))
        };

        // Clamp weights to prevent numerical issues
        weights = Engine.Clamp(weights, NumOps.FromDouble(1e-10), NumOps.FromDouble(1e10));

        return Matrix<T>.CreateDiagonal(weights);
    }

    /// <summary>
    /// Computes the working response for the iteratively reweighted least squares algorithm.
    /// </summary>
    /// <param name="eta">The linear predictor.</param>
    /// <param name="y">The target values vector.</param>
    /// <param name="mu">The vector of predicted mean values.</param>
    /// <returns>The working response vector.</returns>
    /// <remarks>
    /// <para>
    /// The working response is computed as: z = η + (y - μ) × g'(μ)
    /// where g'(μ) is the derivative of the link function.
    ///
    /// - Log link: z = η + (y - μ)/μ
    /// - Power link: z = η + (y - μ)(1-p)μ^(-p)
    /// - Identity link: z = y
    /// </para>
    /// <para>
    /// For Beginners:
    /// The working response is an adjusted target that helps the algorithm converge.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeWorkingResponse(Vector<T> eta, Vector<T> y, Vector<T> mu)
    {
        double p = _options.PowerParameter;
        var muVal = RegressionEngineOps2.ClampMin(mu, NumOps.FromDouble(1e-10));
        var diff = Engine.Subtract(y, muVal);

        return _options.LinkFunction switch
        {
            // g'(μ) = 1/μ, so z = η + (y - μ)/μ
            TweedieLinkFunction.Log => Engine.Add(eta, Engine.Divide(diff, muVal)),
            // g'(μ) = (1-p)μ^(-p), so z = η + (y - μ)(1-p)μ^(-p)
            TweedieLinkFunction.Power => Engine.Add(eta, Engine.Multiply(
                Engine.Multiply(diff, NumOps.FromDouble(1.0 - p)), Engine.Power(muVal, NumOps.FromDouble(-p)))),
            // g'(μ) = 1, so z = η + (y - μ) = y
            TweedieLinkFunction.Identity => y.Clone(),
            _ => Engine.Add(eta, Engine.Divide(diff, muVal))
        };
    }

    /// <summary>
    /// Checks if the algorithm has converged by comparing the change in coefficients.
    /// </summary>
    /// <param name="oldCoefficients">The coefficients from the previous iteration.</param>
    /// <param name="newCoefficients">The coefficients from the current iteration.</param>
    /// <returns>True if converged; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Convergence is determined by the L2 norm of the coefficient change.
    /// </para>
    /// <para>
    /// For Beginners:
    /// Checks if the model has found the best solution by measuring coefficient changes.
    /// </para>
    /// </remarks>
    private bool HasConverged(Vector<T> oldCoefficients, Vector<T> newCoefficients)
    {
        T diff = oldCoefficients.Subtract(newCoefficients).L2Norm();
        return NumOps.LessThan(diff, NumOps.FromDouble(_options.Tolerance));
    }

    /// <summary>
    /// Estimates the dispersion parameter using Pearson residuals after model fitting.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para>
    /// The dispersion parameter is estimated as:
    /// φ = (1/(n-p)) × Σ((y_i - μ_i)²/V(μ_i))
    /// where V(μ) = μ^p is the variance function.
    /// </para>
    /// <para>
    /// For Beginners:
    /// The dispersion parameter measures overall data variability after accounting for the model.
    /// </para>
    /// </remarks>
    private void EstimateDispersion(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = Predict(x);
        int n = y.Length;
        int numParams = Coefficients.Length + 1;
        double power = _options.PowerParameter;

        T floor = NumOps.FromDouble(1e-10);
        var muVal = RegressionEngineOps2.ClampMin(predictions, floor);
        // For Tweedie, variance = μ^p
        var variance = Engine.Power(muVal, NumOps.FromDouble(power));
        var residual = Engine.Subtract(y, muVal);
        var pearsonResidualsSq = Engine.Divide(
            Engine.Multiply(residual, residual), RegressionEngineOps2.ClampMin(variance, floor));
        T sumPearsonResidualsSq = n == 0 ? NumOps.Zero : Engine.Sum(pearsonResidualsSq);

        _dispersion = NumOps.Divide(sumPearsonResidualsSq, NumOps.FromDouble(Math.Max(1, n - numParams)));
    }

    /// <summary>
    /// Makes predictions for the given input data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted mean values for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the linear predictor and applies the inverse link function.
    /// </para>
    /// <para>
    /// For Beginners:
    /// After training, use this to make predictions on new data.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> x)
    {
        // X * Coefficients + Intercept gives the LINEAR PREDICTOR eta, which lives on the link
        // scale. The response is obtained by applying the inverse link — for the default log link,
        // mu = exp(eta). Returning eta directly handed back log-scale numbers for a response-scale
        // quantity, which is what drove R-squared sharply negative on positive data. The omission
        // was invisible while an OLS short-circuit was fitting response-scale coefficients earlier
        // in Train, so eta and mu happened to coincide.
        var eta = RegressionEngineOps2.AddScalar(x.Multiply(Coefficients), Intercept);

        return ApplyInverseLink(eta);
    }
}
