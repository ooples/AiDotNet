using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Attributes;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Online Passive-Aggressive regressor for epsilon-insensitive incremental regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Passive-Aggressive (PA) regression algorithms are a family of online learning algorithms that
/// aggressively update when the prediction error exceeds a threshold but remain passive when
/// the error is within tolerance.
/// </para>
/// <para>
/// <b>For Beginners:</b> PA regressors work like strict teachers for continuous predictions:
///
/// - Passive: When the prediction error is within ε → do nothing
/// - Aggressive: When error exceeds ε → update strongly to reduce the error
///
/// How it works:
/// 1. Compute prediction: ŷ = w·x + b
/// 2. Compute error: e = |y - ŷ|
/// 3. If e ≤ ε (epsilon): Acceptable error → stay passive
/// 4. If e > ε: Too much error → aggressively update weights
///
/// The ε (epsilon) parameter defines an "acceptable error zone" similar to
/// epsilon-insensitive loss in SVR.
///
/// PA variants:
/// - PA: Original, no regularization (can diverge with noise)
/// - PA-I: Adds slack variable, bounds the update size
/// - PA-II: Adds squared penalty, smoother updates
///
/// Advantages over standard online regression:
/// - No learning rate to tune (automatically determined)
/// - Robust to small errors (epsilon-insensitive)
/// - Fast convergence on well-behaved data
///
/// Usage:
/// <code>
/// var regressor = new OnlinePassiveAggressiveRegressor&lt;double&gt;(C: 1.0, epsilon: 0.1);
/// foreach (var (x, y) in dataStream)
/// {
///     regressor.PartialFit(x, y);
///     double prediction = regressor.PredictSingle(x);
/// }
/// </code>
///
/// References:
/// - Crammer et al. (2006). "Online Passive-Aggressive Algorithms"
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Online Passive-Aggressive Algorithms", "https://doi.org/10.1162/jmlr.2006.7.19.551", Year = 2006, Authors = "Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, Yoram Singer")]
public class OnlinePassiveAggressiveRegressor<T> : OnlineLearningModelBase<T>
{
    /// <summary>
    /// The weight vector (coefficients).
    /// </summary>
    private Vector<T>? _weights;

    /// <summary>
    /// The bias (intercept) term.
    /// </summary>
    private T _bias;

    /// <summary>
    /// Regularization parameter (aggressiveness).
    /// </summary>
    private readonly double _c;

    /// <summary>
    /// Epsilon parameter for insensitivity zone.
    /// </summary>
    private readonly double _epsilon;

    /// <summary>
    /// PA variant type.
    /// </summary>
    private readonly PAType _paType;

    /// <summary>
    /// Whether to fit an intercept (bias) term.
    /// </summary>
    private readonly bool _fitIntercept;

    /// <summary>
    /// Number of passes used by the offline <see cref="Train(Matrix{T}, Vector{T})"/> wrapper.
    /// </summary>
    private readonly int _batchEpochs;

    /// <summary>
    /// Gets the model type.
    /// </summary>

    /// <summary>
    /// Initializes a new instance of the OnlinePassiveAggressiveRegressor class.
    /// </summary>
    /// <param name="c">Regularization parameter (aggressiveness). Default is 1.0.</param>
    /// <param name="epsilon">Epsilon for insensitivity zone. Default is 0.1.</param>
    /// <param name="type">PA variant type. Default is PA-I.</param>
    /// <param name="fitIntercept">Whether to fit an intercept (bias) term. Default is true.</param>
    /// <param name="batchEpochs">Number of passes made by the offline Train wrapper. The online PartialFit API always performs one update per sample.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters:
    ///
    /// - C: Controls aggressiveness vs. stability tradeoff
    ///   - Higher C: More aggressive updates (faster learning, may overfit noise)
    ///   - Lower C: Smaller updates (more stable, slower learning)
    ///   - Default (1.0): Good balance for most cases
    ///
    /// - epsilon: Size of the "acceptable error" zone
    ///   - 0.0: Any error triggers an update (like standard loss)
    ///   - 0.1 (default): Errors up to 0.1 are tolerated
    ///   - Larger values: More tolerant of small errors
    ///
    /// - PA variants:
    ///   - PA: No regularization - update = loss / ||x||²
    ///   - PA_I: Bounded update - update = min(C, loss / ||x||²)
    ///   - PA_II: Smooth update - update = loss / (||x||² + 1/(2C))
    ///
    /// PA_I and PA_II are preferred for noisy data.
    /// </para>
    /// </remarks>
    public OnlinePassiveAggressiveRegressor(
        double c = 1.0,
        double epsilon = 0.1,
        PAType type = PAType.PA_I,
        bool fitIntercept = true,
        int batchEpochs = 1)
        : base(1.0, LearningRateSchedule.Constant)  // PA doesn't use learning rate
    {
        if (batchEpochs < 1)
            throw new ArgumentOutOfRangeException(nameof(batchEpochs), "Batch epochs must be at least one.");

        _c = c;
        _epsilon = epsilon;
        _paType = type;
        _fitIntercept = fitIntercept;
        _batchEpochs = batchEpochs;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Updates the model with a single training example.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PA regression update rule:
    /// 1. Compute prediction: ŷ = w·x + b
    /// 2. Compute epsilon-insensitive loss: L = max(0, |y - ŷ| - ε)
    /// 3. If L > 0 (error exceeds epsilon):
    ///    - Compute step size τ based on PA variant
    ///    - Update: w = w + τ × sign(y - ŷ) × x
    ///
    /// The step size τ is computed to minimize ||w_new - w_old|| while ensuring
    /// the new prediction satisfies |y - ŷ_new| ≤ ε.
    /// </para>
    /// </remarks>
    public override void PartialFit(Vector<T> x, T y)
        => PartialFitCore(x, y, _fitIntercept);

    private void PartialFitCore(Vector<T> x, T y, bool updateIntercept)
    {
        EnsureInitialized(x);

        // Ensure weights are initialized
        if (_weights is null)
        {
            _weights = new Vector<T>(NumFeatures);
            for (int i = 0; i < NumFeatures; i++)
            {
                _weights[i] = NumOps.Zero;
            }
        }

        // Compute prediction
        double prediction = ComputePrediction(x);
        double yDouble = NumOps.ToDouble(y);

        // Compute error and epsilon-insensitive loss
        double error = yDouble - prediction;
        double absError = Math.Abs(error);
        double loss = Math.Max(0, absError - _epsilon);

        // Only update if loss > 0 (error exceeds epsilon)
        if (loss > 0)
        {
            // Compute squared norm of x (plus 1 if fitting intercept)
            double normSq = 0;
            for (int i = 0; i < NumFeatures; i++)
            {
                double xi = NumOps.ToDouble(x[i]);
                normSq += xi * xi;
            }
            if (updateIntercept)
            {
                normSq += 1.0;  // intercept contributes 1 to norm squared
            }

            // Compute step size based on PA variant
            double tau = ComputeStepSize(loss, normSq);

            // Direction of update: sign of error
            double sign = error > 0 ? 1.0 : -1.0;

            // Update weights: w = w + τ × sign × x
            for (int i = 0; i < NumFeatures; i++)
            {
                double xi = NumOps.ToDouble(x[i]);
                double wi = NumOps.ToDouble(_weights[i]);
                wi += tau * sign * xi;
                _weights[i] = NumOps.FromDouble(wi);
            }

            // Update bias
            if (updateIntercept)
            {
                double b = NumOps.ToDouble(_bias);
                b += tau * sign;
                _bias = NumOps.FromDouble(b);
            }
        }

        SampleCount++;
    }

    /// <inheritdoc />
    public override void PartialFit(Matrix<T> x, Vector<T> y)
    {
        if (!_fitIntercept || x.Rows == 0)
        {
            base.PartialFit(x, y);
            return;
        }

        if (x.Rows != y.Length)
            throw new ArgumentException("X and y must have the same number of samples.");

        // Center both features and targets before applying the paper PA updates.
        // This isolates slope learning from the intercept, preserves translation
        // equivariance, and avoids biasing the slopes when feature means are nonzero.
        double targetMean = 0.0;
        for (int row = 0; row < y.Length; row++)
            targetMean += NumOps.ToDouble(y[row]);
        targetMean /= y.Length;

        var featureMeans = new double[x.Columns];
        for (int col = 0; col < x.Columns; col++)
        {
            for (int row = 0; row < x.Rows; row++)
                featureMeans[col] += NumOps.ToDouble(x[row, col]);
            featureMeans[col] /= x.Rows;
        }

        var centeredTargets = new Vector<T>(y.Length);
        for (int row = 0; row < y.Length; row++)
            centeredTargets[row] = NumOps.FromDouble(NumOps.ToDouble(y[row]) - targetMean);

        _bias = NumOps.Zero;
        for (int row = 0; row < x.Rows; row++)
        {
            var centeredInput = new Vector<T>(x.Columns);
            for (int col = 0; col < x.Columns; col++)
                centeredInput[col] = NumOps.FromDouble(NumOps.ToDouble(x[row, col]) - featureMeans[col]);
            PartialFitCore(centeredInput, centeredTargets[row], updateIntercept: false);
        }

        if (_weights is null)
            return;

        // For the learned centered weights, the least-squares-optimal intercept
        // is the mean uncentered residual. Assign it after the slope-only updates.
        double residualSum = 0.0;
        for (int row = 0; row < x.Rows; row++)
        {
            double prediction = 0.0;
            for (int col = 0; col < NumFeatures; col++)
                prediction += NumOps.ToDouble(_weights[col]) * NumOps.ToDouble(x[row, col]);
            residualSum += NumOps.ToDouble(y[row]) - prediction;
        }

        _bias = NumOps.FromDouble(residualSum / x.Rows);
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        for (int epoch = 0; epoch < _batchEpochs; epoch++)
            PartialFit(x, y);
    }

    /// <summary>
    /// Computes the step size based on PA variant.
    /// </summary>
    /// <param name="loss">The epsilon-insensitive loss.</param>
    /// <param name="normSq">The squared norm of the feature vector.</param>
    /// <returns>The step size for the update.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The step size determines how much to update the weights.
    /// Different PA variants compute it differently:
    /// - PA: Exact correction (may be large)
    /// - PA-I: Bounded by C (prevents huge updates)
    /// - PA-II: Smoothly dampened by C (gentle regularization)
    /// </para>
    /// </remarks>
    private double ComputeStepSize(double loss, double normSq)
    {
        if (normSq == 0)
        {
            return 0;
        }

        return _paType switch
        {
            PAType.PA => loss / normSq,
            PAType.PA_I => Math.Min(_c, loss / normSq),
            PAType.PA_II => loss / (normSq + 1.0 / (2.0 * _c)),
            _ => loss / normSq
        };
    }

    /// <summary>
    /// Computes the prediction (before epsilon zone).
    /// </summary>
    /// <param name="x">The feature vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes ŷ = w·x + b (dot product of weights and features, plus bias).
    /// </para>
    /// </remarks>
    private double ComputePrediction(Vector<T> x)
    {
        double prediction = NumOps.ToDouble(_bias);
        for (int i = 0; i < NumFeatures; i++)
        {
            prediction += NumOps.ToDouble((_weights ?? throw new InvalidOperationException("_weights has not been initialized."))[i]) * NumOps.ToDouble(x[i]);
        }
        return prediction;
    }

    /// <summary>
    /// Predicts the value for a single sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the predicted continuous value (ŷ = w·x + b).
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> x)
    {
        return NumOps.FromDouble(ComputePrediction(x));
    }

    /// <summary>
    /// Resets the model to its initial state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _weights = null;
        _bias = NumOps.Zero;
    }

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model parameters (weights + bias).
    /// </summary>
    public override Vector<T> GetParameters()
    {
        if (_weights is null)
        {
            return new Vector<T>(0);
        }

        var parameters = new Vector<T>(_weights.Length + 1);
        for (int i = 0; i < _weights.Length; i++)
        {
            parameters[i] = _weights[i];
        }
        parameters[_weights.Length] = _bias;

        return parameters;
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0) return;

        NumFeatures = parameters.Length - 1;
        _weights = new Vector<T>(NumFeatures);
        for (int i = 0; i < NumFeatures; i++)
        {
            _weights[i] = parameters[i];
        }
        _bias = parameters[NumFeatures];
        IsInitialized = true;
    }

    /// <summary>
    /// Creates a new instance with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new OnlinePassiveAggressiveRegressor<T>(_c, _epsilon, _paType, _fitIntercept, _batchEpochs);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OnlinePassiveAggressiveRegressor<T>(_c, _epsilon, _paType, _fitIntercept, _batchEpochs);
    }

    /// <summary>
    /// Gets the feature importance scores (absolute weights).
    /// </summary>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        if (_weights is null) return result;

        for (int i = 0; i < _weights.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(_weights[i])));
        }

        return result;
    }

    #endregion

    #region JIT Compilation Support

    #endregion

    /// <summary>
    /// Gets the weights vector.
    /// </summary>
    public Vector<T>? GetWeights() => _weights;

    /// <summary>
    /// Gets the bias (intercept) term.
    /// </summary>
    public T GetBias() => _bias;

    /// <summary>
    /// Gets the epsilon parameter (insensitivity zone).
    /// </summary>
    public double Epsilon => _epsilon;

    /// <summary>Gets the number of passes used by the offline Train wrapper.</summary>
    /// <remarks>
    /// INTERNAL, not public. Users configure models through AiModelBuilder / AiModelResult, and the
    /// constructor already takes batchEpochs -- a public getter adds a second entry point to the same
    /// value outside the facade. Kept accessible to the test assembly via InternalsVisibleTo.
    /// </remarks>
    internal int BatchEpochs => _batchEpochs;

    /// <summary>
    /// Computes the epsilon-insensitive loss on the provided data.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The average epsilon-insensitive loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Epsilon-insensitive loss is zero when the error is within ε,
    /// and equals |error| - ε when the error exceeds ε. This is the loss used by SVR.
    /// </para>
    /// </remarks>
    public T GetEpsilonInsensitiveLoss(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("X and y must have the same number of samples.");
        }

        double totalLoss = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            var xi = new Vector<T>(x.Columns);
            for (int j = 0; j < x.Columns; j++)
            {
                xi[j] = x[i, j];
            }

            double prediction = ComputePrediction(xi);
            double error = Math.Abs(NumOps.ToDouble(y[i]) - prediction);
            totalLoss += Math.Max(0, error - _epsilon);
        }

        return NumOps.FromDouble(totalLoss / x.Rows);
    }

    /// <summary>
    /// Computes the mean squared error on the provided data.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The mean squared error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MSE measures the average squared difference between
    /// predictions and actual values. Lower is better.
    /// </para>
    /// </remarks>
    public T GetMeanSquaredError(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("X and y must have the same number of samples.");
        }

        double totalSquaredError = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            var xi = new Vector<T>(x.Columns);
            for (int j = 0; j < x.Columns; j++)
            {
                xi[j] = x[i, j];
            }

            double prediction = ComputePrediction(xi);
            double error = NumOps.ToDouble(y[i]) - prediction;
            totalSquaredError += error * error;
        }

        return NumOps.FromDouble(totalSquaredError / x.Rows);
    }
}
