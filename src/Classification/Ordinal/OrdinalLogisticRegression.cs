using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.Ordinal;

/// <summary>
/// Ordinal Logistic Regression (Proportional Odds Model).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Ordinal logistic regression is the standard approach for ordinal data.
/// It's also known as the "proportional odds model" or "cumulative link model".</para>
///
/// <para><b>How it works:</b> Instead of predicting the class directly, it models the cumulative
/// probability of being in class k or lower:
/// <code>
/// log(P(Y ≤ k) / P(Y &gt; k)) = θₖ - β·X
/// </code>
/// Where:
/// <list type="bullet">
/// <item><b>θₖ</b> = threshold (cutpoint) for class k</item>
/// <item><b>β</b> = coefficient vector (same for all classes - "proportional odds")</item>
/// <item><b>X</b> = feature vector</item>
/// </list>
/// </para>
///
/// <para><b>Key assumption - Proportional Odds:</b> The effect of each feature is the same
/// regardless of which threshold we're considering. This means moving from 1→2 stars has
/// the same relationship with features as moving from 4→5 stars.</para>
///
/// <para><b>Example:</b> Rating prediction for a restaurant review:
/// <code>
/// // Train on review features (sentiment, length, etc.) and ratings (1-5)
/// var model = new OrdinalLogisticRegression&lt;double&gt;();
/// model.Train(features, ratings);
///
/// // Predict most likely rating
/// var predicted = model.Predict(newFeatures);
///
/// // Get probability distribution over ratings
/// var probs = model.PredictProbabilities(newFeatures);
/// </code>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>McCullagh, P. (1980). "Regression Models for Ordinal Data"</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Regression Models for Ordinal Data", "https://doi.org/10.1111/j.2517-6161.1980.tb01109.x", Year = 1980, Authors = "Peter McCullagh")]
public partial class OrdinalLogisticRegression<T> : OrdinalClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>, IGradientComputable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>The coefficients followed by the ordinal cut points. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
    /// <summary>
    /// The learned coefficient vector (β).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These coefficients tell us how each feature affects the
    /// ordinal outcome. A positive coefficient means higher values of that feature push
    /// predictions toward higher classes (e.g., more stars).</para>
    /// </remarks>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _coefficients;

    /// <summary>
    /// The gradient-based optimizer for parameter updates.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Matrix<T>, Vector<T>> _paramOptimizer;

    /// <summary>
    /// Maximum iterations for optimization.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// L2 regularization strength.
    /// </summary>
    private readonly double _regularizationStrength;

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Legacy learning rate field — kept for backward compatibility with serialization/clone.
    /// The optimizer handles learning rate internally.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Initializes a new instance of OrdinalLogisticRegression.
    /// </summary>
    /// <param name="optimizer">Gradient-based optimizer for training. Defaults to AdamOptimizer with industry-standard settings.</param>
    /// <param name="maxIterations">Maximum training iterations. Default is 1000.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-6.</param>
    /// <param name="regularization">L2 regularization strength. Default is 0.0.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public OrdinalLogisticRegression(
        IGradientBasedOptimizer<T, Matrix<T>, Vector<T>>? optimizer = null,
        int maxIterations = 1000,
        double tolerance = 1e-6,
        double regularization = 0.0,
        int? seed = null)
        : base()
    {
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Must be >= 1.");
        if (double.IsNaN(tolerance) || double.IsInfinity(tolerance) || tolerance <= 0)
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Must be a positive finite number.");
        if (double.IsNaN(regularization) || double.IsInfinity(regularization) || regularization < 0)
            throw new ArgumentOutOfRangeException(nameof(regularization), "Must be a non-negative finite number.");

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _regularizationStrength = regularization;
        _learningRate = 0.01; // Legacy field for serialization
        _random = RandomHelper.CreateSeededRandom(seed ?? 42);
        // Default to Adam optimizer with learning rate 0.01, which is appropriate for
        // logistic regression models with a small number of parameters. The default Adam
        // LR of 0.001 (designed for neural networks with millions of params) is too small
        // for models with O(features) parameters.
        _paramOptimizer = optimizer ?? new AdamOptimizer<T, Matrix<T>, Vector<T>>(
            null,
            new AdamOptimizerOptions<T, Matrix<T>, Vector<T>> { InitialLearningRate = 0.01 });
    }

    /// <summary>
    /// Legacy constructor for backward compatibility with existing code and serialization.
    /// The learningRate is passed to the Adam optimizer as its initial learning rate.
    /// </summary>
    public OrdinalLogisticRegression(
        double learningRate,
        int maxIterations = 1000,
        double tolerance = 1e-6,
        double regularization = 0.0,
        int? seed = null)
        : base()
    {
        if (double.IsNaN(learningRate) || double.IsInfinity(learningRate) || learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Must be a positive finite number.");
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Must be >= 1.");
        if (double.IsNaN(tolerance) || double.IsInfinity(tolerance) || tolerance <= 0)
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Must be a positive finite number.");
        if (double.IsNaN(regularization) || double.IsInfinity(regularization) || regularization < 0)
            throw new ArgumentOutOfRangeException(nameof(regularization), "Must be a non-negative finite number.");

        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _regularizationStrength = regularization;
        _random = RandomHelper.CreateSeededRandom(seed ?? 42);
        // Use Adam optimizer with the user-specified learning rate
        var adamOptions = new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = learningRate
        };
        _paramOptimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(null, adamOptions);
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.OrdinalLogisticRegression.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns an identifier for this model type,
    /// used internally for serialization and model management.</para>
    /// </remarks>

    /// <summary>
    /// Trains the ordinal logistic regression model.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Ordinal class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training finds:
    /// <list type="number">
    /// <item>Coefficients (β) that show how features affect the ordinal outcome</item>
    /// <item>Thresholds (θ) that separate the ordered classes</item>
    /// </list>
    ///
    /// The training uses gradient descent to minimize the negative log-likelihood:
    /// minimize -Σᵢ log P(Yᵢ = yᵢ | Xᵢ)
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples must match number of labels.");
        }

        NumFeatures = x.Columns;
        ExtractOrderedClasses(y);

        int K = NumClasses;  // Number of classes
        int P = NumFeatures; // Number of features

        // Initialize coefficients and thresholds
        _coefficients = new Vector<T>(P);
        _thresholds = new Vector<T>(K - 1);

        // Initialize thresholds to evenly spaced values: -2 + 4 * (k + 1) / K
        var engine = Engine;
        _thresholds = K <= 1
            ? new Vector<T>(0)
            : engine.TensorAddScalar(
                engine.TensorDivideScalar(engine.TensorMultiplyScalar(engine.TensorArange<T>(1, K), NumOps.FromDouble(4.0)),
                    NumOps.FromDouble(K)),
                NumOps.FromDouble(-2.0)).ToVector();

        // Initialize coefficients to small random values
        var initialCoefficients = new T[P];
        for (int p = 0; p < P; p++)
        {
            initialCoefficients[p] = NumOps.FromDouble(0.01 * (_random.NextDouble() - 0.5));
        }

        _coefficients = Vector<T>.Wrap(initialCoefficients);

        // Convert labels to class indices
        var yIndices = GetOrdinalClassIndices(y);
        var features = Tensor<T>.FromMatrix(x);
        T regularization = NumOps.FromDouble(_regularizationStrength);
        T halfRegularization = NumOps.FromDouble(0.5 * _regularizationStrength);
        T minGap = NumOps.FromDouble(0.001);

        // Optimization using the configured gradient-based optimizer (Adam by default).
        T prevLoss = NumOps.FromDouble(double.MaxValue);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var (gradCoef, gradThresh, loss) = ComputeNegativeLogLikelihoodGradients(features, yIndices, _coefficients, _thresholds);

            // Add L2 regularization to coefficients gradient
            gradCoef = engine.Add(gradCoef, engine.Multiply(_coefficients, regularization));
            loss = NumOps.Add(loss, NumOps.Multiply(halfRegularization, engine.DotProduct(_coefficients, _coefficients)));

            // Pack coefficients + thresholds into a single parameter vector for the optimizer.
            // Pass the raw (summed) gradient — the optimizer's adaptive learning rate (Adam's v_t)
            // naturally scales with gradient magnitude, so pre-averaging by N is redundant and
            // causes underflow when combined with the optimizer's own learning rate.
            var paramVec = Vector<T>.Concatenate(_coefficients, _thresholds);
            var gradVec = Vector<T>.Concatenate(gradCoef, gradThresh);

            // Let the optimizer (Adam by default) handle the update with adaptive learning rates
            var updatedParams = _paramOptimizer.UpdateParameters(paramVec, gradVec);

            // Unpack updated parameters back into coefficients and thresholds
            _coefficients = updatedParams.GetSubVector(0, P);
            _thresholds = updatedParams.GetSubVector(P, K - 1);

            // Ensure thresholds are monotonically increasing
            _thresholds = EnforceIncreasing(_thresholds, minGap);

            // Check convergence
            loss = NumOps.Divide(loss, NumOps.FromDouble(x.Rows));
            if (HasConverged(prevLoss, loss))
            {
                break;
            }
            prevLoss = loss;
        }
    }

    /// <summary>
    /// Whether the loss changed by less than the tolerance.
    /// </summary>
    private bool HasConverged(T previousLoss, T loss)
    {
        return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(previousLoss, loss)), NumOps.FromDouble(_tolerance));
    }

    /// <summary>
    /// Raises every threshold that does not exceed its predecessor to predecessor + <paramref name="gap"/>
    /// (a sequential scan, since each fix depends on the previous one).
    /// </summary>
    private Vector<T> EnforceIncreasing(Vector<T> thresholds, T gap)
    {
        var values = thresholds.ToArray();
        for (int k = 1; k < values.Length; k++)
        {
            if (NumOps.LessThanOrEquals(values[k], values[k - 1]))
            {
                values[k] = NumOps.Add(values[k - 1], gap);
            }
        }

        return Vector<T>.Wrap(values);
    }

    /// <summary>
    /// Summed negative log-likelihood gradients of the cumulative logit model over all samples.
    /// </summary>
    /// <param name="features">The [N, P] features.</param>
    /// <param name="classIndices">The class index of every sample.</param>
    /// <param name="coefficients">The P coefficients.</param>
    /// <param name="thresholds">The K - 1 thresholds.</param>
    /// <returns>The coefficient gradient, the threshold gradient and the summed loss.</returns>
    private (Vector<T> GradCoef, Vector<T> GradThresh, T Loss) ComputeNegativeLogLikelihoodGradients(
        Tensor<T> features, Tensor<int> classIndices, Vector<T> coefficients, Vector<T> thresholds)
    {
        var engine = Engine;
        int n = features.Shape[0];
        int p = features.Shape[1];
        int k = thresholds.Length + 1;
        if (n == 0)
        {
            return (new Vector<T>(p), new Vector<T>(k - 1), NumOps.Zero);
        }

        // Linear predictors eta = X * coef and cumulative probabilities P(Y <= j) = sigmoid(alpha_j - eta)
        var eta = engine.TensorMatMul(features, coefficients.AsTensor().Reshape(p, 1));
        var cumulative = engine.TensorSigmoid(engine.TensorSubtract(thresholds.AsTensor().Reshape(1, k - 1), eta));

        // Padded cumulative probabilities [0, P(Y<=0), ..., P(Y<=K-2), 1] and their derivatives gamma = c(1-c)
        var zeros = new Tensor<T>(new[] { n, 1 });
        var ones = new Tensor<T>(new[] { n, 1 });
        engine.TensorFill(ones, NumOps.One);
        var padded = engine.TensorConcatenate(new[] { zeros, cumulative, ones }, 1).Reshape(n * (k + 1));
        var gammas = engine.TensorMultiply(padded, engine.ScalarMinusTensor(NumOps.One, padded));

        // Row-wise picks at the sample's class: lo = P(Y <= y-1), hi = P(Y <= y)
        var rowStarts = engine.TensorMultiplyScalar(engine.TensorArange<int>(0, n), k + 1);
        var lo = engine.TensorAdd(rowStarts, classIndices.Reshape(n));
        var hi = engine.TensorAddScalar(lo, 1);
        var cumLo = engine.TensorGather(padded, lo, 0);
        var cumHi = engine.TensorGather(padded, hi, 0);
        var gammaLo = engine.TensorGather(gammas, lo, 0);
        var gammaHi = engine.TensorGather(gammas, hi, 0);

        // Class probabilities P(Y = y), kept away from zero
        var probs = engine.TensorClampMin(engine.TensorSubtract(cumHi, cumLo), NumOps.FromDouble(1e-15));
        T loss = engine.TensorSum(engine.TensorNegate(engine.TensorLog(probs)));

        // Coefficient factor per sample: first class 1 - P(Y<=0), last class -P(Y<=K-2),
        // middle classes (gamma_{y-1} - gamma_y) / P(Y=y)
        var firstClass = engine.TensorGather(OneHotVector(0, k), classIndices.Reshape(n), 0);
        var lastClass = engine.TensorGather(OneHotVector(k - 1, k), classIndices.Reshape(n), 0);
        var middleClass = engine.ScalarMinusTensor(NumOps.One, engine.TensorAdd(firstClass, lastClass));
        var factor = engine.TensorAdd(
            engine.TensorAdd(
                engine.TensorMultiply(firstClass, engine.ScalarMinusTensor(NumOps.One, cumHi)),
                engine.TensorMultiply(lastClass, engine.TensorNegate(cumLo))),
            engine.TensorMultiply(middleClass, engine.TensorDivide(engine.TensorSubtract(gammaLo, gammaHi), probs)));
        var gradCoef = engine.TensorMatMul(engine.TensorTranspose(features), factor.Reshape(n, 1)).Reshape(p);

        // Threshold gradients: -gamma_y / P at threshold y, +gamma_{y-1} / P at threshold y-1
        var padIndex = engine.TensorSubtract(lo, rowStarts);
        var thresholdGrad = new Tensor<T>(new[] { k + 1 });
        thresholdGrad = engine.TensorScatterAdd(thresholdGrad, engine.TensorAddScalar(padIndex, 1),
            engine.TensorNegate(engine.TensorDivide(gammaHi, probs)), 0);
        thresholdGrad = engine.TensorScatterAdd(thresholdGrad, padIndex, engine.TensorDivide(gammaLo, probs), 0);

        return (gradCoef.ToVector(), thresholdGrad.Slice(0, 1, k).ToVector(), loss);
    }

    /// <summary>
    /// A length-<paramref name="length"/> vector that is 1 at <paramref name="index"/> and 0 elsewhere.
    /// </summary>
    private Tensor<T> OneHotVector(int index, int length)
    {
        return Engine.TensorOneHot<T>(new Tensor<int>(new[] { index }, new[] { 1 }), length).Reshape(length);
    }

    /// <summary>
    /// Predicts ordinal class labels.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method predicts the most likely class for each sample
    /// by finding the class with the highest probability from the probability distribution.</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        return input.Rows == 0 ? new Vector<T>(0) : ClassIndicesToLabels(ArgMaxRows(probs));
    }

    /// <summary>
    /// Predicts cumulative probabilities P(Y ≤ k).
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <returns>Cumulative probability matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each sample and each threshold, this computes the
    /// probability of being in that class or lower. Uses the logistic function to convert
    /// the linear predictor (threshold - β·X) into a probability.</para>
    /// </remarks>
    public override Matrix<T> PredictCumulativeProbabilities(Matrix<T> features)
    {
        if (_coefficients is null || _thresholds is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int n = features.Rows;
        int K = NumClasses;
        if (n == 0 || K <= 1)
        {
            return new Matrix<T>(n, Math.Max(0, K - 1));
        }

        // P(Y <= k) = sigmoid(threshold_k - x * coef)
        var engine = Engine;
        var x = Tensor<T>.FromMatrix(features);
        if (features.Columns != NumFeatures)
        {
            x = x.Slice(1, 0, NumFeatures);
        }

        var eta = engine.TensorMatMul(x, _coefficients.AsTensor().Reshape(NumFeatures, 1));
        var thresholds = _thresholds.AsTensor().Reshape(1, K - 1);
        return engine.TensorSigmoid(engine.TensorSubtract(thresholds, eta)).ToMatrix();
    }

    /// <summary>
    /// Gets the model parameters (coefficients and thresholds).
    /// </summary>
    /// <returns>Vector containing all model parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This packs all learned parameters into a single vector
    /// for serialization or optimization purposes. The coefficients come first, followed
    /// by the thresholds.</para>
    /// </remarks>
    private Vector<T> PackParameters()
    {
        if (_coefficients is null || _thresholds is null)
        {
            return new Vector<T>(0);
        }

        return Vector<T>.Concatenate(_coefficients, _thresholds);
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Vector containing all model parameters.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This unpacks parameters from a single vector and sets
    /// the internal coefficients and thresholds. Used when loading a saved model.</para>
    /// </remarks>
    private void UnpackParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0)
        {
            return;
        }

        // If we have NumFeatures set, use it to determine split
        if (NumFeatures > 0 && NumClasses > 0)
        {
            int numCoef = NumFeatures;
            int numThresh = NumClasses - 1;

            if (parameters.Length == numCoef + numThresh)
            {
                _coefficients = parameters.GetSubVector(0, numCoef);
                _thresholds = parameters.GetSubVector(numCoef, numThresh);
                return;
            }
        }

        // Fallback: assume roughly equal split between coefficients and thresholds
        int numThresholds = (parameters.Length + 1) / 2;
        int numCoefs = parameters.Length - numThresholds;

        _coefficients = parameters.GetSubVector(0, numCoefs);
        _thresholds = parameters.GetSubVector(numCoefs, numThresholds);

        NumFeatures = numCoefs;
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">Parameters for the new instance.</param>
    /// <returns>New model instance with the specified parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh copy of the model with specific parameter
    /// values. Useful for optimization algorithms that explore different parameter settings.</para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var model = new OrdinalLogisticRegression<T>(_learningRate, _maxIterations, _tolerance, _regularizationStrength);
        model.NumFeatures = NumFeatures;
        model.NumClasses = NumClasses;
        if (ClassLabels is not null)
        {
            model.ClassLabels = ClassLabels.Clone();
        }
        model.SetParameters(parameters);
        return model;
    }

    /// <summary>
    /// Sanitizes random parameters to ensure ordinal constraints are satisfied.
    /// Thresholds (parameters after coefficients) must be monotonically increasing.
    /// </summary>
    public override Vector<T> SanitizeParameters(Vector<T> parameters)
    {
        if (parameters.Length <= NumFeatures) return parameters;

        // Copy coefficients as-is
        var coefficients = parameters.GetSubVector(0, NumFeatures);

        // Sort thresholds to ensure monotonically increasing order, then ensure a minimum gap between them
        int numThresholds = parameters.Length - NumFeatures;
        var sorted = Engine.TensorSort(parameters.GetSubVector(NumFeatures, numThresholds).AsTensor()).Values.ToVector();
        var thresholds = EnforceIncreasing(sorted, NumOps.FromDouble(0.1));

        return Vector<T>.Concatenate(coefficients, thresholds);
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <returns>Gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients indicate how changing each parameter would
    /// affect the prediction error. The optimizer uses these to improve the model.
    /// This computes the derivative of the negative log-likelihood with respect to
    /// each parameter.</para>
    /// </remarks>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        int K = NumClasses;
        int P = NumFeatures;
        var features = Tensor<T>.FromMatrix(input);
        if (input.Columns != P)
        {
            features = features.Slice(1, 0, P);
        }

        // NLL gradients (same sign convention as Train())
        var (gradCoef, gradThresh, _) = ComputeNegativeLogLikelihoodGradients(
            features, GetOrdinalClassIndices(target), FitToLength(_coefficients, P), FitToLength(_thresholds, K - 1));

        // Normalize by batch size
        T n = NumOps.FromDouble(input.Rows);
        return Vector<T>.Concatenate(Engine.Divide(gradCoef, n), Engine.Divide(gradThresh, n));
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate for the update.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This updates the model parameters by moving them in the
    /// opposite direction of the gradients (to reduce error). The learning rate controls
    /// how big the step is.</para>
    /// </remarks>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_coefficients is null || _thresholds is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var engine = Engine;
        int numCoefficients = _coefficients.Length;

        // Update coefficients and thresholds: w = w - lr * g
        _coefficients = engine.Subtract(_coefficients,
            engine.Multiply(gradients.GetSubVector(0, numCoefficients), learningRate));
        _thresholds = engine.Subtract(_thresholds,
            engine.Multiply(gradients.GetSubVector(numCoefficients, _thresholds.Length), learningRate));

        // Ensure thresholds are monotonically increasing
        _thresholds = EnforceIncreasing(_thresholds, NumOps.FromDouble(0.001));
    }

    /// <summary>
    /// Gets feature importance based on coefficient magnitude.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature importance tells you which features have the
    /// biggest impact on predictions. Features with larger absolute coefficient values
    /// are more important. Values are normalized to sum to 1.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        if (_coefficients is null)
        {
            return base.GetFeatureImportance();
        }

        var importance = new Dictionary<string, T>();
        if (_coefficients.Length == 0)
        {
            return importance;
        }

        // |coef| normalized by the sum of absolute coefficients (1 when they are all zero)
        var engine = Engine;
        var magnitudes = engine.TensorAbs(_coefficients.AsTensor());
        T total = engine.TensorSum(magnitudes);
        if (NumOps.Equals(total, NumOps.Zero)) total = NumOps.One;
        var values = engine.TensorDivideScalar(magnitudes, total).ToArray();

        for (int i = 0; i < values.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            importance[name] = values[i];
        }

        return importance;
    }
}
