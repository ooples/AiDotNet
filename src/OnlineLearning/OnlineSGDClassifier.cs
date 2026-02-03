using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Online Stochastic Gradient Descent classifier for incremental learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Online SGD Classifier implements logistic regression with SGD updates, allowing
/// the model to learn incrementally from streaming data.
/// </para>
/// <para>
/// <b>For Beginners:</b> This classifier learns to separate two classes using a linear
/// decision boundary, updating itself one example at a time.
///
/// How it works:
/// 1. Compute prediction: P(y=1) = sigmoid(w·x + b)
/// 2. Compare with true label: error = prediction - truth
/// 3. Update weights: w = w - learning_rate × error × x
///
/// The model "nudges" itself toward correct predictions with each example.
/// Over many examples, it converges to a good decision boundary.
///
/// Advantages:
/// - Handles streaming data naturally
/// - Memory-efficient (doesn't store data)
/// - Can adapt to changing patterns
///
/// Supports:
/// - L1 regularization (sparse weights)
/// - L2 regularization (smooth weights)
/// - Elastic Net (combination)
///
/// Usage:
/// <code>
/// var classifier = new OnlineSGDClassifier&lt;double&gt;(learningRate: 0.01, l2Penalty: 0.001);
/// foreach (var (x, y) in dataStream)
/// {
///     classifier.PartialFit(x, y);
///     double prob = classifier.PredictProbability(x);
/// }
/// </code>
///
/// References:
/// - Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
/// </para>
/// </remarks>
public class OnlineSGDClassifier<T> : OnlineLearningModelBase<T>
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
    /// L1 regularization strength.
    /// </summary>
    private readonly double _l1Penalty;

    /// <summary>
    /// L2 regularization strength.
    /// </summary>
    private readonly double _l2Penalty;

    /// <summary>
    /// Whether to fit an intercept (bias) term.
    /// </summary>
    private readonly bool _fitIntercept;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.OnlineSGDClassifier;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This classifier supports JIT compilation since the prediction
    /// is just a matrix-vector product followed by sigmoid: sigmoid(w·x + b).
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Initializes a new instance of the OnlineSGDClassifier class.
    /// </summary>
    /// <param name="learningRate">Initial learning rate. Default is 0.01.</param>
    /// <param name="learningRateSchedule">Learning rate schedule. Default is InverseScaling.</param>
    /// <param name="l1Penalty">L1 (lasso) regularization strength. Default is 0.</param>
    /// <param name="l2Penalty">L2 (ridge) regularization strength. Default is 0.0001.</param>
    /// <param name="fitIntercept">Whether to fit an intercept (bias) term. Default is true.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters control how the model learns:
    ///
    /// - learningRate: How big each update step is
    /// - learningRateSchedule: How learning rate changes over time
    /// - l1Penalty: Encourages sparse weights (feature selection)
    /// - l2Penalty: Prevents weights from getting too large (regularization)
    /// - fitIntercept: Whether to learn a bias term
    ///
    /// Regularization prevents overfitting - the model won't rely too heavily on any single feature.
    /// </para>
    /// </remarks>
    public OnlineSGDClassifier(
        double learningRate = 0.01,
        LearningRateSchedule learningRateSchedule = LearningRateSchedule.InverseScaling,
        double l1Penalty = 0.0,
        double l2Penalty = 0.0001,
        bool fitIntercept = true)
        : base(learningRate, learningRateSchedule)
    {
        _l1Penalty = l1Penalty;
        _l2Penalty = l2Penalty;
        _fitIntercept = fitIntercept;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Updates the model with a single training example.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each example:
    /// 1. Make a prediction using current weights
    /// 2. Calculate the error (prediction - actual)
    /// 3. Update each weight proportionally to error × feature value
    /// 4. Apply regularization to prevent overfitting
    /// </para>
    /// </remarks>
    public override void PartialFit(Vector<T> x, T y)
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
        double prediction = ComputeProbability(x);
        double yDouble = NumOps.ToDouble(y);

        // Ensure y is 0 or 1
        if (yDouble != 0.0 && yDouble != 1.0)
        {
            yDouble = yDouble > 0.5 ? 1.0 : 0.0;
        }

        // Compute error (gradient of log loss)
        double error = prediction - yDouble;

        // Get current learning rate
        double lr = NumOps.ToDouble(GetLearningRate());

        // Update weights
        for (int i = 0; i < NumFeatures; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            double wi = NumOps.ToDouble(_weights[i]);

            // Gradient of log loss: error × x
            double gradient = error * xi;

            // Add L2 regularization gradient: l2 × w
            gradient += _l2Penalty * wi;

            // Update weight
            wi -= lr * gradient;

            // Apply L1 regularization (proximal gradient / soft thresholding)
            if (_l1Penalty > 0)
            {
                double threshold = lr * _l1Penalty;
                wi = SoftThreshold(wi, threshold);
            }

            _weights[i] = NumOps.FromDouble(wi);
        }

        // Update bias
        if (_fitIntercept)
        {
            double b = NumOps.ToDouble(_bias);
            b -= lr * error;
            _bias = NumOps.FromDouble(b);
        }

        SampleCount++;
    }

    /// <summary>
    /// Soft thresholding operator for L1 regularization.
    /// </summary>
    private static double SoftThreshold(double x, double threshold)
    {
        if (x > threshold)
            return x - threshold;
        if (x < -threshold)
            return x + threshold;
        return 0.0;
    }

    /// <summary>
    /// Computes the probability of class 1.
    /// </summary>
    private double ComputeProbability(Vector<T> x)
    {
        double linearPred = NumOps.ToDouble(_bias);
        for (int i = 0; i < NumFeatures; i++)
        {
            linearPred += NumOps.ToDouble(_weights![i]) * NumOps.ToDouble(x[i]);
        }
        return Sigmoid(linearPred);
    }

    /// <summary>
    /// Sigmoid function.
    /// </summary>
    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        else
        {
            double exp = Math.Exp(x);
            return exp / (1.0 + exp);
        }
    }

    /// <summary>
    /// Predicts the target value for a single sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns 0 or 1 based on whether the probability exceeds 0.5.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> x)
    {
        double prob = ComputeProbability(x);
        return NumOps.FromDouble(prob >= 0.5 ? 1.0 : 0.0);
    }

    /// <summary>
    /// Predicts the probability of class 1.
    /// </summary>
    /// <param name="x">The feature vector.</param>
    /// <returns>Probability between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a continuous probability, not just 0/1.
    /// Useful for ranking, calibration, or when you need confidence levels.
    /// </para>
    /// </remarks>
    public T PredictProbability(Vector<T> x)
    {
        return NumOps.FromDouble(ComputeProbability(x));
    }

    /// <summary>
    /// Predicts probabilities for all samples.
    /// </summary>
    public Vector<T> PredictProbabilities(Matrix<T> x)
    {
        var probs = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            var xi = new Vector<T>(x.Columns);
            for (int j = 0; j < x.Columns; j++)
            {
                xi[j] = x[i, j];
            }
            probs[i] = PredictProbability(xi);
        }
        return probs;
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
        var newModel = new OnlineSGDClassifier<T>(
            InitialLearningRate, LearningRateScheduleType, _l1Penalty, _l2Penalty, _fitIntercept);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OnlineSGDClassifier<T>(
            InitialLearningRate, LearningRateScheduleType, _l1Penalty, _l2Penalty, _fitIntercept);
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

    /// <summary>
    /// Gets the weights vector.
    /// </summary>
    public Vector<T>? GetWeights() => _weights;

    /// <summary>
    /// Gets the bias (intercept) term.
    /// </summary>
    public T GetBias() => _bias;

    /// <summary>
    /// Computes the score used for decision (before sigmoid).
    /// </summary>
    /// <param name="x">Feature vector.</param>
    /// <returns>Linear combination of features with weights.</returns>
    public T DecisionFunction(Vector<T> x)
    {
        if (_weights is null)
        {
            return NumOps.Zero;
        }

        double score = NumOps.ToDouble(_bias);
        for (int i = 0; i < Math.Min(x.Length, _weights.Length); i++)
        {
            score += NumOps.ToDouble(_weights[i]) * NumOps.ToDouble(x[i]);
        }

        return NumOps.FromDouble(score);
    }
}
