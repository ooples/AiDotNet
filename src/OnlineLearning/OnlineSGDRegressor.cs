using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Online Stochastic Gradient Descent regressor for incremental learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Online SGD Regressor implements linear regression with SGD updates, allowing
/// the model to learn incrementally from streaming data.
/// </para>
/// <para>
/// <b>For Beginners:</b> This regressor predicts continuous values using a linear model,
/// updating itself one example at a time.
///
/// How it works:
/// 1. Compute prediction: ŷ = w·x + b
/// 2. Compare with true value: error = prediction - truth
/// 3. Update weights: w = w - learning_rate × error × x
///
/// Over many examples, the model converges to the best linear fit.
///
/// Supports multiple loss functions:
/// - Squared error (default): Sensitive to outliers
/// - Huber loss: Robust to outliers
/// - Epsilon-insensitive: SVR-like, ignores errors smaller than epsilon
///
/// Usage:
/// <code>
/// var regressor = new OnlineSGDRegressor&lt;double&gt;(learningRate: 0.01, loss: SGDLossType.Huber);
/// foreach (var (x, y) in dataStream)
/// {
///     regressor.PartialFit(x, y);
///     double prediction = regressor.PredictSingle(x);
/// }
/// </code>
///
/// References:
/// - Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
/// </para>
/// </remarks>
public class OnlineSGDRegressor<T> : OnlineLearningModelBase<T>
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
    /// The loss function type.
    /// </summary>
    private readonly SGDLossType _lossType;

    /// <summary>
    /// Epsilon for epsilon-insensitive and Huber loss.
    /// </summary>
    private readonly double _epsilon;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.OnlineSGDRegressor;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Initializes a new instance of the OnlineSGDRegressor class.
    /// </summary>
    /// <param name="learningRate">Initial learning rate. Default is 0.01.</param>
    /// <param name="learningRateSchedule">Learning rate schedule. Default is InverseScaling.</param>
    /// <param name="l1Penalty">L1 (lasso) regularization strength. Default is 0.</param>
    /// <param name="l2Penalty">L2 (ridge) regularization strength. Default is 0.0001.</param>
    /// <param name="fitIntercept">Whether to fit an intercept (bias) term. Default is true.</param>
    /// <param name="loss">Loss function type. Default is SquaredError.</param>
    /// <param name="epsilon">Epsilon for Huber and epsilon-insensitive loss. Default is 0.1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters:
    ///
    /// Loss functions:
    /// - SquaredError: (prediction - y)², standard MSE, sensitive to outliers
    /// - Huber: Squared for small errors, linear for large errors (robust)
    /// - EpsilonInsensitive: Ignores errors smaller than epsilon (SVR-like)
    ///
    /// Epsilon:
    /// - For Huber: transition point between squared and linear
    /// - For EpsilonInsensitive: tolerance band around prediction
    /// </para>
    /// </remarks>
    public OnlineSGDRegressor(
        double learningRate = 0.01,
        LearningRateSchedule learningRateSchedule = LearningRateSchedule.InverseScaling,
        double l1Penalty = 0.0,
        double l2Penalty = 0.0001,
        bool fitIntercept = true,
        SGDLossType loss = SGDLossType.SquaredError,
        double epsilon = 0.1)
        : base(learningRate, learningRateSchedule)
    {
        _l1Penalty = l1Penalty;
        _l2Penalty = l2Penalty;
        _fitIntercept = fitIntercept;
        _lossType = loss;
        _epsilon = epsilon;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Updates the model with a single training example.
    /// </summary>
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
        double prediction = ComputePrediction(x);
        double target = NumOps.ToDouble(y);

        // Compute gradient based on loss type
        double gradientMultiplier = ComputeLossGradient(prediction, target);

        // Get current learning rate
        double lr = NumOps.ToDouble(GetLearningRate());

        // Update weights
        for (int i = 0; i < NumFeatures; i++)
        {
            double xi = NumOps.ToDouble(x[i]);
            double wi = NumOps.ToDouble(_weights[i]);

            // Gradient of loss
            double gradient = gradientMultiplier * xi;

            // Add L2 regularization gradient
            gradient += _l2Penalty * wi;

            // Update weight
            wi -= lr * gradient;

            // Apply L1 regularization (soft thresholding)
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
            b -= lr * gradientMultiplier;
            _bias = NumOps.FromDouble(b);
        }

        SampleCount++;
    }

    /// <summary>
    /// Computes the gradient of the loss function.
    /// </summary>
    private double ComputeLossGradient(double prediction, double target)
    {
        double residual = prediction - target;

        return _lossType switch
        {
            SGDLossType.SquaredError => 2.0 * residual,
            SGDLossType.Huber => ComputeHuberGradient(residual),
            SGDLossType.EpsilonInsensitive => ComputeEpsilonInsensitiveGradient(residual),
            _ => 2.0 * residual
        };
    }

    /// <summary>
    /// Huber loss gradient: linear for large errors, quadratic for small.
    /// </summary>
    private double ComputeHuberGradient(double residual)
    {
        double absRes = Math.Abs(residual);
        if (absRes <= _epsilon)
        {
            return 2.0 * residual;
        }
        else
        {
            return 2.0 * _epsilon * Math.Sign(residual);
        }
    }

    /// <summary>
    /// Epsilon-insensitive loss gradient: zero for small errors.
    /// </summary>
    private double ComputeEpsilonInsensitiveGradient(double residual)
    {
        double absRes = Math.Abs(residual);
        if (absRes <= _epsilon)
        {
            return 0.0;
        }
        else
        {
            return Math.Sign(residual);
        }
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
    /// Computes the prediction for a sample.
    /// </summary>
    private double ComputePrediction(Vector<T> x)
    {
        double linearPred = NumOps.ToDouble(_bias);
        for (int i = 0; i < NumFeatures; i++)
        {
            linearPred += NumOps.ToDouble(_weights![i]) * NumOps.ToDouble(x[i]);
        }
        return linearPred;
    }

    /// <summary>
    /// Predicts the target value for a single sample.
    /// </summary>
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
        var newModel = new OnlineSGDRegressor<T>(
            InitialLearningRate, LearningRateScheduleType, _l1Penalty, _l2Penalty,
            _fitIntercept, _lossType, _epsilon);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OnlineSGDRegressor<T>(
            InitialLearningRate, LearningRateScheduleType, _l1Penalty, _l2Penalty,
            _fitIntercept, _lossType, _epsilon);
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

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Online SGD Regressor's prediction is a linear function: y = w·x + b.
    /// This can be JIT compiled for faster batch inference. The computation graph is:
    /// 1. Input X (features)
    /// 2. Weights W (constants after training)
    /// 3. Bias b (constant)
    /// 4. Matrix multiplication: X @ W
    /// 5. Add bias: result + b
    ///
    /// This enables optimized SIMD operations for batch prediction.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!IsInitialized || _weights is null)
        {
            throw new InvalidOperationException(
                "Model must be fitted before exporting computation graph.");
        }

        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        // Create input placeholder for features: [batchSize, numFeatures]
        var inputTensor = new Tensor<T>(new int[] { 1, NumFeatures });
        var inputNode = TensorOperations<T>.Variable(inputTensor, "features");
        inputNodes.Add(inputNode);

        // Create constant node for weights: [numFeatures, 1]
        var weightsTensor = new Tensor<T>(new int[] { NumFeatures, 1 });
        for (int i = 0; i < NumFeatures; i++)
        {
            weightsTensor[i, 0] = _weights[i];
        }
        var weightsNode = TensorOperations<T>.Constant(weightsTensor, "weights");
        inputNodes.Add(weightsNode);

        // Create constant node for bias: [1, 1]
        var biasTensor = new Tensor<T>(new int[] { 1, 1 });
        biasTensor[0, 0] = _bias;
        var biasNode = TensorOperations<T>.Constant(biasTensor, "bias");
        inputNodes.Add(biasNode);

        // Matrix multiplication: linearPred = X @ W, shape [batchSize, 1]
        var linearPredNode = TensorOperations<T>.MatrixMultiply(inputNode, weightsNode);

        // Add bias: prediction = linearPred + bias
        var outputNode = TensorOperations<T>.Add(linearPredNode, biasNode);
        outputNode.Name = "prediction";

        return outputNode;
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
    /// Gets the R-squared score on the provided data.
    /// </summary>
    public T Score(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("X and y must have the same number of samples.");
        }

        // Compute predictions
        var predictions = Predict(x);

        // Compute mean of y
        double meanY = 0;
        for (int i = 0; i < y.Length; i++)
        {
            meanY += NumOps.ToDouble(y[i]);
        }
        meanY /= y.Length;

        // Compute SS_res and SS_tot
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = NumOps.ToDouble(y[i]);
            double pi = NumOps.ToDouble(predictions[i]);
            ssRes += (yi - pi) * (yi - pi);
            ssTot += (yi - meanY) * (yi - meanY);
        }

        double r2 = ssTot > 0 ? 1.0 - ssRes / ssTot : 0;
        return NumOps.FromDouble(Math.Max(0, r2)); // Clamp to non-negative
    }
}

/// <summary>
/// Loss function types for SGD regression.
/// </summary>
public enum SGDLossType
{
    /// <summary>
    /// Squared error loss: (y - ŷ)². Standard MSE, sensitive to outliers.
    /// </summary>
    SquaredError,

    /// <summary>
    /// Huber loss: Squared for small errors, linear for large errors. Robust to outliers.
    /// </summary>
    Huber,

    /// <summary>
    /// Epsilon-insensitive loss: Zero for errors smaller than epsilon. SVR-like behavior.
    /// </summary>
    EpsilonInsensitive
}
