using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Online Passive-Aggressive classifier for margin-based incremental learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Passive-Aggressive (PA) algorithms are a family of online learning algorithms that
/// aggressively update when a mistake is made but remain passive when the prediction is correct.
/// </para>
/// <para>
/// <b>For Beginners:</b> PA classifiers are like strict teachers:
///
/// - Passive: When the prediction is correct with good margin → do nothing
/// - Aggressive: When wrong or uncertain → update strongly to fix the mistake
///
/// How it works:
/// 1. Compute margin: y × (w·x) - how confident and correct is the prediction?
/// 2. If margin >= 1: Correct with good margin → stay passive
/// 3. If margin &lt; 1: Wrong or uncertain → aggressively update
///
/// The update is designed to:
/// - Correct the mistake with minimum change to weights
/// - Maintain a margin of at least 1 after update
///
/// PA variants:
/// - PA: Original, no regularization (can diverge with noise)
/// - PA-I: Adds slack variable, bounds the update size
/// - PA-II: Adds squared penalty, smoother updates
///
/// Advantages over SGD:
/// - No learning rate to tune (automatically determined)
/// - Fast convergence on linearly separable data
/// - Naturally handles the margin
///
/// Usage:
/// <code>
/// var classifier = new OnlinePassiveAggressiveClassifier&lt;double&gt;(C: 1.0, type: PAType.PA_II);
/// foreach (var (x, y) in dataStream)
/// {
///     classifier.PartialFit(x, y);  // y should be +1 or -1
///     double prediction = classifier.PredictSingle(x);
/// }
/// </code>
///
/// References:
/// - Crammer et al. (2006). "Online Passive-Aggressive Algorithms"
/// </para>
/// </remarks>
public class OnlinePassiveAggressiveClassifier<T> : OnlineLearningModelBase<T>
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
    /// PA variant type.
    /// </summary>
    private readonly PAType _paType;

    /// <summary>
    /// Whether to fit an intercept (bias) term.
    /// </summary>
    private readonly bool _fitIntercept;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.OnlinePassiveAggressiveClassifier;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Initializes a new instance of the OnlinePassiveAggressiveClassifier class.
    /// </summary>
    /// <param name="c">Regularization parameter (aggressiveness). Default is 1.0.</param>
    /// <param name="type">PA variant type. Default is PA-I.</param>
    /// <param name="fitIntercept">Whether to fit an intercept (bias) term. Default is true.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters:
    ///
    /// - C: Controls aggressiveness vs. stability tradeoff
    ///   - Higher C: More aggressive updates (faster learning, may overfit noise)
    ///   - Lower C: Smaller updates (more stable, slower learning)
    ///   - Default (1.0): Good balance for most cases
    ///
    /// - PA variants:
    ///   - PA: No regularization - update = hinge_loss / ||x||²
    ///   - PA_I: Bounded update - update = min(C, hinge_loss / ||x||²)
    ///   - PA_II: Smooth update - update = hinge_loss / (||x||² + 1/(2C))
    ///
    /// PA_I and PA_II are preferred for noisy data.
    /// </para>
    /// </remarks>
    public OnlinePassiveAggressiveClassifier(
        double c = 1.0,
        PAType type = PAType.PA_I,
        bool fitIntercept = true)
        : base(1.0, LearningRateSchedule.Constant)  // PA doesn't use learning rate
    {
        _c = c;
        _paType = type;
        _fitIntercept = fitIntercept;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Updates the model with a single training example.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PA update rule:
    /// 1. Compute margin: m = y × (w·x + b)
    /// 2. Compute hinge loss: L = max(0, 1 - m)
    /// 3. If L > 0 (wrong or uncertain):
    ///    - Compute step size τ based on PA variant
    ///    - Update: w = w + τ × y × x
    ///
    /// The step size τ is computed to minimize ||w_new - w_old|| while satisfying
    /// the margin constraint y × (w_new·x) >= 1.
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

        // Convert y to +1 or -1
        double yDouble = NumOps.ToDouble(y);
        double yLabel = yDouble > 0 ? 1.0 : -1.0;

        // Compute score and margin
        double score = ComputeScore(x);
        double margin = yLabel * score;

        // Compute hinge loss
        double hingeLoss = Math.Max(0, 1.0 - margin);

        // Only update if loss > 0 (mistake or insufficient margin)
        if (hingeLoss > 0)
        {
            // Compute squared norm of x (plus 1 if fitting intercept)
            double normSq = 0;
            for (int i = 0; i < NumFeatures; i++)
            {
                double xi = NumOps.ToDouble(x[i]);
                normSq += xi * xi;
            }
            if (_fitIntercept)
            {
                normSq += 1.0;  // intercept contributes 1 to norm squared
            }

            // Compute step size based on PA variant
            double tau = ComputeStepSize(hingeLoss, normSq);

            // Update weights: w = w + τ × y × x
            for (int i = 0; i < NumFeatures; i++)
            {
                double xi = NumOps.ToDouble(x[i]);
                double wi = NumOps.ToDouble(_weights[i]);
                wi += tau * yLabel * xi;
                _weights[i] = NumOps.FromDouble(wi);
            }

            // Update bias
            if (_fitIntercept)
            {
                double b = NumOps.ToDouble(_bias);
                b += tau * yLabel;
                _bias = NumOps.FromDouble(b);
            }
        }

        SampleCount++;
    }

    /// <summary>
    /// Computes the step size based on PA variant.
    /// </summary>
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
    /// Computes the score (before sign function).
    /// </summary>
    private double ComputeScore(Vector<T> x)
    {
        double score = NumOps.ToDouble(_bias);
        for (int i = 0; i < NumFeatures; i++)
        {
            score += NumOps.ToDouble(_weights![i]) * NumOps.ToDouble(x[i]);
        }
        return score;
    }

    /// <summary>
    /// Predicts the class label for a single sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns +1 or -1 based on the sign of the score.
    /// Use DecisionFunction to get the raw score if needed.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> x)
    {
        double score = ComputeScore(x);
        return NumOps.FromDouble(score >= 0 ? 1.0 : -1.0);
    }

    /// <summary>
    /// Gets the decision function value (raw score before sign).
    /// </summary>
    /// <param name="x">The feature vector.</param>
    /// <returns>Raw score - larger positive = more confident class 1.</returns>
    public T DecisionFunction(Vector<T> x)
    {
        return NumOps.FromDouble(ComputeScore(x));
    }

    /// <summary>
    /// Converts predictions to 0/1 format for compatibility.
    /// </summary>
    public T PredictBinary(Vector<T> x)
    {
        double score = ComputeScore(x);
        return NumOps.FromDouble(score >= 0 ? 1.0 : 0.0);
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
        var newModel = new OnlinePassiveAggressiveClassifier<T>(_c, _paType, _fitIntercept);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new OnlinePassiveAggressiveClassifier<T>(_c, _paType, _fitIntercept);
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
    /// <returns>The output computation node representing the raw score (decision function).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Passive-Aggressive classifier's decision function is w·x + b.
    /// The raw score can be JIT compiled for faster batch inference. For classification:
    /// - score > 0 → predict class +1
    /// - score ≤ 0 → predict class -1
    ///
    /// The computation graph returns the raw score, and the caller can apply sign()
    /// for classification or use the raw score for ranking.
    ///
    /// For a differentiable approximation of sign, the graph uses tanh with a steepness
    /// parameter that approximates the step function.
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

        // Add bias: score = linearPred + bias
        var scoreNode = TensorOperations<T>.Add(linearPredNode, biasNode);

        // Apply tanh for differentiable approximation of sign
        // tanh(steepness * x) approaches sign(x) as steepness → ∞
        // Use steepness = 10 for a good approximation that's still differentiable
        var steepnessTensor = new Tensor<T>(new int[] { 1, 1 });
        steepnessTensor[0, 0] = NumOps.FromDouble(10.0);
        var steepnessNode = TensorOperations<T>.Constant(steepnessTensor, "steepness");

        var scaledScoreNode = TensorOperations<T>.ElementwiseMultiply(scoreNode, steepnessNode);
        var outputNode = TensorOperations<T>.Tanh(scaledScoreNode);
        outputNode.Name = "decision_function";

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
    /// Computes the hinge loss on the provided data.
    /// </summary>
    public T GetHingeLoss(Matrix<T> x, Vector<T> y)
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

            double score = ComputeScore(xi);
            double yLabel = NumOps.ToDouble(y[i]) > 0 ? 1.0 : -1.0;
            double margin = yLabel * score;
            totalLoss += Math.Max(0, 1.0 - margin);
        }

        return NumOps.FromDouble(totalLoss / x.Rows);
    }
}

/// <summary>
/// Passive-Aggressive algorithm variants.
/// </summary>
public enum PAType
{
    /// <summary>
    /// Original PA: No regularization. τ = loss / ||x||²
    /// </summary>
    PA,

    /// <summary>
    /// PA-I: Bounded updates. τ = min(C, loss / ||x||²)
    /// </summary>
    PA_I,

    /// <summary>
    /// PA-II: Smooth updates. τ = loss / (||x||² + 1/(2C))
    /// </summary>
    PA_II
}
