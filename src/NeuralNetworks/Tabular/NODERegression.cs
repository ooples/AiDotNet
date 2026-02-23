using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// NODE implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// NODERegression uses an ensemble of differentiable oblivious decision trees
/// for predicting continuous values on tabular data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use NODE for regression when you want:
/// - Interpretable models (see feature importance via GetFeatureImportance())
/// - Tree-based structure with neural network trainability
/// - Good performance on tabular data with continuous targets
///
/// Example:
/// <code>
/// var options = new NODEOptions&lt;double&gt;
/// {
///     NumTrees = 20,
///     TreeDepth = 6
/// };
/// var regressor = new NODERegression&lt;double&gt;(numFeatures: 10, outputDim: 1, options);
///
/// var predictions = regressor.Predict(features);
/// var importance = regressor.GetFeatureImportance();
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NODERegression<T> : NODEBase<T>
{
    private readonly int _outputDimension;
    private readonly FullyConnectedLayer<T> _regressionHead;

    // Cache for backward pass
    private Tensor<T>? _backboneOutputCache;
    private Tensor<T>? _predictionsCache;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public int OutputDimension => _outputDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _regressionHead.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the NODERegression class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="outputDimension">Number of output values to predict (default 1).</param>
    /// <param name="options">Model configuration options.</param>
    public NODERegression(
        int numFeatures,
        int outputDimension = 1,
        NODEOptions<T>? options = null)
        : base(numFeatures, options)
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        // Regression head maps from tree output to predictions
        _regressionHead = new FullyConnectedLayer<T>(
            TreeOutputDimension,
            outputDimension,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Performs the forward pass to get predictions.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Predicted values [batch_size, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> features)
    {
        var backboneOutput = ForwardBackbone(features);
        _backboneOutputCache = backboneOutput;

        var predictions = _regressionHead.Forward(backboneOutput);
        _predictionsCache = predictions;

        return predictions;
    }

    /// <summary>
    /// Makes predictions (alias for Forward).
    /// </summary>
    public Tensor<T> Predict(Tensor<T> features)
    {
        return Forward(features);
    }

    /// <summary>
    /// Computes the Mean Squared Error loss.
    /// </summary>
    public T ComputeMSELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        if (predictions.Length != targets.Length)
        {
            throw new ArgumentException("Predictions and targets must have the same size");
        }

        var totalLoss = NumOps.Zero;

        for (int i = 0; i < predictions.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Computes the Mean Absolute Error loss.
    /// </summary>
    public T ComputeMAELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        if (predictions.Length != targets.Length)
        {
            throw new ArgumentException("Predictions and targets must have the same size");
        }

        var totalLoss = NumOps.Zero;

        for (int i = 0; i < predictions.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            totalLoss = NumOps.Add(totalLoss, NumOps.Abs(diff));
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Performs the backward pass for MSE loss.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> targets)
    {
        if (_predictionsCache == null || _backboneOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _predictionsCache.Shape[0];
        int outputDim = _predictionsCache.Shape.Length > 1 ? _predictionsCache.Shape[1] : 1;

        // Gradient of MSE: 2 * (predictions - targets) / n
        var predictionGrad = new Tensor<T>(_predictionsCache.Shape);
        var scale = NumOps.FromDouble(2.0 / (batchSize * outputDim));

        for (int i = 0; i < _predictionsCache.Length; i++)
        {
            predictionGrad[i] = NumOps.Multiply(
                NumOps.Subtract(_predictionsCache[i], targets[i]),
                scale);
        }

        var backboneGrad = _regressionHead.Backward(predictionGrad);
        return BackwardBackbone(backboneGrad);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    public T TrainStep(Tensor<T> features, Tensor<T> targets, T learningRate)
    {
        var predictions = Forward(features);
        var loss = ComputeMSELoss(predictions, targets);
        _ = Backward(targets);
        UpdateParameters(learningRate);
        ResetState();

        return loss;
    }

    /// <summary>
    /// Computes the R² score (coefficient of determination).
    /// </summary>
    public T ComputeR2Score(Tensor<T> features, Tensor<T> targets)
    {
        var predictions = Predict(features);

        // Compute target mean
        var targetMean = NumOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            targetMean = NumOps.Add(targetMean, targets[i]);
        }
        targetMean = NumOps.Divide(targetMean, NumOps.FromDouble(targets.Length));

        // Compute SS_res and SS_tot
        var ssRes = NumOps.Zero;
        var ssTot = NumOps.Zero;

        for (int i = 0; i < targets.Length; i++)
        {
            var residual = NumOps.Subtract(targets[i], predictions[i]);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(residual, residual));

            var deviation = NumOps.Subtract(targets[i], targetMean);
            ssTot = NumOps.Add(ssTot, NumOps.Multiply(deviation, deviation));
        }

        if (NumOps.Compare(ssTot, NumOps.Zero) == 0)
        {
            return NumOps.Compare(ssRes, NumOps.Zero) == 0 ? NumOps.One : NumOps.Zero;
        }

        return NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));
    }

    /// <summary>
    /// Computes the Root Mean Squared Error.
    /// </summary>
    public T ComputeRMSE(Tensor<T> features, Tensor<T> targets)
    {
        var predictions = Predict(features);
        var mse = ComputeMSELoss(predictions, targets);
        return NumOps.Sqrt(mse);
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        base.UpdateParameters(learningRate);
        _regressionHead.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _backboneOutputCache = null;
        _predictionsCache = null;
        _regressionHead.ResetState();
    }
}
