using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// AutoInt implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// AutoIntRegression uses multi-head self-attention to automatically learn
/// feature interactions for regression on tabular data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use AutoInt for regression when:
/// - Feature interactions matter for prediction
/// - You don't want to manually engineer feature crosses
/// - You want interpretable interaction patterns
///
/// Example:
/// <code>
/// var options = new AutoIntOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 16,
///     NumLayers = 3,
///     NumHeads = 2
/// };
/// var regressor = new AutoIntRegression&lt;double&gt;(numFeatures: 10, outputDim: 1, options);
///
/// // Train and predict
/// var predictions = regressor.Predict(features);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AutoIntRegression<T> : AutoIntBase<T>
{
    private readonly int _outputDimension;
    private readonly FullyConnectedLayer<T> _regressionHead;

    // Cache
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
    /// Initializes a new instance of the AutoIntRegression class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="outputDimension">Number of output values to predict (default 1).</param>
    /// <param name="options">Model configuration options.</param>
    public AutoIntRegression(
        int numNumericalFeatures,
        int outputDimension = 1,
        AutoIntOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        _regressionHead = new FullyConnectedLayer<T>(
            MLPOutputDimension,
            outputDimension,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Performs the forward pass to get predictions.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var backboneOutput = ForwardBackbone(numericalFeatures, categoricalIndices);
        _backboneOutputCache = backboneOutput;

        var predictions = _regressionHead.Forward(backboneOutput);
        _predictionsCache = predictions;

        return predictions;
    }

    /// <summary>
    /// Makes predictions (alias for Forward).
    /// </summary>
    public Tensor<T> Predict(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        return Forward(numericalFeatures, categoricalIndices);
    }

    /// <summary>
    /// Computes the Mean Squared Error loss.
    /// </summary>
    public T ComputeMSELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!predictions.Shape.SequenceEqual(targets.Shape))
        {
            throw new ArgumentException(
                $"Predictions shape [{string.Join(", ", predictions.Shape)}] must match targets shape [{string.Join(", ", targets.Shape)}].");
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
        if (!predictions.Shape.SequenceEqual(targets.Shape))
        {
            throw new ArgumentException(
                $"Predictions shape [{string.Join(", ", predictions.Shape)}] must match targets shape [{string.Join(", ", targets.Shape)}].");
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
    /// Performs the backward pass using MSE loss gradients (2 * (prediction - target) / N).
    /// </summary>
    /// <remarks>
    /// This method computes gradients for <see cref="ComputeMSELoss"/> only.
    /// Do not use after computing <see cref="ComputeMAELoss"/>; the MAE loss is provided
    /// for evaluation purposes and does not have a corresponding backward pass.
    /// </remarks>
    /// <param name="targets">Target values with the same shape as the cached predictions.</param>
    /// <returns>Gradient with respect to the input features.</returns>
    public Tensor<T> Backward(Tensor<T> targets)
    {
        if (_predictionsCache == null || _backboneOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        if (targets.Length != _predictionsCache.Length)
        {
            throw new ArgumentException(
                $"Targets length ({targets.Length}) must match predictions length ({_predictionsCache.Length}).");
        }


        var predictionGrad = new Tensor<T>(_predictionsCache.Shape);
        var scale = NumOps.FromDouble(2.0 / _predictionsCache.Length);

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
    public T TrainStep(Tensor<T> numericalFeatures, Tensor<T> targets, T learningRate, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Forward(numericalFeatures, categoricalIndices);
        var loss = ComputeMSELoss(predictions, targets);
        _ = Backward(targets);
        UpdateParameters(learningRate);
        ResetState();

        return loss;
    }

    /// <summary>
    /// Computes the R² score (coefficient of determination) from precomputed predictions.
    /// </summary>
    /// <remarks>
    /// This overload does not call <see cref="Predict"/> and therefore does not
    /// overwrite internal caches, making it safe to use during a training loop.
    /// </remarks>
    public T ComputeR2Score(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!predictions.Shape.SequenceEqual(targets.Shape))
        {
            throw new ArgumentException(
                $"Predictions shape [{string.Join(", ", predictions.Shape)}] must match targets shape [{string.Join(", ", targets.Shape)}].");
        }

        var targetMean = NumOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            targetMean = NumOps.Add(targetMean, targets[i]);
        }
        targetMean = NumOps.Divide(targetMean, NumOps.FromDouble(targets.Length));

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
    /// Computes the R² score by running a forward pass on the given features.
    /// </summary>
    /// <remarks>
    /// This overload calls <see cref="Predict"/>, which overwrites internal caches
    /// (<c>_backboneOutputCache</c>, <c>_predictionsCache</c>). Do not call this
    /// between <see cref="Forward"/> and <see cref="Backward"/> in a training loop.
    /// Use the <see cref="ComputeR2Score(Tensor{T}, Tensor{T})"/> overload with
    /// precomputed predictions instead.
    /// </remarks>
    public T ComputeR2Score(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        return ComputeR2Score(predictions, targets);
    }

    /// <summary>
    /// Computes the Root Mean Squared Error from precomputed predictions.
    /// </summary>
    /// <remarks>
    /// This overload does not call <see cref="Predict"/> and therefore does not
    /// overwrite internal caches, making it safe to use during a training loop.
    /// </remarks>
    public T ComputeRMSE(Tensor<T> predictions, Tensor<T> targets)
    {
        var mse = ComputeMSELoss(predictions, targets);
        return NumOps.Sqrt(mse);
    }

    /// <summary>
    /// Computes the Root Mean Squared Error by running a forward pass on the given features.
    /// </summary>
    /// <remarks>
    /// This overload calls <see cref="Predict"/>, which overwrites internal caches.
    /// Use the <see cref="ComputeRMSE(Tensor{T}, Tensor{T})"/> overload with
    /// precomputed predictions to avoid side effects during training.
    /// </remarks>
    public T ComputeRMSE(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        return ComputeRMSE(predictions, targets);
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
