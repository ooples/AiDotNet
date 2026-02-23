using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabTransformer implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabTransformerRegression applies transformer attention to categorical features
/// for predicting continuous values. Numerical features are concatenated after
/// the categorical embeddings are transformed.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use TabTransformer for regression when:
/// - You have important categorical features
/// - You believe there are interactions between categorical features
/// - You want to predict continuous values (prices, amounts, etc.)
///
/// Example:
/// <code>
/// var options = new TabTransformerOptions&lt;double&gt;
/// {
///     CategoricalCardinalities = new[] { 5, 10, 20 },
///     EmbeddingDimension = 32,
///     NumLayers = 6
/// };
/// var regressor = new TabTransformerRegression&lt;double&gt;(numNumerical: 5, outputDim: 1, options);
///
/// var predictions = regressor.Predict(numericalFeatures, categoricalIndices);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabTransformerRegression<T> : TabTransformerBase<T>
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
    /// Initializes a new instance of the TabTransformerRegression class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="outputDimension">Number of output values to predict (default 1).</param>
    /// <param name="options">Model configuration options.</param>
    public TabTransformerRegression(
        int numNumericalFeatures,
        int outputDimension = 1,
        TabTransformerOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        // Regression head
        _regressionHead = new FullyConnectedLayer<T>(
            MLPOutputDimension,
            outputDimension,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Performs the forward pass to get predictions.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical indices matrix [batch_size, num_categorical] or null.</param>
    /// <returns>Predicted values [batch_size, output_dim].</returns>
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
    /// Computes the R² score.
    /// </summary>
    public T ComputeR2Score(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);

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
    /// Computes the Root Mean Squared Error.
    /// </summary>
    public T ComputeRMSE(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
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
