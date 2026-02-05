using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabPFN implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabPFNRegression uses in-context learning for tabular regression.
/// It takes training data as context and makes predictions on test data
/// in a single forward pass using attention mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabPFN regression works similarly to classification:
///
/// 1. First, call SetContext() with your training data and target values
/// 2. Then, call Predict() with test data
/// 3. The model uses attention to "learn" the regression pattern from context
///
/// Example:
/// <code>
/// var options = new TabPFNOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     NumLayers = 12
/// };
/// var regressor = new TabPFNRegression&lt;double&gt;(numFeatures: 20, outputDim: 1, options);
///
/// // Set training data as context
/// regressor.SetContext(trainFeatures, trainTargets);
///
/// // Make predictions on test data
/// var predictions = regressor.Predict(testFeatures);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabPFNRegression<T> : TabPFNBase<T>
{
    private readonly int _outputDimension;
    private readonly FullyConnectedLayer<T> _regressionHead;

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
    /// Initializes a new instance of the TabPFNRegression class.
    /// </summary>
    public TabPFNRegression(
        int numNumericalFeatures,
        int outputDimension = 1,
        TabPFNOptions<T>? options = null)
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
    /// Sets the context (training) data for in-context learning.
    /// </summary>
    /// <param name="features">Training features.</param>
    /// <param name="targets">Training target values.</param>
    public new void SetContext(Tensor<T> features, Tensor<T> targets)
    {
        base.SetContext(features, targets);
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
    /// Predicts with ensemble averaging over multiple permutations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble prediction runs the model multiple times
    /// with different orderings of the context data and averages the results.
    /// This can improve prediction accuracy and reliability.
    /// </para>
    /// </remarks>
    public Tensor<T> PredictEnsemble(Tensor<T> numericalFeatures, int numEnsembles = 16, Matrix<int>? categoricalIndices = null)
    {
        var aggregatedPreds = new Tensor<T>([numericalFeatures.Shape[0], _outputDimension]);

        for (int e = 0; e < numEnsembles; e++)
        {
            var preds = Predict(numericalFeatures, categoricalIndices);

            // Accumulate predictions
            for (int i = 0; i < preds.Length; i++)
            {
                aggregatedPreds[i] = NumOps.Add(aggregatedPreds[i], preds[i]);
            }
        }

        // Average
        var scale = NumOps.FromDouble(1.0 / numEnsembles);
        for (int i = 0; i < aggregatedPreds.Length; i++)
        {
            aggregatedPreds[i] = NumOps.Multiply(aggregatedPreds[i], scale);
        }

        return aggregatedPreds;
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
    /// Performs a single training step (for fine-tuning).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Note:</b> TabPFN is designed for zero-shot inference. Training is optional
    /// and primarily useful for domain-specific fine-tuning.
    /// </para>
    /// </remarks>
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
    /// Computes the R² score (coefficient of determination).
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
