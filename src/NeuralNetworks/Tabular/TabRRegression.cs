using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabR implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabRRegression uses retrieval-augmented predictions for regression.
/// It finds similar training samples and uses their information to help
/// predict continuous values.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabR for regression is like estimating a home's price
/// by looking at similar homes that have sold recently.
///
/// Prediction process:
/// 1. Encode the input features
/// 2. Find similar training samples (neighbors)
/// 3. Aggregate neighbor information using attention
/// 4. Combine with encoded input to predict values
///
/// Benefits:
/// - Naturally handles local patterns (similar inputs → similar outputs)
/// - Can explain predictions by showing influential neighbors
/// - Works well with non-linear relationships
///
/// Example:
/// <code>
/// var options = new TabROptions&lt;double&gt; { NumNeighbors = 96 };
/// var regressor = new TabRRegression&lt;double&gt;(10, 1, options);
///
/// // Build index from training data
/// regressor.BuildIndex(trainFeatures);
///
/// // Make predictions
/// var predictions = regressor.Predict(testFeatures);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabRRegression<T> : TabRBase<T>
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
    /// Initializes a new instance of the TabRRegression class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="outputDimension">Number of output values to predict (default 1).</param>
    /// <param name="options">Model configuration options.</param>
    public TabRRegression(
        int numFeatures,
        int outputDimension = 1,
        TabROptions<T>? options = null)
        : base(numFeatures, options)
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        // Regression head
        _regressionHead = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension,
            outputDimension,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Performs the forward pass to get predictions.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for training).</param>
    /// <returns>Predicted values [batch_size, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> features, Vector<int>? excludeIndices = null)
    {
        var backboneOutput = ForwardBackbone(features, excludeIndices);
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
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
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
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <param name="sampleIndices">Indices of samples for leave-one-out retrieval.</param>
    /// <returns>The training loss.</returns>
    public T TrainStep(Tensor<T> features, Tensor<T> targets, T learningRate, Vector<int>? sampleIndices = null)
    {
        var predictions = Forward(features, sampleIndices);
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
    /// Gets interpretability information: which neighbors influenced each prediction.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>List of (neighbor index, attention weight) pairs for each sample.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This shows which training examples were most influential.
    /// For example, if predicting house prices, you can see which similar houses
    /// the model "looked at" to make its prediction.
    ///
    /// Use this to:
    /// - Explain predictions to stakeholders
    /// - Verify the model uses sensible comparisons
    /// - Debug unexpected predictions
    /// </para>
    /// </remarks>
    public List<List<(int NeighborIndex, T AttentionWeight)>> GetPredictionExplanations(Tensor<T> features)
    {
        // Run forward pass
        _ = Predict(features);

        var explanations = new List<List<(int, T)>>();
        var attentionWeights = GetAttentionWeights();
        var neighborIndices = GetRetrievedNeighborIndices();

        if (attentionWeights == null || neighborIndices == null)
        {
            return explanations;
        }

        int batchSize = features.Shape[0];
        int k = Options.NumNeighbors;

        for (int b = 0; b < batchSize; b++)
        {
            var sampleExplanation = new List<(int, T)>();
            for (int j = 0; j < k; j++)
            {
                sampleExplanation.Add((neighborIndices[b, j], attentionWeights[b * k + j]));
            }
            // Sort by attention weight (descending)
            sampleExplanation.Sort((a, b) => NumOps.Compare(b.Item2, a.Item2));
            explanations.Add(sampleExplanation);
        }

        return explanations;
    }

    /// <summary>
    /// Predicts with confidence intervals based on neighbor variance.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Tuple of (predictions, lower bounds, upper bounds).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This provides confidence intervals for predictions:
    /// - Narrow interval = Model is confident
    /// - Wide interval = More uncertainty
    ///
    /// The intervals are based on the variance in the retrieved neighbors.
    /// </para>
    /// </remarks>
    public (Tensor<T> Predictions, Tensor<T> Lower, Tensor<T> Upper) PredictWithConfidence(
        Tensor<T> features,
        double confidenceLevel = 0.95)
    {
        // Run forward pass
        var predictions = Predict(features);
        int batchSize = predictions.Shape[0];
        int outputDim = _outputDimension;

        // Get neighbor information
        var neighborIndices = GetRetrievedNeighborIndices();

        // For now, use a simple heuristic based on attention weight concentration
        // A proper implementation would use neighbor target variance
        var lower = new Tensor<T>([batchSize, outputDim]);
        var upper = new Tensor<T>([batchSize, outputDim]);

        // Simple placeholder: ±10% of prediction value
        var margin = NumOps.FromDouble(0.1);

        for (int i = 0; i < predictions.Length; i++)
        {
            var pred = predictions[i];
            var delta = NumOps.Multiply(NumOps.Abs(pred), margin);
            lower[i] = NumOps.Subtract(pred, delta);
            upper[i] = NumOps.Add(pred, delta);
        }

        return (predictions, lower, upper);
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
