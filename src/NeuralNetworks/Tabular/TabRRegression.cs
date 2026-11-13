using System;
using AiDotNet.Distributions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TabR: Tabular Deep Learning Meets Nearest Neighbors",
    "https://arxiv.org/abs/2307.14338",
    Year = 2024,
    Authors = "Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, Artem Babenko")]
public class TabRRegression<T> : TabRBase<T>
{
    private readonly int _outputDimension;
    private readonly FullyConnectedLayer<T> _regressionHead;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public int OutputDimension => _outputDimension;

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
        : base(numFeatures, options, Math.Max(outputDimension, 0))
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        // Regression head
        _regressionHead = new FullyConnectedLayer<T>(
            EmbeddingDimension,
            outputDimension,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Builds the retrieval index from training features and their targets, so every retrieved neighbour
    /// contributes its target embedding (Gorishniy et al. 2024, eq. 5) and
    /// <see cref="PredictWithConfidence"/> can measure the neighbours' spread.
    /// </summary>
    /// <param name="features">Training features [num_samples, num_features].</param>
    /// <param name="targets">Training targets [num_samples, output_dim].</param>
    public void BuildIndex(Tensor<T> features, Tensor<T> targets)
    {
        BuildIndexWithLabels(features, targets);
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

        var predictions = _regressionHead.Forward(backboneOutput);

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

        return TabularMetrics.MeanSquaredError(predictions, targets);
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

        return TabularMetrics.MeanAbsoluteError(predictions, targets);
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
        return TrainWithTape(
            () => TabularTraining.MeanSquaredErrorObjective(Forward(features, sampleIndices), targets),
            learningRate);
    }

    /// <summary>
    /// Computes the R² score (coefficient of determination).
    /// </summary>
    public T ComputeR2Score(Tensor<T> features, Tensor<T> targets)
    {
        var predictions = Predict(features);

        return TabularMetrics.R2Score(predictions, targets);
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

        // The neighbours actually retrieved: fewer than NumNeighbors when the index (less the excluded sample) is
        // smaller
        int k = attentionWeights.Shape[1];

        // Sort each sample's neighbors by attention weight (descending). The order is computed as a stable
        // sort, which is what List.Sort's insertion-sort path produced for the usual k <= 16 neighbors.
        var weights = TabularMetrics.Flat(attentionWeights, batchSize * k).Reshape(batchSize, k);
        var order = TieBreakArgMax.StableDescendingOrder(weights);
        var sortedWeights = Engine.TensorTakeAlongDim(weights, order, 1).ToArray();
        var neighbors = Tensor<int>.FromMatrix(neighborIndices);
        neighbors = Engine.TensorNarrow(Engine.TensorNarrow(neighbors, 0, 0, batchSize), 1, 0, k);
        var sortedNeighbors = Engine.TensorTakeAlongDim(neighbors, order, 1).ToArray();

        // Packing into the (index, weight) tuples of the public return type is pure data movement.
        for (int b = 0; b < batchSize; b++)
        {
            var sampleExplanation = new List<(int, T)>(k);
            for (int j = 0; j < k; j++)
            {
                sampleExplanation.Add((sortedNeighbors[b * k + j], sortedWeights[b * k + j]));
            }

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
        if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must lie in (0, 1).");
        }

        // Run forward pass
        var predictions = Predict(features);
        int batchSize = predictions.Shape[0];
        int outputDim = _outputDimension;

        var weights = GetAttentionWeights();
        var neighborTargets = GetNeighborLabels();
        if (weights == null || neighborTargets == null)
        {
            throw new InvalidOperationException(
                "Confidence intervals need the neighbours' targets. Build the index with BuildIndex(features, targets).");
        }

        // Retrieval-weighted spread of the neighbour targets: var = sum_i w_i (y_i - sum_j w_j y_j)^2, and the
        // interval prediction +/- z * sqrt(var) with z the two-sided standard normal quantile
        int k = weights.Shape[1];
        var w = weights.Reshape(batchSize, k, 1);
        var neighborMean = Engine.ReduceSum(Engine.TensorMultiply(w, neighborTargets), new[] { 1 }, true);
        var centered = Engine.TensorSubtract(neighborTargets, neighborMean);
        var variance = Engine.ReduceSum(
            Engine.TensorMultiply(w, Engine.TensorMultiply(centered, centered)), new[] { 1 }, false);
        var z = new NormalDistribution<T>().InverseCdf(NumOps.FromDouble(0.5 + confidenceLevel / 2.0));
        var halfWidth = Engine.TensorMultiplyScalar(Engine.TensorSqrt(variance), z);

        var centre = predictions.Reshape(batchSize, outputDim);
        var lower = Engine.TensorSubtract(centre, halfWidth);
        var upper = Engine.TensorAdd(centre, halfWidth);

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
        _regressionHead.ResetState();
    }
}
