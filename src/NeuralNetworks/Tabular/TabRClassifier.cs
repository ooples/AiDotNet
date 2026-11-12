using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabR implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabRClassifier uses retrieval-augmented predictions for classification.
/// It finds similar training samples and uses their information along with
/// a neural network to make class predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabR for classification works like having a smart
/// assistant who remembers all past examples and uses them to help classify
/// new inputs.
///
/// Prediction process:
/// 1. Encode the input features
/// 2. Find similar training samples (neighbors)
/// 3. Use attention to aggregate neighbor information
/// 4. Combine with the encoded input
/// 5. Predict class probabilities
///
/// Example:
/// <code>
/// var options = new TabROptions&lt;double&gt; { NumNeighbors = 96, EmbeddingDimension = 256 };
/// var classifier = new TabRClassifier&lt;double&gt;(10, 3, options);
///
/// // Build index from training data
/// classifier.BuildIndex(trainFeatures);
///
/// // Make predictions
/// var probabilities = classifier.PredictProbabilities(testFeatures);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TabR: Tabular Deep Learning Meets Nearest Neighbors",
    "https://arxiv.org/abs/2307.14338",
    Year = 2024,
    Authors = "Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, Artem Babenko")]
public class TabRClassifier<T> : TabRBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Initializes a new instance of the TabRClassifier class with default configuration.
    /// </summary>
    public TabRClassifier()
        : this(numFeatures: 16, numClasses: 2)
    {
    }

    /// <summary>
    /// Initializes a new instance of the TabRClassifier class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    public TabRClassifier(
        int numFeatures,
        int numClasses,
        TabROptions<T>? options = null)
        : base(numFeatures, options, Math.Max(numClasses, 0))
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head
        _classificationHead = new FullyConnectedLayer<T>(
            EmbeddingDimension,
            numClasses,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Builds the retrieval index from training features and their class labels, so every retrieved
    /// neighbour contributes its label embedding (Gorishniy et al. 2024, eq. 5).
    /// </summary>
    /// <param name="features">Training features [num_samples, num_features].</param>
    /// <param name="labels">Class label of every training sample.</param>
    public void BuildIndex(Tensor<T> features, int[] labels)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        int samples = features.Shape[0];
        if (labels.Length != samples)
            throw new ArgumentException("There must be one label per indexed sample.", nameof(labels));

        BuildIndexWithLabels(features, TabularTraining.OneHotTargets<T>(Vector<int>.Wrap(labels), samples, _numClasses));
    }

    /// <summary>
    /// Performs the forward pass to get class logits.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for training).</param>
    /// <returns>Class logits [batch_size, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> features, Vector<int>? excludeIndices = null)
    {
        var backboneOutput = ForwardBackbone(features, excludeIndices);

        var logits = _classificationHead.Forward(backboneOutput);

        return logits;
    }

    /// <summary>
    /// Predicts class probabilities.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Probability tensor [batch_size, num_classes].</returns>
    public Tensor<T> PredictProbabilities(Tensor<T> features)
    {
        var logits = Forward(features);
        var probabilities = ApplySoftmax(logits);
        return probabilities;
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Predicted class indices [batch_size].</returns>
    public int[] Predict(Tensor<T> features)
    {
        var probabilities = PredictProbabilities(features);
        int batchSize = probabilities.Shape[0];
        return TabularMetrics.ArgMaxRows(probabilities, batchSize, _numClasses).ToArray();
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, int[] targets)
    {
        int batchSize = probabilities.Shape[0];
        var epsilon = NumOps.FromDouble(1e-15);
        return TabularMetrics.MeanNegativeLogLikelihood(probabilities, Vector<int>.Wrap(targets), batchSize, _numClasses, epsilon);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <param name="sampleIndices">Indices of samples for leave-one-out retrieval.</param>
    /// <returns>The training loss.</returns>
    public T TrainStep(Tensor<T> features, int[] targets, T learningRate, Vector<int>? sampleIndices = null)
    {
        int batchSize = features.Shape[0];
        var labels = TabularTraining.OneHotTargets<T>(Vector<int>.Wrap(targets), batchSize, _numClasses);
        var epsilon = NumOps.FromDouble(1e-15);
        return TrainWithTape(
            () => TabularTraining.CrossEntropyObjective(ApplySoftmax(Forward(features, sampleIndices)), labels, epsilon),
            learningRate);
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    public T ComputeAccuracy(Tensor<T> features, int[] targets)
    {
        var predictions = Predict(features);
        int correct = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == targets[i])
                correct++;
        }

        return NumOps.FromDouble((double)correct / predictions.Length);
    }

    /// <summary>
    /// Gets interpretability information: which neighbors influenced each prediction.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>List of (neighbor index, attention weight) pairs for each sample.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This shows which training examples were most influential
    /// for each prediction. Use this to:
    /// - Understand why a prediction was made
    /// - Identify if the model is using sensible examples
    /// - Debug unexpected predictions
    /// </para>
    /// </remarks>
    public List<List<(int NeighborIndex, T AttentionWeight)>> GetPredictionExplanations(Tensor<T> features)
    {
        // Run forward pass
        _ = PredictProbabilities(features);

        var explanations = new List<List<(int, T)>>();
        var attentionWeights = GetAttentionWeights();
        var neighborIndices = GetRetrievedNeighborIndices();

        if (attentionWeights == null || neighborIndices == null)
        {
            return explanations;
        }

        int batchSize = features.Shape[0];
        int k = Options.NumNeighbors;

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

    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        return Engine.Softmax(logits, -1);
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        base.UpdateParameters(learningRate);
        _classificationHead.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _classificationHead.ResetState();
    }
}
