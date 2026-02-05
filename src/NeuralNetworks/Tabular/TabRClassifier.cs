using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

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
public class TabRClassifier<T> : TabRBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

    // Cache for backward pass
    private Tensor<T>? _backboneOutputCache;
    private Tensor<T>? _logitsCache;
    private Tensor<T>? _probabilitiesCache;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _classificationHead.ParameterCount;

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
        : base(numFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head
        _classificationHead = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension,
            numClasses,
            (IActivationFunction<T>?)null);
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
        _backboneOutputCache = backboneOutput;

        var logits = _classificationHead.Forward(backboneOutput);
        _logitsCache = logits;

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
        _probabilitiesCache = probabilities;
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
        var predictions = new int[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            int maxIdx = 0;
            var maxProb = probabilities[b * _numClasses + 0];

            for (int c = 1; c < _numClasses; c++)
            {
                var prob = probabilities[b * _numClasses + c];
                if (NumOps.Compare(prob, maxProb) > 0)
                {
                    maxProb = prob;
                    maxIdx = c;
                }
            }

            predictions[b] = maxIdx;
        }

        return predictions;
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, int[] targets)
    {
        int batchSize = probabilities.Shape[0];
        var totalLoss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-15);

        for (int b = 0; b < batchSize; b++)
        {
            int targetClass = targets[b];
            var prob = probabilities[b * _numClasses + targetClass];
            var safeProb = NumOps.Add(prob, epsilon);
            var loss = NumOps.Negate(NumOps.Log(safeProb));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Performs the backward pass.
    /// </summary>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
    public Tensor<T> Backward(int[] targets)
    {
        if (_probabilitiesCache == null || _logitsCache == null || _backboneOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _probabilitiesCache.Shape[0];

        // Gradient of cross-entropy + softmax
        var logitsGrad = new Tensor<T>(_logitsCache.Shape);
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                var prob = _probabilitiesCache[b * _numClasses + c];
                var oneHot = targets[b] == c ? NumOps.One : NumOps.Zero;
                logitsGrad[b * _numClasses + c] = NumOps.Multiply(
                    NumOps.Subtract(prob, oneHot), scale);
            }
        }

        var backboneGrad = _classificationHead.Backward(logitsGrad);
        return BackwardBackbone(backboneGrad);
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
        // Forward with leave-one-out (exclude each sample from its own retrieval)
        var logits = Forward(features, sampleIndices);
        var probabilities = ApplySoftmax(logits);
        _probabilitiesCache = probabilities;

        var loss = ComputeCrossEntropyLoss(probabilities, targets);
        _ = Backward(targets);
        UpdateParameters(learningRate);
        ResetState();

        return loss;
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

    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        int batchSize = logits.Shape[0];
        int numClasses = logits.Shape[1];
        var probabilities = new Tensor<T>(logits.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            var maxLogit = logits[b * numClasses + 0];
            for (int c = 1; c < numClasses; c++)
            {
                var logit = logits[b * numClasses + c];
                if (NumOps.Compare(logit, maxLogit) > 0)
                    maxLogit = logit;
            }

            var sumExp = NumOps.Zero;
            for (int c = 0; c < numClasses; c++)
            {
                var expVal = NumOps.Exp(NumOps.Subtract(logits[b * numClasses + c], maxLogit));
                probabilities[b * numClasses + c] = expVal;
                sumExp = NumOps.Add(sumExp, expVal);
            }

            for (int c = 0; c < numClasses; c++)
            {
                probabilities[b * numClasses + c] = NumOps.Divide(
                    probabilities[b * numClasses + c], sumExp);
            }
        }

        return probabilities;
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
        _backboneOutputCache = null;
        _logitsCache = null;
        _probabilitiesCache = null;
        _classificationHead.ResetState();
    }
}
