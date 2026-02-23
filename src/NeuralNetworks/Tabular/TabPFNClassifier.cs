using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabPFN implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabPFNClassifier uses in-context learning for tabular classification.
/// It takes training data as context and makes predictions on test data
/// in a single forward pass, similar to how language models complete text.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabPFN classification works differently:
///
/// 1. First, call SetContext() with your training data
/// 2. Then, call Predict() with test data
/// 3. The model uses attention to "learn" from context
/// 4. No training loop needed - it's instant!
///
/// Example:
/// <code>
/// var options = new TabPFNOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     NumLayers = 12
/// };
/// var classifier = new TabPFNClassifier&lt;double&gt;(numFeatures: 20, numClasses: 3, options);
///
/// // Set training data as context
/// classifier.SetContext(trainFeatures, trainLabels);
///
/// // Make predictions on test data
/// var predictions = classifier.Predict(testFeatures);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabPFNClassifier<T> : TabPFNBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

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
    /// Initializes a new instance of the TabPFNClassifier class.
    /// </summary>
    public TabPFNClassifier(
        int numNumericalFeatures,
        int numClasses,
        TabPFNOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        if (numClasses > (options?.MaxClasses ?? 10))
        {
            throw new ArgumentException(
                $"Number of classes ({numClasses}) exceeds maximum supported ({options?.MaxClasses ?? 10})");
        }

        _numClasses = numClasses;

        _classificationHead = new FullyConnectedLayer<T>(
            MLPOutputDimension,
            numClasses,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Sets the context (training) data for in-context learning.
    /// </summary>
    /// <param name="features">Training features.</param>
    /// <param name="labels">Training labels as class indices.</param>
    public void SetContext(Tensor<T> features, Vector<int> labels)
    {
        // Convert integer labels to one-hot encoded tensor
        int numSamples = features.Shape[0];
        var labelTensor = new Tensor<T>([numSamples, _numClasses]);

        for (int i = 0; i < numSamples; i++)
        {
            int classIdx = labels[i];
            if (classIdx >= 0 && classIdx < _numClasses)
            {
                labelTensor[i * _numClasses + classIdx] = NumOps.One;
            }
        }

        base.SetContext(features, labelTensor);
    }

    /// <summary>
    /// Performs the forward pass to get class logits.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var backboneOutput = ForwardBackbone(numericalFeatures, categoricalIndices);
        _backboneOutputCache = backboneOutput;

        var logits = _classificationHead.Forward(backboneOutput);
        _logitsCache = logits;

        return logits;
    }

    /// <summary>
    /// Predicts class probabilities using softmax.
    /// </summary>
    public Tensor<T> PredictProbabilities(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var logits = Forward(numericalFeatures, categoricalIndices);
        var probabilities = ApplySoftmax(logits);
        _probabilitiesCache = probabilities;
        return probabilities;
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    public Vector<int> Predict(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var probabilities = PredictProbabilities(numericalFeatures, categoricalIndices);
        int batchSize = probabilities.Shape[0];
        var predictions = new Vector<int>(batchSize);

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
    /// Predicts with ensemble averaging over multiple permutations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble prediction runs the model multiple times
    /// with different orderings of the context data and averages the results.
    /// This can improve prediction accuracy and reliability.
    /// </para>
    /// </remarks>
    public Vector<int> PredictEnsemble(Tensor<T> numericalFeatures, int numEnsembles = 16, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        var aggregatedProbs = new Tensor<T>([batchSize, _numClasses]);

        for (int e = 0; e < numEnsembles; e++)
        {
            var probs = PredictProbabilities(numericalFeatures, categoricalIndices);

            // Accumulate probabilities
            for (int i = 0; i < probs.Length; i++)
            {
                aggregatedProbs[i] = NumOps.Add(aggregatedProbs[i], probs[i]);
            }
        }

        // Average
        var scale = NumOps.FromDouble(1.0 / numEnsembles);
        for (int i = 0; i < aggregatedProbs.Length; i++)
        {
            aggregatedProbs[i] = NumOps.Multiply(aggregatedProbs[i], scale);
        }

        // Get predictions
        var predictions = new Vector<int>(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            int maxIdx = 0;
            var maxProb = aggregatedProbs[b * _numClasses + 0];

            for (int c = 1; c < _numClasses; c++)
            {
                var prob = aggregatedProbs[b * _numClasses + c];
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
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, Vector<int> targets)
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
    public Tensor<T> Backward(Vector<int> targets)
    {
        if (_probabilitiesCache == null || _logitsCache == null || _backboneOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _probabilitiesCache.Shape[0];

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
    /// Performs a single training step (for fine-tuning).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Note:</b> TabPFN is designed for zero-shot inference. Training is optional
    /// and primarily useful for domain-specific fine-tuning.
    /// </para>
    /// </remarks>
    public T TrainStep(Tensor<T> numericalFeatures, Vector<int> targets, T learningRate, Matrix<int>? categoricalIndices = null)
    {
        var probabilities = PredictProbabilities(numericalFeatures, categoricalIndices);
        var loss = ComputeCrossEntropyLoss(probabilities, targets);
        _ = Backward(targets);
        UpdateParameters(learningRate);
        ResetState();

        return loss;
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    public T ComputeAccuracy(Tensor<T> numericalFeatures, Vector<int> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        int correct = 0;

        for (int i = 0; i < predictions.Length; i++)
        {
            if (predictions[i] == targets[i])
                correct++;
        }

        return NumOps.FromDouble((double)correct / predictions.Length);
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
