using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabTransformer implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabTransformerClassifier applies transformer attention to categorical features
/// for multi-class classification. Numerical features are concatenated after
/// the categorical embeddings are transformed.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use TabTransformer for classification when:
/// - You have important categorical features (like sector, region, category)
/// - You believe there are interactions between categorical features
/// - You want the model to learn these interactions automatically
///
/// Example:
/// <code>
/// var options = new TabTransformerOptions&lt;double&gt;
/// {
///     CategoricalCardinalities = new[] { 5, 10, 20 },  // 3 categorical features
///     EmbeddingDimension = 32,
///     NumLayers = 6
/// };
/// var classifier = new TabTransformerClassifier&lt;double&gt;(numNumerical: 5, numClasses: 3, options);
///
/// var predictions = classifier.Predict(numericalFeatures, categoricalIndices);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabTransformerClassifier<T> : TabTransformerBase<T>
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
    /// Initializes a new instance of the TabTransformerClassifier class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    public TabTransformerClassifier(
        int numNumericalFeatures,
        int numClasses,
        TabTransformerOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head
        _classificationHead = new FullyConnectedLayer<T>(
            MLPOutputDimension,
            numClasses,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Performs the forward pass to get class logits.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical indices matrix [batch_size, num_categorical] or null.</param>
    /// <returns>Class logits [batch_size, num_classes].</returns>
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
    /// Performs a single training step.
    /// </summary>
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
