using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// GANDALF implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// GANDALFClassifier uses gated feature selection with neural decision trees
/// for multi-class classification. The additive ensemble of trees produces
/// class logits that are converted to probabilities via softmax.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use GANDALF for classification when you want:
/// - Automatic feature importance learning
/// - Interpretable predictions (can see feature weights and tree paths)
/// - Good performance on tabular data
///
/// Example:
/// <code>
/// var options = new GANDALFOptions&lt;double&gt; { NumTrees = 20, TreeDepth = 6 };
/// var classifier = new GANDALFClassifier&lt;double&gt;(10, 3, options);
///
/// // Train
/// classifier.TrainStep(features, targets, learningRate);
///
/// // Predict
/// var probabilities = classifier.PredictProbabilities(testFeatures);
/// var predictions = classifier.Predict(testFeatures);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GANDALFClassifier<T> : GANDALFBase<T>
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
    /// Initializes a new instance of the GANDALFClassifier class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    public GANDALFClassifier(
        int numFeatures,
        int numClasses,
        GANDALFOptions<T>? options = null)
        : base(numFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head maps from leaf dimension to number of classes
        _classificationHead = new FullyConnectedLayer<T>(
            Options.LeafDimension,
            numClasses,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Performs the forward pass to get class logits.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Class logits [batch_size, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> features)
    {
        var backboneOutput = ForwardBackbone(features);
        _backboneOutputCache = backboneOutput;

        var logits = _classificationHead.Forward(backboneOutput);
        _logitsCache = logits;

        return logits;
    }

    /// <summary>
    /// Predicts class probabilities using softmax.
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
    public Vector<int> Predict(Tensor<T> features)
    {
        var probabilities = PredictProbabilities(features);
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
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
    public Tensor<T> Backward(Vector<int> targets)
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
    /// <returns>The training loss.</returns>
    public T TrainStep(Tensor<T> features, Vector<int> targets, T learningRate)
    {
        var probabilities = PredictProbabilities(features);
        var loss = ComputeCrossEntropyLoss(probabilities, targets);
        _ = Backward(targets);
        UpdateParameters(learningRate);
        ResetState();

        return loss;
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    public T ComputeAccuracy(Tensor<T> features, Vector<int> targets)
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
