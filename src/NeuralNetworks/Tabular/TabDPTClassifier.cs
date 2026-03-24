using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabDPT implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabDPTClassifier applies foundation model concepts to tabular classification,
/// leveraging pre-trained representations for multi-class prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use TabDPT for classification when:
/// - You have a standard tabular classification problem
/// - You want to leverage foundation model capabilities
/// - Your data has complex feature interactions
///
/// Example:
/// <code>
/// var options = new TabDPTOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     NumLayers = 6,
///     NumHeads = 4
/// };
/// var classifier = new TabDPTClassifier&lt;double&gt;(numFeatures: 20, numClasses: 3, options);
///
/// var predictions = classifier.Predict(features);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("TabDPT: Scaling Tabular Foundation Models",
    "https://arxiv.org/abs/2501.02487",
    Year = 2025,
    Authors = "Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Lacoste, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, Anthony L. Caterini")]
public class TabDPTClassifier<T> : TabDPTBase<T>
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
    /// Initializes a new instance of the TabDPTClassifier class.
    /// </summary>
    public TabDPTClassifier(
        int numNumericalFeatures,
        int numClasses,
        TabDPTOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        _classificationHead = new FullyConnectedLayer<T>(
            MLPOutputDimension,
            numClasses,
            (IActivationFunction<T>?)null);
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

        var logitsGrad = new Tensor<T>(_logitsCache.Shape.ToArray());
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
        _backboneOutputCache = null;
        _logitsCache = null;
        _probabilitiesCache = null;
        _classificationHead.ResetState();
    }
}
