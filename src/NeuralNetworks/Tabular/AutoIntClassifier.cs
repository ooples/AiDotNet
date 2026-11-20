using AiDotNet.Attributes;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// AutoInt implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// AutoIntClassifier uses multi-head self-attention to automatically learn
/// feature interactions for classification on tabular data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use AutoInt for classification when:
/// - Feature interactions matter (e.g., age + product_category)
/// - You don't want to manually engineer feature crosses
/// - You want interpretable interaction patterns
///
/// Example:
/// <code>
/// var options = new AutoIntOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 16,
///     NumLayers = 3,
///     NumHeads = 2
/// };
/// var classifier = new AutoIntClassifier&lt;double&gt;(numFeatures: 10, numClasses: 2, options);
///
/// // Train and predict
/// var predictions = classifier.Predict(features);
///
/// // See learned interactions
/// var interactions = classifier.GetInteractionWeights();
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
[ResearchPaper("AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks",
    "https://arxiv.org/abs/1810.11921",
    Year = 2019,
    Authors = "Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J.")]
public class AutoIntClassifier<T> : AutoIntBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Initializes a new instance of the AutoIntClassifier class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    public AutoIntClassifier(
        int numNumericalFeatures,
        int numClasses,
        AutoIntOptions<T>? options = null)
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

        var logits = _classificationHead.Forward(backboneOutput);

        return logits;
    }

    /// <summary>
    /// Predicts class probabilities using softmax.
    /// </summary>
    public Tensor<T> PredictProbabilities(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        using var inference = InferenceMode.Enter();
        var logits = Forward(numericalFeatures, categoricalIndices);
        var probabilities = ApplySoftmax(logits);
        return probabilities;
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    public Vector<int> Predict(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        var probabilities = PredictProbabilities(numericalFeatures, categoricalIndices);
        int batchSize = probabilities.Shape[0];

        // Per-row argmax over the classes (first index wins ties, as before).
        return Engine.TensorArgMax(Engine.Reshape(probabilities, new[] { batchSize, _numClasses }), 1).ToVector();
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, Vector<int> targets)
    {
        int batchSize = probabilities.Shape[0];

        if (targets.Length != batchSize)
        {
            throw new ArgumentException(
                $"Targets length ({targets.Length}) must match batch size ({batchSize}).");
        }

        var epsilon = NumOps.FromDouble(1e-15);

        // -mean_b log(p[b, target_b] + eps): gather each row's target probability, then reduce.
        var targetProb = Engine.TensorTakeAlongDim(
            Engine.Reshape(probabilities, new[] { batchSize, _numClasses }),
            Tensor<int>.FromVector(targets, new[] { batchSize, 1 }), 1);
        var totalLoss = NumOps.Negate(Engine.TensorSum(Engine.TensorLog(Engine.TensorAddScalar(targetProb, epsilon))));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    public T TrainStep(Tensor<T> numericalFeatures, Vector<int> targets, T learningRate, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        if (batchSize == 0 || targets.Length != batchSize)
            throw new ArgumentException("Targets must match a nonempty input batch.", nameof(targets));
        var targetIndex = Tensor<int>.FromVector(targets, new[] { batchSize });
        if (Engine.TensorMinValue(targetIndex) < 0 || Engine.TensorMaxValue(targetIndex) >= _numClasses)
            throw new ArgumentOutOfRangeException(nameof(targets), "Target class is out of range.");
        var labels = Engine.TensorOneHot<T>(targetIndex, _numClasses);
        return TrainWithTape(() =>
        {
            var probabilities = ApplySoftmax(Forward(numericalFeatures, categoricalIndices));
            var logProbabilities = Engine.TensorLog(Engine.TensorAddScalar(probabilities, NumOps.FromDouble(1e-15)));
            var sum = Engine.ReduceSum(Engine.TensorMultiply(labels, logProbabilities), new[] { 0, 1 }, keepDims: false);
            return Engine.TensorMultiplyScalar(sum, NumOps.FromDouble(-1.0 / batchSize));
        }, learningRate);
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    public T ComputeAccuracy(Tensor<T> numericalFeatures, Vector<int> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        int n = predictions.Length;
        int correct = Engine.TensorSum(Engine.TensorEquals(
            Tensor<int>.FromVector(predictions, new[] { n }),
            Tensor<int>.FromVector(targets.GetSubVector(0, n), new[] { n })));

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
        _classificationHead.ResetState();
    }
}
