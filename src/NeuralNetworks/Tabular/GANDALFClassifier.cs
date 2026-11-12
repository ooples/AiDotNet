using AiDotNet.Attributes;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GANDALF: Gated Adaptive Network for Deep Automated Learning of Features",
    "https://arxiv.org/abs/2207.08548",
    Year = 2022,
    Authors = "Joseph, R. & Raj, H.")]
public class GANDALFClassifier<T> : GANDALFBase<T>
{
    private readonly int _numClasses;
    private readonly FullyConnectedLayer<T> _classificationHead;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

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

        var logits = _classificationHead.Forward(backboneOutput);

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

        // Per-row argmax over the classes (first index wins ties, as before).
        return Engine.TensorArgMax(Engine.Reshape(probabilities, new[] { batchSize, _numClasses }), 1).ToVector();
    }

    /// <summary>
    /// Computes the cross-entropy loss.
    /// </summary>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, Vector<int> targets)
    {
        int batchSize = probabilities.Shape[0];
        var epsilon = NumOps.FromDouble(1e-15);

        // -mean_b log(p[b, target_b] + eps): gather each row's target probability, then reduce.
        var targetProb = Engine.TensorTakeAlongDim(
            Engine.Reshape(probabilities, new[] { batchSize, _numClasses }),
            Tensor<int>.FromVector(targets.GetSubVector(0, batchSize), new[] { batchSize, 1 }), 1);
        var totalLoss = NumOps.Negate(Engine.TensorSum(Engine.TensorLog(Engine.TensorAddScalar(targetProb, epsilon))));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
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
        // Mean cross-entropy -1/B * sum(onehot * log(softmax(logits) + eps)), differentiated on the
        // tape through the head, the trees and the gating network. (This step used to compute the
        // loss and call UpdateParameters with no gradients, so nothing ever trained.)
        int batchSize = features.Shape[0];
        if (batchSize == 0 || targets.Length != batchSize)
            throw new ArgumentException("Targets must match a nonempty input batch.", nameof(targets));
        var targetIndex = Tensor<int>.FromVector(targets, new[] { batchSize });
        if (Engine.TensorMinValue(targetIndex) < 0 || Engine.TensorMaxValue(targetIndex) >= _numClasses)
            throw new ArgumentOutOfRangeException(nameof(targets), "Target class is out of range.");
        var labels = Engine.TensorOneHot<T>(targetIndex, _numClasses);

        return TrainWithTape(() =>
        {
            var probabilities = ApplySoftmax(Forward(features));
            var logProbabilities = Engine.TensorLog(Engine.TensorAddScalar(probabilities, NumOps.FromDouble(1e-15)));
            var sum = Engine.ReduceSum(Engine.TensorMultiply(labels, logProbabilities), new[] { 0, 1 }, keepDims: false);
            return Engine.TensorMultiplyScalar(sum, NumOps.FromDouble(-1.0 / batchSize));
        }, learningRate);
    }

    /// <summary>
    /// Computes classification accuracy.
    /// </summary>
    public T ComputeAccuracy(Tensor<T> features, Vector<int> targets)
    {
        var predictions = Predict(features);
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
