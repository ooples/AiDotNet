using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabM implementation for classification tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabMClassifier uses the TabM architecture with BatchEnsemble layers for multi-class
/// classification. It averages predictions across ensemble members and uses softmax
/// for probability outputs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when you want to classify data into categories
/// with the benefits of an ensemble (multiple models voting).
///
/// How predictions work:
/// 1. Input passes through shared layers with per-member modulation
/// 2. Each ensemble member produces class logits
/// 3. Logits are averaged across members
/// 4. Softmax converts to probabilities
///
/// Benefits over single models:
/// - Reduced overfitting through ensemble averaging
/// - Better calibrated probability estimates
/// - More robust to noisy features
/// - Comparable speed to single models (parameter sharing)
///
/// Example:
/// <code>
/// var options = new TabMOptions&lt;double&gt; { NumEnsembleMembers = 4, HiddenDimensions = [256, 128] };
/// var classifier = new TabMClassifier&lt;double&gt;(10, 3, options);
///
/// var input = new Tensor&lt;double&gt;([32, 10]); // batch of 32 samples
/// var probabilities = classifier.PredictProbabilities(input);
/// var predictions = classifier.Predict(input);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling",
    "https://arxiv.org/abs/2410.24210",
    Year = 2024,
    Authors = "Yury Gorishniy, Akim Kotelnikov, Artem Babenko")]
public class TabMClassifier<T> : TabMBase<T>
{
    private readonly int _numClasses;
    private readonly BatchEnsembleLayer<T> _classificationHead;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Initializes a new instance of the TabMClassifier class with default configuration.
    /// </summary>
    public TabMClassifier()
        : this(numFeatures: 16, numClasses: 2)
    {
    }

    /// <summary>
    /// Initializes a new instance of the TabMClassifier class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="options">Model configuration options.</param>
    public TabMClassifier(
        int numFeatures,
        int numClasses,
        TabMOptions<T>? options = null)
        : base(numFeatures, options)
    {
        if (numClasses < 2)
        {
            throw new ArgumentException("Number of classes must be at least 2", nameof(numClasses));
        }

        _numClasses = numClasses;

        // Classification head: BatchEnsemble layer to class logits
        _classificationHead = new BatchEnsembleLayer<T>(
            GetLastHiddenDim(),
            numClasses,
            Options.NumEnsembleMembers,
            Options.UseBias,
            Options.RankInitScale);
    }

    /// <summary>
    /// Performs the forward pass to get class logits (per member).
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Class logits tensor [batch_size * num_members, num_classes].</returns>
    public Tensor<T> Forward(Tensor<T> features)
    {
        // Forward through backbone
        var backboneOutput = ForwardBackbone(features);

        // Forward through classification head
        var logits = _classificationHead.Forward(backboneOutput);

        return logits;
    }

    /// <summary>
    /// Predicts class probabilities (averaged across ensemble members).
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Probability tensor [batch_size, num_classes].</returns>
    public Tensor<T> PredictProbabilities(Tensor<T> features)
    {
        var memberLogits = Forward(features);

        // Average logits across members first, then softmax
        var avgLogits = AverageMemberOutputs(memberLogits, _numClasses);

        // Apply softmax
        var probabilities = ApplySoftmax(avgLogits);

        return probabilities;
    }

    /// <summary>
    /// Predicts the most likely class for each sample.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
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
    /// <param name="probabilities">Predicted probabilities [batch_size, num_classes].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <returns>The average cross-entropy loss.</returns>
    public T ComputeCrossEntropyLoss(Tensor<T> probabilities, int[] targets)
    {
        int batchSize = probabilities.Shape[0];
        var epsilon = NumOps.FromDouble(1e-15);
        return TabularMetrics.MeanNegativeLogLikelihood(probabilities, Vector<int>.Wrap(targets), batchSize, _numClasses, epsilon);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <param name="targets">Target class indices [batch_size].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <returns>The training loss for this step.</returns>
    public T TrainStep(Tensor<T> features, int[] targets, T learningRate)
    {
        // Forward pass
        var probabilities = PredictProbabilities(features);

        // Compute loss
        var loss = ComputeCrossEntropyLoss(probabilities, targets);

        // Backward pass

        // Update parameters
        UpdateParameters(learningRate);

        // Reset for next iteration
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
            {
                correct++;
            }
        }

        return NumOps.FromDouble((double)correct / predictions.Length);
    }

    /// <summary>
    /// Computes the macro F1 score across all classes.
    /// </summary>
    public T ComputeF1Score(Tensor<T> features, int[] targets)
    {
        var predictions = Predict(features);
        var engine = Engine;
        int n = predictions.Length;

        // One-hot rows ignore labels outside [0, numClasses), exactly as the per-class equality tests did.
        var predicted = engine.TensorOneHot<T>(new Tensor<int>(predictions, new[] { n }), _numClasses);
        var actual = engine.TensorOneHot<T>(
            Tensor<int>.FromVector(Vector<int>.Wrap(targets).GetSubVector(0, n)), _numClasses);

        var truePositives = engine.ReduceSum(engine.TensorMultiply(predicted, actual), new[] { 0 }, false);
        var predictedCounts = engine.ReduceSum(predicted, new[] { 0 }, false);
        var actualCounts = engine.ReduceSum(actual, new[] { 0 }, false);

        // tp is 0 whenever its count is 0, so dividing by max(count, 1) yields the 0 fallback exactly.
        var precision = engine.TensorDivide(truePositives, engine.TensorMax(predictedCounts, NumOps.One));
        var recall = engine.TensorDivide(truePositives, engine.TensorMax(actualCounts, NumOps.One));
        var precisionPlusRecall = engine.TensorAdd(precision, recall);
        var hasScore = engine.TensorGreaterThan(precisionPlusRecall, NumOps.Zero);
        var safeDenominator = engine.TensorWhere(hasScore, precisionPlusRecall, engine.TensorOnesLike(precisionPlusRecall));
        var f1 = engine.TensorDivide(
            engine.TensorMultiplyScalar(engine.TensorMultiply(precision, recall), NumOps.FromDouble(2.0)),
            safeDenominator);

        return NumOps.Divide(engine.TensorSum(f1), NumOps.FromDouble(_numClasses));
    }

    /// <summary>
    /// Gets predictive uncertainty as the entropy of the averaged predictions.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Entropy values [batch_size].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uncertainty tells you how confident the model is:
    /// - Low entropy (near 0): Model is confident (probabilities peaked at one class)
    /// - High entropy: Model is uncertain (probabilities spread across classes)
    ///
    /// This can be useful for:
    /// - Identifying samples that need human review
    /// - Active learning (selecting samples to label)
    /// - Detecting out-of-distribution samples
    /// </para>
    /// </remarks>
    public Vector<T> GetPredictiveUncertainty(Tensor<T> features)
    {
        var probabilities = PredictProbabilities(features);
        int batchSize = probabilities.Shape[0];
        var epsilon = NumOps.FromDouble(1e-15);
        var engine = Engine;

        var p = TabularMetrics.Flat(probabilities, batchSize * _numClasses).Reshape(batchSize, _numClasses);
        var plogp = engine.TensorMultiply(p, engine.TensorLog(engine.TensorAddScalar(p, epsilon)));
        var entropy = engine.TensorNegate(engine.ReduceSum(plogp, new[] { 1 }, false));
        return entropy.ToVector();
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        return Engine.Softmax(logits, -1);
    }

    /// <summary>
    /// Updates all parameters including the classification head.
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
        _classificationHead.ResetGradients();
    }
}
