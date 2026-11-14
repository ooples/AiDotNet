using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.SemiSupervised;

/// <summary>
/// A self-training classifier that iteratively labels high-confidence unlabeled samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Self-training is one of the oldest and simplest semi-supervised learning algorithms.
/// It works by iteratively training a classifier on labeled data, using it to predict
/// labels for unlabeled data, and adding high-confidence predictions to the training set.
/// </para>
/// <para>
/// <b>For Beginners:</b> Self-training is like a student who:
///
/// 1. Studies the examples with answers (labeled data)
/// 2. Makes predictions on practice problems (unlabeled data)
/// 3. Is most confident about some answers
/// 4. Treats those confident answers as correct and studies them too
/// 5. Repeats until no more confident predictions can be made
///
/// The key insight is that the classifier's most confident predictions are likely correct,
/// so we can use them to expand our training set. Over time, the classifier becomes
/// more accurate as it learns from its own confident predictions.
///
/// Algorithm steps:
/// 1. Train base classifier on labeled data
/// 2. Predict probabilities for all unlabeled samples
/// 3. Find predictions with confidence above threshold
/// 4. Add those samples (with predicted labels) to the training set
/// 5. Repeat until no new samples are added or max iterations reached
///
/// References:
/// - Yarowsky, D. (1995). "Unsupervised word sense disambiguation rivaling supervised methods"
/// - Triguero et al. (2015). "Self-labeled techniques for semi-supervised learning"
/// </para>
/// </remarks>
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for selftrainingclassifier classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new SelfTrainingClassifier&lt;double&gt;(new GaussianNaiveBayes&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Semi-Supervised Learning Literature Survey", "https://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf")]
public partial class SelfTrainingClassifier<T> : SemiSupervisedClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>, IGradientComputable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>Whatever the wrapped base classifier exposes, forwarded as before. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }
    private IClassifier<T> _baseClassifier;
    private double _confidenceThreshold;
    private int _maxIterations;
    private int _maxSamplesPerIteration;
    private SelectionCriterion _selectionCriterion;

    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier.
    /// </summary>
    public SelfTrainingClassifier()
        : this(new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// Gets the number of iterations performed during training.
    /// </summary>
    public int IterationsPerformed { get; private set; }

    /// <summary>
    /// Gets the number of samples added from unlabeled data during training.
    /// </summary>
    public int SamplesAdded { get; private set; }

    /// <summary>
    /// Defines how unlabeled samples are selected for labeling.
    /// </summary>
    public enum SelectionCriterion
    {
        /// <summary>
        /// Select all samples above the confidence threshold.
        /// </summary>
        Threshold,

        /// <summary>
        /// Select the top-k most confident samples per iteration.
        /// </summary>
        TopK,

        /// <summary>
        /// Select top-k samples per class to maintain class balance.
        /// </summary>
        TopKPerClass
    }

    /// <summary>
    /// Initializes a new instance of the SelfTrainingClassifier class.
    /// </summary>
    /// <param name="baseClassifier">The base classifier to use for predictions. Must support probability predictions.</param>
    /// <param name="confidenceThreshold">Minimum confidence (0-1) required to add an unlabeled sample to training. Default is 0.75.</param>
    /// <param name="maxIterations">Maximum number of self-training iterations. Default is 10.</param>
    /// <param name="maxSamplesPerIteration">Maximum samples to add per iteration (0 = unlimited). Default is 0.</param>
    /// <param name="selectionCriterion">How to select samples for pseudo-labeling. Default is Threshold.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The parameters control how aggressive the self-training is:
    ///
    /// - confidenceThreshold: Higher = more cautious, only very confident predictions are used
    ///   - 0.9 = very cautious, might not add many samples
    ///   - 0.5 = risky, might add incorrect labels
    ///   - 0.75 (default) = balanced approach
    ///
    /// - maxIterations: How many rounds of self-labeling to perform
    ///   - More iterations = more samples potentially added
    ///   - Too many might cause error propagation
    ///
    /// - maxSamplesPerIteration: Limits how fast training set grows
    ///   - 0 = add all confident samples
    ///   - Small number = gradual, safer growth
    ///
    /// - selectionCriterion: How to pick which samples to add
    ///   - Threshold: All above confidence threshold
    ///   - TopK: Only the k most confident
    ///   - TopKPerClass: k most confident per class (prevents class imbalance)
    /// </para>
    /// </remarks>
    public SelfTrainingClassifier(
        IClassifier<T> baseClassifier,
        double confidenceThreshold = 0.75,
        int maxIterations = 10,
        int maxSamplesPerIteration = 0,
        SelectionCriterion selectionCriterion = SelectionCriterion.Threshold,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        Guard.NotNull(baseClassifier);
        _baseClassifier = baseClassifier;

        if (baseClassifier is not IProbabilisticClassifier<T>)
        {
            throw new ArgumentException(
                "Self-training requires a probabilistic classifier that implements IProbabilisticClassifier<T>. " +
                "Non-probabilistic classifiers cannot provide confidence scores needed for pseudo-labeling.",
                nameof(baseClassifier));
        }

        if (confidenceThreshold < 0 || confidenceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(confidenceThreshold),
                "Confidence threshold must be between 0 and 1.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "Maximum iterations must be at least 1.");
        }

        _confidenceThreshold = confidenceThreshold;
        _maxIterations = maxIterations;
        _maxSamplesPerIteration = maxSamplesPerIteration;
        _selectionCriterion = selectionCriterion;
    }

    /// <summary>
    /// Core implementation of semi-supervised self-training.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the iterative self-training happens:
    ///
    /// 1. Start with the original labeled data
    /// 2. Train the base classifier
    /// 3. Predict labels and confidences for unlabeled data
    /// 4. Select high-confidence predictions
    /// 5. Add them to the training set with their predicted labels
    /// 6. Remove them from the unlabeled set
    /// 7. Repeat until done
    ///
    /// The process stops when:
    /// - No more samples meet the confidence threshold
    /// - Maximum iterations reached
    /// - All unlabeled samples have been labeled
    /// </para>
    /// </remarks>
    protected override void TrainSemiSupervisedCore(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        // The growing training set: the labeled data plus every pseudo-labeled row added so far.
        var engine = AiDotNetEngine.Current;
        var unlabeled = Tensor<T>.FromMatrix(unlabeledX);
        var trainX = Tensor<T>.FromMatrix(labeledX);
        var trainY = Tensor<T>.FromVector(labeledY);

        // Track remaining unlabeled samples
        var remainingUnlabeled = Enumerable.Range(0, unlabeledX.Rows).ToList();

        // Initialize pseudo-labels storage: -1 indicates not yet labeled.
        var pseudoLabels = new Tensor<T>(new[] { unlabeledX.Rows });
        engine.TensorFill(pseudoLabels, NumOps.FromDouble(-1));
        var pseudoConfidences = new Tensor<T>(new[] { unlabeledX.Rows });

        IterationsPerformed = 0;
        SamplesAdded = 0;

        for (int iteration = 0; iteration < _maxIterations && remainingUnlabeled.Count > 0; iteration++)
        {
            IterationsPerformed++;

            // Train the base classifier
            _baseClassifier.Train(trainX.ToMatrix(), trainY.ToVector());

            // Get predictions and confidences for remaining unlabeled samples
            var (selectedIndices, predictedLabels, confidences) = SelectSamplesToAdd(unlabeledX, remainingUnlabeled);

            if (selectedIndices.Count == 0)
            {
                break; // No more samples meet criteria
            }

            // Add selected samples to training set. selectedIndices index into remainingUnlabeled;
            // predictedLabels and confidences are parallel lists aligned with selectedIndices.
            var originals = selectedIndices.Select(idx => remainingUnlabeled[idx]).ToArray();
            var originalIndex = new Tensor<int>(originals, new[] { originals.Length });
            var labels = new Tensor<T>(predictedLabels.ToArray(), new[] { originals.Length });
            trainX = engine.TensorConcatenate(new[] { trainX, engine.TensorIndexSelect(unlabeled, originalIndex, 0) }, 0);
            trainY = engine.TensorConcatenate(new[] { trainY, labels }, 0);

            // Store pseudo-label info
            pseudoLabels = engine.TensorIndexCopy(pseudoLabels, 0, originalIndex, labels);
            pseudoConfidences = engine.TensorIndexCopy(
                pseudoConfidences, 0, originalIndex, new Tensor<T>(confidences.ToArray(), new[] { originals.Length }));

            SamplesAdded += selectedIndices.Count;

            // Remove added samples from remaining unlabeled (in reverse order to maintain indices)
            var sortedIndices = selectedIndices.OrderByDescending(x => x).ToList();
            foreach (int idx in sortedIndices)
            {
                remainingUnlabeled.RemoveAt(idx);
            }
        }

        PseudoLabels = pseudoLabels.ToVector();
        PseudoLabelConfidences = pseudoConfidences.ToVector();

        // Final training with all labeled data (original + pseudo-labeled)
        _baseClassifier.Train(trainX.ToMatrix(), trainY.ToVector());
    }

    /// <summary>
    /// Selects unlabeled samples to add to the training set based on the selection criterion.
    /// </summary>
    private (List<int> indices, List<T> labels, List<T> confidences) SelectSamplesToAdd(
        Matrix<T> unlabeledX, List<int> remainingIndices)
    {
        var selectedIndices = new List<int>();
        var selectedLabels = new List<T>();
        var selectedConfidences = new List<T>();

        if (remainingIndices.Count == 0)
        {
            return (selectedIndices, selectedLabels, selectedConfidences);
        }

        // Build matrix of remaining unlabeled samples
        var engine = AiDotNetEngine.Current;
        var remainingX = engine.TensorIndexSelect(
            Tensor<T>.FromMatrix(unlabeledX),
            new Tensor<int>(remainingIndices.ToArray(), new[] { remainingIndices.Count }), 0).ToMatrix();

        // Get predictions
        var predictions = _baseClassifier.Predict(remainingX);

        // Get confidence scores (using probability predictions if available)
        var confidenceScores = GetConfidenceScores(remainingX, predictions);

        // Candidates: confidence >= threshold, sorted by confidence descending (stable, as OrderByDescending).
        var confidenceValues = Tensor<T>.FromVector(confidenceScores);
        var eligible = AtLeast(confidenceValues, _confidenceThreshold);
        var predictedValues = predictions.ToArray();
        var confidenceArray = confidenceScores.ToArray();
        var candidates = engine.TensorNonzero(eligible).ToArray()
            .OrderByDescending(i => confidenceArray[i], Comparer<T>.Default)
            .Select(i => (index: i, label: predictedValues[i], confidence: confidenceArray[i]))
            .ToList();

        // Apply selection criterion
        int maxToAdd = _maxSamplesPerIteration > 0 ? _maxSamplesPerIteration : candidates.Count;

        switch (_selectionCriterion)
        {
            case SelectionCriterion.Threshold:
                foreach (var (index, label, confidence) in candidates.Take(maxToAdd))
                {
                    selectedIndices.Add(index);
                    selectedLabels.Add(label);
                    selectedConfidences.Add(confidence);
                }
                break;

            case SelectionCriterion.TopK:
                foreach (var (index, label, confidence) in candidates.Take(maxToAdd))
                {
                    selectedIndices.Add(index);
                    selectedLabels.Add(label);
                    selectedConfidences.Add(confidence);
                }
                break;

            case SelectionCriterion.TopKPerClass:
                // Keyed by the label value; ValueTuple<T> satisfies Dictionary's notnull key constraint.
                var perClassCounts = new Dictionary<ValueTuple<T>, int>();
                int perClassLimit = Math.Max(1, maxToAdd / NumClasses);

                foreach (var (index, label, confidence) in candidates)
                {
                    if (!perClassCounts.TryGetValue(new ValueTuple<T>(label), out int count))
                    {
                        count = 0;
                    }

                    if (count < perClassLimit)
                    {
                        selectedIndices.Add(index);
                        selectedLabels.Add(label);
                        selectedConfidences.Add(confidence);
                        perClassCounts[new ValueTuple<T>(label)] = count + 1;
                    }
                }
                break;
        }

        return (selectedIndices, selectedLabels, selectedConfidences);
    }

    /// <summary>
    /// 1 where <paramref name="values"/> is at least the double <paramref name="threshold"/>, else 0, decided exactly
    /// as the double comparison <c>value &gt;= threshold</c> master made.
    /// </summary>
    /// <remarks>
    /// Comparing against the threshold rounded to <typeparamref name="T"/> is not the same test: when the rounding
    /// went down (0.7 becomes 0.699999988 in float) a value equal to the rounded threshold is below the real one
    /// and must be rejected, so the comparison becomes strict.
    /// </remarks>
    internal static Tensor<T> AtLeast(Tensor<T> values, double threshold)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        T rounded = ops.FromDouble(threshold);
        return ops.ToDouble(rounded) < threshold
            ? engine.TensorGreaterThan(values, rounded)
            : engine.TensorGreaterEqual(values, engine.TensorFullLike(values, rounded));
    }

    /// <summary>
    /// Gets confidence scores for predictions.
    /// </summary>
    private Vector<T> GetConfidenceScores(Matrix<T> X, Vector<T> predictions)
    {
        var engine = AiDotNetEngine.Current;

        // Try to get probabilities from the base classifier if it supports it
        if (_baseClassifier is IProbabilisticClassifier<T> probabilisticClassifier)
        {
            // For each sample, confidence is the max probability (floored at zero).
            var probabilities = Tensor<T>.FromMatrix(probabilisticClassifier.PredictProbabilities(X));
            if (probabilities.Shape[1] == 0)
            {
                return new Vector<T>(X.Rows);
            }

            return engine.TensorClampMin(engine.ReduceMax(probabilities, new[] { 1 }, keepDims: false), NumOps.Zero).ToVector();
        }

        // If no probabilities available, use a default confidence of 1.0 for all predictions
        // This means all predictions will be considered equally confident
        var ones = new Tensor<T>(new[] { X.Rows });
        engine.TensorFill(ones, NumOps.One);
        return ones.ToVector();
    }

    /// <summary>
    /// Core implementation of standard supervised training.
    /// </summary>
    protected override void TrainSupervisedCore(Matrix<T> x, Vector<T> y)
    {
        _baseClassifier.Train(x, y);
    }

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        return _baseClassifier.Predict(input);
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    private Vector<T> PackParameters()
    {
        return ((IParameterizable<T, Matrix<T>, Vector<T>>)_baseClassifier).GetParameters();
    }

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newClassifier = new SelfTrainingClassifier<T>(
            (IClassifier<T>)((IParameterizable<T, Matrix<T>, Vector<T>>)_baseClassifier).WithParameters(parameters),
            _confidenceThreshold,
            _maxIterations,
            _maxSamplesPerIteration,
            _selectionCriterion,
            Options,
            Regularization);

        newClassifier.NumFeatures = NumFeatures;
        newClassifier.NumClasses = NumClasses;
        newClassifier.ClassLabels = ClassLabels;
        newClassifier.TaskType = TaskType;

        return newClassifier;
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    private void UnpackParameters(Vector<T> parameters)
    {
        ((IParameterizable<T, Matrix<T>, Vector<T>>)_baseClassifier).SetParameters(parameters);
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return ((IGradientComputable<T, Matrix<T>, Vector<T>>)_baseClassifier).ComputeGradients(input, target, lossFunction ?? DefaultLossFunction);
    }

    /// <summary>
    /// Applies gradients to update the model parameters.
    /// </summary>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        ((IGradientComputable<T, Matrix<T>, Vector<T>>)_baseClassifier).ApplyGradients(gradients, learningRate);
    }
}
