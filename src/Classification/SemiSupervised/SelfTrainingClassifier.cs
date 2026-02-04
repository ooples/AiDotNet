using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

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
public class SelfTrainingClassifier<T> : SemiSupervisedClassifierBase<T>
{
    private readonly IClassifier<T> _baseClassifier;
    private readonly double _confidenceThreshold;
    private readonly int _maxIterations;
    private readonly int _maxSamplesPerIteration;
    private readonly SelectionCriterion _selectionCriterion;

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
        _baseClassifier = baseClassifier ?? throw new ArgumentNullException(nameof(baseClassifier));

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
        // Create mutable copies of the data
        var currentLabeledX = new List<Vector<T>>();
        var currentLabeledY = new List<T>();

        // Initialize with original labeled data
        for (int i = 0; i < labeledX.Rows; i++)
        {
            currentLabeledX.Add(labeledX.GetRow(i));
            currentLabeledY.Add(labeledY[i]);
        }

        // Track remaining unlabeled samples
        var remainingUnlabeled = new List<int>();
        for (int i = 0; i < unlabeledX.Rows; i++)
        {
            remainingUnlabeled.Add(i);
        }

        // Initialize pseudo-labels storage
        PseudoLabels = new Vector<T>(unlabeledX.Rows);
        PseudoLabelConfidences = new Vector<T>(unlabeledX.Rows);
        for (int i = 0; i < unlabeledX.Rows; i++)
        {
            PseudoLabels[i] = NumOps.FromDouble(-1); // -1 indicates not yet labeled
            PseudoLabelConfidences[i] = NumOps.Zero;
        }

        IterationsPerformed = 0;
        SamplesAdded = 0;

        for (int iteration = 0; iteration < _maxIterations && remainingUnlabeled.Count > 0; iteration++)
        {
            IterationsPerformed++;

            // Convert lists to matrices for training
            var trainX = ListToMatrix(currentLabeledX);
            var trainY = new Vector<T>(currentLabeledY.ToArray());

            // Train the base classifier
            _baseClassifier.Train(trainX, trainY);

            // Get predictions and confidences for remaining unlabeled samples
            var (selectedIndices, predictedLabels, confidences) = SelectSamplesToAdd(unlabeledX, remainingUnlabeled);

            if (selectedIndices.Count == 0)
            {
                break; // No more samples meet criteria
            }

            // Add selected samples to training set
            // Note: selectedIndices contains indices into remainingUnlabeled
            // predictedLabels and confidences are parallel lists aligned with selectedIndices
            for (int i = 0; i < selectedIndices.Count; i++)
            {
                int idx = selectedIndices[i]; // Index into remainingUnlabeled
                int originalIndex = remainingUnlabeled[idx];
                currentLabeledX.Add(unlabeledX.GetRow(originalIndex));
                currentLabeledY.Add(predictedLabels[i]); // Use i, not idx

                // Store pseudo-label info
                PseudoLabels[originalIndex] = predictedLabels[i];
                PseudoLabelConfidences[originalIndex] = confidences[i];
            }

            SamplesAdded += selectedIndices.Count;

            // Remove added samples from remaining unlabeled (in reverse order to maintain indices)
            var sortedIndices = selectedIndices.OrderByDescending(x => x).ToList();
            foreach (int idx in sortedIndices)
            {
                remainingUnlabeled.RemoveAt(idx);
            }
        }

        // Final training with all labeled data (original + pseudo-labeled)
        var finalTrainX = ListToMatrix(currentLabeledX);
        var finalTrainY = new Vector<T>(currentLabeledY.ToArray());
        _baseClassifier.Train(finalTrainX, finalTrainY);
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
        var remainingX = new Matrix<T>(remainingIndices.Count, unlabeledX.Columns);
        for (int i = 0; i < remainingIndices.Count; i++)
        {
            var row = unlabeledX.GetRow(remainingIndices[i]);
            for (int j = 0; j < unlabeledX.Columns; j++)
            {
                remainingX[i, j] = row[j];
            }
        }

        // Get predictions
        var predictions = _baseClassifier.Predict(remainingX);

        // Get confidence scores (using probability predictions if available)
        var confidenceScores = GetConfidenceScores(remainingX, predictions);

        // Select samples based on criterion
        var candidates = new List<(int index, T label, T confidence)>();
        for (int i = 0; i < remainingIndices.Count; i++)
        {
            double conf = NumOps.ToDouble(confidenceScores[i]);
            if (conf >= _confidenceThreshold)
            {
                candidates.Add((i, predictions[i], confidenceScores[i]));
            }
        }

        // Sort by confidence (descending)
        candidates = candidates.OrderByDescending(c => NumOps.ToDouble(c.confidence)).ToList();

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
                var perClassCounts = new Dictionary<double, int>();
                int perClassLimit = Math.Max(1, maxToAdd / NumClasses);

                foreach (var (index, label, confidence) in candidates)
                {
                    double labelValue = NumOps.ToDouble(label);
                    if (!perClassCounts.TryGetValue(labelValue, out int count))
                    {
                        count = 0;
                    }

                    if (count < perClassLimit)
                    {
                        selectedIndices.Add(index);
                        selectedLabels.Add(label);
                        selectedConfidences.Add(confidence);
                        perClassCounts[labelValue] = count + 1;
                    }
                }
                break;
        }

        return (selectedIndices, selectedLabels, selectedConfidences);
    }

    /// <summary>
    /// Gets confidence scores for predictions.
    /// </summary>
    private Vector<T> GetConfidenceScores(Matrix<T> X, Vector<T> predictions)
    {
        var confidences = new Vector<T>(X.Rows);

        // Try to get probabilities from the base classifier if it supports it
        if (_baseClassifier is IProbabilisticClassifier<T> probabilisticClassifier)
        {
            var probabilities = probabilisticClassifier.PredictProbabilities(X);

            // For each sample, confidence is the max probability
            for (int i = 0; i < X.Rows; i++)
            {
                T maxProb = NumOps.Zero;
                for (int j = 0; j < probabilities.Columns; j++)
                {
                    if (NumOps.GreaterThan(probabilities[i, j], maxProb))
                    {
                        maxProb = probabilities[i, j];
                    }
                }
                confidences[i] = maxProb;
            }
        }
        else
        {
            // If no probabilities available, use a default confidence of 1.0 for all predictions
            // This means all predictions will be considered equally confident
            for (int i = 0; i < X.Rows; i++)
            {
                confidences[i] = NumOps.One;
            }
        }

        return confidences;
    }

    /// <summary>
    /// Converts a list of vectors to a matrix.
    /// </summary>
    private Matrix<T> ListToMatrix(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
        {
            return new Matrix<T>(0, 0);
        }

        int numFeatures = vectors[0].Length;
        var matrix = new Matrix<T>(vectors.Count, numFeatures);

        for (int i = 0; i < vectors.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                matrix[i, j] = vectors[i][j];
            }
        }

        return matrix;
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
    public override Vector<T> GetParameters()
    {
        return _baseClassifier.GetParameters();
    }

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newClassifier = new SelfTrainingClassifier<T>(
            (IClassifier<T>)_baseClassifier.WithParameters(parameters),
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
    public override void SetParameters(Vector<T> parameters)
    {
        _baseClassifier.SetParameters(parameters);
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return _baseClassifier.ComputeGradients(input, target, lossFunction ?? DefaultLossFunction);
    }

    /// <summary>
    /// Applies gradients to update the model parameters.
    /// </summary>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _baseClassifier.ApplyGradients(gradients, learningRate);
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    protected override ModelType GetModelType()
    {
        return ModelType.SelfTrainingClassifier;
    }

    /// <summary>
    /// Creates a new instance of this classifier.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SelfTrainingClassifier<T>(
            (IClassifier<T>)_baseClassifier.Clone(),
            _confidenceThreshold,
            _maxIterations,
            _maxSamplesPerIteration,
            _selectionCriterion,
            Options,
            Regularization);
    }
}
