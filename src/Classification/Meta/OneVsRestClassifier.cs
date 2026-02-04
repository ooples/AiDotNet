using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// One-vs-Rest (also called One-vs-All) classifier for multi-class classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Trains one binary classifier per class, treating it as the positive class
/// and all other classes as the negative class.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// One-vs-Rest is a simple strategy for multi-class classification:
///
/// For 3 classes (A, B, C):
/// - Classifier 1: Is it A vs not-A?
/// - Classifier 2: Is it B vs not-B?
/// - Classifier 3: Is it C vs not-C?
///
/// For prediction, the class whose classifier gives the highest score wins.
///
/// Advantages:
/// - Simple and effective
/// - Trains K classifiers for K classes
/// - Easily parallelizable
///
/// Disadvantages:
/// - Class imbalance (one class vs all others)
/// - Classifiers don't see inter-class relationships
/// </para>
/// <para>
/// <b>Multi-label usage:</b>
/// This classifier also supports multi-label classification via
/// <see cref="TrainMultiLabel(Matrix{T}, Matrix{T})"/>, along with the
/// <see cref="NumLabels"/> and <see cref="LabelNames"/> properties.
/// When using this class in a multi-label setting, prefer
/// <see cref="TrainMultiLabel(Matrix{T}, Matrix{T})"/> over
/// <see cref="Train(Matrix{T}, Vector{T})"/>.
/// </para>
/// </remarks>
public class OneVsRestClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// The binary classifiers, one per class.
    /// </summary>
    private IClassifier<T>[]? _estimators;

    /// <summary>
    /// Gets the number of labels (same as number of classes for multi-label).
    /// </summary>
    public int NumLabels => NumClasses;

    /// <summary>
    /// Gets or sets the label names if available.
    /// </summary>
    public string[]? LabelNames { get; set; }

    /// <summary>
    /// Initializes a new instance of the OneVsRestClassifier class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base binary classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public OneVsRestClassifier(
        Func<IClassifier<T>> estimatorFactory,
        MetaClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.OneVsRestClassifier;

    /// <summary>
    /// Trains the One-vs-Rest classifier on multi-label data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="yMultiLabel">The multi-label target matrix (rows=samples, cols=labels).</param>
    public void TrainMultiLabel(Matrix<T> x, Matrix<T> yMultiLabel)
    {
        if (x.Rows != yMultiLabel.Rows)
        {
            throw new ArgumentException("Number of samples in X must match number of rows in Y.");
        }

        NumFeatures = x.Columns;
        NumClasses = yMultiLabel.Columns;
        TaskType = ClassificationTaskType.MultiLabel;

        // Create class labels (0 to NumClasses-1)
        ClassLabels = new Vector<T>(NumClasses);
        for (int c = 0; c < NumClasses; c++)
        {
            ClassLabels[c] = NumOps.FromDouble(c);
        }

        // Train one binary classifier per label
        _estimators = new IClassifier<T>[NumClasses];

        for (int c = 0; c < NumClasses; c++)
        {
            // Get labels for this output
            var yLabel = new Vector<T>(x.Rows);
            for (int i = 0; i < x.Rows; i++)
            {
                yLabel[i] = yMultiLabel[i, c];
            }

            // Train binary classifier
            _estimators[c] = CreateBaseEstimator();
            _estimators[c].Train(x, yLabel);
        }
    }

    /// <summary>
    /// Trains the One-vs-Rest classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Train one binary classifier per class
        _estimators = new IClassifier<T>[NumClasses];

        for (int c = 0; c < NumClasses; c++)
        {
            // Create binary labels: 1 for this class, 0 for all others
            var binaryLabels = new Vector<T>(y.Length);
            T classLabel = ClassLabels[c];

            for (int i = 0; i < y.Length; i++)
            {
                binaryLabels[i] = NumOps.Compare(y[i], classLabel) == 0
                    ? NumOps.One
                    : NumOps.Zero;
            }

            // Train binary classifier
            _estimators[c] = CreateBaseEstimator();
            _estimators[c].Train(x, binaryLabels);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            int bestClass = 0;
            T bestScore = probs[i, 0];

            for (int c = 1; c < NumClasses; c++)
            {
                if (NumOps.Compare(probs[i, c], bestScore) > 0)
                {
                    bestScore = probs[i, c];
                    bestClass = c;
                }
            }

            predictions[i] = ClassLabels[bestClass];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var scores = new Matrix<T>(input.Rows, NumClasses);

        // Get decision scores from each binary classifier
        for (int c = 0; c < NumClasses; c++)
        {
            var classProbs = GetEstimatorScores(_estimators[c], input);

            for (int i = 0; i < input.Rows; i++)
            {
                scores[i, c] = classProbs[i];
            }
        }

        // Normalize to probabilities using softmax
        for (int i = 0; i < input.Rows; i++)
        {
            // Find max for numerical stability
            T maxScore = scores[i, 0];
            for (int c = 1; c < NumClasses; c++)
            {
                if (NumOps.Compare(scores[i, c], maxScore) > 0)
                {
                    maxScore = scores[i, c];
                }
            }

            // Compute softmax
            T sumExp = NumOps.Zero;
            for (int c = 0; c < NumClasses; c++)
            {
                T expVal = NumOps.Exp(NumOps.Subtract(scores[i, c], maxScore));
                scores[i, c] = expVal;
                sumExp = NumOps.Add(sumExp, expVal);
            }

            // Normalize
            for (int c = 0; c < NumClasses; c++)
            {
                scores[i, c] = NumOps.Divide(scores[i, c], sumExp);
            }
        }

        return scores;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var logProbs = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                T p = probs[i, c];
                T minP = NumOps.FromDouble(1e-15);
                if (NumOps.Compare(p, minP) < 0)
                {
                    p = minP;
                }
                logProbs[i, c] = NumOps.Log(p);
            }
        }

        return logProbs;
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabel(Matrix<T> input)
    {
        if (_estimators is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // For multi-label, each classifier's prediction is independent
        var predictions = new Matrix<T>(input.Rows, NumClasses);

        for (int c = 0; c < NumClasses; c++)
        {
            var classScores = GetEstimatorScores(_estimators[c], input);

            for (int i = 0; i < input.Rows; i++)
            {
                // Threshold at 0.5 for binary decision
                predictions[i, c] = NumOps.Compare(classScores[i], NumOps.FromDouble(0.5)) >= 0
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return predictions;
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_estimators is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probs = new Matrix<T>(input.Rows, NumClasses);

        for (int c = 0; c < NumClasses; c++)
        {
            var classScores = GetEstimatorScores(_estimators[c], input);

            for (int i = 0; i < input.Rows; i++)
            {
                probs[i, c] = classScores[i];
            }
        }

        return probs;
    }

    /// <summary>
    /// Gets decision scores from an estimator.
    /// </summary>
    private Vector<T> GetEstimatorScores(IClassifier<T> estimator, Matrix<T> input)
    {
        // Try to get probabilities if available
        if (estimator is IProbabilisticClassifier<T> probClassifier)
        {
            var probs = probClassifier.PredictProbabilities(input);
            var scores = new Vector<T>(input.Rows);

            // Get probability of positive class (class 1)
            int posCol = probs.Columns > 1 ? 1 : 0;
            for (int i = 0; i < input.Rows; i++)
            {
                scores[i] = probs[i, posCol];
            }
            return scores;
        }

        // Fall back to binary predictions
        var predictions = estimator.Predict(input);
        var result = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = predictions[i];
        }
        return result;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        if (EstimatorFactory is null)
        {
            throw new InvalidOperationException("Estimator factory is not set.");
        }

        return new OneVsRestClassifier<T>(EstimatorFactory, new MetaClassifierOptions<T>
        {
            NumJobs = Options.NumJobs
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (OneVsRestClassifier<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (ClassLabels is not null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (_estimators is not null)
        {
            clone._estimators = new IClassifier<T>[NumClasses];
            for (int c = 0; c < NumClasses; c++)
            {
                if (_estimators[c] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    clone._estimators[c] = (IClassifier<T>)fullModel.Clone();
                }
                else
                {
                    clone._estimators[c] = _estimators[c];
                }
            }
        }

        return clone;
    }
}
