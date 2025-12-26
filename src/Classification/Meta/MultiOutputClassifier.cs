using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Multi-output classifier for independent multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MultiOutputClassifier fits one classifier per target label, treating
/// each label as independent of the others.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// MultiOutputClassifier is the simplest multi-label approach:
///
/// For labels A, B, C:
/// - Classifier 1: Predict A using features X
/// - Classifier 2: Predict B using features X
/// - Classifier 3: Predict C using features X
///
/// Each classifier is completely independent.
///
/// When to use:
/// - When labels are truly independent
/// - As a simple baseline for multi-label problems
/// - When you don't need to model label correlations
///
/// Note: Unlike ClassifierChain, this does NOT capture label dependencies.
/// </para>
/// </remarks>
public class MultiOutputClassifier<T> : MetaClassifierBase<T>, IMultiLabelClassifier<T>
{
    /// <summary>
    /// The classifiers, one per output.
    /// </summary>
    private IClassifier<T>[]? _classifiers;

    /// <inheritdoc/>
    public int NumLabels => NumClasses;

    /// <summary>
    /// Initializes a new instance of the MultiOutputClassifier class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public MultiOutputClassifier(
        Func<IClassifier<T>> estimatorFactory,
        MetaClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.MultiOutputClassifier;

    /// <summary>
    /// Trains the Multi-output classifier on multi-label data.
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

        _classifiers = new IClassifier<T>[NumClasses];

        // Train one classifier per output
        for (int c = 0; c < NumClasses; c++)
        {
            // Get labels for this output
            var yLabel = new Vector<T>(x.Rows);
            for (int i = 0; i < x.Rows; i++)
            {
                yLabel[i] = yMultiLabel[i, c];
            }

            // Train classifier
            _classifiers[c] = CreateBaseEstimator();
            _classifiers[c].Train(x, yLabel);
        }
    }

    /// <summary>
    /// Standard training method - converts single labels to multi-label format.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        var originalClassLabels = ExtractClassLabels(y);
        int numOriginalClasses = originalClassLabels.Length;
        TaskType = InferTaskType(y);

        // Convert to multi-label format (one-hot encoding)
        var yMultiLabel = new Matrix<T>(y.Length, numOriginalClasses);
        for (int i = 0; i < y.Length; i++)
        {
            for (int c = 0; c < numOriginalClasses; c++)
            {
                if (NumOps.Compare(y[i], originalClassLabels[c]) == 0)
                {
                    yMultiLabel[i, c] = NumOps.One;
                }
            }
        }

        TrainMultiLabel(x, yMultiLabel);

        // Restore original class labels for Predict to return correct label values
        ClassLabels = originalClassLabels;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // For single-label prediction, return the class with highest probability
        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            int bestClass = 0;
            T bestProb = probs[i, 0];

            for (int c = 1; c < NumClasses; c++)
            {
                if (NumOps.Compare(probs[i, c], bestProb) > 0)
                {
                    bestProb = probs[i, c];
                    bestClass = c;
                }
            }

            predictions[i] = ClassLabels![bestClass];
        }

        return predictions;
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabel(Matrix<T> input)
    {
        if (_classifiers is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Matrix<T>(input.Rows, NumClasses);

        // Predict for each output independently
        for (int c = 0; c < NumClasses; c++)
        {
            // Try to get probabilities first, fall back to using predictions directly
            if (_classifiers[c] is IProbabilisticClassifier<T> probClassifier)
            {
                var probs = probClassifier.PredictProbabilities(input);
                for (int i = 0; i < input.Rows; i++)
                {
                    // Use probability of positive class (column 1 for binary classifiers)
                    T posProb = probs.Columns > 1 ? probs[i, 1] : probs[i, 0];
                    predictions[i, c] = NumOps.Compare(posProb, NumOps.FromDouble(0.5)) >= 0
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
            else
            {
                // Fall back to using class predictions (assumes binary 0/1 labels)
                var labelPreds = _classifiers[c].Predict(input);
                for (int i = 0; i < input.Rows; i++)
                {
                    // Compare to NumOps.One to check if positive class was predicted
                    predictions[i, c] = NumOps.Compare(labelPreds[i], NumOps.One) == 0
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
        }

        return predictions;
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_classifiers is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probs = new Matrix<T>(input.Rows, NumClasses);

        // Get probabilities for each output independently
        for (int c = 0; c < NumClasses; c++)
        {
            if (_classifiers[c] is IProbabilisticClassifier<T> probClassifier)
            {
                var labelProbs = probClassifier.PredictProbabilities(input);
                int posCol = labelProbs.Columns > 1 ? 1 : 0;

                for (int i = 0; i < input.Rows; i++)
                {
                    probs[i, c] = labelProbs[i, posCol];
                }
            }
            else
            {
                var labelPreds = _classifiers[c].Predict(input);

                for (int i = 0; i < input.Rows; i++)
                {
                    probs[i, c] = labelPreds[i];
                }
            }
        }

        return probs;
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        return PredictMultiLabelProbabilities(input);
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
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        if (EstimatorFactory is null)
        {
            throw new InvalidOperationException("Estimator factory is not set.");
        }

        return new MultiOutputClassifier<T>(EstimatorFactory, new MetaClassifierOptions<T>
        {
            NumJobs = Options.NumJobs
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (MultiOutputClassifier<T>)CreateNewInstance();

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

        if (_classifiers is not null)
        {
            clone._classifiers = new IClassifier<T>[_classifiers.Length];
            for (int c = 0; c < _classifiers.Length; c++)
            {
                if (_classifiers[c] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    clone._classifiers[c] = (IClassifier<T>)fullModel.Clone();
                }
                else
                {
                    clone._classifiers[c] = _classifiers[c];
                }
            }
        }

        return clone;
    }
}
