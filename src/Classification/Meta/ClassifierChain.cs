using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Classifier Chain for multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Classifier Chain transforms a multi-label problem into a chain of binary
/// classification problems, where each classifier uses the predictions of
/// previous classifiers as additional features.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Classifier Chain captures label dependencies:
///
/// For labels A, B, C:
/// - Classifier 1: Predict A using features X
/// - Classifier 2: Predict B using features X + prediction of A
/// - Classifier 3: Predict C using features X + predictions of A and B
///
/// Benefits:
/// - Captures dependencies between labels
/// - Better than independent binary classifiers
///
/// Trade-offs:
/// - Order of chain matters (can use random order or learned order)
/// - Error propagation (early mistakes affect later predictions)
/// </para>
/// </remarks>
public class ClassifierChain<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Gets the chain-specific options.
    /// </summary>
    protected new ClassifierChainOptions<T> Options => (ClassifierChainOptions<T>)base.Options;

    /// <summary>
    /// The classifiers in the chain, one per label.
    /// </summary>
    private IClassifier<T>[]? _classifiers;

    /// <summary>
    /// The order of labels in the chain.
    /// </summary>
    private int[]? _order;

    /// <summary>
    /// Gets the number of labels (same as number of classes for multi-label).
    /// </summary>
    public int NumLabels => NumClasses;

    /// <summary>
    /// Gets or sets the label names if available.
    /// </summary>
    public string[]? LabelNames { get; set; }

    /// <summary>
    /// Initializes a new instance of the ClassifierChain class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base binary classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public ClassifierChain(
        Func<IClassifier<T>> estimatorFactory,
        ClassifierChainOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new ClassifierChainOptions<T>(), estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.ClassifierChain;

    /// <summary>
    /// Trains the Classifier Chain on multi-label data.
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

        // Determine chain order
        _order = DetermineOrder();
        _classifiers = new IClassifier<T>[NumClasses];

        // Train classifiers in chain order
        for (int i = 0; i < NumClasses; i++)
        {
            int labelIdx = _order[i];

            // Create augmented features: original features + previous predictions
            var xAugmented = CreateAugmentedFeatures(x, yMultiLabel, i);

            // Get labels for this position
            var yLabel = new Vector<T>(x.Rows);
            for (int s = 0; s < x.Rows; s++)
            {
                yLabel[s] = yMultiLabel[s, labelIdx];
            }

            // Train classifier
            _classifiers[labelIdx] = CreateBaseEstimator();
            _classifiers[labelIdx].Train(xAugmented, yLabel);
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
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Convert to multi-label format (one-hot encoding)
        var yMultiLabel = new Matrix<T>(y.Length, NumClasses);
        for (int i = 0; i < y.Length; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                if (NumOps.Compare(y[i], ClassLabels[c]) == 0)
                {
                    yMultiLabel[i, c] = NumOps.One;
                }
            }
        }

        TrainMultiLabel(x, yMultiLabel);
    }

    /// <summary>
    /// Determines the order of labels in the chain.
    /// </summary>
    private int[] DetermineOrder()
    {
        var order = new int[NumClasses];

        if (Options.Order is not null && Options.Order.Length == NumClasses)
        {
            // Use specified order
            Array.Copy(Options.Order, order, NumClasses);
        }
        else if (Options.RandomOrder)
        {
            // Random order
            for (int i = 0; i < NumClasses; i++)
            {
                order[i] = i;
            }

            var random = Options.RandomState.HasValue
                ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
                : RandomHelper.CreateSeededRandom(42);

            for (int i = NumClasses - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }
        else
        {
            // Default order (0, 1, 2, ...)
            for (int i = 0; i < NumClasses; i++)
            {
                order[i] = i;
            }
        }

        return order;
    }

    /// <summary>
    /// Creates augmented features by adding previous predictions.
    /// </summary>
    private Matrix<T> CreateAugmentedFeatures(Matrix<T> x, Matrix<T> yPrevious, int chainPosition)
    {
        int numPrevLabels = chainPosition;
        var xAugmented = new Matrix<T>(x.Rows, NumFeatures + numPrevLabels);

        for (int i = 0; i < x.Rows; i++)
        {
            // Copy original features
            for (int j = 0; j < NumFeatures; j++)
            {
                xAugmented[i, j] = x[i, j];
            }

            // Add previous labels as features
            for (int p = 0; p < numPrevLabels; p++)
            {
                int prevLabelIdx = _order![p];
                xAugmented[i, NumFeatures + p] = yPrevious[i, prevLabelIdx];
            }
        }

        return xAugmented;
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

    /// <summary>
    /// Predicts multi-label output for the given input.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix of binary predictions for each label.</returns>
    public Matrix<T> PredictMultiLabel(Matrix<T> input)
    {
        if (_classifiers is null || _order is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Matrix<T>(input.Rows, NumClasses);

        // Predict in chain order
        for (int i = 0; i < NumClasses; i++)
        {
            int labelIdx = _order[i];

            // Create augmented features with previous predictions
            var xAugmented = new Matrix<T>(input.Rows, NumFeatures + i);

            for (int s = 0; s < input.Rows; s++)
            {
                // Copy original features
                for (int j = 0; j < NumFeatures; j++)
                {
                    xAugmented[s, j] = input[s, j];
                }

                // Add previous predictions as features
                for (int p = 0; p < i; p++)
                {
                    int prevLabelIdx = _order[p];
                    xAugmented[s, NumFeatures + p] = predictions[s, prevLabelIdx];
                }
            }

            // Predict for this label
            var labelPreds = _classifiers[labelIdx].Predict(xAugmented);

            for (int s = 0; s < input.Rows; s++)
            {
                // Threshold at 0.5 for binary
                predictions[s, labelIdx] = NumOps.Compare(labelPreds[s], NumOps.FromDouble(0.5)) >= 0
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return predictions;
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_classifiers is null || _order is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Matrix<T>(input.Rows, NumClasses);
        var probs = new Matrix<T>(input.Rows, NumClasses);

        // Predict in chain order
        for (int i = 0; i < NumClasses; i++)
        {
            int labelIdx = _order[i];

            // Create augmented features with previous predictions
            var xAugmented = new Matrix<T>(input.Rows, NumFeatures + i);

            for (int s = 0; s < input.Rows; s++)
            {
                // Copy original features
                for (int j = 0; j < NumFeatures; j++)
                {
                    xAugmented[s, j] = input[s, j];
                }

                // Add previous predictions as features
                for (int p = 0; p < i; p++)
                {
                    int prevLabelIdx = _order[p];
                    xAugmented[s, NumFeatures + p] = predictions[s, prevLabelIdx];
                }
            }

            // Get probabilities if available
            if (_classifiers[labelIdx] is IProbabilisticClassifier<T> probClassifier)
            {
                var labelProbs = probClassifier.PredictProbabilities(xAugmented);
                int posCol = labelProbs.Columns > 1 ? 1 : 0;

                for (int s = 0; s < input.Rows; s++)
                {
                    probs[s, labelIdx] = labelProbs[s, posCol];
                    predictions[s, labelIdx] = NumOps.Compare(probs[s, labelIdx], NumOps.FromDouble(0.5)) >= 0
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
            else
            {
                var labelPreds = _classifiers[labelIdx].Predict(xAugmented);

                for (int s = 0; s < input.Rows; s++)
                {
                    predictions[s, labelIdx] = NumOps.Compare(labelPreds[s], NumOps.FromDouble(0.5)) >= 0
                        ? NumOps.One
                        : NumOps.Zero;
                    probs[s, labelIdx] = predictions[s, labelIdx];
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

        return new ClassifierChain<T>(EstimatorFactory, new ClassifierChainOptions<T>
        {
            Order = Options.Order,
            RandomOrder = Options.RandomOrder,
            RandomState = Options.RandomState
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (ClassifierChain<T>)CreateNewInstance();

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

        if (_order is not null)
        {
            clone._order = new int[_order.Length];
            Array.Copy(_order, clone._order, _order.Length);
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

/// <summary>
/// Configuration options for Classifier Chain.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class ClassifierChainOptions<T> : MetaClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the explicit order of labels in the chain.
    /// </summary>
    /// <value>Array of label indices defining chain order.</value>
    public int[]? Order { get; set; }

    /// <summary>
    /// Gets or sets whether to use random chain order.
    /// </summary>
    /// <value>True for random order. Default is false.</value>
    public bool RandomOrder { get; set; } = false;

    /// <summary>
    /// Gets or sets the random state for reproducibility.
    /// </summary>
    public int? RandomState { get; set; }
}
