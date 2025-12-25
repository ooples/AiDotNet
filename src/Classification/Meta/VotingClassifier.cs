using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Voting classifier that combines multiple classifiers through voting.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Voting classifier combines predictions from multiple different classifiers
/// using either hard voting (majority vote) or soft voting (average probabilities).
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Voting combines predictions from multiple models:
///
/// Hard Voting:
/// - Each classifier votes for a class
/// - The class with most votes wins
/// - Example: [A, A, B] -> A wins (2 vs 1)
///
/// Soft Voting:
/// - Average the probability predictions
/// - Pick the class with highest average probability
/// - Generally works better when classifiers output calibrated probabilities
///
/// When to use:
/// - To combine different types of classifiers
/// - When you want to reduce the risk of a single bad model
/// - To leverage the strengths of different algorithms
/// </para>
/// </remarks>
public class VotingClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Gets the voting-specific options.
    /// </summary>
    protected new VotingClassifierOptions<T> Options => (VotingClassifierOptions<T>)base.Options;

    /// <summary>
    /// The list of classifiers in the ensemble.
    /// </summary>
    private List<IClassifier<T>>? _estimators;

    /// <summary>
    /// The weights for each classifier.
    /// </summary>
    private double[]? _weights;

    /// <summary>
    /// Initializes a new instance of the VotingClassifier class.
    /// </summary>
    /// <param name="estimators">List of classifiers to combine.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public VotingClassifier(
        IEnumerable<IClassifier<T>> estimators,
        VotingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new VotingClassifierOptions<T>(), null, regularization)
    {
        _estimators = estimators.ToList();
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.VotingClassifier;

    /// <summary>
    /// Trains all classifiers in the voting ensemble.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        if (_estimators is null || _estimators.Count == 0)
        {
            throw new InvalidOperationException("No estimators provided for voting.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Set up weights
        if (Options.Weights is not null && Options.Weights.Length == _estimators.Count)
        {
            _weights = Options.Weights;
        }
        else
        {
            _weights = new double[_estimators.Count];
            for (int i = 0; i < _estimators.Count; i++)
            {
                _weights[i] = 1.0;
            }
        }

        // Normalize weights
        double weightSum = 0;
        for (int i = 0; i < _weights.Length; i++)
        {
            weightSum += _weights[i];
        }
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] /= weightSum;
        }

        // Train each estimator
        foreach (var estimator in _estimators)
        {
            estimator.Train(x, y);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (Options.Voting == VotingType.Soft)
        {
            // Soft voting: use probabilities
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

                predictions[i] = ClassLabels[bestClass];
            }

            return predictions;
        }
        else
        {
            // Hard voting: majority vote
            return HardVotePredict(input);
        }
    }

    /// <summary>
    /// Performs hard voting prediction.
    /// </summary>
    private Vector<T> HardVotePredict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var votes = new double[NumClasses];

            // Get vote from each classifier
            for (int e = 0; e < _estimators!.Count; e++)
            {
                // Extract single sample
                var sample = new Matrix<T>(1, NumFeatures);
                for (int j = 0; j < NumFeatures; j++)
                {
                    sample[0, j] = input[i, j];
                }

                var pred = _estimators[e].Predict(sample);

                // Find which class
                for (int c = 0; c < NumClasses; c++)
                {
                    if (NumOps.Compare(pred[0], ClassLabels![c]) == 0)
                    {
                        votes[c] += _weights![e];
                        break;
                    }
                }
            }

            // Find class with most weighted votes
            int bestClass = 0;
            double maxVotes = votes[0];
            for (int c = 1; c < NumClasses; c++)
            {
                if (votes[c] > maxVotes)
                {
                    maxVotes = votes[c];
                    bestClass = c;
                }
            }

            predictions[i] = ClassLabels![bestClass];
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

        var probs = new Matrix<T>(input.Rows, NumClasses);

        // Weighted average of probabilities from all classifiers
        for (int e = 0; e < _estimators.Count; e++)
        {
            Matrix<T> estProbs;

            if (_estimators[e] is IProbabilisticClassifier<T> probClassifier)
            {
                estProbs = probClassifier.PredictProbabilities(input);
            }
            else
            {
                // Use hard predictions
                var preds = _estimators[e].Predict(input);
                estProbs = new Matrix<T>(input.Rows, NumClasses);

                for (int i = 0; i < input.Rows; i++)
                {
                    for (int c = 0; c < NumClasses; c++)
                    {
                        if (NumOps.Compare(preds[i], ClassLabels![c]) == 0)
                        {
                            estProbs[i, c] = NumOps.One;
                        }
                    }
                }
            }

            // Weighted accumulation
            T weight = NumOps.FromDouble(_weights![e]);
            for (int i = 0; i < input.Rows; i++)
            {
                for (int c = 0; c < NumClasses; c++)
                {
                    T weighted = NumOps.Multiply(estProbs[i, c], weight);
                    probs[i, c] = NumOps.Add(probs[i, c], weighted);
                }
            }
        }

        return probs;
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
        var newEstimators = new List<IClassifier<T>>();

        if (_estimators is not null)
        {
            foreach (var est in _estimators)
            {
                if (est is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    newEstimators.Add((IClassifier<T>)fullModel.Clone());
                }
                else
                {
                    newEstimators.Add(est);
                }
            }
        }

        return new VotingClassifier<T>(newEstimators, new VotingClassifierOptions<T>
        {
            Voting = Options.Voting,
            Weights = Options.Weights
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (VotingClassifier<T>)CreateNewInstance();

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

        if (_weights is not null)
        {
            clone._weights = new double[_weights.Length];
            Array.Copy(_weights, clone._weights, _weights.Length);
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["VotingType"] = Options.Voting.ToString();
        metadata.AdditionalInfo["NumEstimators"] = _estimators?.Count ?? 0;
        return metadata;
    }
}

/// <summary>
/// Configuration options for Voting classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class VotingClassifierOptions<T> : MetaClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the voting type.
    /// </summary>
    /// <value>Hard or soft voting. Default is hard voting.</value>
    public VotingType Voting { get; set; } = VotingType.Hard;

    /// <summary>
    /// Gets or sets the weights for each classifier.
    /// </summary>
    /// <value>Array of weights. If null, all classifiers have equal weight.</value>
    public double[]? Weights { get; set; }
}

/// <summary>
/// Type of voting for ensemble classifiers.
/// </summary>
public enum VotingType
{
    /// <summary>
    /// Hard voting: majority vote of class labels.
    /// </summary>
    Hard,

    /// <summary>
    /// Soft voting: average of predicted probabilities.
    /// </summary>
    Soft
}
