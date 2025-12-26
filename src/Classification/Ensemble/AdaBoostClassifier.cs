using AiDotNet.Classification;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Ensemble;

/// <summary>
/// AdaBoost (Adaptive Boosting) classifier that combines weak learners.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// AdaBoost iteratively trains weak classifiers on re-weighted versions of the data,
/// where incorrectly classified samples receive higher weights in subsequent iterations.
/// The final prediction is a weighted vote of all weak learners.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// AdaBoost works like a learning system that focuses on its mistakes:
///
/// 1. Train a simple classifier on the data
/// 2. See which samples were misclassified
/// 3. Give those samples higher importance
/// 4. Train another classifier with the new importance weights
/// 5. Repeat many times
/// 6. Combine all classifiers with voting
///
/// This creates a powerful classifier from many weak ones.
/// </para>
/// </remarks>
public class AdaBoostClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Gets the AdaBoost specific options.
    /// </summary>
    protected new AdaBoostClassifierOptions<T> Options => (AdaBoostClassifierOptions<T>)base.Options;

    /// <summary>
    /// Weights for each estimator (based on their accuracy).
    /// </summary>
    private Vector<T>? _estimatorWeights;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Initializes a new instance of the AdaBoostClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for AdaBoost.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public AdaBoostClassifier(AdaBoostClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new AdaBoostClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.AdaBoostClassifier;

    /// <summary>
    /// Trains the AdaBoost classifier on the provided data.
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

        _random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators
        Estimators.Clear();
        _estimatorWeights = new Vector<T>(Options.NEstimators);

        // Initialize sample weights uniformly
        int n = x.Rows;
        var sampleWeights = new Vector<T>(n);
        T initialWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));
        for (int i = 0; i < n; i++)
        {
            sampleWeights[i] = initialWeight;
        }

        // Train estimators
        for (int m = 0; m < Options.NEstimators; m++)
        {
            // Create and train a weak learner (decision stump)
            var stumpOptions = new DecisionTreeClassifierOptions<T>
            {
                MaxDepth = 1, // Decision stump
                RandomState = _random.Next()
            };

            var stump = new DecisionTreeClassifier<T>(stumpOptions);

            // Sample with replacement based on weights
            var (xSample, ySample) = SampleWithWeights(x, y, sampleWeights);
            stump.Train(xSample, ySample);

            // Get predictions on full training set
            var predictions = stump.Predict(x);

            // Calculate weighted error
            T error = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                if (NumOps.Compare(predictions[i], y[i]) != 0)
                {
                    error = NumOps.Add(error, sampleWeights[i]);
                }
            }

            // Check if error is too high (random guessing or worse)
            double errorDouble = NumOps.ToDouble(error);
            if (errorDouble >= 0.5)
            {
                // Skip this estimator - it's not better than random
                continue;
            }

            // Calculate estimator weight: alpha = 0.5 * ln((1 - error) / error)
            T oneMinusError = NumOps.Subtract(NumOps.One, error);
            T ratio = NumOps.Divide(oneMinusError, NumOps.Add(error, NumOps.FromDouble(1e-10)));
            T logRatio = NumOps.Log(ratio);
            T alpha = NumOps.Multiply(NumOps.FromDouble(0.5), logRatio);
            alpha = NumOps.Multiply(alpha, NumOps.FromDouble(Options.LearningRate));

            _estimatorWeights[Estimators.Count] = alpha;
            Estimators.Add(stump);

            // Update sample weights
            T sumWeights = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T yi = y[i];
                T pred = predictions[i];

                // exp(-alpha * y * h(x))
                // For correct predictions: y * h(x) > 0 (same sign)
                // For incorrect predictions: y * h(x) < 0 (different signs)
                T indicator = NumOps.Compare(yi, pred) == 0
                    ? NumOps.Negate(NumOps.One)
                    : NumOps.One;
                T exponent = NumOps.Multiply(alpha, indicator);
                sampleWeights[i] = NumOps.Multiply(sampleWeights[i], NumOps.Exp(exponent));
                sumWeights = NumOps.Add(sumWeights, sampleWeights[i]);
            }

            // Normalize weights
            for (int i = 0; i < n; i++)
            {
                sampleWeights[i] = NumOps.Divide(sampleWeights[i], sumWeights);
            }
        }

        // Aggregate feature importances
        AggregateFeatureImportances();
    }

    /// <summary>
    /// Samples data with replacement based on weights.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) SampleWithWeights(Matrix<T> x, Vector<T> y, Vector<T> weights)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }

        int n = x.Rows;
        var indices = new int[n];

        // Build cumulative distribution
        var cumWeights = new double[n];
        cumWeights[0] = NumOps.ToDouble(weights[0]);
        for (int i = 1; i < n; i++)
        {
            cumWeights[i] = cumWeights[i - 1] + NumOps.ToDouble(weights[i]);
        }

        // Sample indices
        for (int i = 0; i < n; i++)
        {
            double r = _random.NextDouble();
            int idx = Array.BinarySearch(cumWeights, r);
            if (idx < 0) idx = ~idx;
            if (idx >= n) idx = n - 1;
            indices[i] = idx;
        }

        // Create sampled data
        var xSample = new Matrix<T>(n, x.Columns);
        var ySample = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            int srcIdx = indices[i];
            for (int j = 0; j < x.Columns; j++)
            {
                xSample[i, j] = x[srcIdx, j];
            }
            ySample[i] = y[srcIdx];
        }

        return (xSample, ySample);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (Estimators.Count == 0 || _estimatorWeights is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // For each sample, compute weighted votes for each class
            var classVotes = new Dictionary<int, T>();
            for (int c = 0; c < NumClasses; c++)
            {
                classVotes[c] = NumOps.Zero;
            }

            // Accumulate weighted votes from each estimator
            for (int m = 0; m < Estimators.Count; m++)
            {
                var estimator = Estimators[m];
                var sample = new Matrix<T>(1, input.Columns);
                for (int j = 0; j < input.Columns; j++)
                {
                    sample[0, j] = input[i, j];
                }

                var pred = estimator.Predict(sample);
                T weight = _estimatorWeights[m];

                // Find which class was predicted
                for (int c = 0; c < NumClasses; c++)
                {
                    if (NumOps.Compare(pred[0], ClassLabels[c]) == 0)
                    {
                        classVotes[c] = NumOps.Add(classVotes[c], weight);
                        break;
                    }
                }
            }

            // Find class with maximum vote
            int bestClass = 0;
            T bestVote = classVotes[0];
            for (int c = 1; c < NumClasses; c++)
            {
                if (NumOps.Compare(classVotes[c], bestVote) > 0)
                {
                    bestVote = classVotes[c];
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
        if (Estimators.Count == 0 || _estimatorWeights is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            // Compute weighted votes for each class
            var classVotes = new T[NumClasses];
            for (int c = 0; c < NumClasses; c++)
            {
                classVotes[c] = NumOps.Zero;
            }

            T totalWeight = NumOps.Zero;

            // Accumulate weighted votes
            for (int m = 0; m < Estimators.Count; m++)
            {
                var estimator = Estimators[m];
                var sample = new Matrix<T>(1, input.Columns);
                for (int j = 0; j < input.Columns; j++)
                {
                    sample[0, j] = input[i, j];
                }

                var pred = estimator.Predict(sample);
                T weight = _estimatorWeights[m];
                totalWeight = NumOps.Add(totalWeight, NumOps.Abs(weight));

                for (int c = 0; c < NumClasses; c++)
                {
                    if (NumOps.Compare(pred[0], ClassLabels[c]) == 0)
                    {
                        classVotes[c] = NumOps.Add(classVotes[c], weight);
                        break;
                    }
                }
            }

            // Normalize to get probabilities
            for (int c = 0; c < NumClasses; c++)
            {
                if (NumOps.Compare(totalWeight, NumOps.Zero) > 0)
                {
                    probabilities[i, c] = NumOps.Divide(classVotes[c], totalWeight);
                    // Ensure non-negative
                    if (NumOps.Compare(probabilities[i, c], NumOps.Zero) < 0)
                    {
                        probabilities[i, c] = NumOps.Zero;
                    }
                }
                else
                {
                    probabilities[i, c] = NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses));
                }
            }
        }

        return probabilities;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new AdaBoostClassifier<T>(new AdaBoostClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            LearningRate = Options.LearningRate,
            Algorithm = Options.Algorithm,
            RandomState = Options.RandomState
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new AdaBoostClassifier<T>(new AdaBoostClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            LearningRate = Options.LearningRate,
            Algorithm = Options.Algorithm,
            RandomState = Options.RandomState
        });

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

        if (_estimatorWeights is not null)
        {
            clone._estimatorWeights = new Vector<T>(_estimatorWeights.Length);
            for (int i = 0; i < _estimatorWeights.Length; i++)
            {
                clone._estimatorWeights[i] = _estimatorWeights[i];
            }
        }

        if (FeatureImportances is not null)
        {
            clone.FeatureImportances = new Vector<T>(FeatureImportances.Length);
            for (int i = 0; i < FeatureImportances.Length; i++)
            {
                clone.FeatureImportances[i] = FeatureImportances[i];
            }
        }

        // Clone all estimators
        foreach (var estimator in Estimators)
        {
            if (estimator is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                clone.Estimators.Add((IClassifier<T>)fullModel.Clone());
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NEstimators"] = Options.NEstimators;
        metadata.AdditionalInfo["LearningRate"] = Options.LearningRate;
        metadata.AdditionalInfo["Algorithm"] = Options.Algorithm.ToString();
        metadata.AdditionalInfo["ActualEstimators"] = Estimators.Count;
        return metadata;
    }
}
