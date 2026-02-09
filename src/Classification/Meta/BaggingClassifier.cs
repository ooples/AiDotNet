using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Bagging (Bootstrap Aggregating) classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bagging trains multiple classifiers on bootstrap samples of the training data
/// and combines their predictions through voting.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Bagging is a technique to reduce overfitting:
///
/// 1. Create N bootstrap samples (random samples with replacement)
/// 2. Train one classifier on each sample
/// 3. For prediction, each classifier votes
/// 4. Final prediction is the majority vote
///
/// Benefits:
/// - Reduces variance (less overfitting)
/// - Works well with high-variance classifiers like decision trees
/// - Easily parallelizable
///
/// When to use:
/// - When your base classifier tends to overfit
/// - When you want more robust predictions
/// - As a simpler alternative to boosting
/// </para>
/// </remarks>
public class BaggingClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Gets the bagging-specific options.
    /// </summary>
    protected new BaggingClassifierOptions<T> Options => (BaggingClassifierOptions<T>)base.Options;

    /// <summary>
    /// The ensemble of classifiers.
    /// </summary>
    private IClassifier<T>[]? _estimators;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Feature indices selected for each estimator.
    /// Used to ensure prediction uses the same features as training.
    /// </summary>
    private int[][]? _featureIndicesPerEstimator;

    /// <summary>
    /// Initializes a new instance of the BaggingClassifier class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public BaggingClassifier(
        Func<IClassifier<T>> estimatorFactory,
        BaggingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new BaggingClassifierOptions<T>(), estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.BaggingClassifier;

    /// <summary>
    /// Trains the Bagging classifier on the provided data.
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

        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        int n = x.Rows;
        int numEstimators = Options.NumEstimators;

        _estimators = new IClassifier<T>[numEstimators];
        _featureIndicesPerEstimator = new int[numEstimators][];

        // Train each estimator on a bootstrap sample
        for (int e = 0; e < numEstimators; e++)
        {
            // Create bootstrap sample
            int sampleSize = (int)(n * Options.MaxSamples);
            var (xSample, ySample) = CreateBootstrapSample(x, y, sampleSize);

            // Optionally sample features randomly
            Matrix<T> xFinal;
            if (Options.MaxFeatures < 1.0)
            {
                int numFeaturesToSample = (int)(NumFeatures * Options.MaxFeatures);
                var (sampledData, featureIndices) = SampleFeaturesRandomly(xSample, numFeaturesToSample);
                xFinal = sampledData;
                _featureIndicesPerEstimator[e] = featureIndices;
            }
            else
            {
                xFinal = xSample;
                _featureIndicesPerEstimator[e] = Enumerable.Range(0, NumFeatures).ToArray();
            }

            // Train classifier
            _estimators[e] = CreateBaseEstimator();
            _estimators[e].Train(xFinal, ySample);
        }
    }

    /// <summary>
    /// Creates a bootstrap sample from the data.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) CreateBootstrapSample(Matrix<T> x, Vector<T> y, int sampleSize)
    {
        var xSample = new Matrix<T>(sampleSize, NumFeatures);
        var ySample = new Vector<T>(sampleSize);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx = _random!.Next(x.Rows);

            for (int j = 0; j < NumFeatures; j++)
            {
                xSample[i, j] = x[idx, j];
            }
            ySample[i] = y[idx];
        }

        return (xSample, ySample);
    }

    /// <summary>
    /// Randomly samples a subset of features.
    /// </summary>
    /// <returns>A tuple of the sampled data matrix and the selected feature indices.</returns>
    private (Matrix<T> data, int[] featureIndices) SampleFeaturesRandomly(Matrix<T> x, int numFeatures)
    {
        // Randomly select feature indices without replacement
        // Sort the selected indices to maintain consistent ordering and avoid bias
        var featureIndices = Enumerable.Range(0, x.Columns)
            .OrderBy(_ => _random!.Next())
            .Take(numFeatures)
            .OrderBy(i => i)
            .ToArray();

        var xSampled = new Matrix<T>(x.Rows, numFeatures);

        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                xSampled[i, j] = x[i, featureIndices[j]];
            }
        }

        return (xSampled, featureIndices);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || ClassLabels is null || _featureIndicesPerEstimator is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Count votes from each estimator
            var votes = new int[NumClasses];

            for (int e = 0; e < _estimators.Length; e++)
            {
                // Extract single sample using the same feature indices used during training
                var featureIndices = _featureIndicesPerEstimator[e];
                var sample = new Matrix<T>(1, featureIndices.Length);
                for (int j = 0; j < featureIndices.Length; j++)
                {
                    sample[0, j] = input[i, featureIndices[j]];
                }

                var pred = _estimators[e].Predict(sample);

                // Find which class this prediction corresponds to
                for (int c = 0; c < NumClasses; c++)
                {
                    if (NumOps.Compare(pred[0], ClassLabels[c]) == 0)
                    {
                        votes[c]++;
                        break;
                    }
                }
            }

            // Find class with most votes
            int bestClass = 0;
            int maxVotes = votes[0];
            for (int c = 1; c < NumClasses; c++)
            {
                if (votes[c] > maxVotes)
                {
                    maxVotes = votes[c];
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
        if (_estimators is null || _featureIndicesPerEstimator is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probs = new Matrix<T>(input.Rows, NumClasses);

        // Average probabilities from all estimators
        for (int e = 0; e < _estimators.Length; e++)
        {
            // Extract features using the same indices used during training for this estimator
            var featureIndices = _featureIndicesPerEstimator[e];
            var filteredInput = new Matrix<T>(input.Rows, featureIndices.Length);
            for (int i = 0; i < input.Rows; i++)
            {
                for (int j = 0; j < featureIndices.Length; j++)
                {
                    filteredInput[i, j] = input[i, featureIndices[j]];
                }
            }

            Matrix<T> estProbs;

            if (_estimators[e] is IProbabilisticClassifier<T> probClassifier)
            {
                estProbs = probClassifier.PredictProbabilities(filteredInput);
            }
            else
            {
                // Use hard predictions
                var preds = _estimators[e].Predict(filteredInput);
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

            // Accumulate
            for (int i = 0; i < input.Rows; i++)
            {
                for (int c = 0; c < NumClasses; c++)
                {
                    probs[i, c] = NumOps.Add(probs[i, c], estProbs[i, c]);
                }
            }
        }

        // Average
        T numEstimatorsT = NumOps.FromDouble(_estimators.Length);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                probs[i, c] = NumOps.Divide(probs[i, c], numEstimatorsT);
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
        if (EstimatorFactory is null)
        {
            throw new InvalidOperationException("Estimator factory is not set.");
        }

        return new BaggingClassifier<T>(EstimatorFactory, new BaggingClassifierOptions<T>
        {
            NumEstimators = Options.NumEstimators,
            MaxSamples = Options.MaxSamples,
            MaxFeatures = Options.MaxFeatures,
            Bootstrap = Options.Bootstrap,
            Seed = Options.Seed
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (BaggingClassifier<T>)CreateNewInstance();

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
            clone._estimators = new IClassifier<T>[_estimators.Length];
            for (int e = 0; e < _estimators.Length; e++)
            {
                if (_estimators[e] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    clone._estimators[e] = (IClassifier<T>)fullModel.Clone();
                }
                else
                {
                    clone._estimators[e] = _estimators[e];
                }
            }
        }

        if (_featureIndicesPerEstimator is not null)
        {
            clone._featureIndicesPerEstimator = new int[_featureIndicesPerEstimator.Length][];
            for (int e = 0; e < _featureIndicesPerEstimator.Length; e++)
            {
                clone._featureIndicesPerEstimator[e] = (int[])_featureIndicesPerEstimator[e].Clone();
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NumEstimators"] = Options.NumEstimators;
        metadata.AdditionalInfo["MaxSamples"] = Options.MaxSamples;
        metadata.AdditionalInfo["MaxFeatures"] = Options.MaxFeatures;
        return metadata;
    }
}

/// <summary>
/// Configuration options for Bagging classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class BaggingClassifierOptions<T> : MetaClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of base estimators.
    /// </summary>
    /// <value>Number of estimators. Default is 10.</value>
    public int NumEstimators { get; set; } = 10;

    /// <summary>
    /// Gets or sets the fraction of samples to draw for each estimator.
    /// </summary>
    /// <value>Fraction of samples (0.0 to 1.0). Default is 1.0.</value>
    public double MaxSamples { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the fraction of features to use for each estimator.
    /// </summary>
    /// <value>Fraction of features (0.0 to 1.0). Default is 1.0.</value>
    public double MaxFeatures { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to sample with replacement.
    /// </summary>
    /// <value>True for bootstrap sampling. Default is true.</value>
    public bool Bootstrap { get; set; } = true;

}
