using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for bagging classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new BaggingClassifier&lt;double&gt;(() => new GaussianNaiveBayes&lt;double&gt;()));
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
[ResearchPaper("Bagging Predictors", "https://doi.org/10.1007/BF00058655", Year = 1996, Authors = "Leo Breiman")]
public partial class BaggingClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator.
    /// </summary>
    public BaggingClassifier()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

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
    private Random _random = Tensors.Helpers.RandomHelper.CreateSecureRandom();

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
        var random = _random ?? throw new InvalidOperationException(
            $"{GetType().Name}: Random not initialized. Call Fit() first.");

        if (sampleSize > x.Rows && !Options.Bootstrap)
        {
            throw new ArgumentException(
                $"Sample size ({sampleSize}) exceeds data size ({x.Rows}) with Bootstrap=false. " +
                "Either enable Bootstrap or reduce MaxSamples.",
                nameof(sampleSize));
        }

        var indices = new int[sampleSize];

        if (Options.Bootstrap)
        {
            // Sample with replacement
            for (int i = 0; i < sampleSize; i++)
            {
                indices[i] = random.Next(x.Rows);
            }
        }
        else
        {
            // Sample without replacement using Fisher-Yates partial shuffle (O(sampleSize), unbiased)
            var pool = Enumerable.Range(0, x.Rows).ToArray();
            for (int k = 0; k < sampleSize; k++)
            {
                int swapIdx = k + random.Next(pool.Length - k);
                (pool[k], pool[swapIdx]) = (pool[swapIdx], pool[k]);
            }
            Array.Copy(pool, 0, indices, 0, sampleSize);
        }

        var (xSample, ySample) = SelectSamples(x, y, indices);
        return (x.Columns == NumFeatures ? xSample : SelectColumns(xSample, Enumerable.Range(0, NumFeatures).ToArray()), ySample);
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
            .OrderBy(_ => _random.Next())
            .Take(numFeatures)
            .OrderBy(i => i)
            .ToArray();

        return (SelectColumns(x, featureIndices), featureIndices);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || ClassLabels is null || _featureIndicesPerEstimator is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // Count votes from each estimator, using the same feature indices used during training
        // (in int: a sum of T one-hot rows stops growing at 2048 for Half)
        var votes = new Tensor<int>(new[] { input.Rows, NumClasses });
        for (int e = 0; e < _estimators.Length; e++)
        {
            var pred = _estimators[e].Predict(SelectColumns(input, _featureIndicesPerEstimator[e]));
            votes = AddClassVotes(votes, pred);
        }

        // Class with most votes (first class wins ties)
        return MajorityVoteLabels(votes);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null || _featureIndicesPerEstimator is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var engine = Engine;
        var probs = new Tensor<T>(new[] { input.Rows, NumClasses });

        // Average probabilities from all estimators
        for (int e = 0; e < _estimators.Length; e++)
        {
            // Extract features using the same indices used during training for this estimator
            var filteredInput = SelectColumns(input, _featureIndicesPerEstimator[e]);

            Tensor<T> estProbs;
            if (_estimators[e] is IProbabilisticClassifier<T> probClassifier)
            {
                var rawProbs = probClassifier.PredictProbabilities(filteredInput);

                // Align estimator probabilities to ensemble label space
                estProbs = Tensor<T>.FromMatrix(rawProbs.Columns == NumClasses
                    ? rawProbs
                    : AlignProbabilities(rawProbs, _estimators[e].ClassLabels));
            }
            else
            {
                // Use hard predictions
                estProbs = ClassMembership(_estimators[e].Predict(filteredInput));
            }

            probs = engine.TensorAdd(probs, estProbs);
        }

        return engine.TensorDivideScalar(probs, NumOps.FromDouble(_estimators.Length)).ToMatrix();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        // log(max(p, 1e-15)) of the averaged probabilities
        return base.PredictLogProbabilities(input);
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
