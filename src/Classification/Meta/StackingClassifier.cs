using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Stacking classifier that uses predictions from base classifiers as features for a meta-classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stacking trains multiple base classifiers and then uses their predictions
/// as features to train a final meta-classifier.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Stacking is a sophisticated ensemble method:
///
/// 1. Train multiple base classifiers
/// 2. Get predictions from each on training data (using cross-validation)
/// 3. Use these predictions as features for a "meta" classifier
/// 4. Train the meta-classifier on these stacked predictions
///
/// For prediction:
/// 1. Get predictions from all base classifiers
/// 2. Stack them as features
/// 3. Feed to meta-classifier for final prediction
///
/// Benefits:
/// - Can combine very different types of classifiers
/// - Often achieves better accuracy than individual classifiers
/// - The meta-classifier learns which base classifiers to trust
///
/// Considerations:
/// - More complex to implement
/// - Risk of overfitting if not using cross-validation
/// - Computationally expensive
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create stacking classifier with base classifiers and meta-learner
/// var options = new StackingClassifierOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train base classifiers and meta-classifier on stacked predictions
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new StackingClassifier&lt;double&gt;())
///     .Build(features, labels);
///
/// // Predict using meta-classifier on base classifier outputs
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Stacked Generalization", "https://doi.org/10.1016/S0893-6080(05)80023-1", Year = 1992, Authors = "David H. Wolpert")]
public partial class StackingClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes estimators.
    /// </summary>
    public StackingClassifier()
        : this(
            new IClassifier<T>[] { new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>() },
            () => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// Gets the stacking-specific options.
    /// </summary>
    protected new StackingClassifierOptions<T> Options => (StackingClassifierOptions<T>)base.Options;

    /// <summary>
    /// The base estimators.
    /// </summary>
    private List<IClassifier<T>> _estimators = new();

    /// <summary>
    /// The meta-classifier (final estimator).
    /// </summary>
    private IClassifier<T>? _finalEstimator;

    /// <summary>
    /// Factory for creating final estimator.
    /// </summary>
    private readonly Func<IClassifier<T>>? _finalEstimatorFactory;

    /// <summary>
    /// Initializes a new instance of the StackingClassifier class.
    /// </summary>
    /// <param name="estimators">List of base classifiers.</param>
    /// <param name="finalEstimator">The meta-classifier for final predictions.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public StackingClassifier(
        IEnumerable<IClassifier<T>> estimators,
        Func<IClassifier<T>> finalEstimator,
        StackingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new StackingClassifierOptions<T>(), null, regularization)
    {
        _estimators = estimators.ToList();
        _finalEstimatorFactory = finalEstimator;
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the stacking classifier.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        if (_estimators is null || _estimators.Count == 0)
        {
            throw new InvalidOperationException("No estimators provided for stacking.");
        }

        if (_finalEstimatorFactory is null)
        {
            throw new InvalidOperationException("Final estimator factory is not set.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Meta-features: numEstimators * (UseProbabilities ? NumClasses : 1) columns, plus the original
        // features when passthrough is enabled.
        // Cross-validation generates out-of-fold predictions; the simple approach trains on all data and
        // predicts on all data (prone to overfitting).
        var metaFeatures = Options.CrossValidationFolds > 1
            ? CreateCrossValidatedMetaFeatures(x, y)
            : CreateSimpleMetaFeatures(x, y);

        // Train base estimators on full training data (for prediction time)
        foreach (var estimator in _estimators)
        {
            estimator.Train(x, y);
        }

        // Train final estimator on meta-features
        _finalEstimator = _finalEstimatorFactory();
        _finalEstimator.Train(metaFeatures, y);
    }

    /// <summary>
    /// Creates meta-features using cross-validation.
    /// </summary>
    private Matrix<T> CreateCrossValidatedMetaFeatures(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int numFolds = Options.CrossValidationFolds;
        int numEstimators = _estimators.Count;
        int metaFeaturesPerEstimator = Options.UseProbabilities ? NumClasses : 1;

        var random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Create fold assignments
        var foldAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            foldAssignments[i] = i % numFolds;
        }

        // Shuffle fold assignments
        for (int i = n - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (foldAssignments[i], foldAssignments[j]) = (foldAssignments[j], foldAssignments[i]);
        }

        // Out-of-fold predictions; rows are scattered back to their original positions
        var engine = Engine;
        var outOfFold = new Tensor<T>(new[] { n, numEstimators * metaFeaturesPerEstimator });

        // Process each fold
        for (int fold = 0; fold < numFolds; fold++)
        {
            // Create train and test splits
            var trainRows = new List<int>(n);
            var testRows = new List<int>(n / Math.Max(1, numFolds) + 1);
            for (int i = 0; i < n; i++)
            {
                if (foldAssignments[i] == fold)
                    testRows.Add(i);
                else
                    trainRows.Add(i);
            }

            var testIndices = testRows.ToArray();
            var (xTrainAll, yTrain) = SelectSamples(x, y, trainRows.ToArray());
            var xTrain = LeadingColumns(xTrainAll, NumFeatures);
            var xTest = LeadingColumns(SelectSamples(x, y, testIndices).X, NumFeatures);

            // Train each estimator and get predictions for test fold
            var foldEstimators = new IClassifier<T>[numEstimators];
            for (int e = 0; e < numEstimators; e++)
            {
                // Clone and train estimator on training data
                if (_estimators[e] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
                {
                    foldEstimators[e] = (IClassifier<T>)fullModel.Clone();
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Estimator at index {e} does not implement IFullModel and cannot be cloned. " +
                        $"Cross-validation requires clonable estimators to avoid shared state corruption. " +
                        $"Set CrossValidationFolds=1 or use clonable estimators.");
                }
                foldEstimators[e].Train(xTrain, yTrain);
            }

            // Get predictions for test fold
            var foldFeatures = BuildMetaFeatures(foldEstimators, xTest, includePassthrough: false);
            if (testIndices.Length > 0 && foldFeatures.Columns > 0)
            {
                outOfFold = engine.TensorScatterAdd(outOfFold,
                    new Tensor<int>(testIndices, new[] { testIndices.Length }), Tensor<T>.FromMatrix(foldFeatures), 0);
            }
        }

        // Add passthrough features if enabled
        return Options.Passthrough
            ? ConcatenateColumns(new[] { outOfFold, Tensor<T>.FromMatrix(LeadingColumns(x, NumFeatures)) }, n)
            : outOfFold.ToMatrix();
    }

    /// <summary>
    /// Creates meta-features without cross-validation.
    /// </summary>
    private Matrix<T> CreateSimpleMetaFeatures(Matrix<T> x, Vector<T> y)
    {
        // Train each estimator and get predictions
        foreach (var estimator in _estimators)
        {
            estimator.Train(x, y);
        }

        // Add passthrough features
        return BuildMetaFeatures(_estimators, x, Options.Passthrough);
    }

    /// <summary>
    /// Concatenates [rows, k] blocks column-wise (an empty list gives a [rows, 0] matrix).
    /// </summary>
    private Matrix<T> ConcatenateColumns(IReadOnlyList<Tensor<T>> blocks, int rows)
    {
        if (blocks.Count == 0)
        {
            return new Matrix<T>(rows, 0);
        }

        return blocks.Count == 1 ? blocks[0].ToMatrix() : Engine.TensorConcatenate(blocks.ToArray(), 1).ToMatrix();
    }

    /// <summary>
    /// Builds the meta-feature matrix from trained estimators: per estimator NumClasses probability
    /// columns (or its prediction followed by zero columns), then optionally the original features.
    /// </summary>
    private Matrix<T> BuildMetaFeatures(IReadOnlyList<IClassifier<T>> estimators, Matrix<T> input, bool includePassthrough)
    {
        int n = input.Rows;
        int metaFeaturesPerEstimator = Options.UseProbabilities ? NumClasses : 1;
        var blocks = new List<Tensor<T>>(estimators.Count + 1);

        // Get predictions from each estimator
        foreach (var estimator in estimators)
        {
            if (Options.UseProbabilities && estimator is IProbabilisticClassifier<T> probClassifier)
            {
                blocks.Add(Tensor<T>.FromMatrix(LeadingColumns(probClassifier.PredictProbabilities(input), NumClasses)));
            }
            else
            {
                blocks.Add(estimator.Predict(input).AsTensor().Reshape(n, 1));
                if (metaFeaturesPerEstimator > 1)
                {
                    blocks.Add(new Tensor<T>(new[] { n, metaFeaturesPerEstimator - 1 }));
                }
            }
        }

        if (includePassthrough)
        {
            blocks.Add(Tensor<T>.FromMatrix(LeadingColumns(input, NumFeatures)));
        }

        return ConcatenateColumns(blocks, n);
    }

    /// <summary>
    /// Creates meta-features for prediction.
    /// </summary>
    private Matrix<T> CreatePredictionMetaFeatures(Matrix<T> input)
    {
        return BuildMetaFeatures(_estimators, input, Options.Passthrough);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || _finalEstimator is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var metaFeatures = CreatePredictionMetaFeatures(input);
        return _finalEstimator.Predict(metaFeatures);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null || _finalEstimator is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var metaFeatures = CreatePredictionMetaFeatures(input);

        if (_finalEstimator is IProbabilisticClassifier<T> probClassifier)
        {
            return probClassifier.PredictProbabilities(metaFeatures);
        }

        // Fall back to hard predictions
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been fitted.");
        }

        return ClassMembership(_finalEstimator.Predict(metaFeatures)).ToMatrix();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        // log(max(p, 1e-15))
        return base.PredictLogProbabilities(input);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NumEstimators"] = _estimators?.Count ?? 0;
        metadata.AdditionalInfo["CrossValidationFolds"] = Options.CrossValidationFolds;
        metadata.AdditionalInfo["UseProbabilities"] = Options.UseProbabilities;
        metadata.AdditionalInfo["Passthrough"] = Options.Passthrough;
        return metadata;
    }
}

/// <summary>
/// Configuration options for Stacking classifier.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class StackingClassifierOptions<T> : MetaClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of cross-validation folds.
    /// </summary>
    /// <value>Number of folds. Use 1 for no cross-validation. Default is 5.</value>
    public int CrossValidationFolds { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use probability predictions as meta-features.
    /// </summary>
    /// <value>True to use probabilities, false to use class predictions. Default is true.</value>
    public bool UseProbabilities { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to pass original features to the final estimator.
    /// </summary>
    /// <value>True to include original features. Default is false.</value>
    public bool Passthrough { get; set; } = false;

}
