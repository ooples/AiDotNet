using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Classification;
using AiDotNet.Enums;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Ensemble;

/// <summary>
/// Extra Trees (Extremely Randomized Trees) classifier.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Extra Trees is an ensemble method that builds multiple decision trees with
/// extra randomization. Unlike Random Forest which finds the best split among
/// random features, Extra Trees picks random splits, leading to more diversity.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Extra Trees takes randomization even further than Random Forest:
///
/// Random Forest: "Look at random features, pick the BEST split"
/// Extra Trees: "Look at random features, pick a RANDOM split"
///
/// Benefits of Extra Trees:
/// - Faster training (no need to find optimal splits)
/// - Often better generalization
/// - More robust to noise
///
/// When Extra Trees might be better:
/// - When you have noisy data
/// - When Random Forest overfits
/// - When you need faster training
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Use AiModelBuilder facade for Extra Trees classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ExtraTreesClassifier&lt;double&gt;(
///         new ExtraTreesClassifierOptions&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Extremely Randomized Trees", "https://doi.org/10.1007/s10994-006-6226-1", Year = 2006, Authors = "Pierre Geurts, Damien Ernst, Louis Wehenkel")]
public class ExtraTreesClassifier<T> : EnsembleClassifierBase<T>, ITreeBasedClassifier<T>
{
    /// <summary>
    /// Gets the Extra Trees specific options.
    /// </summary>
    protected new ExtraTreesClassifierOptions<T> Options => (ExtraTreesClassifierOptions<T>)base.Options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random? _random;

    /// <inheritdoc/>
    public int MaxDepth => Options.MaxDepth ?? CalculateMaxDepth();

    /// <inheritdoc/>
    public int LeafCount => CalculateTotalLeafCount();

    /// <inheritdoc/>
    public int NodeCount => CalculateTotalNodeCount();

    /// <summary>
    /// Initializes a new instance of the ExtraTreesClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for Extra Trees.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public ExtraTreesClassifier(ExtraTreesClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new ExtraTreesClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the Extra Trees classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        // A FEATURELESS MATRIX CANNOT TRAIN A TREE. With NumFeatures = 0 every rule in
        // CalculateMaxFeatures returns 0 or 1 -- Sqrt and Log2 of 0, and All -- so the forest was
        // built on a feature count no split can use, and the failure appeared far from its cause.
        if (x.Columns == 0)
        {
            throw new ArgumentException(
                "Training matrix has no feature columns; a decision tree cannot split on zero "
                + "features.", nameof(x));
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators
        Estimators.Clear();

        // Calculate max features to consider at each split
        int maxFeatures = CalculateMaxFeatures();

        // Train each tree
        for (int i = 0; i < Options.NEstimators; i++)
        {
            Matrix<T> xSample;
            Vector<T> ySample;

            if (Options.Bootstrap)
            {
                // Bootstrap sample
                (xSample, ySample) = CreateBootstrapSample(x, y);
            }
            else
            {
                // Use full dataset (default for Extra Trees)
                xSample = x;
                ySample = y;
            }

            // Create tree with extra randomization
            // Note: We use DecisionTreeClassifier but with random split selection
            // indicated by setting MaxFeatures
            var treeOptions = new DecisionTreeClassifierOptions<T>
            {
                MaxDepth = Options.MaxDepth,
                MinSamplesSplit = Options.MinSamplesSplit,
                MinSamplesLeaf = Options.MinSamplesLeaf,
                MaxFeatures = maxFeatures,
                Criterion = Options.Criterion,
                Seed = _random.Next(),
                MinImpurityDecrease = Options.MinImpurityDecrease
            };

            var tree = new DecisionTreeClassifier<T>(treeOptions);
            tree.Train(xSample, ySample);

            Estimators.Add(tree);
        }

        // Aggregate feature importances
        AggregateFeatureImportances();
    }

    /// <summary>
    /// Creates a bootstrap sample.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) CreateBootstrapSample(Matrix<T> x, Vector<T> y)
    {
        if (_random is null)
        {
            throw new InvalidOperationException("Random number generator not initialized.");
        }

        int n = x.Rows;
        var xSample = new Matrix<T>(n, x.Columns);
        var ySample = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            int idx = _random.Next(n);
            for (int j = 0; j < x.Columns; j++)
            {
                xSample[i, j] = x[idx, j];
            }
            ySample[i] = y[idx];
        }

        return (xSample, ySample);
    }

    /// <summary>
    /// Calculates the number of features to consider at each split.
    /// </summary>
    private int CalculateMaxFeatures()
    {
        // An explicit count wins over the rule.
        if (Options.MaxFeatureCount is int explicitCount)
        {
            if (explicitCount <= 0)
            {
                throw new InvalidOperationException(
                    $"{nameof(Options.MaxFeatureCount)} must be positive when set; got {explicitCount}.");
            }
            return Math.Min(explicitCount, NumFeatures);
        }

        // Exhaustive over the enum, with no catch-all fallback. The string version ended in
        // `_ => sqrt`, so an unrecognized value trained a different model than the caller asked for
        // and said nothing. An unhandled member now fails loudly instead.
        return Options.MaxFeatures switch
        {
            MaxFeatureSelection.Sqrt => (int)Math.Ceiling(Math.Sqrt(NumFeatures)),
            // Math.Max(1, ...): Log2(1) is 0, and a tree cannot split on zero features. The
            // explicit-count path above already rejects a non-positive value; the rule path
            // needs the same floor rather than passing 0 down to every tree.
            MaxFeatureSelection.Log2 => Math.Max(1, (int)Math.Ceiling(Math.Log(NumFeatures, 2))),
            MaxFeatureSelection.All => NumFeatures,
            _ => throw new InvalidOperationException(
                $"Unhandled {nameof(MaxFeatureSelection)} value '{Options.MaxFeatures}'.")
        };
    }

    /// <summary>
    /// Calculates the maximum depth across all trees.
    /// </summary>
    private int CalculateMaxDepth()
    {
        int maxDepth = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                maxDepth = Math.Max(maxDepth, tree.MaxDepth);
            }
        }
        return maxDepth;
    }

    /// <summary>
    /// Calculates the total number of leaf nodes.
    /// </summary>
    private int CalculateTotalLeafCount()
    {
        int total = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                total += tree.LeafCount;
            }
        }
        return total;
    }

    /// <summary>
    /// Calculates the total number of nodes.
    /// </summary>
    private int CalculateTotalNodeCount()
    {
        int total = 0;
        foreach (var estimator in Estimators)
        {
            if (estimator is ITreeBasedClassifier<T> tree)
            {
                total += tree.NodeCount;
            }
        }
        return total;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["NEstimators"] = Options.NEstimators;
        metadata.AdditionalInfo["MaxDepth"] = Options.MaxDepth?.ToString() ?? "unlimited";
        metadata.AdditionalInfo["MaxFeatures"] = Options.MaxFeatures;
        metadata.AdditionalInfo["Criterion"] = Options.Criterion.ToString();
        metadata.AdditionalInfo["Bootstrap"] = Options.Bootstrap;
        metadata.AdditionalInfo["TotalNodes"] = NodeCount;
        metadata.AdditionalInfo["TotalLeaves"] = LeafCount;
        return metadata;
    }
}
