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
/// Random Forest classifier that combines multiple decision trees trained on random subsets.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Random Forest is a meta estimator that fits a number of decision tree classifiers on
/// various sub-samples of the dataset and uses averaging to improve predictive accuracy
/// and control overfitting.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Random Forest is one of the most popular and powerful machine learning algorithms.
/// It works by creating a "forest" of decision trees, where each tree:
///
/// 1. Is trained on a random subset of the data (bootstrap sampling)
/// 2. Considers only a random subset of features at each split
/// 3. Votes on the final prediction
///
/// This randomness makes the trees different from each other, and when combined,
/// they create a robust classifier that:
/// - Is resistant to overfitting
/// - Handles both numerical and categorical features
/// - Works well with default parameters
/// - Provides feature importance scores
///
/// Example: Predicting customer churn
/// - Tree 1 might focus on usage patterns and account age
/// - Tree 2 might focus on customer service calls and billing
/// - Tree 3 might focus on contract type and payment history
/// - Together, they give a more reliable prediction than any single tree
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create random forest classifier with bootstrap aggregation
/// var options = new RandomForestClassifierOptions&lt;double&gt;();
/// var classifier = new RandomForestClassifier&lt;double&gt;(options);
///
/// // Prepare training data
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1.0, 1.1,  1.2, 0.9,  0.8, 1.0,
///     5.0, 5.1,  5.2, 4.9,  4.8, 5.0 });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train multiple trees on random subsets of data and features
/// classifier.Train(features, labels);
///
/// // Predict using majority vote across all trees in the forest
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 1.1, 1.0 });
/// var prediction = classifier.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Random Forests", "https://doi.org/10.1023/A:1010933404324", Year = 2001, Authors = "Leo Breiman")]
public partial class RandomForestClassifier<T> : EnsembleClassifierBase<T>, ITreeBasedClassifier<T>
{
    /// <summary>
    /// Gets the Random Forest specific options.
    /// </summary>
    protected new RandomForestClassifierOptions<T> Options => (RandomForestClassifierOptions<T>)base.Options;

    /// <summary>
    /// Random number generator for bootstrap sampling and feature selection.
    /// </summary>
    private Random? _random;

    /// <summary>
    /// Out-of-bag accuracy score (only available if OobScore is enabled).
    /// </summary>
    public double OobScore_ { get; private set; }

    /// <summary>
    /// Out-of-bag sample indices for each tree.
    /// Used for proper OOB score calculation.
    /// </summary>
    private readonly List<HashSet<int>> _oobIndicesPerTree = new();

    /// <inheritdoc/>
    public int MaxDepth => Options.MaxDepth ?? CalculateMaxDepth();

    /// <inheritdoc/>
    public int LeafCount => CalculateTotalLeafCount();

    /// <inheritdoc/>
    public int NodeCount => CalculateTotalNodeCount();

    /// <summary>
    /// Initializes a new instance of the RandomForestClassifier class.
    /// </summary>
    /// <param name="options">Configuration options for the Random Forest.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public RandomForestClassifier(RandomForestClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new RandomForestClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the Random Forest on the provided data.
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

        // Initialize random number generator
        _random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators and OOB indices
        Estimators.Clear();
        _oobIndicesPerTree.Clear();

        // Calculate max features to consider at each split
        int maxFeatures = CalculateMaxFeatures();

        // Train each tree
        // Note: In a production implementation, this could be parallelized
        for (int i = 0; i < Options.NEstimators; i++)
        {
            // Create bootstrap sample
            var (bootstrapIndices, oobIndices) = CreateBootstrapSample(x.Rows);

            // Create the tree with appropriate options
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

            // Create bootstrap sample matrices
            var (xBootstrap, yBootstrap) = CreateBootstrapData(x, y, bootstrapIndices);

            // Train the tree
            tree.Train(xBootstrap, yBootstrap);

            Estimators.Add(tree);

            // Store OOB indices for this tree (for proper OOB score calculation)
            _oobIndicesPerTree.Add(new HashSet<int>(oobIndices));
        }

        // Aggregate feature importances
        AggregateFeatureImportances();

        // Calculate OOB score if requested
        if (Options.OobScore && Options.Bootstrap)
        {
            CalculateOobScore(x, y);
        }
    }

    /// <summary>
    /// Creates a bootstrap sample of indices.
    /// </summary>
    private (List<int> bootstrap, List<int> oob) CreateBootstrapSample(int nSamples)
    {
        var bootstrapIndices = new List<int>();
        var oobSet = new HashSet<int>(Enumerable.Range(0, nSamples));

        if (Options.Bootstrap)
        {
            // Sample with replacement
            for (int i = 0; i < nSamples; i++)
            {
                int idx = (_random ?? throw new InvalidOperationException("_random has not been initialized.")).Next(nSamples);
                bootstrapIndices.Add(idx);
                oobSet.Remove(idx);
            }
        }
        else
        {
            // Use all samples
            bootstrapIndices.AddRange(Enumerable.Range(0, nSamples));
            oobSet.Clear();
        }

        return (bootstrapIndices, oobSet.ToList());
    }

    /// <summary>
    /// Creates bootstrap sample data matrices.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) CreateBootstrapData(Matrix<T> x, Vector<T> y, List<int> indices)
    {
        var xBootstrap = new Matrix<T>(indices.Count, x.Columns);
        var yBootstrap = new Vector<T>(indices.Count);

        for (int i = 0; i < indices.Count; i++)
        {
            int srcIdx = indices[i];
            for (int j = 0; j < x.Columns; j++)
            {
                xBootstrap[i, j] = x[srcIdx, j];
            }
            yBootstrap[i] = y[srcIdx];
        }

        return (xBootstrap, yBootstrap);
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

        // Exhaustive over the enum. The string version's `_ => sqrt` fallback meant an unrecognized
        // value silently trained a different model than the caller requested.
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
    /// Calculates the out-of-bag score.
    /// </summary>
    private void CalculateOobScore(Matrix<T> x, Vector<T> y)
    {
        // For each sample, aggregate predictions only from trees where it was OOB
        int nSamples = x.Rows;
        int correct = 0;
        int oobSampleCount = 0;

        for (int sampleIdx = 0; sampleIdx < nSamples; sampleIdx++)
        {
            // Collect predictions from trees where this sample was OOB
            var voteCounts = new Dictionary<double, int>();
            int treesVoted = 0;

            for (int treeIdx = 0; treeIdx < Estimators.Count; treeIdx++)
            {
                // Check if this sample was OOB for this tree
                if (_oobIndicesPerTree[treeIdx].Contains(sampleIdx))
                {
                    // Get prediction for this single sample
                    var sample = new Matrix<T>(1, x.Columns);
                    for (int j = 0; j < x.Columns; j++)
                    {
                        sample[0, j] = x[sampleIdx, j];
                    }

                    var pred = Estimators[treeIdx].Predict(sample);
                    double predValue = NumOps.ToDouble(pred[0]);

                    if (!voteCounts.TryGetValue(predValue, out int count))
                    {
                        count = 0;
                    }
                    voteCounts[predValue] = count + 1;
                    treesVoted++;
                }
            }

            // Only count samples that were OOB for at least one tree
            if (treesVoted > 0)
            {
                oobSampleCount++;

                // Find majority vote
                double majorityClass = voteCounts.OrderByDescending(kv => kv.Value).First().Key;
                double actualClass = NumOps.ToDouble(y[sampleIdx]);

                if (Math.Abs(majorityClass - actualClass) < 1e-10)
                {
                    correct++;
                }
            }
        }

        OobScore_ = oobSampleCount > 0 ? (double)correct / oobSampleCount : 0.0;
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
    /// Calculates the total number of leaf nodes across all trees.
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
    /// Calculates the total number of nodes across all trees.
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
        if (Options.OobScore && Options.Bootstrap)
        {
            metadata.AdditionalInfo["OobScore"] = OobScore_;
        }
        metadata.AdditionalInfo["TotalNodes"] = NodeCount;
        metadata.AdditionalInfo["TotalLeaves"] = LeafCount;
        return metadata;
    }
}
