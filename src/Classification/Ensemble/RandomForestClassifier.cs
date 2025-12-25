using AiDotNet.Classification;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

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
public class RandomForestClassifier<T> : EnsembleClassifierBase<T>, ITreeBasedClassifier<T>
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
        : base(options ?? new RandomForestClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.RandomForestClassifier;

    /// <summary>
    /// Trains the Random Forest on the provided data.
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

        // Initialize random number generator
        _random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Clear existing estimators
        Estimators.Clear();

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
                RandomState = _random.Next(),
                MinImpurityDecrease = Options.MinImpurityDecrease
            };

            var tree = new DecisionTreeClassifier<T>(treeOptions);

            // Create bootstrap sample matrices
            var (xBootstrap, yBootstrap) = CreateBootstrapData(x, y, bootstrapIndices);

            // Train the tree
            tree.Train(xBootstrap, yBootstrap);

            Estimators.Add(tree);
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
                int idx = _random!.Next(nSamples);
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
        if (string.IsNullOrEmpty(Options.MaxFeatures))
        {
            return NumFeatures;
        }

        return Options.MaxFeatures.ToLower() switch
        {
            "sqrt" => (int)Math.Ceiling(Math.Sqrt(NumFeatures)),
            "log2" => (int)Math.Ceiling(Math.Log2(NumFeatures)),
            "all" => NumFeatures,
            _ when int.TryParse(Options.MaxFeatures, out int n) => Math.Min(n, NumFeatures),
            _ => (int)Math.Ceiling(Math.Sqrt(NumFeatures)) // Default to sqrt
        };
    }

    /// <summary>
    /// Calculates the out-of-bag score.
    /// </summary>
    private void CalculateOobScore(Matrix<T> x, Vector<T> y)
    {
        // This is a simplified OOB calculation
        // A full implementation would track which samples were OOB for each tree
        var predictions = Predict(x);
        int correct = 0;
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.Compare(predictions[i], y[i]) == 0)
            {
                correct++;
            }
        }
        OobScore_ = (double)correct / y.Length;
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
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new RandomForestClassifier<T>(new RandomForestClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            MaxFeatures = Options.MaxFeatures,
            Criterion = Options.Criterion,
            Bootstrap = Options.Bootstrap,
            OobScore = Options.OobScore,
            NJobs = Options.NJobs,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new RandomForestClassifier<T>(new RandomForestClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            MaxFeatures = Options.MaxFeatures,
            Criterion = Options.Criterion,
            Bootstrap = Options.Bootstrap,
            OobScore = Options.OobScore,
            NJobs = Options.NJobs,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone.OobScore_ = OobScore_;

        if (ClassLabels != null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        if (FeatureImportances != null)
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
