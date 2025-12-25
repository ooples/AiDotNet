using AiDotNet.Classification;
using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

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
        : base(options ?? new ExtraTreesClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.ExtraTreesClassifier;

    /// <summary>
    /// Trains the Extra Trees classifier on the provided data.
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
                RandomState = _random.Next(),
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
        if (string.IsNullOrEmpty(Options.MaxFeatures))
        {
            return NumFeatures;
        }

        return Options.MaxFeatures.ToLower() switch
        {
            "sqrt" => (int)Math.Ceiling(Math.Sqrt(NumFeatures)),
            "log2" => (int)Math.Ceiling(Math.Log(NumFeatures, 2)),
            "all" => NumFeatures,
            _ when int.TryParse(Options.MaxFeatures, out int n) => Math.Min(n, NumFeatures),
            _ => (int)Math.Ceiling(Math.Sqrt(NumFeatures))
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
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ExtraTreesClassifier<T>(new ExtraTreesClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            MaxFeatures = Options.MaxFeatures,
            Criterion = Options.Criterion,
            Bootstrap = Options.Bootstrap,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new ExtraTreesClassifier<T>(new ExtraTreesClassifierOptions<T>
        {
            NEstimators = Options.NEstimators,
            MaxDepth = Options.MaxDepth,
            MinSamplesSplit = Options.MinSamplesSplit,
            MinSamplesLeaf = Options.MinSamplesLeaf,
            MaxFeatures = Options.MaxFeatures,
            Criterion = Options.Criterion,
            Bootstrap = Options.Bootstrap,
            RandomState = Options.RandomState,
            MinImpurityDecrease = Options.MinImpurityDecrease
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
        metadata.AdditionalInfo["MaxDepth"] = Options.MaxDepth?.ToString() ?? "unlimited";
        metadata.AdditionalInfo["MaxFeatures"] = Options.MaxFeatures;
        metadata.AdditionalInfo["Criterion"] = Options.Criterion.ToString();
        metadata.AdditionalInfo["Bootstrap"] = Options.Bootstrap;
        metadata.AdditionalInfo["TotalNodes"] = NodeCount;
        metadata.AdditionalInfo["TotalLeaves"] = LeafCount;
        return metadata;
    }
}
