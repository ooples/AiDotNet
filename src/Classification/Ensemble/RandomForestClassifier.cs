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
/// Console.WriteLine($"Predicted class: {prediction[0]}");
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Random Forests", "https://doi.org/10.1023/A:1010933404324", Year = 2001, Authors = "Leo Breiman")]
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
        : base(options ?? new RandomForestClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
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
            _ => (int)Math.Ceiling(Math.Sqrt(NumFeatures)) // Default to sqrt
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
            Seed = Options.Seed,
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
            Seed = Options.Seed,
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

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "ClassLabels", ClassLabels?.ToArray() ?? Array.Empty<T>() },
            { "RegularizationOptions", Regularization.GetOptions() },
            { "OobScore_", OobScore_ }
        };

        // Serialize FeatureImportances
        if (FeatureImportances is not null)
        {
            var fiArray = new double[FeatureImportances.Length];
            for (int i = 0; i < FeatureImportances.Length; i++)
                fiArray[i] = NumOps.ToDouble(FeatureImportances[i]);
            modelData["FeatureImportances"] = fiArray;
        }

        // Serialize each estimator (DecisionTreeClassifier) as base64
        modelData["EstimatorCount"] = Estimators.Count;
        for (int i = 0; i < Estimators.Count; i++)
        {
            if (Estimators[i] is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                modelData[$"Estimator_{i}"] = Convert.ToBase64String(fullModel.Serialize());
            }
        }

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata == null || modelMetadata.ModelData == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<JObject>(modelDataString);

        if (modelDataObj == null)
            throw new InvalidOperationException("Deserialization failed: The model data is invalid or corrupted.");

        NumClasses = modelDataObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(modelDataObj["TaskType"]?.ToObject<int>() ?? 0);

        var classLabelsToken = modelDataObj["ClassLabels"];
        if (classLabelsToken is not null)
        {
            var classLabelsAsDoubles = classLabelsToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (classLabelsAsDoubles.Length > 0)
            {
                ClassLabels = new Vector<T>(classLabelsAsDoubles.Length);
                for (int i = 0; i < classLabelsAsDoubles.Length; i++)
                    ClassLabels[i] = NumOps.FromDouble(classLabelsAsDoubles[i]);
            }
        }

        OobScore_ = modelDataObj["OobScore_"]?.ToObject<double>() ?? 0.0;

        // Deserialize FeatureImportances
        var fiToken = modelDataObj["FeatureImportances"];
        if (fiToken is not null)
        {
            var fiArray = fiToken.ToObject<double[]>() ?? Array.Empty<double>();
            if (fiArray.Length > 0)
            {
                FeatureImportances = new Vector<T>(fiArray.Length);
                for (int i = 0; i < fiArray.Length; i++)
                    FeatureImportances[i] = NumOps.FromDouble(fiArray[i]);
            }
        }

        // Deserialize estimators
        int estimatorCount = modelDataObj["EstimatorCount"]?.ToObject<int>() ?? 0;
        Estimators.Clear();
        for (int i = 0; i < estimatorCount; i++)
        {
            var estToken = modelDataObj[$"Estimator_{i}"]?.ToObject<string>();
            if (estToken is null)
            {
                throw new InvalidOperationException(
                    $"Deserialization failed: Estimator_{i} is missing (expected {estimatorCount} estimators).");
            }
            var estBytes = Convert.FromBase64String(estToken);
            var tree = new DecisionTreeClassifier<T>();
            tree.Deserialize(estBytes);
            Estimators.Add(tree);
        }
    }
}
