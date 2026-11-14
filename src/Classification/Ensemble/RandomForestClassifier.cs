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
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train multiple trees on random subsets of data and features
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new RandomForestClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using majority vote across all trees in the forest
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
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
    /// For each tree, which training rows its bootstrap sample left out (true = out of bag).
    /// Used for proper OOB score calculation.
    /// </summary>
    private readonly List<bool[]> _oobIndicesPerTree = new();

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

        // Draw every tree's bootstrap sample and seed first, in tree order, so the forest is the same whatever
        // NJobs is.
        var trees = new DecisionTreeClassifier<T>[Options.NEstimators];
        var samples = new int[Options.NEstimators][];
        for (int i = 0; i < Options.NEstimators; i++)
        {
            var (bootstrapIndices, outOfBag) = CreateBootstrapSample(x.Rows);
            samples[i] = bootstrapIndices;

            trees[i] = new DecisionTreeClassifier<T>(new DecisionTreeClassifierOptions<T>
            {
                MaxDepth = Options.MaxDepth,
                MinSamplesSplit = Options.MinSamplesSplit,
                MinSamplesLeaf = Options.MinSamplesLeaf,
                MaxFeatures = maxFeatures,
                Criterion = Options.Criterion,
                Seed = _random.Next(),
                MinImpurityDecrease = Options.MinImpurityDecrease
            });

            // Store OOB indices for this tree (for proper OOB score calculation)
            _oobIndicesPerTree.Add(outOfBag);
        }

        // Rank the data once; each tree grows on its bootstrap rows of it without copying them out.
        var data = DecisionTreeClassifier<T>.CreateDenseData(x);
        Parallel.For(0, trees.Length, new ParallelOptions { MaxDegreeOfParallelism = ResolveJobCount() },
            i => trees[i].TrainOnRows(data, y, samples[i]));
        Estimators.AddRange(trees);

        // Aggregate feature importances
        AggregateFeatureImportances();

        // Calculate OOB score if requested
        if (Options.OobScore && Options.Bootstrap)
        {
            CalculateOobScore(x, y);
        }
    }

    /// <summary>
    /// Creates a bootstrap sample of indices, and marks the rows it left out of the bag.
    /// </summary>
    private (int[] Bootstrap, bool[] OutOfBag) CreateBootstrapSample(int nSamples)
    {
        var bootstrapIndices = new int[nSamples];
        var outOfBag = new bool[nSamples];

        if (Options.Bootstrap)
        {
            var random = _random ?? throw new InvalidOperationException("_random has not been initialized.");
            for (int i = 0; i < nSamples; i++) outOfBag[i] = true;

            // Sample with replacement
            for (int i = 0; i < nSamples; i++)
            {
                int idx = random.Next(nSamples);
                bootstrapIndices[i] = idx;
                outOfBag[idx] = false;
            }
        }
        else
        {
            // Use all samples
            for (int i = 0; i < nSamples; i++) bootstrapIndices[i] = i;
        }

        return (bootstrapIndices, outOfBag);
    }

    /// <summary>Rows per block of a parallel prediction.</summary>
    private const int PredictBlockRows = 256;

    /// <summary>
    /// Averages the trees' class probabilities. For <c>double</c> and <c>float</c> features the trees are walked as
    /// flat arrays over fixed blocks of rows in parallel (<see cref="RandomForestClassifierOptions{T}.NJobs"/>). A block
    /// runs one tree over all its rows before the next, as scikit-learn does, so a tree stays in cache instead of every
    /// row pulling in the whole forest. Each row's probabilities are still summed in tree order and divided by the
    /// tree count exactly as the general path does, so both give the same result.
    /// </summary>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (typeof(T) != typeof(double) && typeof(T) != typeof(float))
            return base.PredictProbabilities(input);

        var trees = new FlatDecisionTree<T>[Estimators.Count];
        for (int t = 0; t < trees.Length; t++)
        {
            // A tree of another type (or not yet trained) takes the general path, which reports it.
            if (Estimators[t].GetType() != typeof(DecisionTreeClassifier<T>)
                || ((DecisionTreeClassifier<T>)Estimators[t]).Flatten() is not { } flat)
            {
                return base.PredictProbabilities(input);
            }

            trees[t] = flat;
        }

        if (trees.Length == 0)
            return base.PredictProbabilities(input);

        int classes = NumClasses;
        int columns = input.Columns;
        T treeCount = NumOps.FromDouble(trees.Length);
        var probabilities = new Matrix<T>(input.Rows, classes);
        int blocks = (input.Rows + PredictBlockRows - 1) / PredictBlockRows;
        Parallel.For(0, blocks, new ParallelOptions { MaxDegreeOfParallelism = ResolveJobCount() }, block =>
        {
            int start = block * PredictBlockRows;
            int rows = Math.Min(input.Rows, start + PredictBlockRows) - start;
            var data = new double[rows * columns];
            var sums = new T[rows * classes];
            for (int r = 0; r < rows; r++)
            {
                var source = input.GetRowReadOnlySpan(start + r);
                for (int j = 0; j < columns; j++) data[r * columns + j] = NumOps.ToDouble(source[j]);
                for (int c = 0; c < classes; c++) sums[r * classes + c] = NumOps.Zero;
            }

            foreach (var tree in trees)
            {
                for (int r = 0; r < rows; r++)
                {
                    var leaf = tree.Leaf(data, r * columns);
                    for (int c = 0; c < classes; c++)
                        sums[r * classes + c] = NumOps.Add(sums[r * classes + c], leaf[c]);
                }
            }

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < classes; c++)
                    probabilities[start + r, c] = NumOps.Divide(sums[r * classes + c], treeCount);
        });

        return probabilities;
    }

    /// <summary>
    /// The number of trees trained at once: <see cref="RandomForestClassifierOptions{T}.NJobs"/> when positive,
    /// otherwise counted back from the processor count as joblib does (-1 all processors, -2 all but one).
    /// </summary>
    private int ResolveJobCount()
    {
        int jobs = Options.NJobs;
        if (jobs == 0)
        {
            throw new InvalidOperationException(
                $"{nameof(Options.NJobs)} must be positive, or negative to count back from the processor count; got 0.");
        }

        return jobs > 0 ? jobs : Math.Max(1, Environment.ProcessorCount + 1 + jobs);
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
        int numTrees = Estimators.Count;
        if (nSamples == 0 || NumClasses == 0)
        {
            OobScore_ = 0.0;
            return;
        }

        // Votes, first voting tree per class and the tie-break key are counted in int: in T a running count
        // stops at 2048 for Half and count * (numTrees + 1) loses integer precision
        var engine = Engine;
        var votes = new Tensor<int>(new[] { nSamples, NumClasses });
        var anyVote = new Tensor<int>(new[] { nSamples });
        var firstVoter = new Tensor<int>(new[] { nSamples, NumClasses });
        engine.TensorFill(firstVoter, numTrees);

        for (int treeIdx = 0; treeIdx < numTrees; treeIdx++)
        {
            var outOfBag = _oobIndicesPerTree[treeIdx];
            var oob = Enumerable.Range(0, outOfBag.Length).Where(i => outOfBag[i]).ToArray();
            if (oob.Length == 0)
            {
                continue;
            }

            // Predict every sample that was OOB for this tree in one call; labels that are not known classes
            // land in a dropped leading column
            var oobTensor = new Tensor<int>(oob, new[] { oob.Length });
            var predictedClass = engine.TensorAddScalar(GetClassIndexTensor(Estimators[treeIdx].Predict(x.GetRows(oob))), 1);
            var predicted = engine.TensorOneHot<int>(predictedClass, NumClasses + 1).Slice(1, 1, NumClasses + 1);
            var treeVotes = engine.TensorScatterAdd(new Tensor<int>(new[] { nSamples, NumClasses }), oobTensor, predicted, 0);
            votes = engine.TensorAdd(votes, treeVotes);
            var ones = new Tensor<int>(new[] { oob.Length });
            engine.TensorFill(ones, 1);
            anyVote = engine.TensorScatterAdd(anyVote, oobTensor, ones, 0);

            // Remember the first tree that voted for each class (ties go to the class voted first)
            var voterIndex = new Tensor<int>(new[] { nSamples, NumClasses });
            engine.TensorFill(voterIndex, treeIdx);
            firstVoter = engine.TensorMin(firstVoter,
                engine.TensorWhere(engine.TensorGreaterThan(treeVotes, 0), voterIndex, firstVoter));
        }

        // Majority vote, ties broken by the earliest vote: maximise count * (numTrees + 1) - firstVoter
        var key = engine.TensorSubtract(
            engine.TensorMultiplyScalar(votes.Cast<long>(), (long)numTrees + 1), firstVoter.Cast<long>());
        var majority = engine.TensorArgMax(key, 1).Reshape(nSamples);
        var actual = GetClassIndexTensor(y);

        // Only count samples that were OOB for at least one tree
        var voted = engine.TensorClampMax(anyVote, 1);
        int oobSampleCount = engine.TensorSum(voted);
        int correct = engine.TensorSum(engine.TensorMultiply(voted, engine.TensorEquals(majority, actual)));

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
