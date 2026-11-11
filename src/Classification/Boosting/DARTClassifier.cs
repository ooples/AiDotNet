using AiDotNet.Attributes;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.Trees;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;

namespace AiDotNet.Classification.Boosting;

/// <summary>
/// DART (Dropouts meet Multiple Additive Regression Trees) classifier.
/// </summary>
/// <remarks>
/// <para>
/// DART applies dropout regularization to gradient boosting for classification. During each
/// iteration, a random subset of existing trees is dropped, and the new tree is fitted to
/// residuals computed only from the non-dropped trees. This prevents overfitting.
/// </para>
/// <para>
/// <b>For Beginners:</b> DART is like gradient boosting classifier with a twist - it randomly
/// "forgets" some of its trees when learning new ones. This prevents the model from becoming
/// too specialized and helps it work better on new data.
///
/// Key concepts:
/// - Dropout: Randomly removing trees during training (like dropout in neural networks)
/// - Normalization: Adjusting predictions after dropout to maintain correct scale
/// - Ensemble: The final prediction uses all trees (no dropout at prediction time)
///
/// When to use DART over regular gradient boosting:
/// - Your model overfits (training error low, validation error high)
/// - You want more robust predictions
/// - You have enough time (DART is slower than regular boosting)
/// </para>
/// <para>
/// Reference: Rashmi, K.V. &amp; Gilad-Bachrach, R. (2015). "DART: Dropouts meet Multiple
/// Additive Regression Trees". AISTATS 2015.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create DART classifier with dropout regularization for gradient boosting
/// var options = new DARTClassifierOptions&lt;double&gt;();
///
/// // Prepare training data
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 1.1 }, { 1.2, 0.9 }, { 0.8, 1.0 }, { 5.0, 5.1 }, { 5.2, 4.9 }, { 4.8, 5.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train with dropout applied to tree ensemble during boosting
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DARTClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict using all trees (no dropout at inference)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.1, 1.0 } });
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("DART: Dropouts meet Multiple Additive Regression Trees", "https://arxiv.org/abs/1505.01866", Year = 2015, Authors = "K. V. Rashmi, Ran Gilad-Bachrach")]
public partial class DARTClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Individual regression trees (trained on pseudo-residuals).
    /// </summary>
    private readonly List<DecisionTreeRegression<T>> _trees;

    /// <summary>
    /// Tree weights (may differ after dropout normalization).
    /// </summary>
    private readonly List<T> _treeWeights;

    /// <summary>
    /// Initial log-odds prediction.
    /// </summary>
    private T _initPrediction;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly DARTClassifierOptions<T> _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random _random;

    /// <summary>
    /// Gets the number of trees in the ensemble.
    /// </summary>
    public int NumberOfTrees => _trees.Count;

    /// <summary>
    /// Initializes a new instance of DART classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public DARTClassifier(DARTClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ??= new DARTClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options;
        _trees = [];
        _treeWeights = [];
        _initPrediction = NumOps.Zero;
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the DART classifier with dropout regularization.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        int n = x.Rows;
        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        _trees.Clear();
        _treeWeights.Clear();
        Estimators.Clear();

        var engine = Engine;

        // Convert to binary labels (for binary classification)
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yBinary = engine.TensorEquals(y.AsTensor(), positiveClass).ToVector();

        // Initialize with log-odds of the prior
        int posCount = engine.TensorCountNonzero(yBinary.AsTensor());
        double p = Math.Max(1e-10, Math.Min(1 - 1e-10, (double)posCount / n));
        _initPrediction = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Current predictions (log-odds)
        var predictions = new Vector<T>(n);
        predictions.Fill(_initPrediction);

        T bestLoss = NumOps.MaxValue;
        int roundsWithoutImprovement = 0;
        T learningRate = NumOps.FromDouble(_options.LearningRate);

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Perform dropout - select trees to drop
            var droppedIndices = new HashSet<int>();
            var keptPredictions = predictions.Clone();
            var droppedTreePreds = new Dictionary<int, Vector<T>>();

            if (_trees.Count > 0)
            {
                int numToDrop = Math.Min(SelectNumDropout(_trees.Count), _trees.Count);

                if (numToDrop > 0)
                {
                    var allIndices = Enumerable.Range(0, _trees.Count).ToList();
                    for (int d = 0; d < numToDrop; d++)
                    {
                        int randomIdx = _random.Next(allIndices.Count);
                        droppedIndices.Add(allIndices[randomIdx]);
                        allIndices.RemoveAt(randomIdx);
                    }

                    foreach (int dropIdx in droppedIndices)
                    {
                        droppedTreePreds[dropIdx] = _trees[dropIdx].Predict(x);
                    }

                    // Subtract dropped tree contributions
                    keptPredictions = engine.Subtract(keptPredictions,
                        DroppedContribution(droppedIndices, droppedTreePreds, learningRate, n));
                }
            }

            // Compute residuals from kept predictions
            var residuals = engine.Subtract(yBinary, engine.Sigmoid(keptPredictions));

            // Train new tree on residuals
            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                SplitCriterion = _options.SplitCriterion,
                Seed = _random.Next()
            };

            var newTree = new DecisionTreeRegression<T>(treeOptions);
            newTree.Train(x, residuals);
            _trees.Add(newTree);

            // Compute normalization factor (DART: new tree gets 1/(k+1), dropped trees scale by k/(k+1))
            T newWeight = ApplyDartNormalization(droppedIndices);
            _treeWeights.Add(newWeight);

            // Update predictions: kept + new tree + dropped trees re-added with their updated weights
            var newTreePred = newTree.Predict(x);
            var newContribution = engine.Multiply(newTreePred, NumOps.Multiply(newWeight, learningRate));
            predictions = engine.Add(engine.Add(keptPredictions, newContribution),
                DroppedContribution(droppedIndices, droppedTreePreds, learningRate, n));

            // Early stopping
            if (_options.EarlyStoppingRounds.HasValue
                && ShouldStopEarly(ComputeLoss(predictions, yBinary), ref bestLoss, ref roundsWithoutImprovement))
            {
                break;
            }

        }

        CalculateFeatureImportances(x.Columns);
    }

    /// <summary>
    /// Predicts class labels for input samples.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Binary classification: P(positive) >= 0.5 selects the last label, otherwise the first.
        return PredictBinaryLabels(PredictProbabilities(input));
    }

    /// <summary>
    /// Predicts class probabilities for input samples.
    /// </summary>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                $"DART classifier currently supports binary classification only (NumClasses={NumClasses}).");
        }

        int n = input.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        var engine = Engine;

        // Compute log-odds predictions: init + sum_t (w_t * lr) * tree_t(x)
        var treePreds = new List<Vector<T>>(_trees.Count);
        foreach (var tree in _trees)
        {
            treePreds.Add(tree.Predict(input));
        }

        var weights = engine.Multiply(Vector<T>.FromList(_treeWeights), NumOps.FromDouble(_options.LearningRate));
        var logOdds = engine.Add(FilledVector(n, _initPrediction), WeightedSumOfVectors(treePreds, weights, n));

        // Convert to probabilities
        return BinaryProbabilitiesFromLogOdds(logOdds);
    }

    /// <summary>
    /// Selects number of trees to drop based on dropout type.
    /// </summary>
    private int SelectNumDropout(int numTrees)
    {
        if (_options.DropoutRate <= 0)
        {
            return 0;
        }

        return _options.DropoutType switch
        {
            DARTDropoutType.Uniform => Math.Max(1, (int)(numTrees * _options.DropoutRate)),
            DARTDropoutType.Binomial => CountBinomialDrops(numTrees, _options.DropoutRate),
            _ => Math.Max(1, (int)(numTrees * _options.DropoutRate))
        };
    }

    /// <summary>
    /// Counts drops using binomial distribution.
    /// </summary>
    private int CountBinomialDrops(int numTrees, double rate)
    {
        int drops = 0;
        for (int i = 0; i < numTrees; i++)
        {
            if (_random.NextDouble() < rate)
            {
                drops++;
            }
        }
        return drops;
    }

    /// <summary>
    /// Sum over the dropped trees of (w_d * learningRate) * prediction_d.
    /// </summary>
    private Vector<T> DroppedContribution(HashSet<int> droppedIndices, Dictionary<int, Vector<T>> droppedTreePreds,
        T learningRate, int n)
    {
        var preds = new List<Vector<T>>(droppedIndices.Count);
        var weights = new List<T>(droppedIndices.Count);
        foreach (int dropIdx in droppedIndices)
        {
            preds.Add(droppedTreePreds[dropIdx]);
            weights.Add(_treeWeights[dropIdx]);
        }

        if (preds.Count == 0)
        {
            return new Vector<T>(n);
        }

        return WeightedSumOfVectors(preds, Engine.Multiply(Vector<T>.FromList(weights), learningRate), n);
    }

    /// <summary>
    /// Applies DART normalization: scales the dropped trees by k/(k+1) and returns the new tree's weight 1/(k+1)
    /// (1 when nothing was dropped).
    /// </summary>
    private T ApplyDartNormalization(HashSet<int> droppedIndices)
    {
        int k = droppedIndices.Count;
        if (k == 0)
        {
            return NumOps.One;
        }

        var order = droppedIndices.ToArray();
        var dropped = new T[k];
        for (int d = 0; d < k; d++)
        {
            dropped[d] = _treeWeights[order[d]];
        }

        T scaleFactor = NumOps.Divide(NumOps.FromDouble(k), NumOps.FromDouble(k + 1));
        var scaled = Engine.Multiply(Vector<T>.Wrap(dropped), scaleFactor).ToArray();
        for (int d = 0; d < k; d++)
        {
            _treeWeights[order[d]] = scaled[d];
        }

        return NumOps.Divide(NumOps.One, NumOps.FromDouble(k + 1));
    }

    /// <summary>
    /// Early-stopping bookkeeping: records an improvement or counts a stale round.
    /// </summary>
    /// <returns>True when training should stop.</returns>
    private bool ShouldStopEarly(T loss, ref T bestLoss, ref int roundsWithoutImprovement)
    {
        if (NumOps.LessThan(loss, bestLoss))
        {
            bestLoss = loss;
            roundsWithoutImprovement = 0;
            return false;
        }

        roundsWithoutImprovement++;
        return roundsWithoutImprovement >= (_options.EarlyStoppingRounds ?? int.MaxValue);
    }

    /// <summary>
    /// Computes log loss for current predictions.
    /// </summary>
    private T ComputeLoss(Vector<T> logOdds, Vector<T> yBinary) => ComputeBinaryLogLoss(logOdds, yBinary);

    /// <summary>
    /// Calculates feature importances from all trees.
    /// </summary>
    private void CalculateFeatureImportances(int featureCount)
    {
        var perTree = new List<Vector<T>>(_trees.Count);
        foreach (var tree in _trees)
        {
            perTree.Add(FitToLength(tree.FeatureImportances, featureCount));
        }

        var importances = WeightedSumOfVectors(perTree, Vector<T>.FromList(_treeWeights), featureCount);

        // Normalize
        T sum = featureCount == 0 ? NumOps.Zero : Engine.Sum(importances);
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            importances = Engine.Divide(importances, sum);
        }

        FeatureImportances = importances;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _trees.Count },
                { "DropoutRate", _options.DropoutRate },
                { "DropoutType", _options.DropoutType.ToString() },
                { "LearningRate", _options.LearningRate },
                { "MaxDepth", _options.MaxDepth }
            }
        };
    }
}
