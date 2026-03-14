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
/// var classifier = new DARTClassifier&lt;double&gt;(options);
///
/// // Prepare training data
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1.0, 1.1,  1.2, 0.9,  0.8, 1.0,
///     5.0, 5.1,  5.2, 4.9,  4.8, 5.0 });
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train with dropout applied to tree ensemble during boosting
/// classifier.Train(features, labels);
///
/// // Predict using all trees (no dropout at inference)
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 1.1, 1.0 });
/// var prediction = classifier.Predict(newSample);
/// Console.WriteLine($"Predicted class: {prediction[0]}");
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("DART: Dropouts meet Multiple Additive Regression Trees", "https://arxiv.org/abs/1505.01866", Year = 2015, Authors = "K. V. Rashmi, Ran Gilad-Bachrach")]
public class DARTClassifier<T> : EnsembleClassifierBase<T>
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
        : base(options ??= new DARTClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
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

        // Convert to binary labels (for binary classification)
        var yBinary = new Vector<T>(n);
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        for (int i = 0; i < n; i++)
        {
            yBinary[i] = NumOps.Compare(y[i], positiveClass) == 0 ? NumOps.One : NumOps.Zero;
        }

        // Initialize with log-odds of the prior
        int posCount = 0;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.Compare(yBinary[i], NumOps.One) == 0)
            {
                posCount++;
            }
        }
        double p = Math.Max(1e-10, Math.Min(1 - 1e-10, (double)posCount / n));
        _initPrediction = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Current predictions (log-odds)
        var predictions = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            predictions[i] = _initPrediction;
        }

        T bestLoss = NumOps.MaxValue;
        int roundsWithoutImprovement = 0;

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Perform dropout - select trees to drop
            var droppedIndices = new HashSet<int>();
            var keptPredictions = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                keptPredictions[i] = predictions[i];
            }
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
                    foreach (int dropIdx in droppedIndices)
                    {
                        T weight = NumOps.Multiply(_treeWeights[dropIdx], NumOps.FromDouble(_options.LearningRate));
                        for (int i = 0; i < n; i++)
                        {
                            keptPredictions[i] = NumOps.Subtract(keptPredictions[i],
                                NumOps.Multiply(weight, droppedTreePreds[dropIdx][i]));
                        }
                    }
                }
            }

            // Compute residuals from kept predictions
            var residuals = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                T prob = Sigmoid(keptPredictions[i]);
                residuals[i] = NumOps.Subtract(yBinary[i], prob);
            }

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

            // Compute normalization factor
            T newWeight = NumOps.One;
            if (droppedIndices.Count > 0)
            {
                // DART normalization: divide new tree contribution
                newWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(droppedIndices.Count + 1));

                // Scale dropped trees
                T scaleFactor = NumOps.Divide(NumOps.FromDouble(droppedIndices.Count),
                    NumOps.FromDouble(droppedIndices.Count + 1));
                foreach (int dropIdx in droppedIndices)
                {
                    _treeWeights[dropIdx] = NumOps.Multiply(_treeWeights[dropIdx], scaleFactor);
                }
            }
            _treeWeights.Add(newWeight);

            // Update predictions
            T learningRate = NumOps.FromDouble(_options.LearningRate);
            var newTreePred = newTree.Predict(x);
            for (int i = 0; i < n; i++)
            {
                predictions[i] = NumOps.Add(keptPredictions[i],
                    NumOps.Multiply(NumOps.Multiply(newWeight, learningRate), newTreePred[i]));

                // Re-add dropped tree contributions with updated weights
                foreach (int dropIdx in droppedIndices)
                {
                    predictions[i] = NumOps.Add(predictions[i],
                        NumOps.Multiply(NumOps.Multiply(_treeWeights[dropIdx], learningRate),
                            droppedTreePreds[dropIdx][i]));
                }
            }

            // Early stopping
            if (_options.EarlyStoppingRounds.HasValue)
            {
                T loss = ComputeLoss(predictions, yBinary);
                if (NumOps.LessThan(loss, bestLoss))
                {
                    bestLoss = loss;
                    roundsWithoutImprovement = 0;
                }
                else
                {
                    roundsWithoutImprovement++;
                    if (roundsWithoutImprovement >= _options.EarlyStoppingRounds.Value)
                    {
                        break;
                    }
                }
            }

        }

        CalculateFeatureImportances(x.Columns);
    }

    /// <summary>
    /// Predicts class labels for input samples.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        var classLabels = ClassLabels ?? throw new InvalidOperationException("Model must be trained before making predictions.");
        T threshold = NumOps.FromDouble(0.5);
        for (int i = 0; i < input.Rows; i++)
        {
            // Binary classification
            predictions[i] = NumOps.Compare(probs[i, 1], threshold) >= 0
                ? classLabels[classLabels.Length - 1]
                : classLabels[0];
        }

        return predictions;
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
        var probs = new Matrix<T>(n, NumClasses);

        // Compute log-odds predictions
        var logOdds = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            logOdds[i] = _initPrediction;
        }

        T learningRate = NumOps.FromDouble(_options.LearningRate);
        for (int t = 0; t < _trees.Count; t++)
        {
            var treePred = _trees[t].Predict(input);
            T weight = NumOps.Multiply(_treeWeights[t], learningRate);
            for (int i = 0; i < n; i++)
            {
                logOdds[i] = NumOps.Add(logOdds[i], NumOps.Multiply(weight, treePred[i]));
            }
        }

        // Convert to probabilities
        for (int i = 0; i < n; i++)
        {
            T prob1 = Sigmoid(logOdds[i]);
            probs[i, 0] = NumOps.Subtract(NumOps.One, prob1);
            probs[i, 1] = prob1;
        }

        return probs;
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
    /// Sigmoid function for converting log-odds to probability.
    /// </summary>
    private T Sigmoid(T x)
    {
        if (NumOps.GreaterThanOrEquals(x, NumOps.Zero))
        {
            T ez = NumOps.Exp(NumOps.Negate(x));
            return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, ez));
        }
        else
        {
            T ez = NumOps.Exp(x);
            return NumOps.Divide(ez, NumOps.Add(NumOps.One, ez));
        }
    }

    /// <summary>
    /// Computes log loss for current predictions.
    /// </summary>
    private T ComputeLoss(Vector<T> logOdds, Vector<T> yBinary)
    {
        T eps = NumOps.FromDouble(1e-10);
        T oneMinusEps = NumOps.Subtract(NumOps.One, eps);
        T loss = NumOps.Zero;
        for (int i = 0; i < logOdds.Length; i++)
        {
            T y = yBinary[i];
            T p = Sigmoid(logOdds[i]);
            // Clamp p to [eps, 1-eps]
            if (NumOps.LessThan(p, eps)) p = eps;
            if (NumOps.GreaterThan(p, oneMinusEps)) p = oneMinusEps;
            // loss -= y * log(p) + (1 - y) * log(1 - p)
            loss = NumOps.Subtract(loss,
                NumOps.Add(
                    NumOps.Multiply(y, NumOps.Log(p)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, y),
                        NumOps.Log(NumOps.Subtract(NumOps.One, p)))));
        }
        return NumOps.Divide(loss, NumOps.FromDouble(logOdds.Length));
    }

    /// <summary>
    /// Calculates feature importances from all trees.
    /// </summary>
    private void CalculateFeatureImportances(int featureCount)
    {
        var importances = new Vector<T>(featureCount);

        for (int t = 0; t < _trees.Count; t++)
        {
            var fi = _trees[t].FeatureImportances;
            T weight = _treeWeights[t];
            int copyCount = Math.Min(featureCount, fi.Length);
            for (int i = 0; i < copyCount; i++)
            {
                importances[i] = NumOps.Add(importances[i],
                    NumOps.Multiply(weight, fi[i]));
            }
        }

        // Normalize
        T sum = NumOps.Zero;
        for (int i = 0; i < featureCount; i++)
        {
            sum = NumOps.Add(sum, importances[i]);
        }
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int i = 0; i < featureCount; i++)
            {
                importances[i] = NumOps.Divide(importances[i], sum);
            }
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

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        writer.Write(NumOps.ToDouble(_initPrediction));
        writer.Write(_options.NumberOfIterations);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
        writer.Write((int)_options.DropoutType);

        writer.Write(_trees.Count);
        for (int t = 0; t < _trees.Count; t++)
        {
            writer.Write(NumOps.ToDouble(_treeWeights[t]));
            byte[] treeData = _trees[t].Serialize();
            writer.Write(treeData.Length);
            writer.Write(treeData);
        }

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseLen);
        base.Deserialize(baseData);

        _initPrediction = NumOps.FromDouble(reader.ReadDouble());
        _options.NumberOfIterations = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
        _options.DropoutType = (DARTDropoutType)reader.ReadInt32();

        int numTrees = reader.ReadInt32();
        _trees.Clear();
        _treeWeights.Clear();
        for (int t = 0; t < numTrees; t++)
        {
            _treeWeights.Add(NumOps.FromDouble(reader.ReadDouble()));
            int treeLen = reader.ReadInt32();
            byte[] treeData = reader.ReadBytes(treeLen);
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions());
            tree.Deserialize(treeData);
            _trees.Add(tree);
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DARTClassifier<T>(_options, Regularization);
    }
}
