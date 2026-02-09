using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.Trees;
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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DARTClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Individual regression trees (trained on pseudo-residuals).
    /// </summary>
    private readonly List<DecisionTreeRegression<T>> _trees;

    /// <summary>
    /// Tree weights (may differ after dropout normalization).
    /// </summary>
    private readonly List<double> _treeWeights;

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
    protected override ModelType GetModelType() => ModelType.DARTClassifier;

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
        var predictions = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            predictions[i] = NumOps.ToDouble(_initPrediction);
        }

        double bestLoss = double.MaxValue;
        int roundsWithoutImprovement = 0;

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Perform dropout - select trees to drop
            var droppedIndices = new HashSet<int>();
            var keptPredictions = new Vector<double>(n);
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
                        double weight = _treeWeights[dropIdx] * _options.LearningRate;
                        for (int i = 0; i < n; i++)
                        {
                            keptPredictions[i] -= weight * NumOps.ToDouble(droppedTreePreds[dropIdx][i]);
                        }
                    }
                }
            }

            // Compute residuals from kept predictions
            var residuals = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                double prob = Sigmoid(keptPredictions[i]);
                double target = NumOps.ToDouble(yBinary[i]);
                residuals[i] = NumOps.FromDouble(target - prob);
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
            double newWeight = 1.0;
            if (droppedIndices.Count > 0)
            {
                // DART normalization: divide new tree contribution
                newWeight = 1.0 / (droppedIndices.Count + 1);

                // Scale dropped trees
                foreach (int dropIdx in droppedIndices)
                {
                    _treeWeights[dropIdx] *= (droppedIndices.Count / (double)(droppedIndices.Count + 1));
                }
            }
            _treeWeights.Add(newWeight);

            // Update predictions
            var newTreePred = newTree.Predict(x);
            for (int i = 0; i < n; i++)
            {
                predictions[i] = keptPredictions[i] + newWeight * _options.LearningRate * NumOps.ToDouble(newTreePred[i]);

                // Re-add dropped tree contributions with updated weights
                foreach (int dropIdx in droppedIndices)
                {
                    predictions[i] += _treeWeights[dropIdx] * _options.LearningRate * NumOps.ToDouble(droppedTreePreds[dropIdx][i]);
                }
            }

            // Early stopping
            if (_options.EarlyStoppingRounds.HasValue)
            {
                double loss = ComputeLoss(predictions, yBinary);
                if (loss < bestLoss)
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

            if (_options.Verbose && (iter + 1) % _options.VerboseEval == 0)
            {
                double loss = ComputeLoss(predictions, yBinary);
                Console.WriteLine($"[{iter + 1}] Loss: {loss:F6}");
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
        var logOdds = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            logOdds[i] = NumOps.ToDouble(_initPrediction);
        }

        for (int t = 0; t < _trees.Count; t++)
        {
            var treePred = _trees[t].Predict(input);
            double weight = _treeWeights[t] * _options.LearningRate;
            for (int i = 0; i < n; i++)
            {
                logOdds[i] += weight * NumOps.ToDouble(treePred[i]);
            }
        }

        // Convert to probabilities
        for (int i = 0; i < n; i++)
        {
            double prob1 = Sigmoid(logOdds[i]);
            probs[i, 0] = NumOps.FromDouble(1 - prob1);
            probs[i, 1] = NumOps.FromDouble(prob1);
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
    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            double ez = Math.Exp(-x);
            return 1.0 / (1.0 + ez);
        }
        else
        {
            double ez = Math.Exp(x);
            return ez / (1.0 + ez);
        }
    }

    /// <summary>
    /// Computes log loss for current predictions.
    /// </summary>
    private double ComputeLoss(Vector<double> logOdds, Vector<T> yBinary)
    {
        double loss = 0;
        for (int i = 0; i < logOdds.Length; i++)
        {
            double y = NumOps.ToDouble(yBinary[i]);
            double p = Sigmoid(logOdds[i]);
            p = Math.Max(1e-10, Math.Min(1 - 1e-10, p));
            loss -= y * Math.Log(p) + (1 - y) * Math.Log(1 - p);
        }
        return loss / logOdds.Length;
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
            double weight = _treeWeights[t];
            int copyCount = Math.Min(featureCount, fi.Length);
            for (int i = 0; i < copyCount; i++)
            {
                importances[i] = NumOps.Add(importances[i],
                    NumOps.Multiply(NumOps.FromDouble(weight), fi[i]));
            }
        }

        // Normalize
        T sum = NumOps.Zero;
        for (int i = 0; i < featureCount; i++)
        {
            sum = NumOps.Add(sum, importances[i]);
        }
        if (NumOps.ToDouble(sum) > 0)
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
            ModelType = ModelType.DARTClassifier,
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
            writer.Write(_treeWeights[t]);
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
            _treeWeights.Add(reader.ReadDouble());
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
