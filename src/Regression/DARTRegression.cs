using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// DART (Dropouts meet Multiple Additive Regression Trees) regression.
/// </summary>
/// <remarks>
/// <para>
/// DART applies dropout regularization to gradient boosting. During each iteration, a random
/// subset of existing trees is dropped, and the new tree is fitted to residuals computed
/// only from the non-dropped trees. This prevents overfitting and improves generalization.
/// </para>
/// <para>
/// <b>For Beginners:</b> DART is like gradient boosting with a twist - it randomly "forgets"
/// some of its trees when learning new ones. This prevents the model from becoming too
/// specialized and helps it work better on new data.
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
/// // Create a DART regression model with dropout regularization
/// var options = new DARTOptions();
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with dropout applied to trees during boosting
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DARTRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new sample (uses all trees without dropout)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("DART: Dropouts meet Multiple Additive Regression Trees", "https://arxiv.org/abs/1505.01866", Year = 2015, Authors = "K. V. Rashmi, Ran Gilad-Bachrach")]
public partial class DARTRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Individual tree structures.
    /// </summary>
    private List<DARTTree> _trees;

    /// <summary>
    /// Tree weights (may differ after normalization).
    /// </summary>
    private List<T> _treeWeights;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly DARTOptions _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <inheritdoc/>
    public override int NumberOfTrees => _trees.Count;

    /// <summary>
    /// Initializes a new instance of DART regression.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public DARTRegression(DARTOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new DARTOptions();
        _trees = [];
        _treeWeights = [];
        _numFeatures = 0;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        _trees = [];
        _treeWeights = [];

        var engine = AiDotNetEngine.Current;

        // Current predictions (starts as mean)
        T meanY = NumOps.Divide(engine.Sum(y), NumOps.FromDouble(n));

        // Add base prediction as first "tree"
        _trees.Add(new DARTTree(NumOps.Zero) { IsConstant = true, ConstantValue = meanY });
        _treeWeights.Add(NumOps.One);

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Dropout: select trees to keep
            var (droppedIndices, keptIndices) = SelectDroppedTrees(iter);

            // Compute predictions from kept trees only
            var droppedPredictions = ComputePartialPredictions(x, keptIndices);

            // Compute residuals (negative gradient for MSE)
            var residuals = engine.Subtract(y, droppedPredictions);

            // Sample features and data points
            var featureIndices = SampleFeatures();
            var sampleIndices = SampleData(n);

            // Fit new tree to residuals
            var newTree = FitTree(x, residuals, sampleIndices, featureIndices, 0);

            // Compute normalization factor
            T normFactor = ComputeNormalizationFactor(droppedIndices);
            T lr = NumOps.FromDouble(_options.LearningRate);

            // Update tree weights based on normalization
            if (_options.Normalization == DARTNormalization.Tree)
            {
                // Scale dropped trees
                ScaleTreeWeights(droppedIndices, normFactor);
                // New tree gets weight based on learning rate and norm factor
                T newWeight = NumOps.Multiply(lr, normFactor);
                _trees.Add(newTree);
                _treeWeights.Add(newWeight);
            }
            else // Forest normalization
            {
                T droppedSum = droppedIndices.Length == 0
                    ? NumOps.Zero
                    : engine.Sum(RegressionEngineOps.Gather(new Vector<T>(_treeWeights), droppedIndices));
                T forestNorm = NumOps.GreaterThan(droppedSum, NumOps.Zero)
                    ? NumOps.Divide(droppedSum, NumOps.Add(droppedSum, lr))
                    : NumOps.One;

                ScaleTreeWeights(droppedIndices, forestNorm);

                T newWeight = lr;
                if (NumOps.GreaterThan(droppedSum, NumOps.Zero))
                {
                    newWeight = NumOps.Multiply(
                        NumOps.Divide(lr, NumOps.Add(droppedSum, lr)),
                        droppedSum);
                }

                _trees.Add(newTree);
                _treeWeights.Add(newWeight);
            }
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Multiplies the weights of the given trees by <paramref name="factor"/>, all at once.
    /// </summary>
    private void ScaleTreeWeights(int[] treeIndices, T factor)
    {
        if (treeIndices.Length == 0)
        {
            return;
        }

        var engine = AiDotNetEngine.Current;
        var multipliers = engine.TensorScatter(
            Tensor<T>.FromVector(engine.Fill(_treeWeights.Count, NumOps.One)),
            new Tensor<int>(treeIndices, new[] { treeIndices.Length }),
            Tensor<T>.FromVector(engine.Fill(treeIndices.Length, factor)),
            0).ToVector();
        _treeWeights = new List<T>(engine.Multiply(new Vector<T>(_treeWeights), multipliers));
    }

    /// <inheritdoc/>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        return await Task.Run(() => Predict(input));
    }

    /// <summary>
    /// Predicts values for input samples.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Vector of predictions.</returns>
    public new Vector<T> Predict(Matrix<T> input)
    {
        // Use all trees at prediction time (no dropout)
        var allIndices = Enumerable.Range(0, _trees.Count).ToArray();
        return ComputePartialPredictions(input, allIndices);
    }

    /// <summary>
    /// Gets the individual tree predictions for a sample.
    /// </summary>
    /// <param name="input">Single input sample.</param>
    /// <returns>Array of weighted tree predictions.</returns>
    public Vector<T> GetTreePredictions(Vector<T> input)
    {
        var rawPredictions = new Vector<T>(_trees.Select(tree => PredictTree(tree, input)));
        return AiDotNetEngine.Current.Multiply(rawPredictions, new Vector<T>(_treeWeights));
    }

    /// <summary>
    /// Gets the tree weights.
    /// </summary>
    /// <returns>Vector of tree weights.</returns>
    public Vector<T> GetTreeWeights()
    {
        return new Vector<T>(_treeWeights);
    }

    /// <summary>
    /// Selects trees to drop for this iteration.
    /// </summary>
    private (int[] dropped, int[] kept) SelectDroppedTrees(int iteration)
    {
        var dropped = new List<int>();
        var kept = new List<int>();

        // Skip dropout for first few iterations
        if (iteration < _options.SkipDropoutIterations || _trees.Count <= 1)
        {
            return ([], Enumerable.Range(0, _trees.Count).ToArray());
        }

        // Apply dropout
        for (int t = 0; t < _trees.Count; t++)
        {
            double dropProb = GetDropProbability(t);

            if (_random.NextDouble() < dropProb)
            {
                dropped.Add(t);
            }
            else
            {
                kept.Add(t);
            }
        }

        // OneDrop: ensure at least one tree is dropped
        if (_options.OneDrop && dropped.Count == 0 && _trees.Count > 0)
        {
            int dropIdx = _random.Next(_trees.Count);
            dropped.Add(dropIdx);
            kept.Remove(dropIdx);
        }

        return ([.. dropped], [.. kept]);
    }

    /// <summary>
    /// Gets the drop probability for a tree based on dropout mode.
    /// </summary>
    private double GetDropProbability(int treeIndex)
    {
        if (_options.DropoutMode == DARTDropoutMode.Weighted)
        {
            T maxWeight = AiDotNetEngine.Current.TensorMaxValue(Tensor<T>.FromVector(new Vector<T>(_treeWeights)));
            double maxW = NumOps.ToDouble(maxWeight);
            return maxW > 0 ? _options.DropoutRate * NumOps.ToDouble(_treeWeights[treeIndex]) / maxW : _options.DropoutRate;
        }

        return _options.DropoutMode switch
        {
            DARTDropoutMode.Uniform => _options.DropoutRate,
            DARTDropoutMode.Age => _options.DropoutRate * (1.0 - (double)treeIndex / _trees.Count),
            _ => _options.DropoutRate
        };
    }

    /// <summary>
    /// Computes normalization factor for dropout.
    /// </summary>
    private T ComputeNormalizationFactor(int[] droppedIndices)
    {
        if (droppedIndices.Length == 0)
        {
            return NumOps.One;
        }

        return NumOps.Divide(
            NumOps.FromDouble(droppedIndices.Length),
            NumOps.FromDouble(droppedIndices.Length + 1));
    }

    /// <summary>
    /// Computes predictions using only specified trees.
    /// </summary>
    private Vector<T> ComputePartialPredictions(Matrix<T> xData, int[] treeIndices)
    {
        int n = xData.Rows;
        var engine = AiDotNetEngine.Current;
        var predictions = engine.FillZero<T>(n);

        foreach (int t in treeIndices)
        {
            T weight = _treeWeights[t];
            var tree = _trees[t];
            var treePredictions = new Vector<T>(Enumerable.Range(0, n).Select(i => PredictTree(tree, xData, i)));
            predictions = engine.Add(predictions, engine.Multiply(treePredictions, weight));
        }

        return predictions;
    }

    /// <summary>
    /// Samples features for this tree.
    /// </summary>
    private int[] SampleFeatures()
    {
        if (_options.FeatureFraction >= 1.0)
        {
            return Enumerable.Range(0, _numFeatures).ToArray();
        }

        int numSample = Math.Max(1, (int)(_numFeatures * _options.FeatureFraction));
        return Enumerable.Range(0, _numFeatures)
            .OrderBy(_ => _random.Next())
            .Take(numSample)
            .ToArray();
    }

    /// <summary>
    /// Samples data points for this tree.
    /// </summary>
    private int[] SampleData(int n)
    {
        if (_options.SubsampleFraction >= 1.0)
        {
            return Enumerable.Range(0, n).ToArray();
        }

        int numSample = Math.Max(1, (int)(n * _options.SubsampleFraction));
        return Enumerable.Range(0, n)
            .OrderBy(_ => _random.Next())
            .Take(numSample)
            .ToArray();
    }

    /// <summary>
    /// Fits a decision tree to the data.
    /// </summary>
    private DARTTree FitTree(Matrix<T> xData, Vector<T> y, int[] sampleIndices, int[] featureIndices, int depth)
    {
        // Check stopping conditions
        if (depth >= _options.MaxDepth || sampleIndices.Length < _options.MinSamplesLeaf * 2)
        {
            return CreateLeaf(y, sampleIndices);
        }

        // Find best split
        var (bestFeature, bestThreshold, bestGain) = FindBestSplit(xData, y, sampleIndices, featureIndices);

        T minGain = NumOps.FromDouble(_options.MinSplitGain);
        if (bestFeature < 0 || NumOps.LessThan(bestGain, minGain))
        {
            return CreateLeaf(y, sampleIndices);
        }

        // Split data
        var (leftIndices, rightIndices) = SplitData(xData, sampleIndices, bestFeature, bestThreshold);

        if (leftIndices.Length < _options.MinSamplesLeaf || rightIndices.Length < _options.MinSamplesLeaf)
        {
            return CreateLeaf(y, sampleIndices);
        }

        // Recursively build subtrees
        var leftChild = FitTree(xData, y, leftIndices, featureIndices, depth + 1);
        var rightChild = FitTree(xData, y, rightIndices, featureIndices, depth + 1);

        return new DARTTree(NumOps.Zero)
        {
            IsLeaf = false,
            SplitFeature = bestFeature,
            SplitThreshold = bestThreshold,
            Left = leftChild,
            Right = rightChild
        };
    }

    /// <summary>
    /// Creates a leaf node.
    /// </summary>
    private DARTTree CreateLeaf(Vector<T> y, int[] indices)
    {
        T sum = indices.Length == 0
            ? NumOps.Zero
            : AiDotNetEngine.Current.Sum(RegressionEngineOps.Gather(y, indices));

        T mean;
        // Apply L2 regularization
        if (_options.L2Regularization > 0)
        {
            mean = NumOps.Divide(sum, NumOps.FromDouble(indices.Length + _options.L2Regularization));
        }
        else
        {
            mean = NumOps.Divide(sum, NumOps.FromDouble(indices.Length));
        }

        return new DARTTree(NumOps.Zero)
        {
            IsLeaf = true,
            LeafValue = mean
        };
    }

    /// <summary>
    /// Finds the best split for a node.
    /// </summary>
    private (int feature, T threshold, T gain) FindBestSplit(
        Matrix<T> xData, Vector<T> y, int[] indices, int[] featureIndices)
    {
        var engine = AiDotNetEngine.Current;
        int count = indices.Length;
        if (count < 2)
        {
            return (-1, NumOps.Zero, NumOps.MinValue);
        }

        T epsilon = NumOps.FromDouble(1e-10);

        // Compute current node's MSE
        var nodeY = RegressionEngineOps.Gather(y, indices);
        T totalSum = engine.Sum(nodeY);
        T totalSumSq = engine.DotProduct(nodeY, nodeY);
        T nT = NumOps.FromDouble(count);
        T totalMse = NumOps.Subtract(totalSumSq, NumOps.Divide(NumOps.Multiply(totalSum, totalSum), nT));

        // Split position s (1..count-1) puts the first s sorted samples left.
        int splits = count - 1;
        var allowed = new bool[splits];
        for (int s = 1; s <= splits; s++)
        {
            // Skip if below minimum samples
            allowed[s - 1] = s >= _options.MinSamplesLeaf && count - s >= _options.MinSamplesLeaf;
        }

        var leftN = engine.TensorCumSum(Tensor<T>.FromVector(engine.Fill(splits, NumOps.One)), 0).ToVector();
        var rightN = engine.Subtract(engine.Fill(splits, nT), leftN);

        var candidateFeatures = new List<int>();
        var candidateGains = new List<T>();
        var candidateThresholds = new List<T>();
        foreach (int f in featureIndices)
        {
            // Sort indices by feature value
            var nodeFeature = RegressionEngineOps.Gather(engine.GetColumn(xData, f), indices);
            var order = RegressionEngineOps.StableAscendingOrder(nodeFeature);
            var sortedFeature = RegressionEngineOps.Gather(nodeFeature, order);
            var sortedY = RegressionEngineOps.Gather(nodeY, order);

            var leftSum = RegressionEngineOps.SplitPrefixSums(sortedY);
            var leftSumSq = RegressionEngineOps.SplitPrefixSums(engine.Multiply(sortedY, sortedY));
            var rightSum = engine.Subtract(engine.Fill(splits, totalSum), leftSum);
            var rightSumSq = engine.Subtract(engine.Fill(splits, totalSumSq), leftSumSq);

            var leftMse = engine.Subtract(leftSumSq, engine.Divide(engine.Multiply(leftSum, leftSum), leftN));
            var rightMse = engine.Subtract(rightSumSq, engine.Divide(engine.Multiply(rightSum, rightSum), rightN));
            var gains = engine.Subtract(engine.Fill(splits, totalMse), engine.Add(leftMse, rightMse));

            // Skip if same feature value as next point (|x[s] - x[s-1]| < epsilon)
            var gaps = engine.Abs(engine.Subtract(sortedFeature.GetSubVector(1, splits), sortedFeature.GetSubVector(0, splits)));
            var distinct = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(Tensor<T>.FromVector(gaps), epsilon));

            var (index, gain, threshold) = RegressionEngineOps.BestSortedSplit(gains, distinct, allowed, sortedFeature);
            if (index >= 0)
            {
                candidateFeatures.Add(f);
                candidateGains.Add(gain);
                candidateThresholds.Add(threshold);
            }
        }

        if (candidateFeatures.Count == 0)
        {
            return (-1, NumOps.Zero, NumOps.MinValue);
        }

        // The first feature with the largest gain wins (strictly-greater sweep order).
        int best = RegressionEngineOps.ArgMaxFirst(candidateGains);
        return (candidateFeatures[best], candidateThresholds[best], candidateGains[best]);
    }

    /// <summary>
    /// Splits data based on a feature and threshold.
    /// </summary>
    private (int[] left, int[] right) SplitData(Matrix<T> xData, int[] indices, int feature, T threshold)
    {
        if (indices.Length == 0)
        {
            return ([], []);
        }

        // Left keeps everything that is not greater than the threshold (NaN included).
        var engine = AiDotNetEngine.Current;
        var values = RegressionEngineOps.Gather(engine.GetColumn(xData, feature), indices);
        var rightMask = engine.TensorGreaterThan(Tensor<T>.FromVector(values), threshold);
        var leftMask = engine.ScalarMinusTensor(NumOps.One, rightMask);
        var left = RegressionEngineOps.NonzeroIndices(leftMask);
        var right = RegressionEngineOps.NonzeroIndices(rightMask);

        return (left.Select(position => indices[position]).ToArray(), right.Select(position => indices[position]).ToArray());
    }

    /// <summary>
    /// Makes a prediction using a single tree for a row of a matrix.
    /// </summary>
    private T PredictTree(DARTTree tree, Matrix<T> x, int row)
    {
        if (tree.IsConstant)
        {
            return tree.ConstantValue;
        }

        while (!tree.IsLeaf)
        {
            if (!NumOps.GreaterThan(x[row, tree.SplitFeature], tree.SplitThreshold))
            {
                if (tree.Left is null) break;
                tree = tree.Left;
            }
            else
            {
                if (tree.Right is null) break;
                tree = tree.Right;
            }
        }

        return tree.LeafValue;
    }

    /// <summary>
    /// Makes a prediction using a single tree for a vector input.
    /// </summary>
    private T PredictTree(DARTTree tree, Vector<T> x)
    {
        if (tree.IsConstant)
        {
            return tree.ConstantValue;
        }

        while (!tree.IsLeaf)
        {
            if (!NumOps.GreaterThan(x[tree.SplitFeature], tree.SplitThreshold))
            {
                if (tree.Left is null) break;
                tree = tree.Left;
            }
            else
            {
                if (tree.Right is null) break;
                tree = tree.Right;
            }
        }

        return tree.LeafValue;
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        // Accumulate split counts for each feature
        foreach (var tree in _trees)
        {
            AccumulateFeatureImportance(tree, importances);
        }

        // Normalize
        var engine = AiDotNetEngine.Current;
        T sum = engine.Sum(importances);
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            importances = engine.Divide(importances, sum);
        }

        FeatureImportances = importances;
        return Task.CompletedTask;
    }

    /// <summary>
    /// Recursively accumulates feature importance from a tree.
    /// </summary>
    private void AccumulateFeatureImportance(DARTTree tree, Vector<T> importances)
    {
        if (tree.IsLeaf || tree.IsConstant)
        {
            return;
        }

        if (tree.SplitFeature >= 0 && tree.SplitFeature < importances.Length)
        {
            importances[tree.SplitFeature] = NumOps.Add(importances[tree.SplitFeature], NumOps.One);
        }

        if (tree.Left != null)
        {
            AccumulateFeatureImportance(tree.Left, importances);
        }
        if (tree.Right != null)
        {
            AccumulateFeatureImportance(tree.Right, importances);
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _trees.Count },
                { "MaxDepth", _options.MaxDepth },
                { "DropoutRate", _options.DropoutRate },
                { "LearningRate", _options.LearningRate },
                { "NumberOfFeatures", _numFeatures }
            }
        };
    }

    private void SerializeTree(BinaryWriter writer, DARTTree tree)
    {
        writer.Write(tree.IsConstant);
        writer.Write(NumOps.ToDouble(tree.ConstantValue));
        writer.Write(tree.IsLeaf);

        if (!tree.IsLeaf && !tree.IsConstant)
        {
            writer.Write(tree.SplitFeature);
            writer.Write(NumOps.ToDouble(tree.SplitThreshold));
            writer.Write(NumOps.ToDouble(tree.LeafValue));

            writer.Write(tree.Left != null);
            if (tree.Left != null)
            {
                SerializeTree(writer, tree.Left);
            }

            writer.Write(tree.Right != null);
            if (tree.Right != null)
            {
                SerializeTree(writer, tree.Right);
            }
        }
        else
        {
            writer.Write(NumOps.ToDouble(tree.LeafValue));
        }
    }

    private DARTTree DeserializeTree(BinaryReader reader)
    {
        var tree = new DARTTree(NumOps.Zero)
        {
            IsConstant = reader.ReadBoolean(),
            ConstantValue = NumOps.FromDouble(reader.ReadDouble()),
            IsLeaf = reader.ReadBoolean()
        };

        if (!tree.IsLeaf && !tree.IsConstant)
        {
            tree.SplitFeature = reader.ReadInt32();
            tree.SplitThreshold = NumOps.FromDouble(reader.ReadDouble());
            tree.LeafValue = NumOps.FromDouble(reader.ReadDouble());

            if (reader.ReadBoolean())
            {
                tree.Left = DeserializeTree(reader);
            }

            if (reader.ReadBoolean())
            {
                tree.Right = DeserializeTree(reader);
            }
        }
        else
        {
            tree.LeafValue = NumOps.FromDouble(reader.ReadDouble());
        }

        return tree;
    }

    /// <summary>
    /// Internal tree structure for DART.
    /// </summary>
    private class DARTTree
    {
        public bool IsConstant { get; set; }
        public T ConstantValue { get; set; }
        public bool IsLeaf { get; set; }
        public int SplitFeature { get; set; }
        public T SplitThreshold { get; set; }
        public T LeafValue { get; set; }
        public DARTTree? Left { get; set; }
        public DARTTree? Right { get; set; }

        public DARTTree(T zero)
        {
            ConstantValue = zero;
            SplitThreshold = zero;
            LeafValue = zero;
        }
    }
}
