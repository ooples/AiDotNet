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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DARTRegression<T> : AsyncDecisionTreeRegressionBase<T>
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

        // Current predictions (starts as mean)
        T meanY = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            meanY = NumOps.Add(meanY, y[i]);
        }
        meanY = NumOps.Divide(meanY, NumOps.FromDouble(n));

        var predictions = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            predictions[i] = meanY;
        }

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
            var residuals = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                residuals[i] = NumOps.Subtract(y[i], droppedPredictions[i]);
            }

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
                foreach (int idx in droppedIndices)
                {
                    _treeWeights[idx] = NumOps.Multiply(_treeWeights[idx], normFactor);
                }
                // New tree gets weight based on learning rate and norm factor
                T newWeight = NumOps.Multiply(lr, normFactor);
                _trees.Add(newTree);
                _treeWeights.Add(newWeight);
            }
            else // Forest normalization
            {
                T droppedSum = NumOps.Zero;
                foreach (int idx in droppedIndices)
                {
                    droppedSum = NumOps.Add(droppedSum, _treeWeights[idx]);
                }
                T forestNorm = NumOps.GreaterThan(droppedSum, NumOps.Zero)
                    ? NumOps.Divide(droppedSum, NumOps.Add(droppedSum, lr))
                    : NumOps.One;

                foreach (int idx in droppedIndices)
                {
                    _treeWeights[idx] = NumOps.Multiply(_treeWeights[idx], forestNorm);
                }

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
        var treePreds = new Vector<T>(_trees.Count);

        for (int t = 0; t < _trees.Count; t++)
        {
            treePreds[t] = NumOps.Multiply(PredictTree(_trees[t], input), _treeWeights[t]);
        }

        return treePreds;
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
            T maxWeight = _treeWeights[0];
            for (int i = 1; i < _treeWeights.Count; i++)
            {
                if (NumOps.GreaterThan(_treeWeights[i], maxWeight))
                    maxWeight = _treeWeights[i];
            }
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
        var predictions = new Vector<T>(n);

        foreach (int t in treeIndices)
        {
            T weight = _treeWeights[t];
            for (int i = 0; i < n; i++)
            {
                predictions[i] = NumOps.Add(predictions[i],
                    NumOps.Multiply(PredictTree(_trees[t], xData, i), weight));
            }
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
        T sum = NumOps.Zero;
        foreach (int idx in indices)
        {
            sum = NumOps.Add(sum, y[idx]);
        }

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
        int bestFeature = -1;
        T bestThreshold = NumOps.Zero;
        T bestGain = NumOps.FromDouble(double.MinValue);
        T epsilon = NumOps.FromDouble(1e-10);
        T two = NumOps.FromDouble(2);

        // Compute current node's MSE
        T totalSum = NumOps.Zero;
        T totalSumSq = NumOps.Zero;
        foreach (int idx in indices)
        {
            totalSum = NumOps.Add(totalSum, y[idx]);
            totalSumSq = NumOps.Add(totalSumSq, NumOps.Multiply(y[idx], y[idx]));
        }
        T nT = NumOps.FromDouble(indices.Length);
        T totalMse = NumOps.Subtract(totalSumSq, NumOps.Divide(NumOps.Multiply(totalSum, totalSum), nT));

        foreach (int f in featureIndices)
        {
            // Sort indices by feature value
            var sortedIndices = indices.OrderBy(i => NumOps.ToDouble(xData[i, f])).ToArray();

            T leftSum = NumOps.Zero;
            T leftSumSq = NumOps.Zero;
            int leftCount = 0;

            for (int i = 0; i < sortedIndices.Length - 1; i++)
            {
                int idx = sortedIndices[i];
                leftSum = NumOps.Add(leftSum, y[idx]);
                leftSumSq = NumOps.Add(leftSumSq, NumOps.Multiply(y[idx], y[idx]));
                leftCount++;

                int rightCount = indices.Length - leftCount;

                // Skip if below minimum samples
                if (leftCount < _options.MinSamplesLeaf || rightCount < _options.MinSamplesLeaf)
                {
                    continue;
                }

                // Skip if same feature value as next point
                T diff = NumOps.Abs(NumOps.Subtract(xData[sortedIndices[i], f], xData[sortedIndices[i + 1], f]));
                if (NumOps.LessThan(diff, epsilon))
                {
                    continue;
                }

                T rightSum = NumOps.Subtract(totalSum, leftSum);
                T rightSumSq = NumOps.Subtract(totalSumSq, leftSumSq);
                T leftN = NumOps.FromDouble(leftCount);
                T rightN = NumOps.FromDouble(rightCount);

                T leftMse = NumOps.Subtract(leftSumSq, NumOps.Divide(NumOps.Multiply(leftSum, leftSum), leftN));
                T rightMse = NumOps.Subtract(rightSumSq, NumOps.Divide(NumOps.Multiply(rightSum, rightSum), rightN));

                T gain = NumOps.Subtract(totalMse, NumOps.Add(leftMse, rightMse));

                if (NumOps.GreaterThan(gain, bestGain))
                {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = NumOps.Divide(
                        NumOps.Add(xData[sortedIndices[i], f], xData[sortedIndices[i + 1], f]), two);
                }
            }
        }

        return (bestFeature, bestThreshold, bestGain);
    }

    /// <summary>
    /// Splits data based on a feature and threshold.
    /// </summary>
    private (int[] left, int[] right) SplitData(Matrix<T> xData, int[] indices, int feature, T threshold)
    {
        var left = new List<int>();
        var right = new List<int>();

        foreach (int idx in indices)
        {
            if (!NumOps.GreaterThan(xData[idx, feature], threshold))
            {
                left.Add(idx);
            }
            else
            {
                right.Add(idx);
            }
        }

        return ([.. left], [.. right]);
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
        T sum = NumOps.Zero;
        for (int f = 0; f < _numFeatures; f++)
        {
            sum = NumOps.Add(sum, importances[f]);
        }
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], sum);
            }
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
            ModelType = ModelType.DART,
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

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Options
        writer.Write(_options.NumberOfIterations);
        writer.Write(_options.LearningRate);
        writer.Write(_options.MaxDepth);
        writer.Write(_options.DropoutRate);
        writer.Write(_numFeatures);

        // Trees
        writer.Write(_trees.Count);
        foreach (var tree in _trees)
        {
            SerializeTree(writer, tree);
        }

        // Tree weights
        foreach (var weight in _treeWeights)
        {
            writer.Write(NumOps.ToDouble(weight));
        }

        return ms.ToArray();
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

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        base.Deserialize(reader.ReadBytes(baseLen));

        _options.NumberOfIterations = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.MaxDepth = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _numFeatures = reader.ReadInt32();

        int numTrees = reader.ReadInt32();
        _trees = [];
        for (int t = 0; t < numTrees; t++)
        {
            _trees.Add(DeserializeTree(reader));
        }

        _treeWeights = [];
        for (int t = 0; t < numTrees; t++)
        {
            _treeWeights.Add(NumOps.FromDouble(reader.ReadDouble()));
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

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DARTRegression<T>(_options, Regularization);
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
