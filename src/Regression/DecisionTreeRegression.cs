namespace AiDotNet.Regression;

public class DecisionTreeRegression<T> : DecisionTreeRegressionBase<T>
{
    private readonly DecisionTreeOptions _options;
    private readonly Random _random;
    private Vector<T> _featureImportances;
    private readonly IRegularization<T> _regularization;

    public override int NumberOfTrees => 1;

    public DecisionTreeRegression(DecisionTreeOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new DecisionTreeOptions();
        _regularization = regularization ?? new NoRegularization<T>();
        _featureImportances = Vector<T>.Empty();
        _random = new Random(_options.Seed ?? Environment.TickCount);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Apply regularization to the input data
        var regularizedX = _regularization.RegularizeMatrix(x);
        var regularizedY = _regularization.RegularizeCoefficients(y);

        Root = BuildTree(regularizedX, regularizedY, 0);
        CalculateFeatureImportances(regularizedX);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        // Apply regularization to the input data
        var regularizedInput = _regularization.RegularizeMatrix(input);

        var predictions = new T[regularizedInput.Rows];
        for (int i = 0; i < regularizedInput.Rows; i++)
        {
            predictions[i] = PredictSingle(regularizedInput.GetRow(i), Root);
        }

        // Apply regularization to the predictions
        return _regularization.RegularizeCoefficients(new Vector<T>(predictions));
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DecisionTree,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "MaxFeatures", _options.MaxFeatures }
            }
        };
    }

    public T GetFeatureImportance(int featureIndex)
    {
        if (_featureImportances.Length == 0)
        {
            throw new InvalidOperationException("Feature importances are not available. Train the model first.");
        }

        if (featureIndex < 0 || featureIndex >= _featureImportances.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index is out of range.");
        }

        return _featureImportances[featureIndex];
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize options
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.Seed ?? -1);

        // Serialize the tree structure
        SerializeNode(Root, writer);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        // Deserialize options
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.MaxFeatures = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize the tree structure
        Root = DeserializeNode(reader);
    }

    public void TrainWithWeights(Matrix<T> x, Vector<T> y, Vector<T> sampleWeights)
    {
        // Validate inputs
        if (x.Rows != y.Length || x.Rows != sampleWeights.Length)
        {
            throw new ArgumentException("Input dimensions mismatch");
        }

        // Initialize the root node
        Root = new DecisionTreeNode<T>();

        // Build the tree recursively
        BuildTreeWithWeights(Root, x, y, sampleWeights, 0);

        // Calculate feature importances
        CalculateFeatureImportances(x);
    }

    private (int featureIndex, T threshold) FindBestSplitWithWeights(Matrix<T> x, Vector<T> y, Vector<T> weights, IEnumerable<int> featureIndices)
    {
        int bestFeatureIndex = -1;
        T bestThreshold = NumOps.Zero;
        T bestScore = NumOps.MinValue;

        foreach (int featureIndex in featureIndices)
        {
            var featureValues = x.GetColumn(featureIndex);
            var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();

            for (int i = 0; i < uniqueValues.Count - 1; i++)
            {
                T threshold = NumOps.Divide(NumOps.Add(uniqueValues[i], uniqueValues[i + 1]), NumOps.FromDouble(2));
                T score = CalculateWeightedSplitScore(featureValues, y, weights, threshold);

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeatureIndex, bestThreshold);
    }

    private T CalculateWeightedSplitScore(Vector<T> featureValues, Vector<T> y, Vector<T> weights, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < featureValues.Length; i++)
        {
            if (NumOps.LessThanOrEquals(featureValues[i], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            return NumOps.MinValue;
        }

        T leftScore = CalculateWeightedVarianceReduction(y.GetElements(leftIndices), weights.GetElements(leftIndices));
        T rightScore = CalculateWeightedVarianceReduction(y.GetElements(rightIndices), weights.GetElements(rightIndices));

        T totalWeight = weights.Sum();
        T leftWeight = weights.GetElements(leftIndices).Sum();
        T rightWeight = weights.GetElements(rightIndices).Sum();

        return NumOps.Subtract(
            NumOps.Multiply(totalWeight, CalculateWeightedVarianceReduction(y, weights)),
            NumOps.Add(
                NumOps.Multiply(leftWeight, leftScore),
                NumOps.Multiply(rightWeight, rightScore)
            )
        );
    }

    private T CalculateWeightedVarianceReduction(Vector<T> y, Vector<T> weights)
    {
        T totalWeight = weights.Sum();
        T weightedMean = NumOps.Divide(y.DotProduct(weights), totalWeight);
        T weightedVariance = NumOps.Zero;

        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], weightedMean);
            weightedVariance = NumOps.Add(weightedVariance, NumOps.Multiply(weights[i], NumOps.Multiply(diff, diff)));
        }

        weightedVariance = NumOps.Divide(weightedVariance, totalWeight);
        return weightedVariance;
    }

    private void BuildTreeWithWeights(DecisionTreeNode<T> node, Matrix<T> x, Vector<T> y, Vector<T> weights, int depth)
    {
        if (depth >= Options.MaxDepth || x.Rows <= Options.MinSamplesSplit)
        {
            // Create a leaf node
            node.Prediction = CalculateWeightedLeafValue(y, weights);
            return;
        }

        int featuresToConsider = (int)Math.Min(x.Columns, Math.Max(1, Options.MaxFeatures * x.Columns));
        var featureIndices = Enumerable.Range(0, x.Columns).OrderBy(_ => _random.Next()).Take(featuresToConsider);

        // Find the best split
        var (featureIndex, threshold) = FindBestSplitWithWeights(x, y, weights, featureIndices);

        if (featureIndex == -1)
        {
            // No valid split found, create a leaf node
            node.Prediction = CalculateWeightedLeafValue(y, weights);
            return;
        }

        // Split the data
        var (leftIndices, rightIndices) = SplitDataWithWeights(x, y, weights, featureIndex, threshold);

        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            // If split results in empty node, create a leaf
            node.Prediction = CalculateWeightedLeafValue(y, weights);
            return;
        }

        // Create child nodes and continue building the tree
        node.FeatureIndex = featureIndex;
        node.Threshold = threshold;
        node.Left = new DecisionTreeNode<T>();
        node.Right = new DecisionTreeNode<T>();

        BuildTreeWithWeights(node.Left, x.GetRows(leftIndices), y.GetElements(leftIndices), weights.GetElements(leftIndices), depth + 1);
        BuildTreeWithWeights(node.Right, x.GetRows(rightIndices), y.GetElements(rightIndices), weights.GetElements(rightIndices), depth + 1);
    }

    private (List<int> leftIndices, List<int> rightIndices) SplitDataWithWeights(Matrix<T> x, Vector<T> y, Vector<T> weights, int featureIndex, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThanOrEquals(x[i, featureIndex], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        return (leftIndices, rightIndices);
    }

    private T CalculateWeightedLeafValue(Vector<T> y, Vector<T> weights)
    {
        T totalWeight = weights.Sum();
        return NumOps.Divide(y.DotProduct(weights), totalWeight);
    }

    private DecisionTreeNode<T>? BuildTree(Matrix<T> x, Vector<T> y, int depth)
    {
        if (depth >= _options.MaxDepth || x.Rows < _options.MinSamplesSplit)
        {
            return new DecisionTreeNode<T>
            {
                IsLeaf = true,
                Prediction = StatisticsHelper<T>.CalculateMean(y)
            };
        }

        int bestFeatureIndex = -1;
        T bestSplitValue = NumOps.Zero;
        T bestScore = NumOps.MinValue;

        int featuresToConsider = (int)Math.Min(x.Columns, Math.Max(1, _options.MaxFeatures * x.Columns));
        var featureIndices = Enumerable.Range(0, x.Columns).OrderBy(_ => _random.Next()).Take(featuresToConsider).ToList();

        foreach (int featureIndex in featureIndices)
        {
            var featureValues = x.GetColumn(featureIndex);
            var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();

            foreach (var splitValue in uniqueValues.Skip(1))
            {
                var leftIndices = new List<int>();
                var rightIndices = new List<int>();

                for (int i = 0; i < x.Rows; i++)
                {
                    if (NumOps.LessThan(x[i, featureIndex], splitValue))
                    {
                        leftIndices.Add(i);
                    }
                    else
                    {
                        rightIndices.Add(i);
                    }
                }

                if (leftIndices.Count == 0 || rightIndices.Count == 0) continue;

                T score = StatisticsHelper<T>.CalculateSplitScore(y, leftIndices, rightIndices, _options.SplitCriterion);

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestFeatureIndex = featureIndex;
                    bestSplitValue = splitValue;
                }
            }
        }

        if (bestFeatureIndex == -1)
        {
            return new DecisionTreeNode<T>
            {
                IsLeaf = true,
                Prediction = StatisticsHelper<T>.CalculateMean(y)
            };
        }

        var leftX = new List<Vector<T>>();
        var leftY = new List<T>();
        var rightX = new List<Vector<T>>();
        var rightY = new List<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThan(x[i, bestFeatureIndex], bestSplitValue))
            {
                leftX.Add(x.GetRow(i));
                leftY.Add(y[i]);
            }
            else
            {
                rightX.Add(x.GetRow(i));
                rightY.Add(y[i]);
            }
        }

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = bestFeatureIndex,
            SplitValue = bestSplitValue,
            Left = BuildTree(new Matrix<T>(leftX), new Vector<T>(leftY), depth + 1),
            Right = BuildTree(new Matrix<T>(rightX), new Vector<T>(rightY), depth + 1),
            LeftSampleCount = leftX.Count,
            RightSampleCount = rightX.Count,
            Samples = [.. x.GetRows().Select((_, i) => new Sample<T>(x.GetRow(i), y[i]))]
        };

        return node;
    }

    private T PredictSingle(Vector<T> input, DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return NumOps.Zero;
        }

        if (node.IsLeaf)
        {
            return node.Prediction;
        }

        if (NumOps.LessThan(input[node.FeatureIndex], node.SplitValue))
        {
            return PredictSingle(input, node.Left);
        }
        else
        {
            return PredictSingle(input, node.Right);
        }
    }

    private void CalculateFeatureImportances(Matrix<T> x)
    {
        _featureImportances = new Vector<T>([.. Enumerable.Repeat(NumOps.Zero, x.Columns)]);
        CalculateFeatureImportancesRecursive(Root, x.Columns);
        NormalizeFeatureImportances();
    }

    private void CalculateFeatureImportancesRecursive(DecisionTreeNode<T>? node, int numFeatures)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        T nodeImportance = CalculateNodeImportance(node);
        _featureImportances[node.FeatureIndex] = NumOps.Add(_featureImportances[node.FeatureIndex], nodeImportance);

        CalculateFeatureImportancesRecursive(node.Left, numFeatures);
        CalculateFeatureImportancesRecursive(node.Right, numFeatures);
    }

    private T CalculateNodeImportance(DecisionTreeNode<T> node)
    {
        if (node.IsLeaf || node.Left == null || node.Right == null)
        {
            return NumOps.Zero;
        }

        T parentVariance = StatisticsHelper<T>.CalculateVariance(node.Samples.Select(s => s.Target));
        T leftVariance = StatisticsHelper<T>.CalculateVariance(node.Left.Samples.Select(s => s.Target));
        T rightVariance = StatisticsHelper<T>.CalculateVariance(node.Right.Samples.Select(s => s.Target));

        T leftWeight = NumOps.Divide(NumOps.FromDouble(node.Left.Samples.Count), NumOps.FromDouble(node.Samples.Count));
        T rightWeight = NumOps.Divide(NumOps.FromDouble(node.Right.Samples.Count), NumOps.FromDouble(node.Samples.Count));

        T varianceReduction = NumOps.Subtract(parentVariance, NumOps.Add(NumOps.Multiply(leftWeight, leftVariance), NumOps.Multiply(rightWeight, rightVariance)));

        // Normalize by the number of samples to give less weight to deeper nodes
        return NumOps.Multiply(varianceReduction, NumOps.FromDouble(node.Samples.Count));
    }

    private void NormalizeFeatureImportances()
    {
        T sum = _featureImportances.Aggregate(NumOps.Zero, (acc, x) => NumOps.Add(acc, x));
    
        if (NumOps.Equals(sum, NumOps.Zero))
        {
            return;
        }

        for (int i = 0; i < _featureImportances.Length; i++)
        {
            _featureImportances[i] = NumOps.Divide(_featureImportances[i], sum);
        }
    }

    private void SerializeNode(DecisionTreeNode<T>? node, BinaryWriter writer)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.FeatureIndex);
        writer.Write(Convert.ToDouble(node.SplitValue));
        writer.Write(Convert.ToDouble(node.Prediction));
        writer.Write(node.IsLeaf);

        SerializeNode(node.Left, writer);
        SerializeNode(node.Right, writer);
    }

    private DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        bool nodeExists = reader.ReadBoolean();
        if (!nodeExists)
        {
            return null;
        }

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = reader.ReadInt32(),
            SplitValue = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T)),
            Prediction = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T)),
            IsLeaf = reader.ReadBoolean(),
            Left = DeserializeNode(reader),
            Right = DeserializeNode(reader)
        };

        return node;
    }

    protected override void CalculateFeatureImportances(int featureCount)
    {
        _featureImportances = new Vector<T>(featureCount);
        T totalImportance = NumOps.Zero;

        // Traverse the tree and calculate feature importances
        CalculateNodeImportance(Root, NumOps.One);

        // Normalize feature importances
        if (!NumOps.Equals(totalImportance, NumOps.Zero))
        {
            for (int i = 0; i < _featureImportances.Length; i++)
            {
                _featureImportances[i] = NumOps.Divide(_featureImportances[i], totalImportance);
            }
        }

        void CalculateNodeImportance(DecisionTreeNode<T>? node, T nodeWeight)
        {
            if (node == null || node.IsLeaf)
            {
                return;
            }

            T improvement = CalculateImpurityImprovement(node);
            T weightedImprovement = NumOps.Multiply(improvement, nodeWeight);

            _featureImportances[node.FeatureIndex] = NumOps.Add(_featureImportances[node.FeatureIndex], weightedImprovement);
            totalImportance = NumOps.Add(totalImportance, weightedImprovement);

            T leftWeight = NumOps.Multiply(nodeWeight, NumOps.FromDouble(node.LeftSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount)));
            T rightWeight = NumOps.Multiply(nodeWeight, NumOps.FromDouble(node.RightSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount)));

            CalculateNodeImportance(node.Left, leftWeight);
            CalculateNodeImportance(node.Right, rightWeight);
        }

        T CalculateImpurityImprovement(DecisionTreeNode<T> node)
        {
            T parentImpurity = CalculateNodeImpurity(node);
            T leftImpurity = CalculateNodeImpurity(node.Left);
            T rightImpurity = CalculateNodeImpurity(node.Right);

            T leftWeight = NumOps.FromDouble(node.LeftSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount));
            T rightWeight = NumOps.FromDouble(node.RightSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount));

            T weightedChildImpurity = NumOps.Add(
                NumOps.Multiply(leftWeight, leftImpurity),
                NumOps.Multiply(rightWeight, rightImpurity)
            );

            return NumOps.Subtract(parentImpurity, weightedChildImpurity);
        }

        T CalculateNodeImpurity(DecisionTreeNode<T>? node)
        {
            if (node == null || node.IsLeaf)
            {
                return NumOps.Zero;
            }

            // For regression trees, we use variance as the impurity measure
            T variance = NumOps.Zero;
            T mean = node.Prediction;
            int sampleCount = node.Samples.Count;

            foreach (var sample in node.Samples)
            {
                T diff = NumOps.Subtract(sample.Target, mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }

            return NumOps.Divide(variance, NumOps.FromDouble(sampleCount));
        }
    }
}