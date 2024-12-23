namespace AiDotNet.Regression;

public class DecisionTreeRegression<T> : ITreeBasedModel<T>
{
    private readonly DecisionTreeOptions _options;
    private readonly Random _random;
    private DecisionTreeNode<T>? _root;
    private Vector<T> _featureImportances;
    private readonly INumericOperations<T> _numOps;

    public int NumberOfTrees => 1;

    public int MaxDepth => _options.MaxDepth;

    public Vector<T> FeatureImportances => _featureImportances;

    public DecisionTreeRegression(DecisionTreeOptions options)
    {
        _options = options;
        _featureImportances = Vector<T>.Empty();
        _random = new Random(options.Seed ?? Environment.TickCount);
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public void Train(Matrix<T> x, Vector<T> y)
    {
        _root = BuildTree(x, y, 0);
        CalculateFeatureImportances(x);
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i), _root);
        }

        return new Vector<T>(predictions);
    }

    public ModelMetadata<T> GetModelMetadata()
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

    public byte[] Serialize()
    {
        using (var ms = new MemoryStream())
        using (var writer = new BinaryWriter(ms))
        {
            // Serialize options
            writer.Write(_options.MaxDepth);
            writer.Write(_options.MinSamplesSplit);
            writer.Write(_options.MaxFeatures);
            writer.Write(_options.Seed ?? -1);

            // Serialize the tree structure
            SerializeNode(_root, writer);

            return ms.ToArray();
        }
    }

    public void Deserialize(byte[] data)
    {
        using (var ms = new MemoryStream(data))
        using (var reader = new BinaryReader(ms))
        {
            // Deserialize options
            _options.MaxDepth = reader.ReadInt32();
            _options.MinSamplesSplit = reader.ReadInt32();
            _options.MaxFeatures = reader.ReadDouble();
            int seed = reader.ReadInt32();
            _options.Seed = seed == -1 ? null : seed;

            // Deserialize the tree structure
            _root = DeserializeNode(reader);
        }
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
        T bestSplitValue = _numOps.Zero;
        T bestScore = _numOps.MinValue;

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
                    if (_numOps.LessThan(x[i, featureIndex], splitValue))
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

                if (_numOps.GreaterThan(score, bestScore))
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
            if (_numOps.LessThan(x[i, bestFeatureIndex], bestSplitValue))
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
            Right = BuildTree(new Matrix<T>(rightX), new Vector<T>(rightY), depth + 1)
        };

        return node;
    }

    private T PredictSingle(Vector<T> input, DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return _numOps.Zero;
        }

        if (node.IsLeaf)
        {
            return node.Prediction;
        }

        if (_numOps.LessThan(input[node.FeatureIndex], node.SplitValue))
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
        _featureImportances = new Vector<T>([.. Enumerable.Repeat(_numOps.Zero, x.Columns)]);
        CalculateFeatureImportancesRecursive(_root, x.Columns);
        NormalizeFeatureImportances();
    }

    private void CalculateFeatureImportancesRecursive(DecisionTreeNode<T>? node, int numFeatures)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        T nodeImportance = CalculateNodeImportance(node);
        _featureImportances[node.FeatureIndex] = _numOps.Add(_featureImportances[node.FeatureIndex], nodeImportance);

        CalculateFeatureImportancesRecursive(node.Left, numFeatures);
        CalculateFeatureImportancesRecursive(node.Right, numFeatures);
    }

    private T CalculateNodeImportance(DecisionTreeNode<T> node)
    {
        if (node.IsLeaf || node.Left == null || node.Right == null)
        {
            return _numOps.Zero;
        }

        T parentVariance = StatisticsHelper<T>.CalculateVariance(node.Samples.Select(s => s.Target));
        T leftVariance = StatisticsHelper<T>.CalculateVariance(node.Left.Samples.Select(s => s.Target));
        T rightVariance = StatisticsHelper<T>.CalculateVariance(node.Right.Samples.Select(s => s.Target));

        T leftWeight = _numOps.Divide(_numOps.FromDouble(node.Left.Samples.Count), _numOps.FromDouble(node.Samples.Count));
        T rightWeight = _numOps.Divide(_numOps.FromDouble(node.Right.Samples.Count), _numOps.FromDouble(node.Samples.Count));

        T varianceReduction = _numOps.Subtract(parentVariance, _numOps.Add(_numOps.Multiply(leftWeight, leftVariance), _numOps.Multiply(rightWeight, rightVariance)));

        // Normalize by the number of samples to give less weight to deeper nodes
        return _numOps.Multiply(varianceReduction, _numOps.FromDouble(node.Samples.Count));
    }

    private void NormalizeFeatureImportances()
    {
        T sum = _featureImportances.Aggregate(_numOps.Zero, (acc, x) => _numOps.Add(acc, x));
    
        if (_numOps.Equals(sum, _numOps.Zero))
        {
            return;
        }

        for (int i = 0; i < _featureImportances.Length; i++)
        {
            _featureImportances[i] = _numOps.Divide(_featureImportances[i], sum);
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
}