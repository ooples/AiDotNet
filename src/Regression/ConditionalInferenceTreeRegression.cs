namespace AiDotNet.Regression;

public class ConditionalInferenceTreeRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    private readonly ConditionalInferenceTreeOptions _options;
    private ConditionalInferenceTreeNode<T>? _root;

    public ConditionalInferenceTreeRegression(ConditionalInferenceTreeOptions options, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
    }

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        var regularizedX = Regularization.RegularizeMatrix(x);
        var regularizedY = Regularization.RegularizeCoefficients(y);

        _root = await BuildTreeAsync(regularizedX, regularizedY, 0);
        await CalculateFeatureImportancesAsync(x.Columns);
    }

    private async Task<ConditionalInferenceTreeNode<T>?> BuildTreeAsync(Matrix<T> x, Vector<T> y, int depth)
    {
        if (x.Rows < _options.MinSamplesSplit || depth >= _options.MaxDepth)
        {
            return new ConditionalInferenceTreeNode<T> { IsLeaf = true, Prediction = y.Mean() };
        }

        var splitResult = await FindBestSplitAsync(x, y);
        if (splitResult == null)
        {
            return new ConditionalInferenceTreeNode<T> { IsLeaf = true, Prediction = y.Mean() };
        }

        var (leftX, leftY, rightX, rightY) = SplitData(x, y, splitResult.Value.Feature, splitResult.Value.Threshold);

        var node = new ConditionalInferenceTreeNode<T>
        {
            FeatureIndex = splitResult.Value.Feature,
            Threshold = splitResult.Value.Threshold,
            PValue = splitResult.Value.PValue
        };

        var buildTasks = new[]
        {
            new Func<Task<ConditionalInferenceTreeNode<T>?>>(() => BuildTreeAsync(leftX, leftY, depth + 1)),
            new Func<Task<ConditionalInferenceTreeNode<T>?>>(() => BuildTreeAsync(rightX, rightY, depth + 1))
        };

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(buildTasks, _options.MaxDegreeOfParallelism);

        node.Left = await results[0];
        node.Right = await results[1];

        return node;
    }

    private (Matrix<T> LeftX, Vector<T> LeftY, Matrix<T> RightX, Vector<T> RightY) SplitData(Matrix<T> x, Vector<T> y, int feature, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThanOrEquals(x[i, feature], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        return (
            x.GetRows(leftIndices),
            y.GetElements(leftIndices),
            x.GetRows(rightIndices),
            y.GetElements(rightIndices)
        );
    }

    private async Task<(int Feature, T Threshold, T PValue)?> FindBestSplitAsync(Matrix<T> x, Vector<T> y)
    {
        var tasks = Enumerable.Range(0, x.Columns)
            .Select(feature => new Func<(int Feature, T Threshold, T PValue)?>(() => FindBestSplitForFeature(x, y, feature)))
            .ToArray();

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, _options.MaxDegreeOfParallelism);
        return results.Where(r => r.HasValue)
                      .OrderBy(r => r!.Value.PValue)
                      .FirstOrDefault();
    }

    private (int Feature, T Threshold, T PValue)? FindBestSplitForFeature(Matrix<T> x, Vector<T> y, int feature)
    {
        var featureValues = x.GetColumn(feature);
        var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();

        if (uniqueValues.Count < 2)
        {
            return null;
        }

        var bestSplit = (Threshold: default(T), PValue: NumOps.MaxValue);

        for (int i = 0; i < uniqueValues.Count - 1; i++)
        {
            var threshold = NumOps.Divide(NumOps.Add(uniqueValues[i], uniqueValues[i + 1]), NumOps.FromDouble(2));
            var leftIndices = featureValues.Select((v, idx) => (Value: v, Index: idx))
                .Where(pair => NumOps.LessThanOrEquals(pair.Value, threshold))
                .Select(pair => pair.Index)
                .ToList();
            var rightIndices = Enumerable.Range(0, y.Length).Except(leftIndices).ToList();

            if (leftIndices.Count == 0 || rightIndices.Count == 0)
            {
                continue;
            }

            var leftY = y.GetElements(leftIndices);
            var rightY = y.GetElements(rightIndices);

            var pValue = StatisticsHelper<T>.CalculatePValue(leftY, rightY, _options.StatisticalTest);

            if (NumOps.LessThan(pValue, bestSplit.PValue))
            {
                bestSplit = (threshold, pValue);
            }
        }

        return NumOps.LessThan(bestSplit.PValue, NumOps.FromDouble(_options.SignificanceLevel))
        ? (Feature: feature, Threshold: bestSplit.Threshold ?? NumOps.Zero, PValue: bestSplit.PValue)
        : null;
    }

    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var regularizedInput = Regularization.RegularizeMatrix(input);
        var tasks = Enumerable.Range(0, input.Rows)
            .Select(i => new Func<T>(() => PredictSingle(regularizedInput.GetRow(i))));

        return new Vector<T>(await ParallelProcessingHelper.ProcessTasksInParallel(tasks, _options.MaxDegreeOfParallelism));
    }

    private T PredictSingle(Vector<T> input)
    {
        var node = _root;
        while (node != null && !node.IsLeaf)
        {
            if (NumOps.LessThanOrEquals(input[node.FeatureIndex], node.Threshold))
            {
                node = (ConditionalInferenceTreeNode<T>?)node.Left;
            }
            else
            {
                node = (ConditionalInferenceTreeNode<T>?)node.Right;
            }
        }

        return node != null ? node.Prediction : NumOps.Zero;
    }

    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[featureCount];
        await CalculateFeatureImportancesRecursiveAsync(_root, importances);
        FeatureImportances = new Vector<T>(importances);
    }

    private async Task CalculateFeatureImportancesRecursiveAsync(ConditionalInferenceTreeNode<T>? node, T[] importances)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        importances[node.FeatureIndex] = NumOps.Add(importances[node.FeatureIndex], NumOps.Subtract(NumOps.One, node.PValue));

        var tasks = new[]
        {
            new Func<Task>(() => CalculateFeatureImportancesRecursiveAsync((ConditionalInferenceTreeNode<T>?)node.Left, importances)),
            new Func<Task>(() => CalculateFeatureImportancesRecursiveAsync((ConditionalInferenceTreeNode<T>?)node.Right, importances))
        };

        await ParallelProcessingHelper.ProcessTasksInParallel(tasks, _options.MaxDegreeOfParallelism);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConditionalInferenceTree,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "SignificanceLevel", _options.SignificanceLevel }
            },
            FeatureImportances = FeatureImportances
        };
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.SignificanceLevel);
        writer.Write(_options.Seed ?? -1);

        // Serialize the tree structure
        SerializeNode(writer, _root);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        foreach (var importance in FeatureImportances)
        {
            writer.Write(Convert.ToDouble(importance));
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.SignificanceLevel = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize the tree structure
        _root = DeserializeNode(reader);

        // Deserialize feature importances
        int importanceCount = reader.ReadInt32();
        var importances = new T[importanceCount];
        for (int i = 0; i < importanceCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);
    }

    private void SerializeNode(BinaryWriter writer, ConditionalInferenceTreeNode<T>? node)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.IsLeaf);
        writer.Write(Convert.ToDouble(node.Prediction));

        if (!node.IsLeaf)
        {
            writer.Write(node.FeatureIndex);
            writer.Write(Convert.ToDouble(node.Threshold));
            writer.Write(Convert.ToDouble(node.PValue));
            SerializeNode(writer, (ConditionalInferenceTreeNode<T>?)node.Left);
            SerializeNode(writer, (ConditionalInferenceTreeNode<T>?)node.Right);
        }
    }

    private ConditionalInferenceTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        if (!reader.ReadBoolean())
        {
            return null;
        }

        var node = new ConditionalInferenceTreeNode<T>
        {
            IsLeaf = reader.ReadBoolean(),
            Prediction = NumOps.FromDouble(reader.ReadDouble())
        };

        if (!node.IsLeaf)
        {
            node.FeatureIndex = reader.ReadInt32();
            node.Threshold = NumOps.FromDouble(reader.ReadDouble());
            node.PValue = NumOps.FromDouble(reader.ReadDouble());
            node.Left = DeserializeNode(reader);
            node.Right = DeserializeNode(reader);
        }

        return node;
    }
}