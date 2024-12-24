namespace AiDotNet.Regression;

public class QuantileRegressionForests<T> : AsyncDecisionTreeRegressionBase<T>
{
    private readonly QuantileRegressionForestsOptions _options;
    private List<DecisionTreeRegression<T>> _trees;
    private Random _random;

    public override int NumberOfTrees => _options.NumberOfTrees;
    public override int MaxDepth => _options.MaxDepth;

    public QuantileRegressionForests(QuantileRegressionForestsOptions options, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _trees = [];
        _random = new Random(_options.Seed ?? Environment.TickCount);
    }

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _trees.Clear();
        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => new Func<DecisionTreeRegression<T>>(() =>
        {
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                Seed = _random.Next()
            }, Regularization);

            var (sampledX, sampledY) = SampleWithReplacement(x, y);
            tree.Train(sampledX, sampledY);
            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks, _options.MaxDegreeOfParallelism);
        await CalculateFeatureImportancesAsync(x.Columns);
    }

    public async Task<Vector<T>> PredictQuantileAsync(Matrix<T> input, double quantile)
    {
        if (quantile < 0 || quantile > 1)
        {
            throw new ArgumentException("Quantile must be between 0 and 1", nameof(quantile));
        }

        var regularizedInput = Regularization.RegularizeMatrix(input);
        var predictionTasks = _trees.Select(tree => new Func<Vector<T>>(() => tree.Predict(regularizedInput)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks, _options.MaxDegreeOfParallelism);

        var result = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            var samplePredictions = predictions.Select(p => p[i]).OrderBy(v => v).ToList();
            int index = (int)Math.Floor(quantile * (samplePredictions.Count - 1));
            result[i] = samplePredictions[index];
        }

        var quantilePredictions = new Vector<T>(result, NumOps);
        return Regularization.RegularizeCoefficients(quantilePredictions);
    }

    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // For standard prediction, we'll use the median (0.5 quantile)
        return await PredictQuantileAsync(input, 0.5);
    }

    private (Matrix<T> sampledX, Vector<T> sampledY) SampleWithReplacement(Matrix<T> x, Vector<T> y)
    {
        var sampledIndices = new List<int>();
        for (int i = 0; i < x.Rows; i++)
        {
            sampledIndices.Add(_random.Next(0, x.Rows));
        }

        var sampledX = new Matrix<T>(sampledIndices.Select(i => x.GetRow(i)).ToList());
        var sampledY = new Vector<T>([.. sampledIndices.Select(i => y[i])], NumOps);

        return (sampledX, sampledY);
    }

    protected override async Task CalculateFeatureImportancesAsync(int numFeatures)
    {
        FeatureImportances = new Vector<T>(new T[numFeatures], NumOps);
        var importanceTasks = _trees.Select(tree => new Func<Vector<T>>(() => tree.FeatureImportances));
        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks, _options.MaxDegreeOfParallelism);

        for (int i = 0; i < numFeatures; i++)
        {
            FeatureImportances[i] = NumOps.Divide(
                allImportances.Aggregate(NumOps.Zero, (acc, imp) => NumOps.Add(acc, imp[i])),
                NumOps.FromDouble(_trees.Count)
            );
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.QuantileRegressionForests,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _trees.Count },
                { "MaxDepth", _options.MaxDepth }
            },
            FeatureImportances = FeatureImportances
        };
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(_options.NumberOfTrees);
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.Seed ?? -1);
        writer.Write((int)_options.SplitCriterion);
        writer.Write(_options.MaxDegreeOfParallelism);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        foreach (var importance in FeatureImportances)
        {
            writer.Write(Convert.ToDouble(importance));
        }

        // Serialize trees
        writer.Write(_trees.Count);
        foreach (var tree in _trees)
        {
            var treeData = tree.Serialize();
            writer.Write(treeData.Length);
            writer.Write(treeData);
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        _options.NumberOfTrees = reader.ReadInt32();
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.MaxFeatures = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;
        _options.SplitCriterion = (SplitCriterion)reader.ReadInt32();
        _options.MaxDegreeOfParallelism = reader.ReadInt32();

        // Deserialize feature importances
        int featureCount = reader.ReadInt32();
        var importances = new T[featureCount];
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);

        // Deserialize trees
        int treeCount = reader.ReadInt32();
        _trees = new List<DecisionTreeRegression<T>>(treeCount);
        for (int i = 0; i < treeCount; i++)
        {
            int treeDataLength = reader.ReadInt32();
            byte[] treeData = reader.ReadBytes(treeDataLength);
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions(), Regularization);
            tree.Deserialize(treeData);
            _trees.Add(tree);
        }

        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }
}