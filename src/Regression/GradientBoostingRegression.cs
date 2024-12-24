namespace AiDotNet.Regression;

public class GradientBoostingRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    private List<DecisionTreeRegression<T>> _trees;
    private T _initialPrediction;
    private readonly GradientBoostingRegressionOptions _options;
    private Random _random;

    public override int NumberOfTrees => _trees.Count;

    public GradientBoostingRegression(GradientBoostingRegressionOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new();
        _trees = [];
        _initialPrediction = NumOps.Zero;
        _random = new Random(Options.Seed ?? Environment.TickCount);
    }

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        // Apply regularization to the feature matrix
        x = Regularization.RegularizeMatrix(x);

        _initialPrediction = NumOps.Divide(y.Sum(), NumOps.FromDouble(y.Length)); // Mean of y
        var residuals = y.Subtract(Vector<T>.CreateDefault(y.Length, _initialPrediction));

        FeatureImportances = new Vector<T>(x.Columns);

        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => Task.Run(() =>
        {
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                SplitCriterion = _options.SplitCriterion,
                Seed = _random.Next()
            });

            // Subsample the data if SubsampleRatio < 1
            Matrix<T> xSubsample = x;
            Vector<T> ySubsample = residuals;

            if (_options.SubsampleRatio < 1)
            {
                int subsampleSize = (int)(x.Rows * _options.SubsampleRatio);
                int[] sampleIndices = SamplingHelper.SampleWithoutReplacement(x.Rows, subsampleSize);
                xSubsample = x.GetRows(sampleIndices);
                ySubsample = residuals.GetElements(sampleIndices);
            }

            tree.Train(xSubsample, ySubsample);

            // Update residuals
            var predictions = tree.Predict(x);
            for (int i = 0; i < residuals.Length; i++)
            {
                residuals[i] = NumOps.Subtract(residuals[i], NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), predictions[i]));
            }

            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks);

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // Apply regularization to the input matrix
        input = Regularization.RegularizeMatrix(input);

        var predictions = Vector<T>.CreateDefault(input.Rows, _initialPrediction);

        var treePredictions = await ParallelProcessingHelper.ProcessTasksInParallel(
            _trees.Select(tree => Task.Run(() => tree.Predict(input))));

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < _trees.Count; j++)
            {
                predictions[i] = NumOps.Add(predictions[i], NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), treePredictions[j][i]));
            }
        }

        // Apply regularization to the final predictions
        predictions = Regularization.RegularizeCoefficients(predictions);

        return predictions;
    }

    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[featureCount];

        // Calculate importances in parallel for each tree
        var importanceTasks = _trees.Select(tree => Task.Run(() =>
        {
            var treeImportances = new T[featureCount];
            for (int i = 0; i < featureCount; i++)
            {
                treeImportances[i] = tree.FeatureImportances[i];
            }
            return treeImportances;
        }));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks);

        // Aggregate importances
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = allImportances.Aggregate(NumOps.Zero, (acc, treeImportance) => NumOps.Add(acc, treeImportance[i]));
        }

        // Normalize feature importances
        T sum = importances.Aggregate(NumOps.Zero, NumOps.Add);
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.Divide(importances[i], sum);
        }

        FeatureImportances = new Vector<T>(importances);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GradientBoosting,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _options.NumberOfTrees },
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "LearningRate", _options.LearningRate },
                { "SubsampleRatio", _options.SubsampleRatio },
                { "MaxFeatures", _options.MaxFeatures }
            }
        };
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GradientBoostingRegression specific data
        writer.Write(_options.NumberOfTrees);
        writer.Write(_options.LearningRate);
        writer.Write(_options.SubsampleRatio);
        writer.Write(Convert.ToDouble(_initialPrediction));

        // Serialize trees
        writer.Write(_trees.Count);
        foreach (var tree in _trees)
        {
            byte[] treeData = tree.Serialize();
            writer.Write(treeData.Length);
            writer.Write(treeData);
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);
        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GradientBoostingRegression specific data
        _options.NumberOfTrees = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.SubsampleRatio = reader.ReadDouble();
        _initialPrediction = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize trees
        int treeCount = reader.ReadInt32();
        _trees = new List<DecisionTreeRegression<T>>(treeCount);
        for (int i = 0; i < treeCount; i++)
        {
            int treeDataLength = reader.ReadInt32();
            byte[] treeData = reader.ReadBytes(treeDataLength);
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions());
            tree.Deserialize(treeData);
            _trees.Add(tree);
        }

        // Reinitialize other fields
        _random = new Random(_options.Seed ?? Environment.TickCount);
    }
}