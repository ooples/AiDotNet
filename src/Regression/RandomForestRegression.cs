namespace AiDotNet.Regression;

public class RandomForestRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    private RandomForestRegressionOptions _options;
    private List<DecisionTreeRegression<T>> _trees;
    private Random _random;

    public override int NumberOfTrees => _options.NumberOfTrees;
    public override int MaxDepth => _options.MaxDepth;

    public RandomForestRegression(RandomForestRegressionOptions options, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _trees = [];
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _trees.Clear();
        var numFeatures = x.Columns;
        var numSamples = x.Rows;
        var featuresToConsider = (int)Math.Max(1, Math.Round(_options.MaxFeatures * numFeatures));

        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => Task.Run(() =>
        {
            var bootstrapIndices = GetBootstrapSampleIndices(numSamples);
            var bootstrapX = x.GetRows(bootstrapIndices);
            var bootstrapY = y.GetElements(bootstrapIndices);

            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = featuresToConsider / (double)numFeatures,
                Seed = _random.Next(),
                SplitCriterion = _options.SplitCriterion
            };
            var tree = new DecisionTreeRegression<T>(treeOptions, Regularization);
            tree.Train(bootstrapX, bootstrapY);
            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks);

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var regularizedInput = Regularization.RegularizeMatrix(input);
        var predictionTasks = _trees.Select(tree => Task.Run(() => tree.Predict(regularizedInput)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks);

        var result = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = NumOps.Divide(
                predictions.Aggregate(NumOps.Zero, (acc, p) => NumOps.Add(acc, p[i])),
                NumOps.FromDouble(_trees.Count)
            );
        }

        var regularizedPredictions = new Vector<T>(result, NumOps);
        return Regularization.RegularizeCoefficients(regularizedPredictions);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.RandomForest,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _options.NumberOfTrees },
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "MaxFeatures", _options.MaxFeatures },
                { "FeatureImportances", FeatureImportances },
                { "RegularizationType", Regularization.GetType().Name }
            }
        };
    }

    private int[] GetBootstrapSampleIndices(int numSamples)
    {
        var indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            indices[i] = _random.Next(numSamples);
        }

        return indices;
    }

    protected override async Task CalculateFeatureImportancesAsync(int numFeatures)
    {
        var importances = new T[numFeatures];

        // Calculate importances in parallel for each tree
        var importanceTasks = _trees.Select(tree => Task.Run(() =>
        {
            var treeImportances = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                treeImportances[i] = tree.GetFeatureImportance(i);
            }
            return treeImportances;
        }));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks);

        // Aggregate importances
        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = allImportances.Aggregate(NumOps.Zero, (acc, treeImportance) => NumOps.Add(acc, treeImportance[i]));
        }

        // Normalize importances
        T sum = importances.Aggregate(NumOps.Zero, NumOps.Add);
        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = NumOps.Divide(importances[i], sum);
        }

        FeatureImportances = new Vector<T>(importances);
    }

    public override byte[] Serialize()
    {
        var serializableModel = new
        {
            Options = _options,
            Trees = _trees.Select(tree => Convert.ToBase64String(tree.Serialize())).ToList(),
            Regularization = Regularization.GetType().Name
        };

        var json = JsonConvert.SerializeObject(serializableModel, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    public override void Deserialize(byte[] data)
    {
        var json = Encoding.UTF8.GetString(data);
        var deserializedModel = JsonConvert.DeserializeAnonymousType(json, new
        {
            Options = new RandomForestRegressionOptions(),
            Trees = new List<string>(),
            Regularization = ""
        });

        if (deserializedModel == null)
        {
            throw new InvalidOperationException("Failed to deserialize the model");
        }

        _options = deserializedModel.Options;

        _trees = [.. deserializedModel.Trees.Select(treeData =>
        {
            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                Seed = _options.Seed,
                SplitCriterion = _options.SplitCriterion
            };
            var tree = new DecisionTreeRegression<T>(treeOptions, Regularization);
            tree.Deserialize(Convert.FromBase64String(treeData));
            return tree;
        })];

        // Reinitialize other fields
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }
}