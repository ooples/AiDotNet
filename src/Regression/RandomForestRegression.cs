namespace AiDotNet.Regression;

public class RandomForestRegression<T> : IAsyncTreeBasedModel<T>
{
    private readonly RandomForestRegressionOptions _options;
    private List<DecisionTreeRegression<T>> _trees;
    private INumericOperations<T> _numOps;
    private Random _random;

    public int NumberOfTrees => _options.NumberOfTrees;
    public int MaxDepth => _options.MaxDepth;
    public Vector<T> FeatureImportances { get; private set; }

    public RandomForestRegression(RandomForestRegressionOptions options)
    {
        _options = options;
        _trees = [];
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
        FeatureImportances = Vector<T>.Empty();
    }

    public async Task TrainAsync(Matrix<T> x, Vector<T> y)
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
                Seed = _random.Next()
            };
            var tree = new DecisionTreeRegression<T>(treeOptions);
            tree.Train(bootstrapX, bootstrapY);
            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks);

        CalculateFeatureImportances(x.Columns);
    }

    public void Train(Matrix<T> x, Vector<T> y)
    {
        TrainAsync(x, y).GetAwaiter().GetResult();
    }

    public async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var predictionTasks = _trees.Select(tree => Task.Run(() => tree.Predict(input)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks);

        var result = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = _numOps.Divide(
                predictions.Aggregate(_numOps.Zero, (acc, p) => _numOps.Add(acc, p[i])),
                _numOps.FromDouble(_trees.Count)
            );
        }

        return new Vector<T>(result, _numOps);
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        return PredictAsync(input).GetAwaiter().GetResult();
    }

    public ModelMetadata<T> GetModelMetadata()
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
                { "FeatureImportances", FeatureImportances }
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

    private void CalculateFeatureImportances(int numFeatures)
    {
        var importances = new T[numFeatures];
        foreach (var tree in _trees)
        {
            for (int i = 0; i < numFeatures; i++)
            {
                importances[i] = _numOps.Add(importances[i], tree.GetFeatureImportance(i));
            }
        }

        // Normalize importances
        T sum = importances.Aggregate(_numOps.Zero, (acc, x) => _numOps.Add(acc, x));
        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = _numOps.Divide(importances[i], sum);
        }

        FeatureImportances = new Vector<T>(importances);
    }

    public byte[] Serialize()
    {
        var serializableModel = new
        {
            Options = _options,
            Trees = _trees.Select(tree => Convert.ToBase64String(tree.Serialize())).ToList()
        };

        var json = JsonConvert.SerializeObject(serializableModel, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    public void Deserialize(byte[] data)
    {
        var json = Encoding.UTF8.GetString(data);
        var deserializedModel = JsonConvert.DeserializeAnonymousType(json, new
        {
            Options = new RandomForestRegressionOptions(),
            Trees = new List<string>()
        });

        if (deserializedModel == null)
        {
            throw new InvalidOperationException("Failed to deserialize the model");
        }

        if (!_options.Equals(deserializedModel.Options))
        {
            throw new InvalidOperationException("Deserialized options do not match the current model's options");
        }

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
            var tree = new DecisionTreeRegression<T>(treeOptions);
            tree.Deserialize(Convert.FromBase64String(treeData));
            return tree;
        })];

        // Reinitialize other fields if necessary
        _numOps = MathHelper.GetNumericOperations<T>();
    }
}