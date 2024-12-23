namespace AiDotNet.Regression;

public class GradientBoostingRegression<T> : IAsyncTreeBasedModel<T>
{
    private List<DecisionTreeRegression<T>> _trees;
    private T _initialPrediction;
    private readonly GradientBoostingRegressionOptions _options;
    private readonly INumericOperations<T> _numOps;
    private Vector<T> _featureImportances;
    private IFitnessCalculator<T> _fitnessCalculator;
    private Random _random;

    public int NumberOfTrees => _trees.Count;
    public int MaxDepth => _options.MaxDepth;
    public Vector<T> FeatureImportances => _featureImportances;

    public GradientBoostingRegression(GradientBoostingRegressionOptions? options = null)
    {
        _options = options ?? new GradientBoostingRegressionOptions();
        _trees = [];
        _numOps = MathHelper.GetNumericOperations<T>();
        _featureImportances = Vector<T>.Empty();
        _initialPrediction = _numOps.Zero;
        _fitnessCalculator = FitnessCalculatorFactory.CreateFitnessCalculator<T>(_options.FitnessCalculatorType);
        _random = new Random(_options.Seed ?? Environment.TickCount);
    }

    public void Train(Matrix<T> x, Vector<T> y)
    {
        TrainAsync(x, y).GetAwaiter().GetResult();
    }

    public async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _initialPrediction = _numOps.Divide(y.Sum(), _numOps.FromDouble(y.Length)); // Mean of y
        var residuals = y.Subtract(Vector<T>.CreateDefault(y.Length, _initialPrediction));

        _featureImportances = new Vector<T>(x.Columns);

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
                residuals[i] = _numOps.Subtract(residuals[i], _numOps.Multiply(_numOps.FromDouble(_options.LearningRate), predictions[i]));
            }

            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks);

        CalculateFeatureImportances();
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        return PredictAsync(input).GetAwaiter().GetResult();
    }

    public async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var predictions = Vector<T>.CreateDefault(input.Rows, _initialPrediction);

        var treePredictions = await ParallelProcessingHelper.ProcessTasksInParallel(
            _trees.Select(tree => Task.Run(() => tree.Predict(input))));

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < _trees.Count; j++)
            {
                predictions[i] = _numOps.Add(predictions[i], _numOps.Multiply(_numOps.FromDouble(_options.LearningRate), treePredictions[j][i]));
            }
        }

        return predictions;
    }

    private void CalculateFeatureImportances()
    {
        _featureImportances = new Vector<T>(new T[_trees[0].FeatureImportances.Length]);
        foreach (var tree in _trees)
        {
            for (int i = 0; i < _featureImportances.Length; i++)
            {
                _featureImportances[i] = _numOps.Add(_featureImportances[i], tree.FeatureImportances[i]);
            }
        }

        // Normalize feature importances
        T sum = _featureImportances.Sum();
        for (int i = 0; i < _featureImportances.Length; i++)
        {
            _featureImportances[i] = _numOps.Divide(_featureImportances[i], sum);
        }
    }

    public ModelMetadata<T> GetModelMetadata()
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

    public byte[] Serialize()
    {
        using (var ms = new MemoryStream())
        using (var writer = new BinaryWriter(ms))
        {
            // Serialize options
            writer.Write(_options.NumberOfTrees);
            writer.Write(_options.MaxDepth);
            writer.Write(_options.MinSamplesSplit);
            writer.Write(_options.LearningRate);
            writer.Write(_options.SubsampleRatio);
            writer.Write(_options.MaxFeatures);
            writer.Write(_options.Seed ?? -1);

            // Serialize initial prediction
            writer.Write(Convert.ToDouble(_initialPrediction));

            // Serialize trees
            writer.Write(_trees.Count);
            foreach (var tree in _trees)
            {
                byte[] treeData = tree.Serialize();
                writer.Write(treeData.Length);
                writer.Write(treeData);
            }

            // Serialize feature importances
            writer.Write(_featureImportances.Length);
            for (int i = 0; i < _featureImportances.Length; i++)
            {
                writer.Write(Convert.ToDouble(_featureImportances[i]));
            }

            return ms.ToArray();
        }
    }

    public void Deserialize(byte[] modelData)
    {
        using (var ms = new MemoryStream(modelData))
        using (var reader = new BinaryReader(ms))
        {
            // Deserialize options
            _options.NumberOfTrees = reader.ReadInt32();
            _options.MaxDepth = reader.ReadInt32();
            _options.MinSamplesSplit = reader.ReadInt32();
            _options.LearningRate = reader.ReadDouble();
            _options.SubsampleRatio = reader.ReadDouble();
            _options.MaxFeatures = reader.ReadDouble();
            int seed = reader.ReadInt32();
            _options.Seed = seed == -1 ? null : seed;

            // Deserialize initial prediction
            _initialPrediction = _numOps.FromDouble(reader.ReadDouble());

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

            // Deserialize feature importances
            int featureCount = reader.ReadInt32();
            T[] importances = new T[featureCount];
            for (int i = 0; i < featureCount; i++)
            {
                importances[i] = _numOps.FromDouble(reader.ReadDouble());
            }
            _featureImportances = new Vector<T>(importances);

            // Reinitialize other fields
            _random = new Random(_options.Seed ?? Environment.TickCount);
            _fitnessCalculator = FitnessCalculatorFactory.CreateFitnessCalculator<T>(_options.FitnessCalculatorType);
        }
    }
}