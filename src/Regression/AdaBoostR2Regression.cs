namespace AiDotNet.Regression;

public class AdaBoostR2Regression<T> : AsyncDecisionTreeRegressionBase<T>
{
    private AdaBoostR2RegressionOptions _options;
    private List<(DecisionTreeRegression<T> Tree, T Weight)> _ensemble;
    private Random _random;

    public override int NumberOfTrees => _options.NumberOfEstimators;
    public override int MaxDepth => _options.MaxDepth;

    public AdaBoostR2Regression(AdaBoostR2RegressionOptions options, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _ensemble = [];
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _ensemble.Clear();
        var sampleWeights = Vector<T>.CreateDefault(y.Length, NumOps.One);
        var numSamples = x.Rows;

        for (int i = 0; i < _options.NumberOfEstimators; i++)
        {
            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                Seed = _random.Next(),
                SplitCriterion = _options.SplitCriterion
            };

            var tree = new DecisionTreeRegression<T>(treeOptions, Regularization);
            tree.TrainWithWeights(x, y, sampleWeights);

            var predictions = tree.Predict(x);
            var errors = CalculateErrors(y, predictions);
            var averageError = CalculateAverageError(errors, sampleWeights);

            if (NumOps.GreaterThanOrEquals(averageError, NumOps.FromDouble(0.5)))
            {
                break; // Stop if the error is too high
            }

            var beta = NumOps.Divide(averageError, NumOps.Subtract(NumOps.One, averageError));
            var weight = NumOps.Log(NumOps.Divide(NumOps.One, beta));

            _ensemble.Add((tree, weight));

            // Update sample weights
            sampleWeights = UpdateSampleWeights(sampleWeights, errors, beta);
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var regularizedInput = Regularization.RegularizeMatrix(input);
        var sumWeights = _ensemble.Aggregate(NumOps.Zero, (acc, e) => NumOps.Add(acc, e.Weight));
        var result = new T[input.Rows];

        var tasks = _ensemble.Select(treeWeight => Task.Run(() =>
        {
            var (tree, weight) = treeWeight;
            var prediction = tree.Predict(regularizedInput);
            return (prediction, weight);
        }));

        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = predictions.Aggregate(NumOps.Zero, (acc, p) => 
                NumOps.Add(acc, NumOps.Multiply(p.prediction[i], p.weight)));
            result[i] = NumOps.Divide(result[i], sumWeights);
        }

        var finalPredictions = new Vector<T>(result, NumOps);
        return Regularization.RegularizeCoefficients(finalPredictions);
    }

    private Vector<T> CalculateErrors(Vector<T> y, Vector<T> predictions)
    {
        return new Vector<T>(y.Select((yi, i) => NumOps.Abs(NumOps.Subtract(yi, predictions[i]))));
    }

    private T CalculateAverageError(Vector<T> errors, Vector<T> sampleWeights)
    {
        var maxError = errors.Max();
        var weightedErrors = errors.Select((e, i) => 
            NumOps.Multiply(NumOps.Divide(e, maxError), sampleWeights[i]));

        return NumOps.Divide(weightedErrors.Aggregate(NumOps.Zero, NumOps.Add), sampleWeights.Sum());
    }

    private Vector<T> UpdateSampleWeights(Vector<T> sampleWeights, Vector<T> errors, T beta)
    {
        var maxError = errors.Max();
        var updatedWeights = sampleWeights.Select((w, i) => 
            NumOps.Multiply(w, NumOps.Power(beta, NumOps.Subtract(NumOps.One, NumOps.Divide(errors[i], maxError)))));
        var sumWeights = updatedWeights.Aggregate(NumOps.Zero, NumOps.Add);

        return new Vector<T>(updatedWeights.Select(w => NumOps.Divide(w, sumWeights)));
    }

    protected override async Task CalculateFeatureImportancesAsync(int numFeatures)
    {
        var importances = new T[numFeatures];
        var totalWeight = _ensemble.Aggregate(NumOps.Zero, (acc, e) => NumOps.Add(acc, e.Weight));

        var tasks = _ensemble.Select(treeWeight => Task.Run(() =>
        {
            var (tree, weight) = treeWeight;
            var treeImportances = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                treeImportances[i] = NumOps.Multiply(tree.GetFeatureImportance(i), NumOps.Divide(weight, totalWeight));
            }
            return treeImportances;
        }));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = allImportances.Aggregate(NumOps.Zero, (acc, imp) => NumOps.Add(acc, imp[i]));
        }

        FeatureImportances = new Vector<T>(importances);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.AdaBoostR2,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfEstimators", _options.NumberOfEstimators },
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "MaxFeatures", _options.MaxFeatures },
                { "FeatureImportances", FeatureImportances },
                { "RegularizationType", Regularization.GetType().Name }
            }
        };
    }

    public override byte[] Serialize()
    {
        var serializableModel = new
        {
            Options = _options,
            Ensemble = _ensemble.Select(e => new 
            { 
                Tree = Convert.ToBase64String(e.Tree.Serialize()),
                Weight = e.Weight
            }).ToList(),
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
            Options = new AdaBoostR2RegressionOptions(),
            Ensemble = new List<dynamic>(),
            Regularization = ""
        });

        if (deserializedModel == null)
        {
            throw new InvalidOperationException("Failed to deserialize the model");
        }

        _options = deserializedModel.Options;

        _ensemble = [.. deserializedModel.Ensemble.Select(e =>
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
            tree.Deserialize(Convert.FromBase64String((string)e.Tree));
            return (Tree: tree, Weight: (T)e.Weight);
        })];

        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }
}