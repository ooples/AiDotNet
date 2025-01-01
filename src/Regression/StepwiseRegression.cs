namespace AiDotNet.Regression;

public class StepwiseRegression<T> : RegressionBase<T>
{
    private readonly StepwiseRegressionOptions<T> _options;
    private readonly IFitnessCalculator<T> _fitnessCalculator;
    private List<int> _selectedFeatures;
    private readonly IModelEvaluator<T> _modelEvaluator;

    public StepwiseRegression(StepwiseRegressionOptions<T>? options = null, 
        PredictionStatsOptions? predictionOptions = null, 
        IFitnessCalculator<T>? fitnessCalculator = null, 
        IRegularization<T>? regularization = null, 
        IModelEvaluator<T>? modelEvaluator = null)
        : base(options, regularization)
    {
        _options = options ?? new StepwiseRegressionOptions<T>();
        _fitnessCalculator = fitnessCalculator ?? new AdjustedRSquaredFitnessCalculator<T>();
        _selectedFeatures = [];
        _modelEvaluator = modelEvaluator ?? new ModelEvaluator<T>();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        ValidationHelper<T>.ValidateInputData(x, y);

        if (_options.Method == StepwiseMethod.Forward)
        {
            ForwardSelection(x, y);
        }
        else if (_options.Method == StepwiseMethod.Backward)
        {
            BackwardElimination(x, y);
        }
        else
        {
            throw new NotSupportedException("Unsupported stepwise method.");
        }

        // Train the final model using selected features
        Matrix<T> selectedX = x.GetColumns(_selectedFeatures);
        var finalRegression = new MultipleRegression<T>(Options, Regularization);
        finalRegression.Train(selectedX, y);

        Coefficients = finalRegression.Coefficients;
        Intercept = finalRegression.Intercept;
    }

    private void ForwardSelection(Matrix<T> x, Vector<T> y)
    {
        List<int> remainingFeatures = [.. Enumerable.Range(0, x.Columns)];
        _selectedFeatures.Clear();

        while (_selectedFeatures.Count < Math.Min(_options.MaxFeatures, x.Columns))
        {
            var (bestFeature, bestScore) = EvaluateFeatures(x, y, remainingFeatures, true);

            if (bestFeature != -1)
            {
                _selectedFeatures.Add(bestFeature);
                remainingFeatures.Remove(bestFeature);

                if (NumOps.LessThan(bestScore, NumOps.FromDouble(_options.MinImprovement)))
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }

    private void BackwardElimination(Matrix<T> x, Vector<T> y)
    {
        _selectedFeatures = [.. Enumerable.Range(0, x.Columns)];

        while (_selectedFeatures.Count > _options.MinFeatures)
        {
            var (worstFeature, bestScore) = EvaluateFeatures(x, y, _selectedFeatures, false);

            if (worstFeature != -1)
            {
                _selectedFeatures.RemoveAt(worstFeature);

                if (NumOps.LessThan(bestScore, NumOps.FromDouble(_options.MinImprovement)))
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }

    private (int bestFeatureIndex, T bestScore) EvaluateFeatures(Matrix<T> x, Vector<T> y, List<int> featuresToEvaluate, bool isForwardSelection)
    {
        int bestFeatureIndex = -1;
        T bestScore = _fitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue;

        for (int i = 0; i < featuresToEvaluate.Count; i++)
        {
            List<int> currentFeatures = [.. _selectedFeatures];
            if (isForwardSelection)
            {
                currentFeatures.Add(featuresToEvaluate[i]);
            }
            else
            {
                currentFeatures.RemoveAt(i);
            }

            Matrix<T> currentX = x.GetColumns(currentFeatures);
            var regression = new MultipleRegression<T>(Options, Regularization);
            regression.Train(currentX, y);

            var input = new ModelEvaluationInput<T>
            {
                InputData = OptimizerHelper.CreateOptimizationInputData(currentX, y, currentX, y, currentX, y)
            };
            var evaluationData = _modelEvaluator.EvaluateModel(input);
            var score = _fitnessCalculator.CalculateFitnessScore(evaluationData);

            if (_fitnessCalculator.IsBetterFitness(score, bestScore))
            {
                bestScore = score;
                bestFeatureIndex = i;
            }
        }

        return (bestFeatureIndex, bestScore);
    }

    protected override ModelType GetModelType()
    {
        return ModelType.StepwiseRegression;
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize StepwiseRegression specific data
        writer.Write((int)_options.Method);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.MinFeatures);
        writer.Write(Convert.ToDouble(_options.MinImprovement));

        // Serialize selected features
        writer.Write(_selectedFeatures.Count);
        foreach (var feature in _selectedFeatures)
        {
            writer.Write(feature);
        }

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize StepwiseRegression specific data
        _options.Method = (StepwiseMethod)reader.ReadInt32();
        _options.MaxFeatures = reader.ReadInt32();
        _options.MinFeatures = reader.ReadInt32();
        _options.MinImprovement = Convert.ToDouble(reader.ReadDouble());

        // Deserialize selected features
        int featureCount = reader.ReadInt32();
        _selectedFeatures = new List<int>(featureCount);
        for (int i = 0; i < featureCount; i++)
        {
            _selectedFeatures.Add(reader.ReadInt32());
        }
    }
}