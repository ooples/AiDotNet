namespace AiDotNet.Optimizers;

public class AdamOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private AdamOptimizerOptions _options;
    private Vector<T> _m;
    private Vector<T> _v;
    private int _t;
    private T _currentLearningRate;
    private T _currentBeta1;
    private T _currentBeta2;

    public AdamOptimizer(
        AdamOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        _options = options ?? new();
        _currentLearningRate = NumOps.Zero;
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        _currentLearningRate = NumOps.FromDouble(_options.LearningRate);
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();

        _m = new Vector<T>(currentSolution.Coefficients.Length);
        _v = new Vector<T>(currentSolution.Coefficients.Length);
        _t = 0;

        InitializeAdaptiveParameters();

        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Adam-specific adaptive parameter updates
        if (_options.UseAdaptiveLearningRate)
        {
            _currentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), _currentLearningRate));
        }

        if (_options.UseAdaptiveBetas)
        {
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        for (int i = 0; i < gradient.Length; i++)
        {
            _m[i] = NumOps.Add(NumOps.Multiply(_currentBeta1, _m[i]), NumOps.Multiply(NumOps.Subtract(NumOps.One, _currentBeta1), gradient[i]));
            _v[i] = NumOps.Add(NumOps.Multiply(_currentBeta2, _v[i]), NumOps.Multiply(NumOps.Subtract(NumOps.One, _currentBeta2), NumOps.Multiply(gradient[i], gradient[i])));

            var mHat = NumOps.Divide(_m[i], NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta1, NumOps.FromDouble(_t))));
            var vHat = NumOps.Divide(_v[i], NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta2, NumOps.FromDouble(_t))));

            var update = NumOps.Divide(NumOps.Multiply(_currentLearningRate, mHat), NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon)));

            currentSolution.Coefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], update);
        }

        return currentSolution;
    }

    public override Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        for (int i = 0; i < parameters.Length; i++)
        {
            _m[i] = NumOps.Add(
                NumOps.Multiply(_m[i], NumOps.FromDouble(_options.Beta1)),
                NumOps.Multiply(gradient[i], NumOps.FromDouble(1 - _options.Beta1))
            );

            _v[i] = NumOps.Add(
                NumOps.Multiply(_v[i], NumOps.FromDouble(_options.Beta2)),
                NumOps.Multiply(NumOps.Multiply(gradient[i], gradient[i]), NumOps.FromDouble(1 - _options.Beta2))
            );

            T mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
            T vHat = NumOps.Divide(_v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            T update = NumOps.Divide(
                mHat,
                NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
            );

            parameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
            );
        }

        return parameters;
    }

    public override Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Rows * parameters.Columns)
        {
            _m = new Vector<T>(parameters.Rows * parameters.Columns);
            _v = new Vector<T>(parameters.Rows * parameters.Columns);
            _t = 0;
        }

        _t++;

        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        int index = 0;

        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                T g = gradient[i, j];

                _m[index] = NumOps.Add(
                    NumOps.Multiply(_m[index], NumOps.FromDouble(_options.Beta1)),
                    NumOps.Multiply(g, NumOps.FromDouble(1 - _options.Beta1))
                );

                _v[index] = NumOps.Add(
                    NumOps.Multiply(_v[index], NumOps.FromDouble(_options.Beta2)),
                    NumOps.Multiply(NumOps.Multiply(g, g), NumOps.FromDouble(1 - _options.Beta2))
                );

                T mHat = NumOps.Divide(_m[index], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
                T vHat = NumOps.Divide(_v[index], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

                T update = NumOps.Divide(
                    mHat,
                    NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
                );

                updatedMatrix[i, j] = NumOps.Subtract(
                    parameters[i, j],
                    NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
                );

                index++;
            }
        }

        return updatedMatrix;
    }

    public override void Reset()
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is AdamOptimizerOptions adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdamOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize AdamOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize Adam-specific data
            writer.Write(_t);
            writer.Write(_m.Length);
            foreach (var value in _m)
            {
                writer.Write(Convert.ToDouble(value));
            }
            writer.Write(_v.Length);
            foreach (var value in _v)
            {
                writer.Write(Convert.ToDouble(value));
            }

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize AdamOptimizerOptions
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdamOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize Adam-specific data
            _t = reader.ReadInt32();
            int mLength = reader.ReadInt32();
            _m = new Vector<T>(mLength);
            for (int i = 0; i < mLength; i++)
            {
                _m[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            int vLength = reader.ReadInt32();
            _v = new Vector<T>(vLength);
            for (int i = 0; i < vLength; i++)
            {
                _v[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adam_{_options.LearningRate}_{_options.MaxIterations}";
    }
}