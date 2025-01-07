global using AiDotNet.GaussianProcesses;

namespace AiDotNet.Optimizers;

public class BayesianOptimizer<T> : OptimizerBase<T>
{
    private BayesianOptimizerOptions<T> _options;
    private Random _random;
    private Matrix<T> _sampledPoints;
    private Vector<T> _sampledValues;
    private IGaussianProcess<T> _gaussianProcess;

    public BayesianOptimizer(
        BayesianOptimizerOptions<T>? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGaussianProcess<T>? gaussianProcess = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _options = options ?? new BayesianOptimizerOptions<T>();
        _random = new Random(_options.Seed);
        _sampledPoints = Matrix<T>.Empty();
        _sampledValues = Vector<T>.Empty();
        _gaussianProcess = gaussianProcess ?? new StandardGaussianProcess<T>(_options.KernelFunction);
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _sampledPoints = Matrix<T>.Empty();
        _sampledValues = Vector<T>.Empty();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();

        // Initial random sampling
        _sampledPoints = new Matrix<T>(_options.InitialSamples, inputData.XTrain.Columns, NumOps);
        _sampledValues = new Vector<T>(_options.InitialSamples, NumOps);

        for (int i = 0; i < _options.InitialSamples; i++)
        {
            var randomSolution = InitializeRandomSolution(inputData.XTrain.Columns);
            var stepData = EvaluateSolution(randomSolution, inputData);
            UpdateBestSolution(stepData, ref bestStepData);
    
            // Set values in the matrix and vector using indexing
            for (int j = 0; j < randomSolution.Coefficients.Length; j++)
            {
                _sampledPoints[i, j] = randomSolution.Coefficients[j];
            }
            _sampledValues[i] = stepData.FitnessScore;
        }

        for (int iteration = _options.InitialSamples; iteration < _options.MaxIterations; iteration++)
        {
            // Fit Gaussian Process to observed data
            _gaussianProcess.Fit(_sampledPoints, _sampledValues);

            // Find next point to sample using acquisition function
            var nextPoint = OptimizeAcquisitionFunction(inputData.XTrain.Columns);
            var currentSolution = new VectorModel<T>(nextPoint);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            // Resize _sampledPoints and _sampledValues
            int newSize = _sampledPoints.Rows + 1;
            var newSampledPoints = new Matrix<T>(newSize, inputData.XTrain.Columns, NumOps);
            var newSampledValues = new Vector<T>(newSize, NumOps);

            // Copy existing data
            for (int i = 0; i < _sampledPoints.Rows; i++)
            {
                for (int j = 0; j < _sampledPoints.Columns; j++)
                {
                    newSampledPoints[i, j] = _sampledPoints[i, j];
                }
                newSampledValues[i] = _sampledValues[i];
            }

            // Add new point and value
            for (int j = 0; j < nextPoint.Length; j++)
            {
                newSampledPoints[newSize - 1, j] = nextPoint[j];
            }
            newSampledValues[newSize - 1] = currentStepData.FitnessScore;

            // Replace old data with new data
            _sampledPoints = newSampledPoints;
            _sampledValues = newSampledValues;

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private Vector<T> OptimizeAcquisitionFunction(int dimensions)
    {
        Vector<T> bestPoint = Vector<T>.Empty();
        T bestValue = NumOps.MinValue;

        for (int i = 0; i < _options.AcquisitionOptimizationSamples; i++)
        {
            var candidatePoint = GenerateRandomPoint(dimensions);
            var acquisitionValue = CalculateAcquisitionFunction(candidatePoint);

            if (NumOps.GreaterThan(acquisitionValue, bestValue))
            {
                bestPoint = candidatePoint;
                bestValue = acquisitionValue;
            }
        }

        return bestPoint;
    }

    private Vector<T> GenerateRandomPoint(int dimensions)
    {
        var point = new T[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            point[i] = NumOps.FromDouble(_random.NextDouble() * (_options.UpperBound - _options.LowerBound) + _options.LowerBound);
        }

        return new Vector<T>(point);
    }

    private T CalculateAcquisitionFunction(Vector<T> point)
    {
        var (mean, variance) = _gaussianProcess.Predict(point);
        var stdDev = NumOps.Sqrt(variance);

        switch (_options.AcquisitionFunction)
        {
            case AcquisitionFunctionType.UpperConfidenceBound:
                return NumOps.Add(mean, NumOps.Multiply(NumOps.FromDouble(_options.ExplorationFactor), stdDev));
            case AcquisitionFunctionType.ExpectedImprovement:
                var maxObservedValue = _sampledValues.Max();
                var improvement = NumOps.Subtract(mean, maxObservedValue);
                var z = NumOps.Divide(improvement, stdDev);
                var cdf = StatisticsHelper<T>.CalculateNormalCDF(mean, stdDev, z);
                return NumOps.Multiply(improvement, cdf);
            default:
                throw new NotImplementedException("Unsupported acquisition function.");
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is BayesianOptimizerOptions<T> bayesianOptions)
        {
            _options = bayesianOptions;
            _gaussianProcess.UpdateKernel(_options.KernelFunction);
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected BayesianOptimizerOptions.");
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
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize _sampledPoints
            writer.Write(_sampledPoints.Rows);
            writer.Write(_sampledPoints.Columns);
            for (int i = 0; i < _sampledPoints.Rows; i++)
            {
                for (int j = 0; j < _sampledPoints.Columns; j++)
                {
                    writer.Write(Convert.ToDouble(_sampledPoints[i, j]));
                }
            }

            // Serialize _sampledValues
            writer.Write(_sampledValues.Length);
            for (int i = 0; i < _sampledValues.Length; i++)
            {
                writer.Write(Convert.ToDouble(_sampledValues[i]));
            }

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<BayesianOptimizerOptions<T>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _sampledPoints
            int rows = reader.ReadInt32();
            int columns = reader.ReadInt32();
            _sampledPoints = new Matrix<T>(rows, columns, NumOps);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    _sampledPoints[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            // Deserialize _sampledValues
            int valueCount = reader.ReadInt32();
            _sampledValues = new Vector<T>(valueCount, NumOps);
            for (int i = 0; i < valueCount; i++)
            {
                _sampledValues[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            _gaussianProcess = new StandardGaussianProcess<T>(_options.KernelFunction);
            _random = new Random(_options.Seed);
        }
    }
}