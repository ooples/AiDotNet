namespace AiDotNet.Optimizers;

public class PowellOptimizer<T> : OptimizerBase<T>
{
    private PowellOptimizerOptions _options;
    private int _iteration;
    private T _adaptiveStepSize;

    public PowellOptimizer(
        PowellOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _options = options ?? new PowellOptimizerOptions();
        _adaptiveStepSize = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _adaptiveStepSize = NumOps.FromDouble(_options.InitialStepSize);
        _iteration = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        int n = inputData.XTrain.Columns;
        var currentSolution = InitializeRandomSolution(n);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;

            var directions = GenerateDirections(n);
            var newSolution = currentSolution;

            for (int i = 0; i < n; i++)
            {
                var lineSearchResult = LineSearch(newSolution, directions[i], inputData);
                newSolution = lineSearchResult.OptimalSolution;
            }

            var extrapolatedPoint = ExtrapolatePoint(currentSolution, newSolution);
            var extrapolatedStepData = EvaluateSolution(extrapolatedPoint, inputData);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            if (NumOps.GreaterThan(extrapolatedStepData.FitnessScore, currentStepData.FitnessScore))
            {
                currentSolution = extrapolatedPoint;
                currentStepData = extrapolatedStepData;
            }
            else
            {
                currentSolution = newSolution;
                currentStepData = EvaluateSolution(newSolution, inputData);
            }

            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private List<Vector<T>> GenerateDirections(int n)
    {
        var directions = new List<Vector<T>>();
        for (int i = 0; i < n; i++)
        {
            var direction = new Vector<T>(n, NumOps);
            direction[i] = NumOps.One;
            directions.Add(direction);
        }
        return directions;
    }

    private (ISymbolicModel<T> OptimalSolution, T OptimalStep) LineSearch(ISymbolicModel<T> currentSolution, Vector<T> direction, OptimizationInputData<T> inputData)
    {
        var a = NumOps.FromDouble(-1.0);
        var b = NumOps.One;
        var goldenRatio = NumOps.FromDouble((Math.Sqrt(5) - 1) / 2);

        while (NumOps.GreaterThan(NumOps.Subtract(b, a), _adaptiveStepSize))
        {
            var c = NumOps.Subtract(b, NumOps.Multiply(goldenRatio, NumOps.Subtract(b, a)));
            var d = NumOps.Add(a, NumOps.Multiply(goldenRatio, NumOps.Subtract(b, a)));

            var fc = EvaluateSolution(MoveInDirection(currentSolution, direction, c), inputData).FitnessScore;
            var fd = EvaluateSolution(MoveInDirection(currentSolution, direction, d), inputData).FitnessScore;

            if (NumOps.GreaterThan(fc, fd))
            {
                b = d;
            }
            else
            {
                a = c;
            }
        }

        var optimalStep = NumOps.Divide(NumOps.Add(a, b), NumOps.FromDouble(2.0));
        return (MoveInDirection(currentSolution, direction, optimalStep), optimalStep);
    }

    private ISymbolicModel<T> MoveInDirection(ISymbolicModel<T> solution, Vector<T> direction, T step)
    {
        var newCoefficients = new Vector<T>(solution.Coefficients.Length, NumOps);
        for (int i = 0; i < solution.Coefficients.Length; i++)
        {
            newCoefficients[i] = NumOps.Add(solution.Coefficients[i], NumOps.Multiply(direction[i], step));
        }
        return new VectorModel<T>(newCoefficients);
    }

    private ISymbolicModel<T> ExtrapolatePoint(ISymbolicModel<T> oldPoint, ISymbolicModel<T> newPoint)
    {
        var extrapolatedCoefficients = new Vector<T>(oldPoint.Coefficients.Length, NumOps);
        for (int i = 0; i < oldPoint.Coefficients.Length; i++)
        {
            extrapolatedCoefficients[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(2.0), newPoint.Coefficients[i]),
                NumOps.Negate(oldPoint.Coefficients[i])
            );
        }
        return new VectorModel<T>(extrapolatedCoefficients);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveStepSize)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _adaptiveStepSize = NumOps.Multiply(_adaptiveStepSize, NumOps.Add(NumOps.One, adaptationRate));
            }
            else
            {
                _adaptiveStepSize = NumOps.Multiply(_adaptiveStepSize, NumOps.Subtract(NumOps.One, adaptationRate));
            }

            _adaptiveStepSize = MathHelper.Clamp(_adaptiveStepSize, NumOps.FromDouble(_options.MinStepSize), NumOps.FromDouble(_options.MaxStepSize));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is PowellOptimizerOptions powellOptions)
        {
            _options = powellOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected PowellOptimizerOptions.");
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

            writer.Write(_iteration);
            writer.Write(Convert.ToDouble(_adaptiveStepSize));

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
            _options = JsonConvert.DeserializeObject<PowellOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _adaptiveStepSize = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}