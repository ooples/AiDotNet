namespace AiDotNet.Optimizers;

public class NelderMeadOptimizer<T> : OptimizerBase<T>
{
    private NelderMeadOptimizerOptions _options;
    private int _iteration;
    private T _alpha;
    private T _beta;
    private T _gamma;
    private T _delta;

    public NelderMeadOptimizer(
        NelderMeadOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _options = options ?? new NelderMeadOptimizerOptions();
        _alpha = NumOps.Zero;
        _beta = NumOps.Zero;
        _gamma = NumOps.Zero;
        _delta = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _alpha = NumOps.FromDouble(_options.InitialAlpha);
        _beta = NumOps.FromDouble(_options.InitialBeta);
        _gamma = NumOps.FromDouble(_options.InitialGamma);
        _delta = NumOps.FromDouble(_options.InitialDelta);
        _iteration = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        int n = inputData.XTrain.Columns;
        var simplex = InitializeSimplex(n);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;

            // Sort simplex points by fitness
            simplex = [.. simplex.OrderBy(p => EvaluateSolution(p, inputData).FitnessScore)];

            var centroid = CalculateCentroid(simplex, n);
            var reflected = Reflect(simplex[n], centroid);
            var reflectedStepData = EvaluateSolution(reflected, inputData);

            if (NumOps.GreaterThan(reflectedStepData.FitnessScore, EvaluateSolution(simplex[0], inputData).FitnessScore) &&
                NumOps.LessThan(reflectedStepData.FitnessScore, EvaluateSolution(simplex[n - 1], inputData).FitnessScore))
            {
                simplex[n] = reflected;
            }
            else if (NumOps.GreaterThan(reflectedStepData.FitnessScore, EvaluateSolution(simplex[0], inputData).FitnessScore))
            {
                var expanded = Expand(centroid, reflected);
                var expandedStepData = EvaluateSolution(expanded, inputData);

                simplex[n] = NumOps.GreaterThan(expandedStepData.FitnessScore, reflectedStepData.FitnessScore) ? expanded : reflected;
            }
            else
            {
                var contracted = Contract(simplex[n], centroid);
                var contractedStepData = EvaluateSolution(contracted, inputData);

                if (NumOps.GreaterThan(contractedStepData.FitnessScore, EvaluateSolution(simplex[n], inputData).FitnessScore))
                {
                    simplex[n] = contracted;
                }
                else
                {
                    Shrink(simplex);
                }
            }

            var currentStepData = EvaluateSolution(simplex[0], inputData);
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

    private List<ISymbolicModel<T>> InitializeSimplex(int n)
    {
        var simplex = new List<ISymbolicModel<T>>();
        for (int i = 0; i <= n; i++)
        {
            simplex.Add(InitializeRandomSolution(n));
        }
        return simplex;
    }

    private ISymbolicModel<T> CalculateCentroid(List<ISymbolicModel<T>> simplex, int n)
    {
        var centroidCoefficients = new Vector<T>(n, NumOps);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                centroidCoefficients[j] = NumOps.Add(centroidCoefficients[j], simplex[i].Coefficients[j]);
            }
        }
        for (int j = 0; j < n; j++)
        {
            centroidCoefficients[j] = NumOps.Divide(centroidCoefficients[j], NumOps.FromDouble(n));
        }
        return new VectorModel<T>(centroidCoefficients);
    }

    private ISymbolicModel<T> Reflect(ISymbolicModel<T> worst, ISymbolicModel<T> centroid)
    {
        return PerformVectorOperation(centroid, worst, _alpha, (a, b, c) => NumOps.Add(a, NumOps.Multiply(c, NumOps.Subtract(a, b))));
    }

    private ISymbolicModel<T> Expand(ISymbolicModel<T> centroid, ISymbolicModel<T> reflected)
    {
        return PerformVectorOperation(centroid, reflected, _gamma, (a, b, c) => NumOps.Add(a, NumOps.Multiply(c, NumOps.Subtract(b, a))));
    }

    private ISymbolicModel<T> Contract(ISymbolicModel<T> worst, ISymbolicModel<T> centroid)
    {
        return PerformVectorOperation(centroid, worst, _beta, (a, b, c) => NumOps.Add(a, NumOps.Multiply(c, NumOps.Subtract(b, a))));
    }

    private void Shrink(List<ISymbolicModel<T>> simplex)
    {
        var best = simplex[0];
        for (int i = 1; i < simplex.Count; i++)
        {
            simplex[i] = PerformVectorOperation(best, simplex[i], _delta, (a, b, c) => NumOps.Add(a, NumOps.Multiply(c, NumOps.Subtract(b, a))));
        }
    }

    private ISymbolicModel<T> PerformVectorOperation(ISymbolicModel<T> a, ISymbolicModel<T> b, T factor, Func<T, T, T, T> operation)
    {
        var newCoefficients = new Vector<T>(a.Coefficients.Length, NumOps);
        for (int i = 0; i < a.Coefficients.Length; i++)
        {
            newCoefficients[i] = operation(a.Coefficients[i], b.Coefficients[i], factor);
        }
        return new VectorModel<T>(newCoefficients);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveParameters)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            _alpha = NumOps.Add(_alpha, NumOps.Multiply(adaptationRate, improvement));
            _beta = NumOps.Add(_beta, NumOps.Multiply(adaptationRate, improvement));
            _gamma = NumOps.Add(_gamma, NumOps.Multiply(adaptationRate, improvement));
            _delta = NumOps.Add(_delta, NumOps.Multiply(adaptationRate, improvement));

            _alpha = MathHelper.Clamp(_alpha, NumOps.FromDouble(_options.MinAlpha), NumOps.FromDouble(_options.MaxAlpha));
            _beta = MathHelper.Clamp(_beta, NumOps.FromDouble(_options.MinBeta), NumOps.FromDouble(_options.MaxBeta));
            _gamma = MathHelper.Clamp(_gamma, NumOps.FromDouble(_options.MinGamma), NumOps.FromDouble(_options.MaxGamma));
            _delta = MathHelper.Clamp(_delta, NumOps.FromDouble(_options.MinDelta), NumOps.FromDouble(_options.MaxDelta));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is NelderMeadOptimizerOptions nmOptions)
        {
            _options = nmOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NelderMeadOptimizerOptions.");
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
            writer.Write(Convert.ToDouble(_alpha));
            writer.Write(Convert.ToDouble(_beta));
            writer.Write(Convert.ToDouble(_gamma));
            writer.Write(Convert.ToDouble(_delta));

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
            _options = JsonConvert.DeserializeObject<NelderMeadOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _alpha = NumOps.FromDouble(reader.ReadDouble());
            _beta = NumOps.FromDouble(reader.ReadDouble());
            _gamma = NumOps.FromDouble(reader.ReadDouble());
            _delta = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}