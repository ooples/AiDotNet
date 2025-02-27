namespace AiDotNet.Optimizers;

public class NewtonMethodOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private NewtonMethodOptimizerOptions _options;
    private int _iteration;

    public NewtonMethodOptimizer(
        NewtonMethodOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new NewtonMethodOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var hessian = CalculateHessian(currentSolution, inputData);
            var direction = CalculateDirection(gradient, hessian);
            var newSolution = UpdateSolution(currentSolution, direction);

            var currentStepData = EvaluateSolution(newSolution, inputData);
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

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private Vector<T> CalculateDirection(Vector<T> gradient, Matrix<T> hessian)
    {
        try
        {
            var inverseHessian = hessian.Inverse();
            return inverseHessian.Multiply(gradient).Transform(x => NumOps.Negate(x));
        }
        catch (InvalidOperationException)
        {
            // If Hessian is not invertible, fall back to gradient descent
            return gradient.Transform(x => NumOps.Negate(x));
        }
    }

    private Matrix<T> CalculateHessian(ISymbolicModel<T> model, OptimizationInputData<T> inputData)
    {
        int n = model.Coefficients.Length;
        var hessian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                hessian[i, j] = CalculateSecondPartialDerivative(model, inputData, i, j);
            }
        }

        return hessian;
    }

    private T CalculateSecondPartialDerivative(ISymbolicModel<T> model, OptimizationInputData<T> inputData, int i, int j)
    {
        var epsilon = NumOps.FromDouble(1e-5);
        var originalI = model.Coefficients[i];
        var originalJ = model.Coefficients[j];

        // f(x+h, y+h)
        model.Coefficients[i] = NumOps.Add(originalI, epsilon);
        model.Coefficients[j] = NumOps.Add(originalJ, epsilon);
        var fhh = CalculateLoss(model, inputData);

        // f(x+h, y-h)
        model.Coefficients[j] = NumOps.Subtract(originalJ, epsilon);
        var fhm = CalculateLoss(model, inputData);

        // f(x-h, y+h)
        model.Coefficients[i] = NumOps.Subtract(originalI, epsilon);
        model.Coefficients[j] = NumOps.Add(originalJ, epsilon);
        var fmh = CalculateLoss(model, inputData);

        // f(x-h, y-h)
        model.Coefficients[j] = NumOps.Subtract(originalJ, epsilon);
        var fmm = CalculateLoss(model, inputData);

        // Reset coefficients
        model.Coefficients[i] = originalI;
        model.Coefficients[j] = originalJ;

        // Calculate second partial derivative
        var numerator = NumOps.Subtract(NumOps.Add(fhh, fmm), NumOps.Add(fhm, fmh));
        var denominator = NumOps.Multiply(NumOps.FromDouble(4), NumOps.Multiply(epsilon, epsilon));
        return NumOps.Divide(numerator, denominator);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> direction)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], NumOps.Multiply(CurrentLearningRate, direction[i]));
        }
        return new VectorModel<T>(newCoefficients);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate, 
                NumOps.FromDouble(_options.MinLearningRate), 
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is NewtonMethodOptimizerOptions newtonOptions)
        {
            _options = newtonOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NewtonMethodOptimizerOptions.");
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
            _options = JsonConvert.DeserializeObject<NewtonMethodOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
        }
    }
}