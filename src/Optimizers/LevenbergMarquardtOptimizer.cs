namespace AiDotNet.Optimizers;

public class LevenbergMarquardtOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private LevenbergMarquardtOptimizerOptions<T> _options;
    private int _iteration;
    private T _dampingFactor;

    public LevenbergMarquardtOptimizer(
        LevenbergMarquardtOptimizerOptions<T>? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new LevenbergMarquardtOptimizerOptions<T>();
        _dampingFactor = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _dampingFactor = NumOps.FromDouble(_options.InitialDampingFactor);
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
            var jacobian = CalculateJacobian(currentSolution, inputData.XTrain);
            var residuals = CalculateResiduals(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, jacobian, residuals);

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

    private Matrix<T> CalculateJacobian(ISymbolicModel<T> model, Matrix<T> X)
    {
        int m = X.Rows;
        int n = model.Coefficients.Length;
        var jacobian = new Matrix<T>(m, n);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                jacobian[i, j] = CalculatePartialDerivative(model, X.GetRow(i), j);
            }
        }

        return jacobian;
    }

    private T CalculatePartialDerivative(ISymbolicModel<T> model, Vector<T> x, int paramIndex)
    {
        var epsilon = NumOps.FromDouble(1e-8);
        var originalParam = model.Coefficients[paramIndex];

        model.Coefficients[paramIndex] = NumOps.Add(originalParam, epsilon);
        var yPlus = model.Predict(new Matrix<T>(new[] { x }))[0];

        model.Coefficients[paramIndex] = NumOps.Subtract(originalParam, epsilon);
        var yMinus = model.Predict(new Matrix<T>(new[] { x }))[0];

        model.Coefficients[paramIndex] = originalParam;

        return NumOps.Divide(NumOps.Subtract(yPlus, yMinus), NumOps.FromDouble(2 * 1e-8));
    }

    private Vector<T> CalculateResiduals(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var predictions = model.Predict(X);
        return y.Subtract(predictions);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Matrix<T> jacobian, Vector<T> residuals)
    {
        var jTj = jacobian.Transpose().Multiply(jacobian);
        var diagonal = Matrix<T>.CreateDiagonal(jTj.Diagonal());
        var jTr = jacobian.Transpose().Multiply(residuals);

        var lhs = jTj.Add(diagonal.Multiply(_dampingFactor));
        var delta = SolveLinearSystem(lhs, jTr);

        var newCoefficients = currentSolution.Coefficients.Add(delta);
        return new VectorModel<T>(newCoefficients);
    }

    private Vector<T> SolveLinearSystem(Matrix<T> lhs, Vector<T> jTr)
    {
        if (_options.CustomDecomposition != null)
        {
            // Use the custom decomposition if provided
            return _options.CustomDecomposition.Solve(jTr);
        }
        else
        {
            // Use the default method if no custom decomposition is provided
            try
            {
                return MatrixSolutionHelper.SolveLinearSystem(lhs, jTr, MatrixDecompositionType.Lu);
            }
            catch (InvalidOperationException)
            {
                return MatrixSolutionHelper.SolveLinearSystem(lhs, jTr, MatrixDecompositionType.Svd);
            }
        }
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveDampingFactor)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                _dampingFactor = NumOps.Multiply(_dampingFactor, NumOps.FromDouble(_options.DampingFactorDecreaseFactor));
            }
            else
            {
                _dampingFactor = NumOps.Multiply(_dampingFactor, NumOps.FromDouble(_options.DampingFactorIncreaseFactor));
            }

            _dampingFactor = MathHelper.Clamp(_dampingFactor, 
                NumOps.FromDouble(_options.MinDampingFactor), 
                NumOps.FromDouble(_options.MaxDampingFactor));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is LevenbergMarquardtOptimizerOptions<T> lmOptions)
        {
            _options = lmOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LevenbergMarquardtOptimizerOptions.");
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
            writer.Write(Convert.ToDouble(_dampingFactor));

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
            _options = JsonConvert.DeserializeObject<LevenbergMarquardtOptimizerOptions<T>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _dampingFactor = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}