namespace AiDotNet.Optimizers;

public class ADMMOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private ADMMOptimizerOptions _options;
    private int _iteration;
    private IRegularization<T> _regularization;
    private Vector<T> _z;
    private Vector<T> _u;

    public ADMMOptimizer(
        ADMMOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null,
        IRegularization<T>? regularization = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new ADMMOptimizerOptions();
        _regularization = regularization ?? new NoRegularization<T>();
        _z = Vector<T>.Empty();
        _u = Vector<T>.Empty();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _iteration = 0;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        _z = new Vector<T>(currentSolution.Coefficients.Length);
        _u = new Vector<T>(currentSolution.Coefficients.Length);

        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;

            // ADMM steps
            currentSolution = UpdateX(currentSolution, inputData.XTrain, inputData.YTrain);
            UpdateZ(currentSolution.Coefficients);
            UpdateU(currentSolution.Coefficients);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (CheckConvergence(currentSolution.Coefficients))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateX(ISymbolicModel<T> currentSolution, Matrix<T> X, Vector<T> y)
    {
        // Solve (X^T X + rho I)x = X^T y + rho(z - u)
        var XTranspose = X.Transpose();
        var XTX = XTranspose.Multiply(X);
        var rhoI = Matrix<T>.CreateIdentity(XTX.Rows).Multiply(NumOps.FromDouble(_options.Rho));
        var leftSide = XTX.Add(rhoI);

        var XTy = XTranspose.Multiply(y);
        var zMinusU = _z.Subtract(_u);
        var rhoZMinusU = zMinusU.Multiply(NumOps.FromDouble(_options.Rho));
        var rightSide = XTy.Add(rhoZMinusU);
        var newCoefficients = MatrixSolutionHelper.SolveLinearSystem(leftSide, rightSide, _options.DecompositionType);

        return new VectorModel<T>(newCoefficients);
    }

    private void UpdateZ(Vector<T> x)
    {
        var xPlusU = x.Add(_u);
        var scaledXPlusU = xPlusU.Multiply(NumOps.FromDouble(1.0 / _options.Rho));
        _z = _regularization.RegularizeCoefficients(scaledXPlusU);
    }

    private void UpdateU(Vector<T> x)
    {
        _u = _u.Add(x.Subtract(_z));
    }

    private bool CheckConvergence(Vector<T> x)
    {
        var primalResidual = x.Subtract(_z);
        var dualResidual = _z.Subtract(_z.Subtract(_u));

        var primalNorm = primalResidual.Norm();
        var dualNorm = dualResidual.Norm();

        return NumOps.LessThan(primalNorm, NumOps.FromDouble(_options.AbsoluteTolerance)) &&
               NumOps.LessThan(dualNorm, NumOps.FromDouble(_options.AbsoluteTolerance));
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveRho)
        {
            var primalResidual = currentStepData.Solution.Coefficients.Subtract(_z);
            var dualResidual = _z.Subtract(_z.Subtract(_u));

            var primalNorm = primalResidual.Norm();
            var dualNorm = dualResidual.Norm();

            if (NumOps.GreaterThan(primalNorm, NumOps.Multiply(NumOps.FromDouble(_options.AdaptiveRhoFactor), dualNorm)))
            {
                _options.Rho *= _options.AdaptiveRhoIncrease;
            }
            else if (NumOps.GreaterThan(dualNorm, NumOps.Multiply(NumOps.FromDouble(_options.AdaptiveRhoFactor), primalNorm)))
            {
                _options.Rho /= _options.AdaptiveRhoDecrease;
            }
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is ADMMOptimizerOptions admmOptions)
        {
            _options = admmOptions;
            _regularization = GetRegularizationFromOptions(admmOptions);
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ADMMOptimizerOptions.");
        }
    }

    private IRegularization<T> GetRegularizationFromOptions(ADMMOptimizerOptions options)
    {
        return options.RegularizationType switch
        {
            RegularizationType.L1 => new L1Regularization<T>(new RegularizationOptions { Strength = options.RegularizationStrength }),
            RegularizationType.L2 => new L2Regularization<T>(new RegularizationOptions { Strength = options.RegularizationStrength }),
            RegularizationType.ElasticNet => new ElasticNetRegularization<T>(new RegularizationOptions { Strength = options.RegularizationStrength, L1Ratio = options.ElasticNetMixing }),
            _ => new NoRegularization<T>()
        };
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
            writer.Write(_z.Serialize());
            writer.Write(_u.Serialize());

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
            _options = JsonConvert.DeserializeObject<ADMMOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _z = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));
            _u = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));

            _regularization = GetRegularizationFromOptions(_options);
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_ADMM_{_options.Rho}_{_regularization.GetType().Name}_{_options.AbsoluteTolerance}_{_iteration}";
    }
}