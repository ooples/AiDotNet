namespace AiDotNet.Optimizers;

public class TrustRegionOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private TrustRegionOptimizerOptions _options;
    private T _trustRegionRadius;
    private int _iteration;

    public TrustRegionOptimizer(
        TrustRegionOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new TrustRegionOptimizerOptions();
        _trustRegionRadius = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _trustRegionRadius = NumOps.FromDouble(_options.InitialTrustRegionRadius);
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

            var stepDirection = SolveSubproblem(gradient, hessian);
            var proposedSolution = MoveInDirection(currentSolution, stepDirection, NumOps.One);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            var proposedStepData = EvaluateSolution(proposedSolution, inputData);

            var actualReduction = NumOps.Subtract(currentStepData.FitnessScore, proposedStepData.FitnessScore);
            var predictedReduction = CalculatePredictedReduction(gradient, hessian, stepDirection);

            var rho = NumOps.Divide(actualReduction, predictedReduction);

            if (NumOps.GreaterThan(rho, NumOps.FromDouble(_options.AcceptanceThreshold)))
            {
                currentSolution = proposedSolution;
                currentStepData = proposedStepData;
                UpdateTrustRegionRadius(rho);
            }
            else
            {
                ShrinkTrustRegionRadius();
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

    private Matrix<T> CalculateHessian(ISymbolicModel<T> currentSolution, OptimizationInputData<T> inputData)
    {
        var coefficients = currentSolution.Coefficients;
        var hessian = new Matrix<T>(coefficients.Length, coefficients.Length);
        var epsilon = NumOps.FromDouble(1e-8);

        for (int i = 0; i < coefficients.Length; i++)
        {
            for (int j = i; j < coefficients.Length; j++)
            {
                var perturbed1 = currentSolution.Copy();
                var perturbed2 = currentSolution.Copy();
                var perturbed3 = currentSolution.Copy();
                var perturbed4 = currentSolution.Copy();

                var coeffs1 = coefficients.Copy();
                var coeffs2 = coefficients.Copy();
                var coeffs3 = coefficients.Copy();
                var coeffs4 = coefficients.Copy();

                coeffs1[i] = NumOps.Add(coeffs1[i], epsilon);
                coeffs1[j] = NumOps.Add(coeffs1[j], epsilon);

                coeffs2[i] = NumOps.Add(coeffs2[i], epsilon);
                coeffs2[j] = NumOps.Subtract(coeffs2[j], epsilon);

                coeffs3[i] = NumOps.Subtract(coeffs3[i], epsilon);
                coeffs3[j] = NumOps.Add(coeffs3[j], epsilon);

                coeffs4[i] = NumOps.Subtract(coeffs4[i], epsilon);
                coeffs4[j] = NumOps.Subtract(coeffs4[j], epsilon);

                perturbed1 = perturbed1.UpdateCoefficients(coeffs1);
                perturbed2 = perturbed2.UpdateCoefficients(coeffs2);
                perturbed3 = perturbed3.UpdateCoefficients(coeffs3);
                perturbed4 = perturbed4.UpdateCoefficients(coeffs4);

                var f11 = EvaluateSolution(perturbed1, inputData).FitnessScore;
                var f12 = EvaluateSolution(perturbed2, inputData).FitnessScore;
                var f21 = EvaluateSolution(perturbed3, inputData).FitnessScore;
                var f22 = EvaluateSolution(perturbed4, inputData).FitnessScore;

                var secondDerivative = NumOps.Divide(
                    NumOps.Subtract(
                        NumOps.Add(f11, f22),
                        NumOps.Add(f12, f21)
                    ),
                    NumOps.Multiply(NumOps.FromDouble(4), NumOps.Multiply(epsilon, epsilon))
                );

                hessian[i, j] = secondDerivative;
                hessian[j, i] = secondDerivative;
            }
        }

        return hessian;
    }

    private ISymbolicModel<T> MoveInDirection(ISymbolicModel<T> currentSolution, Vector<T> direction, T stepSize)
    {
        var newModel = currentSolution.Copy();
        var currentCoefficients = newModel.Coefficients;
        var newCoefficients = new Vector<T>(currentCoefficients.Length);

        for (int i = 0; i < currentCoefficients.Length; i++)
        {
            newCoefficients[i] = NumOps.Add(currentCoefficients[i], NumOps.Multiply(direction[i], stepSize));
        }

        return newModel.UpdateCoefficients(newCoefficients);
    }

    private Vector<T> SolveSubproblem(Vector<T> gradient, Matrix<T> hessian)
    {
        var z = new Vector<T>(gradient.Length, NumOps);
        var r = gradient.Copy();
        var d = r.Copy();
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = NumOps.Negate(d[i]);
        }
        var g0 = gradient.DotProduct(gradient);

        for (int i = 0; i < _options.MaxCGIterations; i++)
        {
            var Hd = hessian.Multiply(d);
            var dHd = d.DotProduct(Hd);

            if (NumOps.LessThanOrEquals(dHd, NumOps.Zero))
            {
                return ComputeBoundaryStep(z, d);
            }

            var alpha = NumOps.Divide(r.DotProduct(r), dHd);
            var zNext = z.Add(d.Multiply(alpha));

            if (NumOps.GreaterThan(zNext.Norm(), _trustRegionRadius))
            {
                return ComputeBoundaryStep(z, d);
            }

            z = zNext;
            var rNext = r.Add(Hd.Multiply(alpha));
            var beta = NumOps.Divide(rNext.DotProduct(rNext), r.DotProduct(r));
        
            var dNext = rNext.Copy();
            for (int j = 0; j < dNext.Length; j++)
            {
                dNext[j] = NumOps.Negate(dNext[j]);
            }
            d = dNext.Add(d.Multiply(beta));
        
            r = rNext;

            if (NumOps.LessThan(r.Norm(), NumOps.Multiply(NumOps.FromDouble(_options.CGTolerance), g0)))
            {
                break;
            }
        }

        return z;
    }

    private Vector<T> ComputeBoundaryStep(Vector<T> z, Vector<T> d)
    {
        var a = d.DotProduct(d);
        var b = NumOps.Multiply(NumOps.FromDouble(2), z.DotProduct(d));
        var c = NumOps.Subtract(z.DotProduct(z), NumOps.Multiply(_trustRegionRadius, _trustRegionRadius));
        var tau = SolveQuadratic(a, b, c);

        return z.Add(d.Multiply(tau));
    }

    private T SolveQuadratic(T a, T b, T c)
    {
        var discriminant = NumOps.Subtract(NumOps.Multiply(b, b), NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(4), a), c));
        var sqrtDiscriminant = NumOps.Sqrt(discriminant);
        var tau1 = NumOps.Divide(NumOps.Add(NumOps.Negate(b), sqrtDiscriminant), NumOps.Multiply(NumOps.FromDouble(2), a));
        var tau2 = NumOps.Divide(NumOps.Subtract(NumOps.Negate(b), sqrtDiscriminant), NumOps.Multiply(NumOps.FromDouble(2), a));

        return NumOps.GreaterThan(tau1, NumOps.Zero) ? tau1 : tau2;
    }

    private T CalculatePredictedReduction(Vector<T> gradient, Matrix<T> hessian, Vector<T> stepDirection)
    {
        var linearTerm = gradient.DotProduct(stepDirection);
        var quadraticTerm = stepDirection.DotProduct(hessian.Multiply(stepDirection));
        return NumOps.Add(linearTerm, NumOps.Multiply(NumOps.FromDouble(0.5), quadraticTerm));
    }

    private void UpdateTrustRegionRadius(T rho)
    {
        if (NumOps.GreaterThan(rho, NumOps.FromDouble(_options.VerySuccessfulThreshold)))
        {
            _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ExpansionFactor));
        }
        else if (NumOps.LessThan(rho, NumOps.FromDouble(_options.UnsuccessfulThreshold)))
        {
            _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ContractionFactor));
        }

        _trustRegionRadius = MathHelper.Clamp(_trustRegionRadius, NumOps.FromDouble(_options.MinTrustRegionRadius), NumOps.FromDouble(_options.MaxTrustRegionRadius));
    }

    private void ShrinkTrustRegionRadius()
    {
        _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ContractionFactor));
        _trustRegionRadius = MathHelper.Clamp(_trustRegionRadius, NumOps.FromDouble(_options.MinTrustRegionRadius), NumOps.FromDouble(_options.MaxTrustRegionRadius));
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveTrustRegionRadius)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.Add(NumOps.One, adaptationRate));
            }
            else
            {
                _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.Subtract(NumOps.One, adaptationRate));
            }

            _trustRegionRadius = MathHelper.Clamp(_trustRegionRadius, NumOps.FromDouble(_options.MinTrustRegionRadius), NumOps.FromDouble(_options.MaxTrustRegionRadius));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is TrustRegionOptimizerOptions trustRegionOptions)
        {
            _options = trustRegionOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected TrustRegionOptimizerOptions.");
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
            writer.Write(Convert.ToDouble(_trustRegionRadius));

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
            _options = JsonConvert.DeserializeObject<TrustRegionOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _trustRegionRadius = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}