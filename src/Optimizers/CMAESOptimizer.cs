namespace AiDotNet.Optimizers;

public class CMAESOptimizer<T> : OptimizerBase<T>
{
    private CMAESOptimizerOptions<T> _options;
    private Random _random;
    private Matrix<T> _population;
    private Vector<T> _mean;
    private Matrix<T> _C;
    private Vector<T> _pc;
    private Vector<T> _ps;
    private T _sigma;

    public CMAESOptimizer(
        CMAESOptimizerOptions<T>? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _options = options ?? new CMAESOptimizerOptions<T>();
        _random = new Random(_options.Seed);
        _population = Matrix<T>.Empty();
        _mean = Vector<T>.Empty();
        _C = Matrix<T>.Empty();
        _pc = Vector<T>.Empty();
        _ps = Vector<T>.Empty();
        _sigma = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _population = Matrix<T>.Empty();
        _mean = Vector<T>.Empty();
        _C = Matrix<T>.Empty();
        _pc = Vector<T>.Empty();
        _ps = Vector<T>.Empty();
        _sigma = NumOps.FromDouble(_options.InitialStepSize);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();
        InitializeCMAESParameters(inputData.XTrain.Columns);

        for (int generation = 0; generation < _options.MaxGenerations; generation++)
        {
            var population = GeneratePopulation();
            var fitnessValues = EvaluatePopulation(population, inputData);

            UpdateDistribution(population, fitnessValues);

            var currentSolution = new VectorModel<T>(_mean);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(generation, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(_sigma, NumOps.FromDouble(_options.StopTolerance)))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private void InitializeCMAESParameters(int dimensions)
    {
        _mean = InitializeRandomSolution(dimensions).Coefficients;
        _C = Matrix<T>.CreateIdentity(dimensions);
        _pc = new Vector<T>(dimensions);
        _ps = new Vector<T>(dimensions);
    }

    private Matrix<T> GeneratePopulation()
    {
        int dimensions = _mean.Length;
        var population = new Matrix<T>(_options.PopulationSize, dimensions);

        for (int i = 0; i < _options.PopulationSize; i++)
        {
            var sample = GenerateMultivariateNormalSample(dimensions);
            for (int j = 0; j < dimensions; j++)
            {
                population[i, j] = NumOps.Add(_mean[j], NumOps.Multiply(_sigma, sample[j]));
            }
        }

        return population;
    }

    private Vector<T> GenerateMultivariateNormalSample(int dimensions)
    {
        // Generate a vector of standard normal samples
        var standardNormal = new Vector<T>(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            standardNormal[i] = NumOps.FromDouble(GenerateStandardNormal());
        }

        if (!_C.IsPositiveDefiniteMatrix())
        {
            // If the matrix is not positive definite, add a small value to the diagonal
            var epsilon = NumOps.FromDouble(1e-6);
            for (int i = 0; i < dimensions; i++)
            {
                _C[i, i] = NumOps.Add(_C[i, i], epsilon);
            }
        }

        // Perform Cholesky decomposition of the covariance matrix
        var choleskyDecomposition = new CholeskyDecomposition<T>(_C);

        // Transform the standard normal samples using the Cholesky decomposition
        var lowerTriangular = choleskyDecomposition.L;

        return lowerTriangular.Multiply(standardNormal);
    }

    private double GenerateStandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private Vector<T> EvaluatePopulation(Matrix<T> population, OptimizationInputData<T> inputData)
    {
        var fitnessValues = new Vector<T>(population.Rows);
        for (int i = 0; i < population.Rows; i++)
        {
            var solution = new VectorModel<T>(population.GetRow(i));
            var stepData = EvaluateSolution(solution, inputData);
            fitnessValues[i] = stepData.FitnessScore;
        }

        return fitnessValues;
    }

    private void UpdateDistribution(Matrix<T> population, Vector<T> fitnessValues)
    {
        int dimensions = _mean.Length;
        int lambda = _options.PopulationSize;
        int mu = lambda / 2;

        // Sort and select the best individuals
        var sortedIndices = fitnessValues.Argsort().Reverse().ToArray();
        var selectedPopulation = new Matrix<T>(mu, dimensions);
        for (int i = 0; i < mu; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                selectedPopulation[i, j] = population[sortedIndices[i], j];
            }
        }

        // Calculate weights
        var weights = new Vector<T>(mu);
        T sumWeights = NumOps.Zero;
        for (int i = 0; i < mu; i++)
        {
            weights[i] = NumOps.Log(NumOps.Add(NumOps.FromDouble(mu + 0.5), NumOps.FromDouble(i)));
            weights[i] = NumOps.Subtract(NumOps.FromDouble(mu + 0.5), weights[i]);
            sumWeights = NumOps.Add(sumWeights, weights[i]);
        }
        weights = weights.Divide(sumWeights);

        // Calculate effective mu
        T muEff = NumOps.Zero;
        for (int i = 0; i < mu; i++)
        {
            muEff = NumOps.Add(muEff, NumOps.Square(weights[i]));
        }
        muEff = NumOps.Divide(NumOps.One, muEff);

        // Update mean
        var oldMean = _mean;
        _mean = selectedPopulation.Transpose().Multiply(weights);

        // Calculate learning rates
        T c1 = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.Add(NumOps.FromDouble(Math.Pow(dimensions + 1.3, 2)), muEff));
        T cmu = MathHelper.Min(
            NumOps.Subtract(NumOps.One, c1),
            NumOps.Divide(
                NumOps.Multiply(NumOps.FromDouble(2), NumOps.Add(NumOps.Subtract(muEff, NumOps.FromDouble(2)), NumOps.Divide(NumOps.One, muEff))),
                NumOps.Add(NumOps.FromDouble(Math.Pow(dimensions + 2, 2)), muEff)
            )
        );
        T cc = NumOps.Divide(
            NumOps.Add(NumOps.FromDouble(4), NumOps.Divide(muEff, NumOps.FromDouble(dimensions))),
            NumOps.Add(NumOps.FromDouble(dimensions + 4), NumOps.Multiply(NumOps.FromDouble(2), NumOps.Divide(muEff, NumOps.FromDouble(dimensions))))
        );
        T cs = NumOps.Divide(
            NumOps.Add(muEff, NumOps.FromDouble(2)),
            NumOps.Add(NumOps.FromDouble(dimensions), NumOps.Add(muEff, NumOps.FromDouble(5)))
        );

        // Update evolution paths
        var y = _mean.Subtract(oldMean).Divide(_sigma);
        _ps = _ps.Multiply(NumOps.Subtract(NumOps.One, cs)).Add(
            y.Multiply(NumOps.Sqrt(NumOps.Multiply(cs, NumOps.Subtract(NumOps.FromDouble(2), cs)))).Multiply(NumOps.Sqrt(muEff)));

        T hsig = NumOps.LessThan(
            NumOps.Divide(
                _ps.Norm(),
                NumOps.Sqrt(NumOps.Subtract(NumOps.One, NumOps.Power(NumOps.Subtract(NumOps.One, cs), NumOps.FromDouble(2 * _options.MaxGenerations))))
            ),
            NumOps.FromDouble(1.4 + 2 / (dimensions + 1.0))
        ) ? NumOps.One : NumOps.Zero;

        _pc = _pc.Multiply(NumOps.Subtract(NumOps.One, cc)).Add(
            y.Multiply(NumOps.Sqrt(NumOps.Multiply(cc, NumOps.Subtract(NumOps.FromDouble(2), cc)))).Multiply(NumOps.Sqrt(muEff)).Multiply(hsig));

        // Update covariance matrix
        var artmp = selectedPopulation.Subtract(_mean.Repeat(mu).Reshape(mu, dimensions)).Divide(_sigma);
        _C = _C.Multiply(NumOps.Subtract(NumOps.One, NumOps.Add(c1, cmu)))
            .Add(_pc.OuterProduct(_pc).Multiply(c1))
            .Add(artmp.Transpose().Multiply(weights.CreateDiagonal()).Multiply(artmp).Multiply(cmu));

        // Update step size
        T damps = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2), 
            MathHelper.Max(NumOps.Zero, NumOps.Subtract(NumOps.Sqrt(NumOps.Divide(NumOps.Subtract(muEff, NumOps.One), NumOps.FromDouble(dimensions + 1))), NumOps.One))
        ));
        damps = NumOps.Add(damps, cs);
        _sigma = NumOps.Multiply(_sigma, NumOps.Exp(NumOps.Multiply(
            NumOps.Divide(cs, damps),
            NumOps.Subtract(
                NumOps.Divide(
                    _ps.Norm(),
                    NumOps.Sqrt(NumOps.FromDouble(dimensions))
                ),
                NumOps.One
            )
        )));
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is CMAESOptimizerOptions<T> cmaesOptions)
        {
            _options = cmaesOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected CMAESOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        string optionsJson = JsonConvert.SerializeObject(_options);
        writer.Write(optionsJson);

        // Serialize CMA-ES specific data
        SerializationHelper<T>.SerializeMatrix(writer, _population);
        SerializationHelper<T>.SerializeVector(writer, _mean);
        SerializationHelper<T>.SerializeMatrix(writer, _C);
        SerializationHelper<T>.SerializeVector(writer, _pc);
        SerializationHelper<T>.SerializeVector(writer, _ps);
        SerializationHelper<T>.WriteValue(writer, _sigma);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<CMAESOptimizerOptions<T>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize CMA-ES specific data
        _population = SerializationHelper<T>.DeserializeMatrix(reader);
        _mean = SerializationHelper<T>.DeserializeVector(reader);
        _C = SerializationHelper<T>.DeserializeMatrix(reader);
        _pc = SerializationHelper<T>.DeserializeVector(reader);
        _ps = SerializationHelper<T>.DeserializeVector(reader);
        _sigma = SerializationHelper<T>.ReadValue(reader);

        _random = new Random(_options.Seed);
    }
}