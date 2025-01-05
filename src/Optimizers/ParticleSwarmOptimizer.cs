namespace AiDotNet.Optimizers;

public class ParticleSwarmOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random;
    private ParticleSwarmOptimizationOptions _psoOptions;
    private double _currentInertia;
    private double _currentCognitiveWeight;
    private double _currentSocialWeight;

    public ParticleSwarmOptimizer(
        ParticleSwarmOptimizationOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _random = new Random();
        _psoOptions = options ?? new ParticleSwarmOptimizationOptions();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        _currentInertia = _psoOptions.InitialInertia;
        _currentCognitiveWeight = _psoOptions.InitialCognitiveWeight;
        _currentSocialWeight = _psoOptions.InitialSocialWeight;
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        int dimensions = inputData.XTrain.Columns;
        var swarm = InitializeSwarm(dimensions, _psoOptions.SwarmSize);
        var velocities = InitializeVelocities(dimensions, _psoOptions.SwarmSize);
        var personalBests = new List<OptimizationStepData<T>>();
        var globalBest = new OptimizationStepData<T>();

        // Initialize personal bests and global best
        for (int i = 0; i < _psoOptions.SwarmSize; i++)
        {
            var particle = swarm[i];
            var velocity = velocities[i].ToVector();
            var position = particle.ToVector();
            var personalBest = personalBests[i].Solution.ToVector();
            var globalBestVector = globalBest.Solution.ToVector();

            // Update velocity
            UpdateVelocity(velocity, position, personalBest, globalBestVector);

            // Update particle position
            for (int j = 0; j < velocity.Length; j++)
            {
                position[j] = NumOps.Add(position[j], velocity[j]);
            }

            // Update particle model
            particle.UpdateFromVelocity(velocity);

            // Evaluate new position
            var newStepData = EvaluateSolution(particle, inputData);

            // Update personal best
            if (_fitnessCalculator.IsBetterFitness(newStepData.FitnessScore, personalBests[i].FitnessScore))
            {
                personalBests[i] = newStepData;
            }

            // Update global best
            UpdateGlobalBest(newStepData, ref globalBest);
        }

        var previousStepData = new OptimizationStepData<T>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            for (int i = 0; i < _psoOptions.SwarmSize; i++)
            {
                var personalBest = personalBests[i];
                UpdateParticle(swarm[i], ref personalBest, globalBest, inputData);
            }

            UpdateAdaptiveParameters(globalBest, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, globalBest))
            {
                break;
            }

            previousStepData = globalBest;
        }

        return CreateOptimizationResult(globalBest, inputData);
    }

    private void UpdateGlobalBest(OptimizationStepData<T> stepData, ref OptimizationStepData<T> globalBest)
    {
        if (globalBest.Solution == null || _fitnessCalculator.IsBetterFitness(stepData.FitnessScore, globalBest.FitnessScore))
        {
            globalBest = stepData;
        }
    }

    private List<ISymbolicModel<T>> InitializeSwarm(int dimensions, int swarmSize)
    {
        var swarm = new List<ISymbolicModel<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var particle = SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions, NumOps);
            
            if (particle is VectorModel<T> vectorModel)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    vectorModel.Coefficients[j] = NumOps.FromDouble(_random.NextDouble() * 2 - 1);
                }
            }

            swarm.Add(particle);
        }

        return swarm;
    }

    private void UpdateVelocity(Vector<T> velocity, Vector<T> position, Vector<T> personalBest, Vector<T> globalBest)
    {
        var inertia = NumOps.FromDouble(_currentInertia);
        var cognitiveWeight = NumOps.FromDouble(_currentCognitiveWeight);
        var socialWeight = NumOps.FromDouble(_currentSocialWeight);

        for (int i = 0; i < velocity.Length; i++)
        {
            var cognitive = NumOps.Multiply(cognitiveWeight, NumOps.FromDouble(_random.NextDouble()));
            var social = NumOps.Multiply(socialWeight, NumOps.FromDouble(_random.NextDouble()));

            velocity[i] = NumOps.Add(
                NumOps.Multiply(inertia, velocity[i]),
                NumOps.Add(
                    NumOps.Multiply(cognitive, NumOps.Subtract(personalBest[i], position[i])),
                    NumOps.Multiply(social, NumOps.Subtract(globalBest[i], position[i]))
                )
            );
        }
    }

    private List<ISymbolicModel<T>> InitializeVelocities(int dimensions, int swarmSize)
    {
        var velocities = new List<ISymbolicModel<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var velocity = SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions, NumOps);

            // Initialize velocity with small random values
            var randomVector = new Vector<T>(dimensions);
            for (int j = 0; j < dimensions; j++)
            {
                randomVector[j] = NumOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
            }

            velocity.UpdateFromVelocity(randomVector);

            velocities.Add(velocity);
        }

        return velocities;
    }

    private Vector<T> CalculateVelocity(ISymbolicModel<T> particle, ISymbolicModel<T> personalBest, ISymbolicModel<T> globalBest)
    {
        var particleVector = particle.ToVector();
        var personalBestVector = personalBest.ToVector();
        var globalBestVector = globalBest.ToVector();
        var velocity = new Vector<T>(particleVector.Length);

        for (int i = 0; i < particleVector.Length; i++)
        {
            T cognitive = NumOps.Multiply(NumOps.FromDouble(_psoOptions.CognitiveParameter), 
                NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), 
                    NumOps.Subtract(personalBestVector[i], particleVector[i])));

            T social = NumOps.Multiply(NumOps.FromDouble(_psoOptions.SocialParameter), 
                NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), 
                    NumOps.Subtract(globalBestVector[i], particleVector[i])));

            velocity[i] = NumOps.Add(cognitive, social);
        }

        return velocity;
    }

    private void UpdateParticle(ISymbolicModel<T> particle, ref OptimizationStepData<T> personalBest, 
        OptimizationStepData<T> globalBest, OptimizationInputData<T> inputData)
    {
        var velocity = CalculateVelocity(particle, personalBest.Solution, globalBest.Solution);
        particle.UpdateFromVelocity(velocity);

        var currentStepData = EvaluateSolution(particle, inputData);

        if (NumOps.GreaterThan(currentStepData.FitnessScore, personalBest.FitnessScore))
        {
            personalBest = currentStepData;
        }
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Update PSO-specific parameters
        _currentInertia = Math.Max(_psoOptions.MinInertia, Math.Min(_psoOptions.MaxInertia, _currentInertia * _psoOptions.InertiaDecayRate));
        _currentCognitiveWeight = Math.Max(_psoOptions.MinCognitiveWeight, Math.Min(_psoOptions.MaxCognitiveWeight, _currentCognitiveWeight * _psoOptions.CognitiveWeightAdaptationRate));
        _currentSocialWeight = Math.Max(_psoOptions.MinSocialWeight, Math.Min(_psoOptions.MaxSocialWeight, _currentSocialWeight * _psoOptions.SocialWeightAdaptationRate));
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is ParticleSwarmOptimizationOptions psoOptions)
        {
            _psoOptions = psoOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ParticleSwarmOptimizationOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _psoOptions;
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

            // Serialize ParticleSwarmOptimizationOptions
            string optionsJson = JsonConvert.SerializeObject(_psoOptions);
            writer.Write(optionsJson);

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

            // Deserialize ParticleSwarmOptimizationOptions
            string optionsJson = reader.ReadString();
            _psoOptions = JsonConvert.DeserializeObject<ParticleSwarmOptimizationOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}