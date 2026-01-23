using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements a Particle Swarm Optimization algorithm for finding optimal solutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Particle Swarm Optimization (PSO) is a population-based stochastic optimization technique inspired by the social
/// behavior of birds flocking or fish schooling. The algorithm maintains a population (swarm) of candidate solutions
/// (particles) that move around in the search space according to simple mathematical formulas that consider the
/// particle's position and velocity.
/// </para>
/// <para><b>For Beginners:</b> Particle Swarm Optimization is like a group of birds searching for food.
/// 
/// Imagine a flock of birds looking for the best food source in a field:
/// - Each bird is a "particle" in the swarm
/// - Each bird remembers where it personally found the most food
/// - The flock shares information about where the most food has been found overall
/// - Birds adjust their flight based on their own experience and what they learn from others
/// - Over time, the whole flock converges on the best food source
/// 
/// This approach is very effective for finding good solutions to complex problems where
/// traditional methods might get stuck in suboptimal areas.
/// </para>
/// </remarks>
public class ParticleSwarmOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Random number generator for stochastic components of the algorithm.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Configuration options specific to Particle Swarm Optimization.
    /// </summary>
    private ParticleSwarmOptimizationOptions<T, TInput, TOutput> _psoOptions;

    /// <summary>
    /// The current inertia weight that controls a particle's tendency to continue its current trajectory.
    /// </summary>
    private double _currentInertia;

    /// <summary>
    /// The current cognitive weight that controls a particle's attraction to its personal best position.
    /// </summary>
    private double _currentCognitiveWeight;

    /// <summary>
    /// The current social weight that controls a particle's attraction to the global best position.
    /// </summary>
    private double _currentSocialWeight;

    /// <summary>
    /// Initializes a new instance of the ParticleSwarmOptimizer class with the specified options.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The particle swarm optimization options, or null to use default options.</param>
    public ParticleSwarmOptimizer(
        IFullModel<T, TInput, TOutput> model,
        ParticleSwarmOptimizationOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _random = RandomHelper.CreateSecureRandom();
        _psoOptions = options ?? new ParticleSwarmOptimizationOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the PSO algorithm.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        _currentInertia = _psoOptions.InitialInertia;
        _currentCognitiveWeight = _psoOptions.InitialCognitiveWeight;
        _currentSocialWeight = _psoOptions.InitialSocialWeight;
    }

    /// <summary>
    /// Performs the particle swarm optimization to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize particles using InitializeRandomSolution
        var swarm = new List<IFullModel<T, TInput, TOutput>>();
        for (int i = 0; i < _psoOptions.SwarmSize; i++)
        {
            swarm.Add(InitializeRandomSolution(inputData.XTrain));
        }

        // Use model's parameter count instead of input size for velocity dimensions
        int dimensions = swarm.Count > 0 ? swarm[0].ParameterCount : InputHelper<T, TInput>.GetInputSize(inputData.XTrain);

        // Initialize velocities for each particle
        var velocities = new List<Vector<T>>();
        for (int i = 0; i < _psoOptions.SwarmSize; i++)
        {
            var velocity = new Vector<T>(dimensions);
            for (int j = 0; j < dimensions; j++)
            {
                velocity[j] = NumOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
            }
            velocities.Add(velocity);
        }

        // Initialize personal bests with current particles
        var personalBests = new List<OptimizationStepData<T, TInput, TOutput>>();

        // Evaluate first particle to initialize globalBest properly
        // (Can't use default OptimizationStepData because its FitnessScore of 0 is "perfect" for MSE)
        var firstStepData = EvaluateSolution(swarm[0], inputData);
        personalBests.Add(firstStepData);
        var globalBest = firstStepData;

        // Evaluate remaining particles and initialize personal/global bests
        for (int i = 1; i < _psoOptions.SwarmSize; i++)
        {
            var stepData = EvaluateSolution(swarm[i], inputData);
            personalBests.Add(stepData);

            // Update global best if this particle is better
            if (FitnessCalculator.IsBetterFitness(stepData.FitnessScore, globalBest.FitnessScore))
            {
                globalBest = stepData;
            }
        }

        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var currentIterationBest = new OptimizationStepData<T, TInput, TOutput>();

        // Main optimization loop
        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            currentIterationBest = new OptimizationStepData<T, TInput, TOutput>();

            for (int i = 0; i < _psoOptions.SwarmSize; i++)
            {
                // Get current position and best positions
                var position = swarm[i].GetParameters();
                var personalBestVector = personalBests[i].Solution.GetParameters();
                var globalBestVector = globalBest.Solution.GetParameters();
                var velocity = velocities[i];

                // Update velocity and store the new velocity vector
                velocities[i] = UpdateVelocity(velocity, position, personalBestVector, globalBestVector);

                // === Vectorized Position Update using IEngine (Phase B: US-GPU-015) ===
                // position = position + velocity
                var newPosition = (Vector<T>)Engine.Add(position, velocities[i]);

                // Update the particle model with new position
                swarm[i] = swarm[i].WithParameters(newPosition);

                // Evaluate the updated particle
                var stepData = EvaluateSolution(swarm[i], inputData);

                // Update personal best if better
                if (FitnessCalculator.IsBetterFitness(stepData.FitnessScore, personalBests[i].FitnessScore))
                {
                    personalBests[i] = stepData;
                }

                // Update current iteration's best solution
                if (currentIterationBest.Solution == null ||
                    FitnessCalculator.IsBetterFitness(stepData.FitnessScore, currentIterationBest.FitnessScore))
                {
                    currentIterationBest = stepData;
                }

                // Update global best
                if (FitnessCalculator.IsBetterFitness(stepData.FitnessScore, globalBest.FitnessScore))
                {
                    globalBest = stepData;
                }
            }

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentIterationBest, previousStepData);

            // Check early stopping criteria
            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, globalBest))
            {
                break;
            }

            previousStepData = currentIterationBest;
        }

        return CreateOptimizationResult(globalBest, inputData);
    }

    /// <summary>
    /// Updates the velocity of a particle based on inertia, cognitive, and social components.
    /// </summary>
    /// <param name="velocity">The velocity vector to update.</param>
    /// <param name="position">The current position vector of the particle.</param>
    /// <param name="personalBest">The personal best position vector of the particle.</param>
    /// <param name="globalBest">The global best position vector of the swarm.</param>
    private Vector<T> UpdateVelocity(Vector<T> velocity, Vector<T> position, Vector<T> personalBest, Vector<T> globalBest)
    {
        // === Partially Vectorized Velocity Update using IEngine (Phase B: US-GPU-015) ===
        // velocity = inertia * velocity + cognitive * (personalBest - position) + social * (globalBest - position)

        var inertia = NumOps.FromDouble(_currentInertia);
        var cognitiveWeight = NumOps.FromDouble(_currentCognitiveWeight);
        var socialWeight = NumOps.FromDouble(_currentSocialWeight);

        // Vectorized position differences
        var personalDiff = (Vector<T>)Engine.Subtract(personalBest, position);
        var globalDiff = (Vector<T>)Engine.Subtract(globalBest, position);

        // Vectorized velocity update with random factors
        // Generate random vectors for cognitive and social components
        var cognitiveRandom = new Vector<T>(velocity.Length);
        var socialRandom = new Vector<T>(velocity.Length);
        for (int i = 0; i < velocity.Length; i++)
        {
            cognitiveRandom[i] = NumOps.Multiply(cognitiveWeight, NumOps.FromDouble(_random.NextDouble()));
            socialRandom[i] = NumOps.Multiply(socialWeight, NumOps.FromDouble(_random.NextDouble()));
        }

        // Vectorized computation: velocity = inertia*velocity + cognitive*personalDiff + social*globalDiff
        var inertiaVel = (Vector<T>)Engine.Multiply(velocity, Engine.Fill<T>(velocity.Length, inertia));
        var cognitiveTerm = (Vector<T>)Engine.Multiply(personalDiff, cognitiveRandom);
        var socialTerm = (Vector<T>)Engine.Multiply(globalDiff, socialRandom);
        var cogSocSum = (Vector<T>)Engine.Add(cognitiveTerm, socialTerm);
        return (Vector<T>)Engine.Add(inertiaVel, cogSocSum);
    }

    /// <summary>
    /// Updates adaptive parameters based on optimization progress.
    /// </summary>
    /// <param name="currentStepData">The data from the current optimization step.</param>
    /// <param name="previousStepData">The data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Update PSO-specific parameters
        _currentInertia = Math.Max(_psoOptions.MinInertia,
            Math.Min(_psoOptions.MaxInertia, _currentInertia * _psoOptions.InertiaDecayRate));

        _currentCognitiveWeight = Math.Max(_psoOptions.MinCognitiveWeight,
            Math.Min(_psoOptions.MaxCognitiveWeight, _currentCognitiveWeight * _psoOptions.CognitiveWeightAdaptationRate));

        _currentSocialWeight = Math.Max(_psoOptions.MinSocialWeight,
            Math.Min(_psoOptions.MaxSocialWeight, _currentSocialWeight * _psoOptions.SocialWeightAdaptationRate));
    }

    /// <summary>
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the options are not of the expected type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is ParticleSwarmOptimizationOptions<T, TInput, TOutput> psoOptions)
        {
            _psoOptions = psoOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ParticleSwarmOptimizationOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current particle swarm optimization options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _psoOptions;
    }

    /// <summary>
    /// Serializes the particle swarm optimizer to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer.</returns>
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

            // Serialize current adaptive parameters
            writer.Write(_currentInertia);
            writer.Write(_currentCognitiveWeight);
            writer.Write(_currentSocialWeight);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Reconstructs the particle swarm optimizer from a serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when the options cannot be deserialized.</exception>
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
            _psoOptions = JsonConvert.DeserializeObject<ParticleSwarmOptimizationOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize current adaptive parameters
            _currentInertia = reader.ReadDouble();
            _currentCognitiveWeight = reader.ReadDouble();
            _currentSocialWeight = reader.ReadDouble();
        }
    }
}
