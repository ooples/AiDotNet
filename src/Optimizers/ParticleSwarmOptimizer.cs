/// <summary>
/// Implements a Particle Swarm Optimization algorithm for finding optimal solutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
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
public class ParticleSwarmOptimizer<T> : OptimizerBase<T>
{
    /// <summary>
    /// Random number generator for stochastic components of the algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field provides a source of randomness for the PSO algorithm, which is essential for
    /// adding exploration capability to the particle movement. It is used to generate random
    /// coefficients in the velocity update calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a dice roller for the algorithm.
    /// 
    /// The random number generator:
    /// - Adds unpredictability to how particles move
    /// - Helps particles explore different areas of the search space
    /// - Prevents particles from always making the same decisions
    /// 
    /// Without randomness, the algorithm would be too rigid and might get stuck in suboptimal solutions.
    /// </para>
    /// </remarks>
    private readonly Random _random;

    /// <summary>
    /// Configuration options specific to Particle Swarm Optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters for the PSO algorithm, such as swarm size,
    /// inertia weights, cognitive and social parameters, and adaptation rates. These parameters
    /// control the behavior of the particles and how they explore the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the rule book for the swarm.
    /// 
    /// The PSO options control:
    /// - How many particles are in the swarm
    /// - How strongly particles are pulled toward their own best position
    /// - How strongly particles are pulled toward the swarm's best position
    /// - How much momentum particles maintain while moving
    /// - How these behaviors change over time
    /// 
    /// Adjusting these settings can help the algorithm work better for different types of problems.
    /// </para>
    /// </remarks>
    private ParticleSwarmOptimizationOptions _psoOptions;

    /// <summary>
    /// The current inertia weight that controls a particle's tendency to continue its current trajectory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current inertia weight, which determines how much of a particle's previous
    /// velocity is retained when updating its velocity. A higher inertia value means particles resist
    /// changing direction, while a lower value makes them more responsive to cognitive and social pulls.
    /// This value typically decreases over time to favor exploitation over exploration.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the momentum of a bird in flight.
    /// 
    /// The inertia weight controls:
    /// - How much a particle tends to keep moving in its current direction
    /// - A high value means particles are harder to redirect (good for exploration)
    /// - A low value means particles change direction more easily (good for fine-tuning)
    /// 
    /// The algorithm usually starts with higher inertia to explore broadly, then reduces it
    /// to focus on refining good solutions.
    /// </para>
    /// </remarks>
    private double _currentInertia;

    /// <summary>
    /// The current cognitive weight that controls a particle's attraction to its personal best position.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current cognitive weight, which determines how strongly a particle is drawn
    /// toward its own personal best position. A higher cognitive weight emphasizes a particle's individual
    /// experience, promoting exploration of different areas of the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a bird's memory of where it personally found food.
    /// 
    /// The cognitive weight controls:
    /// - How strongly a particle is pulled toward the best position it has personally found
    /// - A high value means particles strongly favor their own experience
    /// - A low value means particles may ignore their previous discoveries
    /// 
    /// This parameter helps maintain diversity in the swarm by having particles partially
    /// follow their own distinct paths.
    /// </para>
    /// </remarks>
    private double _currentCognitiveWeight;

    /// <summary>
    /// The current social weight that controls a particle's attraction to the global best position.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current social weight, which determines how strongly a particle is drawn
    /// toward the global best position found by any particle in the swarm. A higher social weight
    /// emphasizes collective knowledge, promoting convergence toward promising areas.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a bird's awareness of where the flock found the most food.
    /// 
    /// The social weight controls:
    /// - How strongly a particle is pulled toward the best position found by any particle
    /// - A high value means particles quickly converge toward the current best solution
    /// - A low value means particles are more independent from the group
    /// 
    /// This parameter helps the swarm concentrate its search in promising areas
    /// while sharing knowledge between particles.
    /// </para>
    /// </remarks>
    private double _currentSocialWeight;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParticleSwarmOptimizer{T}"/> class with the specified options and components.
    /// </summary>
    /// <param name="options">The particle swarm optimization options, or null to use default options.</param>
    /// <param name="predictionOptions">The prediction statistics options, or null to use default options.</param>
    /// <param name="modelOptions">The model statistics options, or null to use default options.</param>
    /// <param name="modelEvaluator">The model evaluator, or null to use the default evaluator.</param>
    /// <param name="fitDetector">The fit detector, or null to use the default detector.</param>
    /// <param name="fitnessCalculator">The fitness calculator, or null to use the default calculator.</param>
    /// <param name="modelCache">The model cache, or null to use the default cache.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new particle swarm optimizer with the specified options and components.
    /// If any parameter is null, a default implementation is used. The constructor also initializes a random
    /// number generator and sets up the initial adaptive parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    /// 
    /// Think of it like setting up a new game:
    /// - You can provide custom settings (options) or use the default ones
    /// - You can provide custom tools (evaluators, calculators) or use the default ones
    /// - It gets everything ready so you can start the optimization process
    /// 
    /// The options control things like how many "birds" are in the flock, how fast they move,
    /// and how much they rely on their own experience versus the group's experience.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the adaptive parameters used by the PSO algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to initialize PSO-specific adaptive parameters.
    /// It sets the initial inertia, cognitive weight, and social weight values from the PSO options.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting behavior values for the particles.
    /// 
    /// These parameters control how particles move:
    /// - Inertia: How much particles tend to keep moving in their current direction
    /// - Cognitive weight: How much particles are drawn to their own best found position
    /// - Social weight: How much particles are drawn to the swarm's best found position
    /// 
    /// Getting these values right helps the swarm explore effectively without getting stuck
    /// or flying around chaotically.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method implements the main PSO algorithm. It initializes a swarm of particles and their velocities,
    /// then iteratively updates them based on their personal best positions and the global best position.
    /// The algorithm continues until either the maximum number of iterations is reached or early stopping
    /// criteria are met.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main search process where particles look for the best solution.
    /// 
    /// The process works like this:
    /// 1. Create a group of particles (the swarm) at random positions
    /// 2. For each iteration:
    ///    - Each particle evaluates how good its current position is
    ///    - Each particle remembers its own best position found so far
    ///    - The swarm tracks the overall best position found by any particle
    ///    - Each particle adjusts its velocity based on inertia, its own experience, and the swarm's experience
    ///    - Each particle moves to a new position based on its velocity
    /// 3. After enough iterations, or when no more improvement is happening, return the best solution found
    /// 
    /// This approach is powerful because it balances:
    /// - Exploration: Particles spread out to search different areas
    /// - Exploitation: Particles concentrate around promising areas
    /// </para>
    /// </remarks>
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
        var currentIterationBest = new OptimizationStepData<T>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            currentIterationBest = new OptimizationStepData<T>();

            for (int i = 0; i < _psoOptions.SwarmSize; i++)
            {
                var personalBest = personalBests[i];
                UpdateParticle(swarm[i], ref personalBest, globalBest, inputData);
        
                // Update current iteration's best solution
                if (currentIterationBest.Solution == null || 
                    _fitnessCalculator.IsBetterFitness(personalBest.FitnessScore, currentIterationBest.FitnessScore))
                {
                    currentIterationBest = personalBest;
                }

                // Update global best
                UpdateGlobalBest(personalBest, ref globalBest);
            }

            UpdateAdaptiveParameters(currentIterationBest, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, globalBest))
            {
                break;
            }

            previousStepData = currentIterationBest;
        }

        return CreateOptimizationResult(globalBest, inputData);
    }

    /// <summary>
    /// Updates the global best solution if the current step data has a better fitness score.
    /// </summary>
    /// <param name="stepData">The current step data to evaluate.</param>
    /// <param name="globalBest">The global best solution, passed by reference to be updated if needed.</param>
    /// <remarks>
    /// <para>
    /// This method checks if the provided step data has a better fitness score than the current global best.
    /// If it does, the global best is updated to the new step data.
    /// </para>
    /// <para><b>For Beginners:</b> This method keeps track of the best position found by any particle.
    /// 
    /// It's like keeping track of the best restaurant anyone in your friend group has found:
    /// - If someone finds a better restaurant than the current favorite
    /// - The group's favorite restaurant gets updated
    /// - Everyone in the group now knows about this better option
    /// 
    /// This shared knowledge helps guide all the particles toward promising areas.
    /// </para>
    /// </remarks>
    private void UpdateGlobalBest(OptimizationStepData<T> stepData, ref OptimizationStepData<T> globalBest)
    {
        if (globalBest.Solution == null || _fitnessCalculator.IsBetterFitness(stepData.FitnessScore, globalBest.FitnessScore))
        {
            globalBest = stepData;
        }
    }

    /// <summary>
    /// Initializes the swarm with random particles.
    /// </summary>
    /// <param name="dimensions">The number of dimensions in the search space.</param>
    /// <param name="swarmSize">The number of particles in the swarm.</param>
    /// <returns>A list of symbolic models representing the particles in the swarm.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a swarm of particles with random initial positions. Each particle is a symbolic model
    /// with random coefficients. The type of model created depends on the optimizer options.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the group of particles that will search for solutions.
    /// 
    /// It's like:
    /// - Deciding how many birds will be in your flock (swarmSize)
    /// - Placing each bird at a random starting position in the search space
    /// - Making sure each bird has random initial characteristics
    /// 
    /// These random starting positions help the swarm explore different parts of the
    /// solution space right from the beginning.
    /// </para>
    /// </remarks>
    private List<ISymbolicModel<T>> InitializeSwarm(int dimensions, int swarmSize)
    {
        var swarm = new List<ISymbolicModel<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var particle = SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions);
            
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

    /// <summary>
    /// Updates the velocity of a particle based on inertia, cognitive, and social components.
    /// </summary>
    /// <param name="velocity">The velocity vector to update.</param>
    /// <param name="position">The current position vector of the particle.</param>
    /// <param name="personalBest">The personal best position vector of the particle.</param>
    /// <param name="globalBest">The global best position vector of the swarm.</param>
    /// <remarks>
    /// <para>
    /// This method updates the velocity of a particle according to the PSO velocity update formula.
    /// The new velocity is a combination of:
    /// 1. Inertia component: the particle's tendency to continue moving in the same direction
    /// 2. Cognitive component: the particle's tendency to move toward its personal best position
    /// 3. Social component: the particle's tendency to move toward the global best position
    /// </para>
    /// <para><b>For Beginners:</b> This method determines how a particle changes its direction and speed.
    /// 
    /// It works like a bird adjusting its flight based on three factors:
    /// 1. Momentum: The bird tends to keep flying in the same direction it's already going (inertia)
    /// 2. Personal Experience: The bird is drawn toward the best place it personally found food (cognitive)
    /// 3. Social Influence: The bird is drawn toward where the flock found the most food (social)
    /// 
    /// The weights of these factors and some randomness determine exactly how the bird adjusts its flight.
    /// This balance of factors helps the birds explore new areas while also focusing on promising locations.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the velocities for all particles in the swarm.
    /// </summary>
    /// <param name="dimensions">The number of dimensions in the search space.</param>
    /// <param name="swarmSize">The number of particles in the swarm.</param>
    /// <returns>A list of symbolic models representing the velocities of the particles.</returns>
    /// <remarks>
    /// <para>
    /// This method creates initial velocities for all particles in the swarm. Each velocity is a symbolic model
    /// with small random values. The small initial velocities help the swarm start with controlled movement.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets how fast and in what direction each particle starts moving.
    /// 
    /// It's like determining:
    /// - The initial speed and direction for each bird in the flock
    /// - Using small random values so birds don't fly too wildly at first
    /// - Making sure each bird has its own unique flight pattern
    /// 
    /// The velocities control how particles move through the search space, and starting with
    /// small random velocities helps the search begin in a controlled way.
    /// </para>
    /// </remarks>
    private List<ISymbolicModel<T>> InitializeVelocities(int dimensions, int swarmSize)
    {
        var velocities = new List<ISymbolicModel<T>>();
        for (int i = 0; i < swarmSize; i++)
        {
            var velocity = SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions);

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

    /// <summary>
    /// Calculates the velocity for a particle based on its current position, personal best, and the global best.
    /// </summary>
    /// <param name="particle">The particle whose velocity is being calculated.</param>
    /// <param name="personalBest">The particle's personal best solution.</param>
    /// <param name="globalBest">The global best solution found by the swarm.</param>
    /// <returns>A velocity vector for the particle.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates a new velocity for a particle based on the particle's current position,
    /// its personal best position, and the global best position. The calculation includes random factors
    /// to add exploration capability to the algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines how a particle should adjust its movement.
    /// 
    /// Imagine a bird deciding how to adjust its flight:
    /// - It looks at where it personally found the most food (personal best)
    /// - It looks at where the flock found the most food (global best)
    /// - It calculates a flight adjustment that balances going to its best spot and the flock's best spot
    /// - Some randomness is added so the bird doesn't always make the same decision
    /// 
    /// This combination of personal experience, social information, and randomness
    /// helps the particles efficiently explore the search space.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Updates a particle's position and personal best information.
    /// </summary>
    /// <param name="particle">The particle to update.</param>
    /// <param name="personalBest">The particle's personal best data, passed by reference to be updated if needed.</param>
    /// <param name="globalBest">The global best data found by the swarm.</param>
    /// <param name="inputData">The input data to evaluate the particle against.</param>
    /// <remarks>
    /// <para>
    /// This method updates a particle's position by calculating and applying a new velocity.
    /// It then evaluates the particle's new position and updates its personal best if the new position
    /// is better than its previous personal best.
    /// </para>
    /// <para><b>For Beginners:</b> This method moves a particle to a new position and checks if it's the best yet.
    /// 
    /// It's like a bird in flight:
    /// 1. The bird calculates a new direction based on its experience and the flock's experience
    /// 2. The bird flies to a new position
    /// 3. The bird checks if this new position has more food than any spot it's found before
    /// 4. If it's better, the bird remembers this as its new personal best location
    /// 
    /// This process of moving and evaluating helps each particle explore the space
    /// and remember the best positions it finds.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Updates adaptive parameters based on optimization progress.
    /// </summary>
    /// <param name="currentStepData">The data from the current optimization step.</param>
    /// <param name="previousStepData">The data from the previous optimization step.</param>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update PSO-specific adaptive parameters
    /// in addition to the base adaptive parameters. It adjusts the inertia, cognitive weight, and
    /// social weight according to their adaptation rates, while keeping them within their specified
    /// minimum and maximum values.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how particles behave as the search progresses.
    /// 
    /// It's like a flock of birds adjusting their search strategy over time:
    /// - Inertia decreases gradually, making birds less likely to overshoot good areas
    /// - Cognitive and social weights change to balance personal exploration vs. group convergence
    /// - All parameters stay within reasonable limits to prevent chaotic behavior
    /// 
    /// These adjustments help the swarm start with broad exploration and gradually
    /// focus on refining the most promising solutions.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Update PSO-specific parameters
        _currentInertia = Math.Max(_psoOptions.MinInertia, Math.Min(_psoOptions.MaxInertia, _currentInertia * _psoOptions.InertiaDecayRate));
        _currentCognitiveWeight = Math.Max(_psoOptions.MinCognitiveWeight, Math.Min(_psoOptions.MaxCognitiveWeight, _currentCognitiveWeight * _psoOptions.CognitiveWeightAdaptationRate));
        _currentSocialWeight = Math.Max(_psoOptions.MinSocialWeight, Math.Min(_psoOptions.MaxSocialWeight, _currentSocialWeight * _psoOptions.SocialWeightAdaptationRate));
    }

    /// <summary>
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the options are not of the expected type.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update the PSO-specific options.
    /// It checks that the provided options are of the correct type (ParticleSwarmOptimizationOptions)
    /// and throws an exception if they are not.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the settings that control how the optimizer works.
    /// 
    /// It's like changing the game settings:
    /// - You provide a set of options to use
    /// - The method checks that these are the right kind of options for a PSO optimizer
    /// - If they are, it applies these new settings
    /// - If not, it lets you know there's a problem
    /// 
    /// This ensures that only appropriate settings are used with this specific optimizer.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current particle swarm optimization options.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to return the PSO-specific options.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the current settings of the optimizer.
    /// 
    /// It's like checking what game settings are currently active:
    /// - You can see the current swarm size
    /// - You can see the current weights for different behaviors
    /// - You can see the current limits and adaptation rates
    /// 
    /// This is useful for understanding how the optimizer is currently configured
    /// or for making a copy of the settings to modify and apply later.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _psoOptions;
    }

    /// <summary>
    /// Serializes the particle swarm optimizer to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to include PSO-specific information in the serialization.
    /// It first serializes the base class data, then adds the PSO options.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer so it can be restored later.
    /// 
    /// It's like taking a snapshot of the optimizer:
    /// - First, it saves all the general optimizer information
    /// - Then, it saves the PSO-specific settings
    /// - It packages everything into a format that can be saved to a file or sent over a network
    /// 
    /// This allows you to:
    /// - Save a trained optimizer to use later
    /// - Share an optimizer with others
    /// - Create a backup before making changes
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Reconstructs the particle swarm optimizer from a serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when the options cannot be deserialized.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to handle PSO-specific information during deserialization.
    /// It first deserializes the base class data, then reconstructs the PSO options.
    /// </para>
    /// <para><b>For Beginners:</b> This method restores the optimizer from a previously saved state.
    /// 
    /// It's like restoring from a snapshot:
    /// - First, it loads all the general optimizer information
    /// - Then, it loads the PSO-specific settings
    /// - It reconstructs the optimizer to the exact state it was in when saved
    /// 
    /// This allows you to:
    /// - Continue working with an optimizer you previously saved
    /// - Use an optimizer that someone else created and shared
    /// - Revert to a backup if needed
    /// </para>
    /// </remarks>
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