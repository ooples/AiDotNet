using Newtonsoft.Json;

/// <summary>
/// Implements the Simulated Annealing optimization algorithm, a probabilistic technique for finding global optima.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Simulated Annealing is a metaheuristic optimization algorithm inspired by the annealing process in metallurgy,
/// where materials are heated and then slowly cooled to reduce defects. The algorithm starts with a high "temperature"
/// that allows it to accept worse solutions with a certain probability, helping it escape local optima. As the
/// temperature gradually decreases, the algorithm becomes more selective, eventually converging to a good solution.
/// </para>
/// <para><b>For Beginners:</b> Simulated Annealing is like a hiker who sometimes deliberately goes uphill to avoid getting stuck in a valley.
/// 
/// Imagine a hiker exploring a mountain range in fog, trying to find the lowest point:
/// - When it's hot (high temperature), the hiker is willing to climb uphill sometimes
/// - This helps avoid getting stuck in small valleys that aren't the true lowest point
/// - As it gets cooler (temperature decreases), the hiker becomes less willing to go uphill
/// - Eventually, when it's cold, the hiker only moves downhill to find the precise low point
/// 
/// This balance between exploration (accepting worse solutions) and exploitation (refining good solutions)
/// helps the algorithm find better solutions in complex landscapes with many local optima.
/// </para>
/// </remarks>
public class SimulatedAnnealingOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Random number generator for stochastic decision-making.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field provides a source of randomness for the Simulated Annealing algorithm. It is used
    /// to generate random perturbations when creating neighbor solutions and to make probabilistic
    /// decisions about accepting worse solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the hiker's dice that helps make unpredictable choices.
    /// 
    /// The random number generator:
    /// - Creates random variations when generating new solutions to try
    /// - Determines whether to accept a worse solution based on probability
    /// - Adds the element of chance that is essential to the algorithm's effectiveness
    /// 
    /// This randomness is crucial for the algorithm to escape local optima and explore the solution space effectively.
    /// </para>
    /// </remarks>
    private readonly Random _random;

    /// <summary>
    /// Configuration options specific to the Simulated Annealing algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters for the Simulated Annealing algorithm, such as
    /// the initial temperature, cooling rate, and neighbor generation parameters. These parameters control
    /// the behavior of the optimizer and affect its exploration vs. exploitation balance.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the rule book for the optimizer.
    /// 
    /// The options control:
    /// - How "hot" to start (initial temperature)
    /// - How quickly to "cool down" (cooling rate)
    /// - How far to look for new solutions (neighbor generation range)
    /// - When to stop the optimization process (maximum iterations)
    /// 
    /// Adjusting these settings can help the algorithm work better for different types of problems.
    /// </para>
    /// </remarks>
    private SimulatedAnnealingOptions<T, TInput, TOutput> _saOptions;

    /// <summary>
    /// The current temperature controlling the acceptance probability of worse solutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current temperature of the Simulated Annealing process. At high temperatures,
    /// the algorithm is more likely to accept worse solutions, promoting exploration. As the temperature
    /// decreases, the algorithm becomes more selective, focusing on exploitation of good solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the weather temperature that affects the hiker's willingness to climb uphill.
    /// 
    /// The current temperature:
    /// - When high, allows the algorithm to accept worse solutions with higher probability
    /// - When low, makes the algorithm more likely to reject worse solutions
    /// - Decreases over time as the optimization progresses
    /// - Helps balance between exploring widely and refining good solutions
    /// 
    /// This temperature mechanism is what gives Simulated Annealing its power to escape local optima.
    /// </para>
    /// </remarks>
    private T _currentTemperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="SimulatedAnnealingOptimizer{T}"/> class with the specified options and components.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The simulated annealing options, or null to use default options.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Simulated Annealing optimizer with the specified options and components.
    /// If any parameter is null, a default implementation is used. The constructor initializes the random
    /// number generator, options, and starting temperature.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    ///
    /// Think of it like preparing for a hiking expedition:
    /// - You can provide custom settings (options) or use the default ones
    /// - You can provide specialized tools (evaluators, calculators) or use the basic ones
    /// - It initializes the random number generator for making probabilistic decisions
    /// - It sets the starting temperature to begin the annealing process
    ///
    /// This constructor gets everything ready so you can start the optimization process.
    /// </para>
    /// </remarks>
    public SimulatedAnnealingOptimizer(
        IFullModel<T, TInput, TOutput> model,
        SimulatedAnnealingOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _random = RandomHelper.CreateSecureRandom();
        _saOptions = options ?? new SimulatedAnnealingOptions<T, TInput, TOutput>();
        _currentTemperature = NumOps.FromDouble(_saOptions.InitialTemperature);
    }

    /// <summary>
    /// Performs the Simulated Annealing optimization to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main Simulated Annealing algorithm. It starts from a random solution and
    /// iteratively generates neighbor solutions. Better solutions are always accepted, while worse solutions
    /// are accepted with a probability that depends on how much worse they are and the current temperature.
    /// The temperature gradually decreases, making the algorithm less likely to accept worse solutions over time.
    /// The process continues until either the maximum number of iterations is reached or early stopping
    /// criteria are met.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main search process where the algorithm looks for the best solution.
    /// 
    /// The process works like this:
    /// 1. Start at a random position on the "landscape"
    /// 2. For each iteration:
    ///    - Generate a nearby position to consider (neighbor solution)
    ///    - Always move there if it's better than the current position
    ///    - Sometimes move there even if it's worse, based on temperature and how much worse it is
    ///    - Keep track of the best position found so far
    ///    - Lower the temperature, making it less likely to accept worse moves as time goes on
    ///    - Update other parameters based on progress
    /// 3. Return the best solution found during the entire process
    /// 
    /// This approach helps find good solutions even in complex landscapes with many local optima.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        InitializeAdaptiveParameters();
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        for (int iteration = 0; iteration < _saOptions.MaxIterations; iteration++)
        {
            var newSolution = GenerateNeighborSolution(currentSolution);
            var currentStepData = EvaluateSolution(newSolution, inputData);

            if (AcceptNewSolution(previousStepData.FitnessScore, currentStepData.FitnessScore))
            {
                currentSolution = newSolution;
                UpdateBestSolution(currentStepData, ref bestStepData);
            }

            UpdateAdaptiveParameters(currentStepData, previousStepData);
            _currentTemperature = CoolDown(_currentTemperature);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates adaptive parameters based on optimization progress.
    /// </summary>
    /// <param name="currentStepData">The data from the current optimization step.</param>
    /// <param name="previousStepData">The data from the previous optimization step.</param>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update Simulated Annealing-specific adaptive parameters
    /// in addition to the base adaptive parameters. It updates the temperature and neighbor generation parameters
    /// based on whether the current solution is better than the previous one.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm searches based on its progress.
    /// 
    /// It's like a hiker changing their approach:
    /// - Based on whether they're finding better spots, they adjust how willing they are to climb uphill
    /// - They also adjust how far they look for new spots to try
    /// - These adjustments help the algorithm be more efficient as the search progresses
    /// 
    /// This adaptive behavior helps balance exploration and exploitation dynamically.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        UpdateTemperature(currentStepData.FitnessScore, previousStepData.FitnessScore);
        UpdateNeighborGenerationParameters(currentStepData.FitnessScore, previousStepData.FitnessScore);
    }

    /// <summary>
    /// Updates the temperature based on the comparison of current and previous fitness scores.
    /// </summary>
    /// <param name="currentFitness">The fitness score of the current solution.</param>
    /// <param name="previousFitness">The fitness score of the previous solution.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the temperature based on whether the current solution is better than the previous one.
    /// If the current solution is better, the temperature is decreased to focus more on exploitation.
    /// If the current solution is worse, the temperature is increased to allow more exploration.
    /// The temperature is kept within the specified minimum and maximum limits.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how willing the algorithm is to accept worse solutions.
    /// 
    /// Think of it like adjusting a thermostat:
    /// - If you're finding better solutions, you turn down the temperature, becoming more selective
    /// - If you're not finding better solutions, you turn up the temperature, becoming more willing to try different paths
    /// - The temperature always stays between minimum and maximum values to avoid extremes
    /// 
    /// This adaptive temperature helps the algorithm balance exploration and exploitation based on recent progress.
    /// </para>
    /// </remarks>
    private void UpdateTemperature(T currentFitness, T previousFitness)
    {
        if (FitnessCalculator.IsBetterFitness(currentFitness, previousFitness))
        {
            _currentTemperature = NumOps.Multiply(_currentTemperature, NumOps.FromDouble(_saOptions.CoolingRate));
        }
        else
        {
            _currentTemperature = NumOps.Divide(_currentTemperature, NumOps.FromDouble(_saOptions.CoolingRate));
        }

        _currentTemperature = MathHelper.Clamp(_currentTemperature,
            NumOps.FromDouble(_saOptions.MinTemperature),
            NumOps.FromDouble(_saOptions.MaxTemperature));
    }

    /// <summary>
    /// Updates the neighbor generation parameters based on the comparison of current and previous fitness scores.
    /// </summary>
    /// <param name="currentFitness">The fitness score of the current solution.</param>
    /// <param name="previousFitness">The fitness score of the previous solution.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the range used for generating neighbor solutions based on whether the current solution
    /// is better than the previous one. If the current solution is better, the range is decreased to focus more
    /// on local refinement. If the current solution is worse, the range is increased to explore more widely.
    /// The range is kept within the specified minimum and maximum limits.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how far the algorithm looks for new solutions.
    /// 
    /// Think of it like a hiker adjusting their search radius:
    /// - If you're finding better spots, you narrow your search to look more carefully nearby
    /// - If you're not finding better spots, you widen your search to look farther away
    /// - The search radius always stays between minimum and maximum values to avoid extremes
    /// 
    /// This adaptive search radius helps the algorithm focus on promising areas while still being able to explore widely when needed.
    /// </para>
    /// </remarks>
    private void UpdateNeighborGenerationParameters(T currentFitness, T previousFitness)
    {
        if (FitnessCalculator.IsBetterFitness(currentFitness, previousFitness))
        {
            _saOptions.NeighborGenerationRange *= 0.95;
        }
        else
        {
            _saOptions.NeighborGenerationRange *= 1.05;
        }

        _saOptions.NeighborGenerationRange = MathHelper.Clamp(_saOptions.NeighborGenerationRange,
            _saOptions.MinNeighborGenerationRange,
            _saOptions.MaxNeighborGenerationRange);
    }

    /// <summary>
    /// Reduces the temperature according to the cooling schedule.
    /// </summary>
    /// <param name="temperature">The current temperature.</param>
    /// <returns>The new temperature after cooling.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the cooling schedule for the Simulated Annealing algorithm. It reduces
    /// the current temperature by multiplying it by the cooling rate. This gradual reduction in temperature
    /// causes the algorithm to become more selective over time, transitioning from exploration to exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This method gradually lowers the "temperature" of the system.
    /// 
    /// It's like the natural cooling process:
    /// - The temperature decreases by a certain percentage each time
    /// - This is controlled by the cooling rate parameter
    /// - Lower temperatures make the algorithm less willing to accept worse solutions
    /// - This gradual transition helps the algorithm converge to a good solution
    /// 
    /// The cooling schedule is a critical part of the Simulated Annealing process, as it controls
    /// the balance between exploration and exploitation over time.
    /// </para>
    /// </remarks>
    private T CoolDown(T temperature)
    {
        return NumOps.Multiply(temperature, NumOps.FromDouble(_saOptions.CoolingRate));
    }

    /// <summary>
    /// Determines whether to accept a new solution based on its fitness compared to the current solution.
    /// </summary>
    /// <param name="currentFitness">The fitness of the current solution.</param>
    /// <param name="newFitness">The fitness of the new candidate solution.</param>
    /// <returns>True if the new solution should be accepted; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the acceptance criterion of the Simulated Annealing algorithm. Better solutions
    /// are always accepted. Worse solutions are accepted with a probability that depends on how much worse
    /// they are and the current temperature. Higher temperatures make it more likely to accept worse solutions,
    /// while lower temperatures make it less likely.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides whether to move to a new position or stay put.
    /// 
    /// The decision works like this:
    /// - If the new position is better, always move there
    /// - If the new position is worse, sometimes move there anyway
    /// - The probability of moving to a worse position depends on:
    ///   - How much worse it is (smaller difference means higher probability)
    ///   - The current temperature (higher temperature means higher probability)
    /// 
    /// This occasional acceptance of worse solutions is what allows the algorithm to escape local optima
    /// and potentially find better solutions elsewhere.
    /// </para>
    /// </remarks>
    private bool AcceptNewSolution(T currentFitness, T newFitness)
    {
        if (FitnessCalculator.IsBetterFitness(newFitness, currentFitness))
        {
            return true;
        }

        var acceptanceProbability = Math.Exp(Convert.ToDouble(NumOps.Divide(
            NumOps.Subtract(currentFitness, newFitness),
            _currentTemperature
        )));

        return _random.NextDouble() < acceptanceProbability;
    }

    /// <summary>
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the options are not of the expected type.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update the Simulated Annealing-specific options.
    /// It checks that the provided options are of the correct type (SimulatedAnnealingOptions)
    /// and throws an exception if they are not.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the settings that control how the optimizer works.
    /// 
    /// It's like changing the rule book:
    /// - You provide a set of options to use
    /// - The method checks that these are the right kind of options for a Simulated Annealing optimizer
    /// - If they are, it applies these new settings
    /// - If not, it lets you know there's a problem
    /// 
    /// This ensures that only appropriate settings are used with this specific optimizer.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is SimulatedAnnealingOptions<T, TInput, TOutput> saOptions)
        {
            _saOptions = saOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected SimulatedAnnealingOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current simulated annealing options.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to return the Simulated Annealing-specific options.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the current settings of the optimizer.
    /// 
    /// It's like checking what settings are currently active:
    /// - You can see the current temperature settings
    /// - You can see the current neighbor generation settings
    /// - You can see all the other parameters that control the optimizer
    /// 
    /// This is useful for understanding how the optimizer is currently configured
    /// or for making a copy of the settings to modify and apply later.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _saOptions;
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Simulated Annealing algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to initialize Simulated Annealing-specific adaptive parameters.
    /// It sets the current temperature to the initial temperature specified in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the optimizer for a fresh start.
    /// 
    /// It's like resetting the thermostat:
    /// - Setting the temperature back to its starting value
    /// - Getting ready to begin a new annealing process
    /// 
    /// This initialization ensures that each optimization run starts with the same initial conditions.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _currentTemperature = NumOps.FromDouble(_saOptions.InitialTemperature);
    }

    /// <summary>
    /// Generates a new solution by perturbing the coefficients of the current solution.
    /// </summary>
    /// <param name="currentSolution">The current solution to be perturbed.</param>
    /// <returns>A new ISymbolicModel representing the neighbor solution.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new solution by adding small random changes (perturbations) to each coefficient
    /// of the current solution. The size of these perturbations is controlled by the NeighborGenerationRange
    /// parameter in the SimulatedAnnealingOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like taking a small step in a random direction from your current position.
    /// 
    /// Imagine you're on a hill trying to find the highest point:
    /// - Your current position is the 'currentSolution'
    /// - You take a small step in a random direction to see if it's higher
    /// - The size of your step is controlled by 'NeighborGenerationRange'
    /// - You do this for each direction you can move in (each 'coefficient')
    /// 
    /// This helps the algorithm explore nearby solutions that might be better than the current one.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> GenerateNeighborSolution(IFullModel<T, TInput, TOutput> currentSolution)
    {
        var parameters = currentSolution.GetParameters();

        // Generate random perturbations vectorized using Engine
        var perturbations = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            perturbations[i] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * _saOptions.NeighborGenerationRange);
        }

        // Add perturbations to parameters using vectorized Engine operation
        var newCoefficients = (Vector<T>)Engine.Add(parameters, perturbations);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Serializes the current state of the SimulatedAnnealingOptimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its base class data,
    /// SimulatedAnnealingOptions, and current temperature, into a byte array. This allows the
    /// optimizer's state to be stored or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like taking a snapshot of the optimizer's current setup.
    /// 
    /// Think of it as saving a game:
    /// - It saves all the current settings and progress
    /// - This saved data can be used later to continue from where you left off
    /// - It includes information from the parent class (base data), specific settings for this optimizer,
    ///   and the current "temperature" of the system
    /// 
    /// This is useful for saving progress, sharing the optimizer's state, or creating checkpoints in the optimization process.
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

            // Serialize SimulatedAnnealingOptions
            string optionsJson = JsonConvert.SerializeObject(_saOptions);
            writer.Write(optionsJson);

            // Serialize current temperature
            writer.Write(Convert.ToDouble(_currentTemperature));

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the state of the SimulatedAnnealingOptimizer.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the optimizer from a byte array, including its base class data,
    /// SimulatedAnnealingOptions, and current temperature. It's the counterpart to the Serialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like loading a saved game to continue where you left off.
    /// 
    /// Imagine unpacking a suitcase:
    /// - You're taking out all the pieces of information that were saved earlier
    /// - First, you unpack the basic information (base class data)
    /// - Then, you unpack the specific settings for this optimizer (SimulatedAnnealingOptions)
    /// - Finally, you set the current "temperature" to what it was when saved
    /// 
    /// This allows you to recreate the exact state of the optimizer from a previous point in time.
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

            // Deserialize SimulatedAnnealingOptions
            string optionsJson = reader.ReadString();
            _saOptions = JsonConvert.DeserializeObject<SimulatedAnnealingOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize current temperature
            _currentTemperature = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}
