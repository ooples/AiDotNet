using Newtonsoft.Json;

/// <summary>
/// Implements Powell's method, a derivative-free optimization algorithm for finding local minima or maxima.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Powell's method is a direction-set method that performs sequential line minimizations along a set of directions,
/// starting from an initial point. The algorithm iteratively updates these directions to achieve faster convergence.
/// It does not require derivatives of the objective function, making it suitable for optimizing complex functions
/// where derivatives are unavailable or difficult to compute.
/// </para>
/// <para><b>For Beginners:</b> Powell's method is like a hiker trying to find the highest point in a mountain range.
/// 
/// Imagine a hiker exploring mountains:
/// - The hiker starts from a random position
/// - The hiker first tries moving north, south, east, and west to find better spots
/// - After trying all directions, the hiker figures out a new, better direction to try
/// - The hiker keeps trying different directions until they find the highest peak
/// 
/// This approach is efficient because:
/// - It doesn't need to know the slope at each point (no derivatives needed)
/// - It systematically explores the landscape by moving in different directions
/// - It can discover and follow valleys or ridges to reach peaks more quickly
/// </para>
/// </remarks>
public class PowellOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options specific to Powell's optimization method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters for the Powell algorithm, such as the maximum number
    /// of iterations, tolerance for convergence, initial step size, and adaptation parameters. These parameters
    /// control the behavior of the optimizer and affect its performance and convergence properties.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the rule book for the optimizer.
    /// 
    /// The Powell options control:
    /// - How many attempts the algorithm makes before giving up
    /// - How precise the final answer needs to be
    /// - How big steps the algorithm takes when exploring
    /// - How the algorithm adjusts its behavior as it goes
    /// 
    /// Adjusting these settings can help the algorithm work better for different types of problems.
    /// </para>
    /// </remarks>
    private PowellOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field keeps track of the number of completed iterations in the optimization process.
    /// It is used to enforce the maximum iteration limit and can also be useful for monitoring
    /// the progress of the optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like counting how many steps the hiker has taken.
    /// 
    /// The iteration counter:
    /// - Keeps track of how many rounds of optimization have been completed
    /// - Helps decide when to stop if progress is too slow
    /// - Can be used to monitor how efficiently the algorithm is working
    /// 
    /// This is important because we don't want the algorithm to run forever.
    /// </para>
    /// </remarks>
    private int _iteration;

    /// <summary>
    /// The current step size used in the line search process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current step size used during line search operations. The step size
    /// determines how far the algorithm moves along each direction when searching for improvements.
    /// When adaptive step size is enabled, this value changes based on the progress of the optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the length of the hiker's stride.
    /// 
    /// The adaptive step size controls:
    /// - How far the algorithm moves in each step when exploring
    /// - A larger step size covers more ground but might miss details
    /// - A smaller step size is more precise but slower
    /// 
    /// The algorithm can automatically adjust this value - taking bigger steps when moving in promising
    /// directions and smaller steps when fine-tuning a solution.
    /// </para>
    /// </remarks>
    private T _adaptiveStepSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="PowellOptimizer{T}"/> class with the specified options and components.
    /// </summary>
    /// <param name="options">The Powell optimization options, or null to use default options.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Powell optimizer with the specified options and components.
    /// If any parameter is null, a default implementation is used. The constructor initializes
    /// the adaptive step size to zero and sets up the initial adaptive parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    /// 
    /// Think of it like preparing for a hiking expedition:
    /// - You can provide custom gear (options) or use the standard equipment
    /// - You can provide specialized tools (evaluators, calculators) or use the basic ones
    /// - It gets everything ready so you can start the optimization journey
    /// 
    /// The options control things like how many steps to take, how precise to be, and when to stop.
    /// </para>
    /// </remarks>
    public PowellOptimizer(
        IFullModel<T, TInput, TOutput> model,
        PowellOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new PowellOptimizerOptions<T, TInput, TOutput>();
        _adaptiveStepSize = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Powell algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to initialize Powell-specific adaptive parameters.
    /// It sets the initial step size from the options and resets the iteration counter to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the optimizer for a fresh start.
    /// 
    /// It's like a hiker preparing for a new day of exploration:
    /// - Setting their stride length to a comfortable starting value
    /// - Resetting their step counter to zero
    /// - Getting ready to begin a new search pattern
    /// 
    /// These initial settings help the algorithm start with a balanced approach that
    /// can be adjusted as it learns more about the landscape.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _adaptiveStepSize = NumOps.FromDouble(_options.InitialStepSize);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the Powell optimization to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main Powell optimization algorithm. It starts from a random solution and
    /// iteratively improves it by performing line searches along a set of directions. After each cycle through
    /// all directions, it extrapolates a new point and evaluates whether this extrapolation provides further
    /// improvement. The algorithm continues until either the maximum number of iterations is reached, early
    /// stopping criteria are met, or the improvement falls below the specified tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main search process where the algorithm looks for the best solution.
    /// 
    /// The process works like this:
    /// 1. Start at a random position in the landscape
    /// 2. For each iteration:
    ///    - Try moving along each basic direction (north, south, east, west) to find better spots
    ///    - Once all directions are tried, try a jump in a new direction based on the overall movement
    ///    - Remember the best position found so far
    ///    - Adjust how the search is performed based on progress
    /// 3. Stop when enough iterations are done, when no more improvement is happening, or when the
    ///    improvement is very small
    /// 
    /// This systematic approach helps find good solutions efficiently, even when the landscape is complex.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        // Use model's parameter count instead of input size for direction dimensions
        int n = currentSolution.ParameterCount;
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;

            var directions = GenerateDirections(n);
            var newSolution = currentSolution;

            for (int i = 0; i < n; i++)
            {
                var lineSearchResult = LineSearch(newSolution, directions[i], inputData);
                newSolution = lineSearchResult.OptimalSolution;
            }

            var extrapolatedPoint = ExtrapolatePoint(currentSolution, newSolution);
            var extrapolatedStepData = EvaluateSolution(extrapolatedPoint, inputData);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            if (NumOps.GreaterThan(extrapolatedStepData.FitnessScore, currentStepData.FitnessScore))
            {
                currentSolution = extrapolatedPoint;
                currentStepData = extrapolatedStepData;
            }
            else
            {
                currentSolution = newSolution;
                currentStepData = EvaluateSolution(newSolution, inputData);
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

    /// <summary>
    /// Generates a set of orthogonal search directions.
    /// </summary>
    /// <param name="n">The dimensionality of the search space.</param>
    /// <returns>A list of unit vectors representing the search directions.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a set of n orthogonal unit vectors that form the initial search directions
    /// for Powell's method. These directions correspond to the standard basis vectors (e.g., [1,0,0], [0,1,0], etc.),
    /// which are aligned with the coordinate axes of the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a compass for the algorithm to navigate with.
    /// 
    /// Think of it like a hiker deciding which directions to explore:
    /// - If we're in a 2D landscape, the directions would be "north" and "east"
    /// - If we're in a 3D landscape, the directions would be "north", "east", and "up"
    /// - For higher dimensions, we need more directions to cover all possible movements
    /// 
    /// These basic directions help the algorithm systematically explore the search space
    /// by moving along one dimension at a time.
    /// </para>
    /// </remarks>
    private List<Vector<T>> GenerateDirections(int n)
    {
        var directions = new List<Vector<T>>();
        for (int i = 0; i < n; i++)
        {
            var direction = new Vector<T>(n);
            direction[i] = NumOps.One;
            directions.Add(direction);
        }
        return directions;
    }

    /// <summary>
    /// Performs a line search to find the optimal step size along a given direction.
    /// </summary>
    /// <param name="currentSolution">The current solution from which to start the line search.</param>
    /// <param name="direction">The direction vector along which to search.</param>
    /// <param name="inputData">The input data to evaluate solutions against.</param>
    /// <returns>A tuple containing the optimal solution found and the optimal step size.</returns>
    /// <remarks>
    /// <para>
    /// This method implements a golden section search to find the optimal step size along a specified direction.
    /// It repeatedly narrows the search interval until the interval width is less than the adaptive step size,
    /// at which point it returns the midpoint of the interval as the optimal step size.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a hiker looking for the highest point along a specific path.
    /// 
    /// Imagine a hiker standing on a hill:
    /// - They want to know how far to walk along a certain direction to reach the highest point
    /// - They check a few spots along that path to narrow down where the peak might be
    /// - They keep focusing on smaller and smaller sections of the path until they find the best spot
    /// - The golden ratio helps them efficiently decide which spots to check
    /// 
    /// This process helps find the best position along a single direction before moving to the next direction.
    /// </para>
    /// </remarks>
    private (IFullModel<T, TInput, TOutput> OptimalSolution, T OptimalStep) LineSearch(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var a = NumOps.FromDouble(-1.0);
        var b = NumOps.One;
        var goldenRatio = NumOps.FromDouble((Math.Sqrt(5) - 1) / 2);

        while (NumOps.GreaterThan(NumOps.Subtract(b, a), _adaptiveStepSize))
        {
            var c = NumOps.Subtract(b, NumOps.Multiply(goldenRatio, NumOps.Subtract(b, a)));
            var d = NumOps.Add(a, NumOps.Multiply(goldenRatio, NumOps.Subtract(b, a)));

            var fc = EvaluateSolution(MoveInDirection(currentSolution, direction, c), inputData).FitnessScore;
            var fd = EvaluateSolution(MoveInDirection(currentSolution, direction, d), inputData).FitnessScore;

            if (NumOps.GreaterThan(fc, fd))
            {
                b = d;
            }
            else
            {
                a = c;
            }
        }

        var optimalStep = NumOps.Divide(NumOps.Add(a, b), NumOps.FromDouble(2.0));
        return (MoveInDirection(currentSolution, direction, optimalStep), optimalStep);
    }

    /// <summary>
    /// Creates a new solution by moving the current solution along a specified direction with a given step size.
    /// </summary>
    /// <param name="solution">The current solution to move from.</param>
    /// <param name="direction">The direction vector to move along.</param>
    /// <param name="step">The step size determining how far to move.</param>
    /// <returns>A new solution representing the moved position.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new solution by moving the current solution along a specified direction
    /// with a given step size. It computes the new coefficients by adding the product of the direction
    /// vector and the step size to the current coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a hiker taking a step in a specific direction.
    /// 
    /// Imagine a hiker:
    /// - Standing at their current position (the current solution)
    /// - Facing a specific direction (like north, east, etc.)
    /// - Taking a step of a specific length (the step size)
    /// - Arriving at a new position (the new solution)
    /// 
    /// This simple movement is the basic building block of how the algorithm explores the search space.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> MoveInDirection(IFullModel<T, TInput, TOutput> solution, Vector<T> direction, T step)
    {
        // === Vectorized Move in Direction using IEngine (Phase B: US-GPU-015) ===
        // newCoefficients = parameters + step * direction

        var parameters = solution.GetParameters();
        var scaledDirection = (Vector<T>)AiDotNetEngine.Current.Multiply(direction, step);
        var newCoefficients = (Vector<T>)AiDotNetEngine.Current.Add(parameters, scaledDirection);

        return solution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Extrapolates a new point based on the movement from an old point to a new point.
    /// </summary>
    /// <param name="oldPoint">The starting point of the movement.</param>
    /// <param name="newPoint">The ending point of the movement.</param>
    /// <returns>An extrapolated point continuing the movement trend.</returns>
    /// <remarks>
    /// <para>
    /// This method creates an extrapolated point by continuing the movement from an old point to a new point.
    /// It computes the extrapolated coefficients by reflecting the old point through the new point,
    /// effectively doubling the movement vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a hiker predicting where they'd end up if they kept going.
    /// 
    /// Imagine a hiker:
    /// - Who was at point A (the old point)
    /// - And has now moved to point B (the new point)
    /// - Wondering "If I keep going in this same direction and same distance, where would I end up?"
    /// - That predicted position is point C (the extrapolated point)
    /// 
    /// This extrapolation helps the algorithm make bigger jumps in promising directions,
    /// potentially speeding up the search process.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> ExtrapolatePoint(IFullModel<T, TInput, TOutput> oldPoint, IFullModel<T, TInput, TOutput> newPoint)
    {
        // === Vectorized Extrapolation using IEngine (Phase B: US-GPU-015) ===
        // extrapolated = new + (new - old) = 2*new - old

        var parameters = newPoint.GetParameters();
        var oldParameters = oldPoint.GetParameters();

        // Calculate the direction vector: new - old
        var direction = (Vector<T>)AiDotNetEngine.Current.Subtract(parameters, oldParameters);
        // Extrapolate: new + direction = new + (new - old)
        var extrapolatedCoefficients = (Vector<T>)AiDotNetEngine.Current.Add(parameters, direction);

        // Use the newPoint as a template to create a new model with the extrapolated coefficients
        return newPoint.WithParameters(extrapolatedCoefficients);
    }

    /// <summary>
    /// Updates adaptive parameters based on optimization progress.
    /// </summary>
    /// <param name="currentStepData">The data from the current optimization step.</param>
    /// <param name="previousStepData">The data from the previous optimization step.</param>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update Powell-specific adaptive parameters
    /// in addition to the base adaptive parameters. It adjusts the adaptive step size based on whether
    /// the algorithm is making progress. If there is improvement, the step size increases; otherwise,
    /// it decreases. The step size is kept within specified minimum and maximum limits.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm searches based on its progress.
    /// 
    /// It's like a hiker changing their approach:
    /// - If they're finding better viewpoints, they might take bigger steps to explore more quickly
    /// - If they're not finding improvements, they might take smaller steps to search more carefully
    /// - The step size always stays between minimum and maximum values to avoid extremes
    /// 
    /// These adaptive adjustments help the algorithm be more efficient by being bold when things
    /// are going well and cautious when progress is difficult.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveStepSize)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _adaptiveStepSize = NumOps.Multiply(_adaptiveStepSize, NumOps.Add(NumOps.One, adaptationRate));
            }
            else
            {
                _adaptiveStepSize = NumOps.Multiply(_adaptiveStepSize, NumOps.Subtract(NumOps.One, adaptationRate));
            }

            _adaptiveStepSize = MathHelper.Clamp(_adaptiveStepSize, NumOps.FromDouble(_options.MinStepSize), NumOps.FromDouble(_options.MaxStepSize));
        }
    }

    /// <summary>
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the options are not of the expected type.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to update the Powell-specific options.
    /// It checks that the provided options are of the correct type (PowellOptimizerOptions)
    /// and throws an exception if they are not.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the settings that control how the optimizer works.
    /// 
    /// It's like changing the rules of the game:
    /// - You provide a set of options to use
    /// - The method checks that these are the right kind of options for a Powell optimizer
    /// - If they are, it applies these new settings
    /// - If not, it lets you know there's a problem
    /// 
    /// This ensures that only appropriate settings are used with this specific optimizer.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is PowellOptimizerOptions<T, TInput, TOutput> powellOptions)
        {
            _options = powellOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected PowellOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current Powell optimization options.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to return the Powell-specific options.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the current settings of the optimizer.
    /// 
    /// It's like checking what game settings are currently active:
    /// - You can see the current step size settings
    /// - You can see the current tolerance and iteration limits
    /// - You can see all the other parameters that control the optimizer
    /// 
    /// This is useful for understanding how the optimizer is currently configured
    /// or for making a copy of the settings to modify and apply later.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the Powell optimizer to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to include Powell-specific information in the serialization.
    /// It first serializes the base class data, then adds the Powell options, iteration count, and adaptive step size.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer so it can be restored later.
    /// 
    /// It's like taking a snapshot of the optimizer:
    /// - First, it saves all the general optimizer information
    /// - Then, it saves the Powell-specific settings and state
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
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_iteration);
            writer.Write(Convert.ToDouble(_adaptiveStepSize));

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Reconstructs the Powell optimizer from a serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when the options cannot be deserialized.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to handle Powell-specific information during deserialization.
    /// It first deserializes the base class data, then reconstructs the Powell options, iteration count, and adaptive step size.
    /// </para>
    /// <para><b>For Beginners:</b> This method restores the optimizer from a previously saved state.
    /// 
    /// It's like restoring from a snapshot:
    /// - First, it loads all the general optimizer information
    /// - Then, it loads the Powell-specific settings and state
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
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<PowellOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _adaptiveStepSize = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}
