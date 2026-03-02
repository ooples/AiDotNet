namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Simulated Annealing optimization algorithm, a probabilistic technique
/// for approximating the global optimum of a given function.
/// </summary>
/// <remarks>
/// <para>
/// Simulated Annealing is a probabilistic optimization algorithm inspired by the annealing process in metallurgy, 
/// where metals are heated and then slowly cooled to reduce defects. In optimization, it uses a similar approach 
/// to find the global minimum of a function by occasionally accepting worse solutions to escape local minima. 
/// The algorithm starts with a high "temperature" that allows for large random moves in the solution space, 
/// including accepting worse solutions with high probability. As the temperature decreases according to a cooling 
/// schedule, the algorithm becomes more selective, gradually focusing on promising regions of the solution space. 
/// This class provides configuration options for controlling the annealing process, including temperature parameters, 
/// iteration limits, and neighborhood generation settings. Simulated Annealing is particularly useful for complex 
/// optimization problems with many local optima where deterministic methods might get trapped.
/// </para>
/// <para><b>For Beginners:</b> Simulated Annealing is an optimization technique inspired by metallurgy.
/// 
/// When metalworkers want to remove defects from metals, they:
/// - Heat the metal to a high temperature (atoms move freely)
/// - Slowly cool it down (atoms gradually settle into a low-energy state)
/// 
/// Simulated Annealing works similarly to find the best solution to a problem:
/// - It starts with a high "temperature" where it makes big, sometimes random changes
/// - It gradually "cools down," making smaller, more careful adjustments
/// - This approach helps it avoid getting stuck in mediocre solutions
/// 
/// The key insight is that sometimes accepting a worse solution temporarily
/// helps you find a better solution in the long run:
/// - At high temperatures, it frequently accepts worse solutions to explore widely
/// - At low temperatures, it rarely accepts worse solutions, focusing on refinement
/// 
/// This is particularly useful for complex problems with many possible solutions,
/// like route optimization, scheduling, or parameter tuning.
/// 
/// This class lets you configure exactly how the algorithm heats, cools, and explores
/// the solution space.
/// </para>
/// </remarks>
public class SimulatedAnnealingOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the SimulatedAnnealingOptions class with appropriate defaults.
    /// </summary>
    public SimulatedAnnealingOptions()
    {
        // SA typically needs more iterations than the base default of 100
        MaxIterations = 10000;
    }

    /// <summary>
    /// Gets or sets the initial temperature of the annealing process.
    /// </summary>
    /// <value>A positive double value, defaulting to 100.0.</value>
    /// <remarks>
    /// <para>
    /// The initial temperature determines how likely the algorithm is to accept worse solutions at the beginning 
    /// of the optimization process. A higher initial temperature increases the probability of accepting worse 
    /// solutions, promoting more exploration of the solution space. This is crucial for avoiding getting trapped 
    /// in local optima early in the search. The default value of 100.0 provides a good balance between exploration 
    /// and exploitation for many problems. For complex problems with many local optima, a higher initial temperature 
    /// might be appropriate to ensure sufficient exploration. For simpler problems or when a good initial solution 
    /// is provided, a lower initial temperature might be sufficient. The appropriate value depends on the scale of 
    /// the objective function and the complexity of the optimization landscape.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how "hot" the system starts - how willing it is to make random moves initially.
    /// 
    /// The temperature in Simulated Annealing determines:
    /// - How likely the algorithm is to accept a worse solution temporarily
    /// - Higher temperature = more randomness and exploration
    /// 
    /// The default value of 100.0 means:
    /// - The algorithm starts in a highly exploratory state
    /// - It will frequently accept worse solutions to avoid getting trapped in local optima
    /// 
    /// Think of it like this:
    /// - High temperature (e.g., 1000): Very random, like shaking a box of puzzle pieces vigorously
    /// - Medium temperature (e.g., 100): Moderately random, allowing significant exploration
    /// - Low temperature (e.g., 1): More focused, making mostly improvements with occasional random moves
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 500 or 1000) for complex problems with many local optima
    /// - Decrease it (e.g., to 10 or 50) when you have a good starting point or simpler problems
    /// 
    /// The ideal value depends on your specific problem and objective function scale.
    /// </para>
    /// </remarks>
    public double InitialTemperature { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets the cooling rate for the temperature reduction schedule.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.995.</value>
    /// <remarks>
    /// <para>
    /// The cooling rate determines how quickly the temperature decreases during the annealing process. After each 
    /// iteration or set of iterations, the current temperature is multiplied by this rate. A value closer to 1 
    /// results in slower cooling, allowing more exploration but requiring more iterations to converge. A value 
    /// further from 1 results in faster cooling, potentially converging more quickly but with a higher risk of 
    /// getting trapped in local optima. The default value of 0.995 provides a relatively slow cooling schedule 
    /// that works well for many problems. For complex problems requiring extensive exploration, a value even 
    /// closer to 1 (e.g., 0.999) might be appropriate. For simpler problems or when computational resources are 
    /// limited, a lower value (e.g., 0.99 or 0.98) might be used to accelerate convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the system "cools down" and becomes less random.
    /// 
    /// The cooling rate determines:
    /// - How quickly the algorithm transitions from exploration to exploitation
    /// - How many iterations it will take to reach the minimum temperature
    /// 
    /// The default value of 0.995 means:
    /// - After each iteration, the temperature is multiplied by 0.995
    /// - This creates a gradual cooling effect (temperature after n iterations = InitialTemperature × 0.995n)
    /// 
    /// Think of it like this:
    /// - Values closer to 1.0 (e.g., 0.999): Very slow cooling, more thorough exploration
    /// - Values further from 1.0 (e.g., 0.98): Faster cooling, quicker convergence but might miss optimal solutions
    /// 
    /// When to adjust this value:
    /// - Increase it (closer to 1, e.g., 0.999) for complex problems that need thorough exploration
    /// - Decrease it (further from 1, e.g., 0.99) when you need faster results and have simpler problems
    /// 
    /// For example, with InitialTemperature=100 and CoolingRate=0.995:
    /// - After 100 iterations: Temperature ˜ 60.6
    /// - After 500 iterations: Temperature ˜ 8.2
    /// - After 1000 iterations: Temperature ˜ 0.7
    /// </para>
    /// </remarks>
    public double CoolingRate { get; set; } = 0.995;

    /// <summary>
    /// Gets or sets the minimum temperature at which the annealing process stops.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-8 (0.00000001).</value>
    /// <remarks>
    /// <para>
    /// The minimum temperature defines a stopping criterion for the annealing process. When the current temperature 
    /// falls below this threshold, the algorithm transitions to a pure hill-climbing approach, accepting only 
    /// improvements to the current solution. This effectively ends the exploration phase of the algorithm. The 
    /// default value of 1e-8 is very small, ensuring that the algorithm has ample opportunity to explore the 
    /// solution space before focusing exclusively on exploitation. For problems requiring less exploration or 
    /// when computational resources are limited, a higher minimum temperature might be appropriate to terminate 
    /// the annealing process earlier. The appropriate value depends on the initial temperature, cooling rate, 
    /// and the specific characteristics of the optimization problem.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines when the system is considered "cold" and stops accepting worse solutions.
    /// 
    /// The minimum temperature:
    /// - Acts as a stopping condition for the cooling process
    /// - When reached, the algorithm essentially becomes a hill-climbing method (only accepting improvements)
    /// 
    /// The default value of 1e-8 (0.00000001) means:
    /// - The algorithm continues exploring until the temperature becomes extremely low
    /// - At this temperature, the probability of accepting worse solutions is virtually zero
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.1 or 1.0): The algorithm stops exploring earlier, potentially saving computation time
    /// - Lower values (e.g., 1e-10): The algorithm continues the annealing process longer, potentially finding better solutions
    /// 
    /// When to adjust this value:
    /// - Increase it when you want faster convergence and are less concerned about finding the absolute best solution
    /// - Decrease it when you want to ensure the algorithm thoroughly explores the solution space
    /// 
    /// This parameter works together with MaxIterations to determine when the algorithm stops -
    /// whichever condition is met first.
    /// </para>
    /// </remarks>
    public double MinTemperature { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the maximum temperature allowed during the annealing process.
    /// </summary>
    /// <value>A positive double value, defaulting to 1000.0.</value>
    /// <remarks>
    /// <para>
    /// The maximum temperature defines an upper bound for the temperature during the annealing process. This is 
    /// particularly relevant when using adaptive temperature schedules that might increase the temperature in 
    /// certain situations, such as when the algorithm appears to be trapped in a local optimum. The default value 
    /// of 1000.0 is significantly higher than the default initial temperature, allowing for substantial temperature 
    /// increases if needed. For problems requiring more aggressive exploration, a higher maximum temperature might 
    /// be appropriate. For problems where more controlled exploration is desired, a lower maximum temperature closer 
    /// to the initial temperature might be used. This parameter helps prevent the algorithm from becoming too random 
    /// in adaptive schemes while still allowing for temperature increases when beneficial.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how "hot" the system can get, even with adaptive temperature adjustments.
    /// 
    /// While the algorithm typically starts at InitialTemperature and cools down:
    /// - Some advanced implementations might temporarily increase temperature to escape local optima
    /// - This parameter sets an upper limit on how high the temperature can go
    /// 
    /// The default value of 1000.0 means:
    /// - The temperature will never exceed 1000, even if the algorithm tries to increase it
    /// - This prevents excessive randomness while still allowing for adaptive behavior
    /// 
    /// When to adjust this value:
    /// - Increase it if you're using adaptive temperature schedules and want to allow more dramatic
    ///   "reheating" to escape difficult local optima
    /// - Decrease it (closer to InitialTemperature) if you want more controlled exploration
    /// 
    /// This is primarily useful for advanced implementations of Simulated Annealing that use
    /// adaptive cooling schedules rather than monotonically decreasing temperatures.
    /// </para>
    /// </remarks>
    public double MaxTemperature { get; set; } = 1000.0;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the annealing process.
    /// </summary>
    /// <value>A positive integer, defaulting to 10000.</value>
    /// <remarks>
    /// <para>
    /// This property overrides the MaxIterations property inherited from the base class to provide a more 
    /// appropriate default value for Simulated Annealing. It specifies the maximum number of iterations allowed 
    /// before the algorithm terminates, regardless of whether other stopping criteria (such as reaching the 
    /// minimum temperature) have been met. The default value of 10000 is sufficient for many problems of moderate 
    /// complexity. For more complex problems with large solution spaces, a higher value might be needed to ensure 
    /// adequate exploration. For simpler problems or when computational resources are limited, a lower value might 
    /// be appropriate. This parameter helps prevent the algorithm from running indefinitely in cases where the 
    /// temperature decreases very slowly or other stopping criteria are difficult to meet.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits the total number of solution attempts the algorithm will make.
    /// 
    /// The maximum iterations:
    /// - Ensures the algorithm eventually stops, even if other stopping conditions aren't met
    /// - Provides a hard limit on computational resources used
    /// 
    /// The default value of 10000 means:
    /// - The algorithm will try at most 10000 different solutions
    /// - This is usually sufficient for problems of moderate complexity
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 50000 or 100000): More thorough exploration, better chance of finding optimal solution
    /// - Lower values (e.g., 1000 or 5000): Faster execution, but might not find the best solution
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems with large solution spaces
    /// - Decrease it when you need faster results or have simpler problems
    /// 
    /// This parameter works together with MinTemperature - the algorithm stops when either
    /// the temperature falls below MinTemperature OR the number of iterations reaches MaxIterations.
    /// </para>
    /// </remarks>
    // Note: SimulatedAnnealing default MaxIterations is set in the constructor
    // to avoid the 'new' keyword which causes JSON serialization issues with
    // duplicate property names in the base and derived class.

    /// <summary>
    /// Gets or sets the maximum number of consecutive iterations without improvement before early stopping.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of consecutive iterations allowed without improvement to the best 
    /// solution before the algorithm terminates early. This is an additional stopping criterion that can help save 
    /// computational resources when the algorithm appears to have stagnated. The default value of 1000 allows for 
    /// substantial exploration without improvement before concluding that further iterations are unlikely to yield 
    /// better results. For problems with complex landscapes where improvements might be rare but significant, a 
    /// higher value might be appropriate to prevent premature termination. For simpler problems or when computational 
    /// efficiency is a priority, a lower value might be used to terminate the search earlier when it appears to have 
    /// stagnated.
    /// </para>
    /// <para><b>For Beginners:</b> This setting allows the algorithm to stop early if it's not making progress.
    /// 
    /// While exploring the solution space:
    /// - The algorithm keeps track of the best solution found so far
    /// - This parameter counts how many iterations it will continue without finding a better solution
    /// 
    /// The default value of 1000 means:
    /// - If 1000 consecutive iterations pass without improving the best solution, the algorithm stops
    /// - This prevents wasting computation time when progress has stalled
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 5000): More patient, continues searching longer without improvement
    /// - Lower values (e.g., 500): More aggressive early stopping, saves computation time
    /// 
    /// When to adjust this value:
    /// - Increase it for difficult problems where improvements might be rare but valuable
    /// - Decrease it when you want faster results and are willing to accept potentially suboptimal solutions
    /// 
    /// This is an efficiency feature that helps balance thoroughness with computational cost.
    /// </para>
    /// </remarks>
    public int MaxIterationsWithoutImprovement { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the initial range for generating neighboring solutions.
    /// </summary>
    /// <value>A positive double value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property defines the initial scale or range used when generating neighboring solutions from the current 
    /// solution. In many implementations, this value represents a fraction of the parameter range or a step size for 
    /// perturbations. A larger value allows for bigger jumps in the solution space, promoting exploration, while a 
    /// smaller value focuses on local refinement. The default value of 0.1 (10% of the parameter range) provides a 
    /// moderate step size that works well for many problems. For problems requiring more fine-grained exploration, a 
    /// smaller value might be appropriate, while for problems with large or sparse solution spaces, a larger value 
    /// might be needed. Some implementations may adaptively adjust this range during the optimization process, using 
    /// this value as the starting point and respecting the minimum and maximum range constraints.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how different each new solution is from the current one.
    /// 
    /// When exploring the solution space:
    /// - The algorithm needs to generate "neighboring" solutions to consider
    /// - This parameter determines how far away these neighbors are from the current solution
    /// 
    /// The default value of 0.1 means:
    /// - New solutions are generated by changing the current solution by approximately 10%
    /// - This provides a moderate balance between small refinements and larger jumps
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 0.5): Bigger jumps, exploring more distant solutions
    /// - Smaller values (e.g., 0.05): Smaller steps, focusing on refining the current area
    /// 
    /// When to adjust this value:
    /// - Increase it when the solution space is large or sparse and you need to explore more widely
    /// - Decrease it when fine-tuning is important or when the optimal solution is likely near the starting point
    /// 
    /// Some implementations may adaptively adjust this range during optimization, starting with
    /// this value and changing it based on success rates or temperature.
    /// </para>
    /// </remarks>
    public double NeighborGenerationRange { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum range for generating neighboring solutions.
    /// </summary>
    /// <value>A positive double value, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// This property defines the lower bound for the range used when generating neighboring solutions. In adaptive 
    /// implementations that adjust the neighbor generation range during the optimization process, this value ensures 
    /// that the range doesn't become too small, which could limit the algorithm's ability to escape local optima or 
    /// make meaningful progress. The default value of 0.001 (0.1% of the parameter range) allows for fine-grained 
    /// local search while still permitting meaningful steps. For problems requiring extremely precise optimization, 
    /// a smaller value might be appropriate, while for problems where very small steps are not meaningful or efficient, 
    /// a larger value might be used. This parameter is particularly relevant in the later stages of the annealing 
    /// process when the temperature is low and the algorithm is focusing on exploitation rather than exploration.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the algorithm from taking steps that are too tiny to be useful.
    /// 
    /// In adaptive implementations:
    /// - The neighbor generation range might decrease over time or based on success rates
    /// - This parameter sets a lower limit on how small that range can become
    /// 
    /// The default value of 0.001 means:
    /// - The algorithm will never generate neighbors by changing the current solution by less than 0.1%
    /// - This ensures that even in the final stages, meaningful steps are still taken
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.01): Prevents the algorithm from making very small adjustments
    /// - Lower values (e.g., 0.0001): Allows for extremely fine-tuning in the final stages
    /// 
    /// When to adjust this value:
    /// - Increase it when very small changes don't meaningfully affect your objective function
    /// - Decrease it when extremely precise fine-tuning is important for your problem
    /// 
    /// This parameter is most relevant in the later stages of optimization when the algorithm
    /// is refining a promising solution rather than exploring broadly.
    /// </para>
    /// </remarks>
    public double MinNeighborGenerationRange { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the maximum range for generating neighboring solutions.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This property defines the upper bound for the range used when generating neighboring solutions. In adaptive 
    /// implementations that adjust the neighbor generation range during the optimization process, this value ensures 
    /// that the range doesn't become too large, which could make the search too random and inefficient. The default 
    /// value of 1.0 (100% of the parameter range) allows for very large steps that can jump across the entire solution 
    /// space when appropriate. For problems where more controlled exploration is desired, a smaller value might be 
    /// appropriate. For problems with very large or disconnected solution spaces, a larger value might be needed to 
    /// allow jumps between distant regions. This parameter is particularly relevant in the early stages of the annealing 
    /// process when the temperature is high and the algorithm is focusing on exploration rather than exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the algorithm from taking steps that are too large to be efficient.
    /// 
    /// In adaptive implementations:
    /// - The neighbor generation range might increase in certain situations (like when stuck in a local optimum)
    /// - This parameter sets an upper limit on how large that range can become
    /// 
    /// The default value of 1.0 means:
    /// - The algorithm will never generate neighbors by changing the current solution by more than 100%
    /// - This allows for large jumps across the solution space while preventing excessive randomness
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 2.0): Allows for very large jumps, potentially across the entire solution space
    /// - Lower values (e.g., 0.5): Keeps exploration more controlled and focused
    /// 
    /// When to adjust this value:
    /// - Increase it when the solution space is very large or has disconnected promising regions
    /// - Decrease it when more controlled exploration is desired or when large jumps are unlikely to be helpful
    /// 
    /// This parameter is most relevant in the early stages of optimization or when the algorithm
    /// is trying to escape local optima by making larger moves.
    /// </para>
    /// </remarks>
    public double MaxNeighborGenerationRange { get; set; } = 1.0;
}
