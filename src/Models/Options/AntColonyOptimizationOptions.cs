namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Ant Colony Optimization algorithm, which is inspired by the foraging behavior of ants
/// to find optimal paths through a search space.
/// </summary>
/// <remarks>
/// <para>
/// Ant Colony Optimization (ACO) is a probabilistic technique for solving computational problems that can be reduced to finding
/// good paths through graphs. It simulates the behavior of ants seeking a path between their colony and a food source.
/// </para>
/// <para><b>For Beginners:</b> Ant Colony Optimization works like real ants finding food. When ants find food, they leave
/// a chemical trail (pheromone) on their way back to the nest. Other ants follow strong trails, making them stronger.
/// Trails that don't lead to food gradually fade away. This algorithm mimics this behavior to solve problems by having
/// virtual "ants" explore possible solutions and leave "trails" on good solutions that guide future searches.
/// It's especially good at finding efficient routes or combinations when there are many possibilities to consider.</para>
/// </remarks>
public class AntColonyOptimizationOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the number of artificial ants used in each iteration of the algorithm.
    /// </summary>
    /// <value>The number of ants, defaulting to 50.</value>
    /// <remarks>
    /// <para>
    /// Each ant constructs a potential solution to the problem. More ants can explore the solution space more thoroughly
    /// but require more computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This is simply how many virtual "ants" will search for solutions at the same time.
    /// More ants means a more thorough search but takes more computing power. The default value of 50 is a good balance
    /// for most problems. Think of it like sending out 50 scouts to explore different paths - enough to cover a good area
    /// without overwhelming your computer.</para>
    /// </remarks>
    public int AntCount { get; set; } = 50;

    /// <summary>
    /// Gets or sets the importance of heuristic information (problem-specific knowledge) relative to pheromone trails.
    /// </summary>
    /// <value>The beta parameter, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// Beta controls the balance between exploitation (following strong pheromone trails) and exploration (trying new paths
    /// based on heuristic information). Higher values give more weight to the heuristic information.
    /// </para>
    /// <para><b>For Beginners:</b> Beta determines how much the ants rely on their "instinct" (problem-specific knowledge)
    /// versus following existing trails. A higher value (like the default 2.0) means ants will pay more attention to what
    /// looks like a good path based on the problem's characteristics, rather than just following where others have gone before.
    /// It's like deciding whether to follow a map (instinct) or follow footprints (trails) when exploring.</para>
    /// </remarks>
    public double Beta { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the initial rate at which pheromone trails evaporate over time.
    /// </summary>
    /// <value>The initial pheromone evaporation rate, defaulting to 0.1 (10% per iteration).</value>
    /// <remarks>
    /// <para>
    /// Pheromone evaporation prevents the algorithm from converging too quickly to suboptimal solutions by reducing
    /// the influence of older pheromone deposits.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the "trails" left by ants fade away. The default value of 0.1
    /// means that 10% of each trail disappears after each round of searching. If trails didn't fade, the algorithm might get
    /// stuck following the first decent path it found instead of discovering better ones. Think of it like footprints in sand
    /// that gradually get washed away by waves, ensuring that only consistently good paths remain visible over time.</para>
    /// </remarks>
    public double InitialPheromoneEvaporationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the initial strength of pheromone deposits left by ants.
    /// </summary>
    /// <value>The initial pheromone intensity, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This value determines how strongly each ant's solution influences future iterations. The actual pheromone deposit
    /// is typically scaled by the quality of the solution found.
    /// </para>
    /// <para><b>For Beginners:</b> This is how strong the initial "trails" are when ants find a good path. The default
    /// value of 1.0 provides a baseline intensity. When an ant finds a good solution, it leaves a trail with this strength
    /// (adjusted by how good the solution is). Think of it as how much ink each ant has in its pen to mark the path - 
    /// enough to be visible but not so much that a single ant's path dominates everything else.</para>
    /// </remarks>
    public double InitialPheromoneIntensity { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the factor by which to decrease the pheromone evaporation rate when adaptation is needed.
    /// </summary>
    /// <value>The pheromone evaporation rate decay factor, defaulting to 0.95 (5% reduction).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm needs to focus more on exploitation of known good solutions, the evaporation rate can be decreased
    /// by this factor to preserve existing pheromone trails longer.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to slow down the trail fading process when the algorithm needs
    /// to focus more on using what it has already learned. The default value of 0.95 means reducing the fade rate by 5%.
    /// It's like deciding to preserve existing paths a bit longer so ants can make the most of promising discoveries before
    /// they fade away.</para>
    /// </remarks>
    public double PheromoneEvaporationRateDecay { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the factor by which to increase the pheromone evaporation rate when adaptation is needed.
    /// </summary>
    /// <value>The pheromone evaporation rate increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm needs to focus more on exploration of new solutions, the evaporation rate can be increased
    /// by this factor to make existing pheromone trails fade faster.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to speed up the trail fading process when the algorithm needs
    /// to explore new possibilities. The default value of 1.05 means increasing the fade rate by 5%. It's like deciding
    /// to erase existing paths more quickly to encourage ants to discover completely new routes instead of just refining
    /// known ones.</para>
    /// </remarks>
    public double PheromoneEvaporationRateIncrease { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the minimum allowed value for the pheromone evaporation rate.
    /// </summary>
    /// <value>The minimum pheromone evaporation rate, defaulting to 0.01 (1% per iteration).</value>
    /// <remarks>
    /// <para>
    /// This prevents the evaporation rate from becoming too small during adaptive adjustments, which could cause
    /// the algorithm to over-exploit existing solutions and get stuck in local optima.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a floor for how slowly trails can fade. Even if the algorithm wants
    /// to preserve trails for a long time, they will still fade at least 1% per round. This ensures that very old
    /// paths don't stick around forever, preventing the ants from getting permanently stuck following a decent but
    /// not optimal path.</para>
    /// </remarks>
    public double MinPheromoneEvaporationRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum allowed value for the pheromone evaporation rate.
    /// </summary>
    /// <value>The maximum pheromone evaporation rate, defaulting to 0.5 (50% per iteration).</value>
    /// <remarks>
    /// <para>
    /// This prevents the evaporation rate from becoming too large during adaptive adjustments, which could cause
    /// the algorithm to forget good solutions too quickly and fail to converge.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a ceiling for how quickly trails can fade. Even if the algorithm wants
    /// to encourage lots of exploration, trails won't fade more than 50% per round. This ensures that promising paths
    /// have a chance to be reinforced before disappearing completely. It's like making sure footprints don't vanish
    /// instantly in a sandstorm, giving other ants at least some chance to follow them.</para>
    /// </remarks>
    public double MaxPheromoneEvaporationRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the factor by which to decrease the pheromone intensity when adaptation is needed.
    /// </summary>
    /// <value>The pheromone intensity decay factor, defaulting to 0.95 (5% reduction).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm needs to reduce the influence of new solutions, the pheromone intensity can be decreased
    /// by this factor to make new pheromone deposits less significant.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to reduce the strength of new trails when the algorithm
    /// wants to be more conservative. The default value of 0.95 means reducing new trail strength by 5%. It's like
    /// giving ants less ink for their pens when you want them to make more subtle marks that don't overwhelm existing
    /// information.</para>
    /// </remarks>
    public double PheromoneIntensityDecay { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the factor by which to increase the pheromone intensity when adaptation is needed.
    /// </summary>
    /// <value>The pheromone intensity increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm needs to increase the influence of new solutions, the pheromone intensity can be increased
    /// by this factor to make new pheromone deposits more significant.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to increase the strength of new trails when the algorithm
    /// wants to emphasize new discoveries. The default value of 1.05 means increasing new trail strength by 5%. It's like
    /// giving ants more vibrant ink when you want new findings to stand out and have a stronger influence on future
    /// exploration.</para>
    /// </remarks>
    public double PheromoneIntensityIncrease { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the minimum allowed value for pheromone intensity on any path.
    /// </summary>
    /// <value>The minimum pheromone intensity, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This prevents pheromone levels from becoming too low during the optimization process, ensuring that all paths
    /// maintain at least some probability of being selected. This helps maintain diversity in the search process.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a floor for how faint a trail can become. Even if a path hasn't been used 
    /// for a while, it will still have at least this much "scent" left on it. The default value of 0.1 ensures that no 
    /// path is ever completely forgotten, giving the algorithm a chance to revisit previously unexplored or abandoned 
    /// paths. Think of it like making sure every path has at least a faint trail marker, so ants might occasionally 
    /// check it again rather than completely ignoring it forever.</para>
    /// </remarks>
    public double MinPheromoneIntensity { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed value for pheromone intensity on any path.
    /// </summary>
    /// <value>The maximum pheromone intensity, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This prevents pheromone levels from becoming excessively high during the optimization process, which could cause
    /// the algorithm to over-commit to certain paths and lose diversity in the search process.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a ceiling for how strong a trail can become. Even if a path is used 
    /// repeatedly, its "scent" won't get stronger than this value. The default value of 10.0 prevents any single path 
    /// from becoming so attractive that ants ignore all other possibilities. It's like limiting how much ink can 
    /// accumulate on a popular trail, ensuring that the ants don't become so fixated on one path that they completely 
    /// stop exploring alternatives that might actually be better in the long run.</para>
    /// </remarks>
    public double MaxPheromoneIntensity { get; set; } = 10.0;
}
