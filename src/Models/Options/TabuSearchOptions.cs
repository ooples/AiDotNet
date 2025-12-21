namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Tabu Search, a metaheuristic optimization algorithm that enhances
/// local search by using memory structures to avoid revisiting previously explored solutions.
/// </summary>
/// <remarks>
/// <para>
/// Tabu Search is an iterative neighborhood search algorithm that enhances local search by avoiding points 
/// in the search space that have already been visited. The main feature of Tabu Search is the use of explicit 
/// memory (the tabu list) with two goals: to prevent the search from revisiting previously visited solutions, 
/// and to explore unvisited areas of the solution space. This approach helps escape local optima and avoid 
/// cycling in the search process. Tabu Search is particularly effective for combinatorial optimization problems 
/// where the search space is discrete and contains many local optima. This class inherits from 
/// GeneticAlgorithmOptimizerOptions and adds parameters specific to Tabu Search, such as tabu list size, 
/// neighborhood size, and various adaptive parameters for controlling the search process.
/// </para>
/// <para><b>For Beginners:</b> Tabu Search is like exploring a maze while keeping a list of recently visited paths to avoid.
/// 
/// When solving optimization problems:
/// - Simple local search methods often get stuck in "local optima" (solutions that are better than their neighbors but not the best overall)
/// - They can also waste time revisiting the same solutions repeatedly
/// 
/// Tabu Search solves this by:
/// - Keeping a "tabu list" of recently visited solutions that are temporarily forbidden
/// - Forcing the search to explore new areas even if they initially seem worse
/// - Adaptively adjusting its parameters based on search progress
/// - Combining elements of memory, neighborhood exploration, and strategic oscillation
/// 
/// This approach offers several benefits:
/// - Effectively escapes local optima
/// - Avoids cycling through the same solutions
/// - Efficiently explores the solution space
/// - Works well for complex combinatorial problems
/// 
/// This class lets you configure how the Tabu Search algorithm behaves.
/// </para>
/// </remarks>
public class TabuSearchOptions<T, TInput, TOutput> : GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the size of the tabu list.
    /// </summary>
    /// <value>A positive integer, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of recently visited solutions to keep in the tabu list. Solutions in 
    /// the tabu list are forbidden from being revisited for a certain number of iterations, preventing the search 
    /// from cycling through the same solutions. A larger tabu list provides a longer memory and more effectively 
    /// prevents cycling, but might be too restrictive and prevent the search from exploring promising areas. A 
    /// smaller tabu list provides less memory but more flexibility in the search. The default value of 10 provides 
    /// a moderate tabu list size suitable for many applications, balancing memory and flexibility. The optimal 
    /// value depends on the size and structure of the search space and the tendency of the problem to have cycles.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many recently visited solutions the algorithm remembers and avoids.
    /// 
    /// The tabu list size:
    /// - Determines how many recent solutions are "forbidden" (tabu)
    /// - Prevents the algorithm from revisiting the same solutions repeatedly
    /// - Helps the search escape local optima by forcing exploration of new areas
    /// 
    /// The default value of 10 means:
    /// - The 10 most recently visited solutions are remembered and avoided
    /// - This provides a moderate memory that works well for many problems
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 50): Longer memory, better at avoiding cycles, but might be too restrictive
    /// - Smaller values (e.g., 5): Shorter memory, more flexible search, but might revisit solutions
    /// 
    /// When to adjust this value:
    /// - Increase it when the algorithm keeps cycling through the same solutions
    /// - Decrease it when the search seems too restricted and can't find good solutions
    /// - Scale it with the size of your search space
    /// 
    /// For example, in a complex scheduling problem with many possible solutions,
    /// you might increase this to 20-30 to prevent cycling through similar schedules.
    /// </para>
    /// </remarks>
    public int TabuListSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the size of the neighborhood to explore in each iteration.
    /// </summary>
    /// <value>A positive integer, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of neighboring solutions to generate and evaluate in each iteration of 
    /// the search. The neighborhood consists of solutions that can be reached from the current solution by applying 
    /// a small modification or move. A larger neighborhood provides a more thorough exploration of the local area 
    /// but requires more computation per iteration. A smaller neighborhood requires less computation but might miss 
    /// promising solutions. The default value of 20 provides a moderate neighborhood size suitable for many 
    /// applications, balancing exploration and computational efficiency. The optimal value depends on the size and 
    /// structure of the search space and the computational resources available.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many nearby solutions the algorithm examines in each step.
    /// 
    /// The neighborhood size:
    /// - Determines how many alternative solutions are generated and evaluated in each iteration
    /// - Affects both the quality of the search and the computational cost
    /// 
    /// The default value of 20 means:
    /// - In each iteration, 20 neighboring solutions are created and evaluated
    /// - This provides a good balance between exploration and speed for many problems
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 50): More thorough exploration, better chance of finding good moves, but slower
    /// - Smaller values (e.g., 10): Faster iterations, but might miss good solutions
    /// 
    /// When to adjust this value:
    /// - Increase it when solution quality is more important than speed
    /// - Decrease it when computational resources are limited or for simpler problems
    /// - Scale it with the complexity of your problem
    /// 
    /// For example, in a vehicle routing problem with many constraints,
    /// you might increase this to 30-40 to ensure good routing options are considered.
    /// </para>
    /// </remarks>
    public int NeighborhoodSize { get; set; } = 20;

    /// <summary>
    /// Gets or sets the factor for perturbing solutions to generate neighbors.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the magnitude of the perturbation applied to the current solution to generate 
    /// neighboring solutions. It controls how different the neighbors are from the current solution. A larger 
    /// perturbation factor creates more diverse neighbors, potentially allowing for larger jumps in the search 
    /// space but with less focus on local improvement. A smaller perturbation factor creates more similar neighbors, 
    /// focusing more on local improvement but potentially limiting the ability to escape local optima. The default 
    /// value of 0.1 (10%) provides a moderate perturbation suitable for many applications, balancing local 
    /// improvement and diversity. The optimal value depends on the structure of the search space and the desired 
    /// balance between exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how different the neighboring solutions are from the current solution.
    /// 
    /// The perturbation factor:
    /// - Determines how much change is applied when generating neighboring solutions
    /// - Affects the balance between exploring nearby solutions and making larger jumps
    /// 
    /// The default value of 0.1 means:
    /// - Neighbors differ from the current solution by about 10%
    /// - This creates moderately different solutions that are still related to the current one
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 0.3): More diverse neighbors, bigger changes, better for escaping local optima
    /// - Smaller values (e.g., 0.05): More similar neighbors, smaller changes, better for fine-tuning
    /// 
    /// When to adjust this value:
    /// - Increase it when the algorithm seems stuck in local optima
    /// - Decrease it when the algorithm is finding good areas but not refining solutions well
    /// - Consider using adaptive strategies that vary this value during the search
    /// 
    /// For example, in a feature selection problem, a value of 0.1 might mean
    /// changing about 10% of the selected features when generating neighbors.
    /// </para>
    /// </remarks>
    public double PerturbationFactor { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the probability of mutation in the search process.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the probability of applying random mutations to solutions during the search process. 
    /// Mutations introduce random changes to solutions, helping to maintain diversity and explore new regions of 
    /// the search space. A higher mutation rate increases exploration but might disrupt good solutions, while a 
    /// lower rate preserves good solutions but might lead to premature convergence. The default value of 0.1 (10%) 
    /// provides a moderate mutation rate suitable for many applications, balancing exploration and exploitation. 
    /// This property overrides the MutationRate property inherited from the base class to provide a more appropriate 
    /// default value for Tabu Search. The optimal value depends on the complexity of the problem and the desired 
    /// balance between exploring new solutions and refining existing ones.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how often random changes are introduced to solutions.
    /// 
    /// The mutation rate:
    /// - Determines the probability of making random changes to solutions
    /// - Helps the algorithm explore new areas and maintain diversity
    /// - Prevents the search from becoming too focused on a single region
    /// 
    /// The default value of 0.1 means:
    /// - There's a 10% chance of applying a random mutation to a solution
    /// - This provides a good balance between exploration and stability for many problems
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.3): More exploration, more diversity, but may disrupt good solutions
    /// - Lower values (e.g., 0.05): More stability, better refinement of good solutions, but may get stuck
    /// 
    /// When to adjust this value:
    /// - Increase it when the algorithm seems to be converging too quickly to suboptimal solutions
    /// - Decrease it when good solutions are being found but need refinement
    /// - This setting overrides the value inherited from the base class
    /// 
    /// For example, in a complex optimization problem with many local optima,
    /// you might increase this to 0.2 to encourage more exploration.
    /// </para>
    /// </remarks>
    public new double MutationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum ratio of features to consider in the search.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum proportion of features or variables that should be considered in the 
    /// search process. It is particularly relevant for feature selection problems, where the goal is to identify 
    /// a subset of features that optimize some objective function. A smaller ratio allows for more compact solutions 
    /// with fewer features, while a larger ratio requires more features to be included. The default value of 0.1 (10%) 
    /// provides a lower bound that ensures at least some features are always included, preventing degenerate solutions. 
    /// The optimal value depends on the specific problem and the expected proportion of relevant features in the dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit on how many features or variables must be included in a solution.
    /// 
    /// The minimum feature ratio:
    /// - Ensures solutions don't become too sparse or simplified
    /// - Particularly important in feature selection problems
    /// - Prevents the algorithm from selecting too few features
    /// 
    /// The default value of 0.1 means:
    /// - Solutions must include at least 10% of the available features
    /// - For example, with 100 features, at least 10 must be selected
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.3): Forces more features to be included, potentially more comprehensive solutions
    /// - Lower values (e.g., 0.05): Allows more sparse solutions, potentially more efficient
    /// 
    /// When to adjust this value:
    /// - Increase it when you need more comprehensive solutions that consider more factors
    /// - Decrease it when you want to find the most minimal effective solution
    /// - Set based on domain knowledge about how many features are likely relevant
    /// 
    /// For example, in a medical diagnosis model where you know at least 20% of the
    /// symptoms are relevant, you might set this to 0.2.
    /// </para>
    /// </remarks>
    public double MinFeatureRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum ratio of features to consider in the search.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum proportion of features or variables that should be considered in the 
    /// search process. It is particularly relevant for feature selection problems, where the goal is to identify 
    /// a subset of features that optimize some objective function. A smaller ratio restricts solutions to fewer 
    /// features, potentially leading to more efficient models, while a larger ratio allows more features to be 
    /// included, potentially capturing more information. The default value of 0.9 (90%) provides an upper bound 
    /// that ensures at least some features are always excluded, preventing overly complex solutions. The optimal 
    /// value depends on the specific problem and the expected proportion of relevant features in the dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit on how many features or variables can be included in a solution.
    /// 
    /// The maximum feature ratio:
    /// - Ensures solutions don't become too complex or overfit
    /// - Particularly important in feature selection problems
    /// - Prevents the algorithm from selecting too many features
    /// 
    /// The default value of 0.9 means:
    /// - Solutions can include at most 90% of the available features
    /// - For example, with 100 features, at most 90 can be selected
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.95): Allows more comprehensive solutions, but potentially more complex
    /// - Lower values (e.g., 0.7): Forces more feature elimination, potentially more efficient solutions
    /// 
    /// When to adjust this value:
    /// - Increase it when you want to allow more comprehensive solutions
    /// - Decrease it when you want to force more feature elimination
    /// - Set based on domain knowledge about how many features are likely irrelevant
    /// 
    /// For example, in a text classification problem where you know many words are
    /// irrelevant, you might decrease this to 0.7 to force more feature elimination.
    /// </para>
    /// </remarks>
    public double MaxFeatureRatio { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the initial size of the tabu list.
    /// </summary>
    /// <value>A positive integer, defaulting to 50.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the initial size of the tabu list at the beginning of the search process. In adaptive 
    /// Tabu Search, the tabu list size might change during the search based on the search progress. Starting with a 
    /// larger tabu list provides more memory and more effectively prevents cycling in the early stages of the search, 
    /// when exploration is typically emphasized. The default value of 50 provides a relatively large initial tabu list 
    /// suitable for many applications, emphasizing exploration in the early stages. As the search progresses, the tabu 
    /// list size might be reduced to allow for more exploitation of promising areas. The optimal value depends on the 
    /// size and structure of the search space and the desired balance between exploration and exploitation in the early 
    /// stages of the search.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many solutions the algorithm initially remembers and avoids.
    /// 
    /// The initial tabu list size:
    /// - Sets the starting size of the tabu list before any adaptive adjustments
    /// - Affects how aggressively the algorithm explores in the early stages
    /// - Is typically larger than the regular tabu list size
    /// 
    /// The default value of 50 means:
    /// - The search starts by remembering and avoiding the 50 most recent solutions
    /// - This encourages broad exploration in the early stages
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 100): More aggressive initial exploration, better for complex problems
    /// - Smaller values (e.g., 20): Less restrictive initial search, better for simpler problems
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems with many local optima
    /// - Decrease it for simpler problems or when computational resources are limited
    /// - This value is particularly important if you're using adaptive tabu list sizing
    /// 
    /// For example, in a complex network optimization problem,
    /// you might increase this to 100 to ensure thorough initial exploration.
    /// </para>
    /// </remarks>
    public int InitialTabuListSize { get; set; } = 50;

    /// <summary>
    /// Gets or sets the initial size of the neighborhood.
    /// </summary>
    /// <value>A positive integer, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the initial size of the neighborhood at the beginning of the search process. In adaptive 
    /// Tabu Search, the neighborhood size might change during the search based on the search progress. Starting with a 
    /// specific neighborhood size allows for calibrating the initial exploration behavior. The default value of 20 
    /// provides a moderate initial neighborhood size suitable for many applications. As the search progresses, the 
    /// neighborhood size might be adjusted to balance exploration and exploitation based on the search progress. The 
    /// optimal value depends on the size and structure of the search space and the desired balance between exploration 
    /// and computational efficiency in the early stages of the search.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many neighboring solutions the algorithm examines in each step at the beginning of the search.
    /// 
    /// The initial neighborhood size:
    /// - Sets the starting number of neighbors generated in each iteration
    /// - Affects the initial balance between exploration thoroughness and speed
    /// - May be adjusted adaptively as the search progresses
    /// 
    /// The default value of 20 means:
    /// - The search starts by examining 20 neighboring solutions in each iteration
    /// - This provides a moderate level of exploration for many problems
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 40): More thorough initial exploration, better chance of finding good directions
    /// - Smaller values (e.g., 10): Faster initial iterations, but might miss promising directions
    /// 
    /// When to adjust this value:
    /// - Increase it when solution quality is more important than speed
    /// - Decrease it when computational resources are limited
    /// - This value is particularly important if you're using adaptive neighborhood sizing
    /// 
    /// For example, in a complex scheduling problem with many constraints,
    /// you might increase this to 30-40 to ensure good initial direction.
    /// </para>
    /// </remarks>
    public int InitialNeighborhoodSize { get; set; } = 20;

    /// <summary>
    /// Gets or sets the initial mutation rate.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the initial probability of applying random mutations to solutions at the beginning of 
    /// the search process. In adaptive Tabu Search, the mutation rate might change during the search based on the 
    /// search progress. Starting with a specific mutation rate allows for calibrating the initial exploration behavior. 
    /// The default value of 0.1 (10%) provides a moderate initial mutation rate suitable for many applications. As the 
    /// search progresses, the mutation rate might be adjusted to balance exploration and exploitation based on the 
    /// search progress. The optimal value depends on the complexity of the problem and the desired balance between 
    /// exploring new solutions and refining existing ones in the early stages of the search.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how often random changes are introduced to solutions at the beginning of the search.
    /// 
    /// The initial mutation rate:
    /// - Sets the starting probability of random mutations
    /// - Affects how exploratory the algorithm is in its early stages
    /// - May be adjusted adaptively as the search progresses
    /// 
    /// The default value of 0.1 means:
    /// - The search starts with a 10% chance of applying random mutations
    /// - This provides a moderate level of exploration for many problems
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.2): More exploration in early stages, better for complex problems
    /// - Lower values (e.g., 0.05): More stability in early stages, better when good starting solutions exist
    /// 
    /// When to adjust this value:
    /// - Increase it when the problem has many local optima and needs thorough exploration
    /// - Decrease it when you have good initial solutions that need refinement
    /// - This value is particularly important if you're using adaptive mutation rates
    /// 
    /// For example, in a complex optimization problem with no good initial solution,
    /// you might increase this to 0.2 to encourage more diverse exploration early on.
    /// </para>
    /// </remarks>
    public double InitialMutationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum mutation rate.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lower bound for the mutation rate when using adaptive mutation rates. It ensures 
    /// that the mutation rate doesn't become too small, which would limit the algorithm's ability to explore new 
    /// regions of the search space. The default value of 0.01 (1%) provides a small but non-negligible minimum 
    /// mutation rate, ensuring some level of exploration is maintained throughout the search. This property overrides 
    /// the MinMutationRate property inherited from the base class to provide a more appropriate default value for 
    /// Tabu Search. The optimal value depends on the complexity of the problem and the minimum level of exploration 
    /// desired even in the exploitation phases of the search.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit on how often random changes are introduced to solutions.
    /// 
    /// The minimum mutation rate:
    /// - Ensures the mutation rate doesn't drop too low during adaptive adjustments
    /// - Maintains some level of exploration even in later stages
    /// - Prevents the search from becoming too focused and getting stuck
    /// 
    /// The default value of 0.01 means:
    /// - The mutation rate will never drop below 1%
    /// - This ensures at least some exploration continues throughout the search
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.05): More guaranteed exploration, less chance of getting stuck
    /// - Lower values (e.g., 0.001): More potential for fine-tuning, but higher risk of getting stuck
    /// 
    /// When to adjust this value:
    /// - Increase it for problems with many local optima where exploration is always important
    /// - Decrease it when fine-tuning solutions is more important than exploration
    /// - This setting overrides the value inherited from the base class
    /// 
    /// For example, in a deceptive optimization problem where local optima are far from the global optimum,
    /// you might increase this to 0.05 to ensure continued exploration.
    /// </para>
    /// </remarks>
    public new double MinMutationRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum mutation rate.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the upper bound for the mutation rate when using adaptive mutation rates. It ensures 
    /// that the mutation rate doesn't become too large, which would make the search too random and disrupt good 
    /// solutions excessively. The default value of 0.5 (50%) provides a relatively high maximum mutation rate, 
    /// allowing for significant exploration when needed but preventing the search from becoming completely random. 
    /// This property overrides the MaxMutationRate property inherited from the base class to provide a more 
    /// appropriate default value for Tabu Search. The optimal value depends on the complexity of the problem and 
    /// the maximum level of exploration desired even in the most exploratory phases of the search.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit on how often random changes are introduced to solutions.
    /// 
    /// The maximum mutation rate:
    /// - Ensures the mutation rate doesn't rise too high during adaptive adjustments
    /// - Prevents the search from becoming too random
    /// - Maintains some level of stability even in exploratory phases
    /// 
    /// The default value of 0.5 means:
    /// - The mutation rate will never exceed 50%
    /// - This prevents excessive randomness while still allowing significant exploration
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.7): Allows more aggressive exploration when needed
    /// - Lower values (e.g., 0.3): Maintains more stability, less disruption to good solutions
    /// 
    /// When to adjust this value:
    /// - Increase it for very difficult problems where aggressive exploration is sometimes needed
    /// - Decrease it when stability is important and good solutions should be preserved
    /// - This setting overrides the value inherited from the base class
    /// 
    /// For example, in a highly multimodal function with many deceptive local optima,
    /// you might increase this to 0.7 to allow more aggressive exploration when needed.
    /// </para>
    /// </remarks>
    public new double MaxMutationRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the decay factor for the tabu list size.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factor by which the tabu list size is reduced when the search appears to be 
    /// converging or when more exploitation is desired. In adaptive Tabu Search, the tabu list size might be 
    /// reduced to allow for more exploitation of promising areas as the search progresses. A value closer to 1 
    /// results in a slower decay, maintaining a larger tabu list for longer, while a value closer to 0 results in 
    /// a faster decay, quickly reducing the tabu list size. The default value of 0.95 (95%) provides a gradual 
    /// decay suitable for many applications, slowly transitioning from exploration to exploitation. The optimal 
    /// value depends on the desired rate of transition from exploration to exploitation and the characteristics 
    /// of the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the tabu list size decreases during the search.
    /// 
    /// The tabu list size decay:
    /// - Determines the rate at which the tabu list shrinks when reduction is triggered
    /// - Affects how quickly the search transitions from exploration to exploitation
    /// - Used in adaptive strategies to adjust the search behavior
    /// 
    /// The default value of 0.95 means:
    /// - When triggered, the tabu list size is multiplied by 0.95 (reduced by 5%)
    /// - This provides a gradual reduction that slowly increases focus on promising areas
    /// 
    /// Think of it like this:
    /// - Values closer to 1 (e.g., 0.98): Slower decay, more gradual transition to exploitation
    /// - Values further from 1 (e.g., 0.9): Faster decay, quicker transition to exploitation
    /// 
    /// When to adjust this value:
    /// - Increase it (closer to 1) when you want a more gradual transition
    /// - Decrease it (further from 1) when you want a faster transition
    /// - This is particularly important in longer searches where adaptive behavior matters
    /// 
    /// For example, in a long-running optimization where gradual refinement is important,
    /// you might increase this to 0.98 for a very slow transition to exploitation.
    /// </para>
    /// </remarks>
    public double TabuListSizeDecay { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the increase factor for the tabu list size.
    /// </summary>
    /// <value>A double value greater than 1, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factor by which the tabu list size is increased when the search appears to be 
    /// cycling or when more exploration is desired. In adaptive Tabu Search, the tabu list size might be increased 
    /// to prevent cycling and encourage exploration of new areas when the search gets stuck. A value closer to 1 
    /// results in a slower increase, gradually enlarging the tabu list, while a larger value results in a faster 
    /// increase, quickly enlarging the tabu list. The default value of 1.05 (105%) provides a gradual increase 
    /// suitable for many applications, slowly transitioning from exploitation to exploration when needed. The 
    /// optimal value depends on the desired rate of transition from exploitation to exploration and the 
    /// characteristics of the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the tabu list size increases during the search.
    /// 
    /// The tabu list size increase:
    /// - Determines the rate at which the tabu list grows when expansion is triggered
    /// - Affects how quickly the search transitions from exploitation to exploration
    /// - Used in adaptive strategies to adjust the search behavior
    /// 
    /// The default value of 1.05 means:
    /// - When triggered, the tabu list size is multiplied by 1.05 (increased by 5%)
    /// - This provides a gradual increase that slowly enhances exploration
    /// 
    /// Think of it like this:
    /// - Values closer to 1 (e.g., 1.02): Slower increase, more gradual transition to exploration
    /// - Values further from 1 (e.g., 1.1): Faster increase, quicker transition to exploration
    /// 
    /// When to adjust this value:
    /// - Decrease it (closer to 1) when you want a more gradual transition
    /// - Increase it (further from 1) when you want a faster transition
    /// - This is particularly important when the search gets stuck and needs to diversify
    /// 
    /// For example, if your algorithm frequently gets stuck in local optima,
    /// you might increase this to 1.1 to more aggressively expand the tabu list when needed.
    /// </para>
    /// </remarks>
    public double TabuListSizeIncrease { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the minimum size of the tabu list.
    /// </summary>
    /// <value>A positive integer, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lower bound for the tabu list size when using adaptive tabu list sizing. It ensures 
    /// that the tabu list doesn't become too small, which would limit the algorithm's ability to prevent cycling. The 
    /// default value of 10 provides a moderate minimum tabu list size suitable for many applications, ensuring some 
    /// level of memory is maintained throughout the search. The optimal value depends on the minimum level of memory 
    /// desired even in the most exploitative phases of the search and the minimum number of solutions needed to 
    /// prevent cycling in the specific problem.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit on how small the tabu list can become.
    /// 
    /// The minimum tabu list size:
    /// - Ensures the tabu list doesn't become too small during adaptive adjustments
    /// - Maintains a minimum level of memory to prevent cycling
    /// - Acts as a safety mechanism in adaptive strategies
    /// 
    /// The default value of 10 means:
    /// - The tabu list will never contain fewer than 10 solutions
    /// - This ensures some protection against cycling even in exploitative phases
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 20): More guaranteed protection against cycling
    /// - Lower values (e.g., 5): More potential for exploitation, but higher risk of cycling
    /// 
    /// When to adjust this value:
    /// - Increase it for problems where cycling is a significant concern
    /// - Decrease it for problems where exploitation is more important than preventing cycling
    /// - Scale it with the size of your search space
    /// 
    /// For example, in a complex combinatorial problem with many similar solutions,
    /// you might increase this to 20 to ensure better protection against cycling.
    /// </para>
    /// </remarks>
    public int MinTabuListSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum size of the tabu list.
    /// </summary>
    /// <value>A positive integer, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the upper bound for the tabu list size when using adaptive tabu list sizing. It ensures 
    /// that the tabu list doesn't become too large, which would be computationally expensive and might be too 
    /// restrictive, preventing the search from exploring promising areas. The default value of 100 provides a 
    /// relatively large maximum tabu list size suitable for many applications, allowing for significant memory when 
    /// needed but preventing excessive restrictions. The optimal value depends on the maximum level of memory desired 
    /// even in the most exploratory phases of the search and the computational resources available.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit on how large the tabu list can become.
    /// 
    /// The maximum tabu list size:
    /// - Ensures the tabu list doesn't become too large during adaptive adjustments
    /// - Prevents excessive memory usage and computational overhead
    /// - Avoids over-restricting the search
    /// 
    /// The default value of 100 means:
    /// - The tabu list will never contain more than 100 solutions
    /// - This provides substantial memory for complex problems while avoiding excess
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 200): Allows more memory for complex problems, better at preventing cycling
    /// - Lower values (e.g., 50): More computationally efficient, less restrictive
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems with large search spaces where cycling is a major concern
    /// - Decrease it when computational resources are limited or when the search space is smaller
    /// - Scale it with the size of your search space
    /// 
    /// For example, in a very large optimization problem with millions of possible solutions,
    /// you might increase this to 200 to allow for more memory when needed.
    /// </para>
    /// </remarks>
    public int MaxTabuListSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the decay factor for the neighborhood size.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factor by which the neighborhood size is reduced when the search appears to be 
    /// converging or when more exploitation is desired. In adaptive Tabu Search, the neighborhood size might be 
    /// reduced to focus on promising areas as the search progresses. A value closer to 1 results in a slower decay, 
    /// maintaining a larger neighborhood for longer, while a value closer to 0 results in a faster decay, quickly 
    /// reducing the neighborhood size. The default value of 0.95 (95%) provides a gradual decay suitable for many 
    /// applications, slowly transitioning from exploration to exploitation. The optimal value depends on the desired 
    /// rate of transition from exploration to exploitation and the characteristics of the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the neighborhood size decreases during the search.
    /// 
    /// The neighborhood size decay:
    /// - Determines the rate at which the neighborhood shrinks when reduction is triggered
    /// - Affects how quickly the search transitions from broad exploration to focused exploitation
    /// - Used in adaptive strategies to adjust the search behavior
    /// 
    /// The default value of 0.95 means:
    /// - When triggered, the neighborhood size is multiplied by 0.95 (reduced by 5%)
    /// - This provides a gradual reduction that slowly increases focus on promising areas
    /// 
    /// Think of it like this:
    /// - Values closer to 1 (e.g., 0.98): Slower decay, more gradual transition to exploitation
    /// - Values further from 1 (e.g., 0.9): Faster decay, quicker transition to exploitation
    /// 
    /// When to adjust this value:
    /// - Increase it (closer to 1) when you want a more gradual transition
    /// - Decrease it (further from 1) when you want a faster transition
    /// - This is particularly important in longer searches where adaptive behavior matters
    /// 
    /// For example, in a long-running optimization where gradual refinement is important,
    /// you might increase this to 0.98 for a very slow transition to exploitation.
    /// </para>
    /// </remarks>
    public double NeighborhoodSizeDecay { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the increase factor for the neighborhood size.
    /// </summary>
    /// <value>A double value greater than 1, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factor by which the neighborhood size is increased when the search appears to be 
    /// stuck or when more exploration is desired. In adaptive Tabu Search, the neighborhood size might be increased 
    /// to explore more alternatives when the search gets stuck. A value closer to 1 results in a slower increase, 
    /// gradually enlarging the neighborhood, while a larger value results in a faster increase, quickly enlarging 
    /// the neighborhood. The default value of 1.05 (105%) provides a gradual increase suitable for many applications, 
    /// slowly transitioning from exploitation to exploration when needed. The optimal value depends on the desired 
    /// rate of transition from exploitation to exploration and the characteristics of the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the neighborhood size increases during the search.
    /// 
    /// The neighborhood size increase:
    /// - Determines the rate at which the neighborhood grows when expansion is triggered
    /// - Affects how quickly the search transitions from focused exploitation to broader exploration
    /// - Used in adaptive strategies to adjust the search behavior
    /// 
    /// The default value of 1.05 means:
    /// - When triggered, the neighborhood size is multiplied by 1.05 (increased by 5%)
    /// - This provides a gradual increase that slowly enhances exploration
    /// 
    /// Think of it like this:
    /// - Values closer to 1 (e.g., 1.02): Slower increase, more gradual transition to exploration
    /// - Values further from 1 (e.g., 1.1): Faster increase, quicker transition to exploration
    /// 
    /// When to adjust this value:
    /// - Decrease it (closer to 1) when you want a more gradual transition
    /// - Increase it (further from 1) when you want a faster transition
    /// - This is particularly important when the search gets stuck and needs to diversify
    /// 
    /// For example, if your algorithm frequently gets stuck in local optima,
    /// you might increase this to 1.1 to more aggressively expand the neighborhood when needed.
    /// </para>
    /// </remarks>
    public double NeighborhoodSizeIncrease { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the minimum size of the neighborhood.
    /// </summary>
    /// <value>A positive integer, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lower bound for the neighborhood size when using adaptive neighborhood sizing. 
    /// It ensures that the neighborhood doesn't become too small, which would limit the algorithm's ability to 
    /// explore alternatives and potentially lead to premature convergence. The default value of 5 provides a 
    /// moderate minimum neighborhood size suitable for many applications, ensuring some level of exploration is 
    /// maintained throughout the search. The optimal value depends on the minimum level of exploration desired 
    /// even in the most exploitative phases of the search and the minimum number of alternatives needed to make 
    /// meaningful progress in the specific problem.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit on how small the neighborhood can become.
    /// 
    /// The minimum neighborhood size:
    /// - Ensures the neighborhood doesn't become too small during adaptive adjustments
    /// - Maintains a minimum level of exploration even in exploitative phases
    /// - Acts as a safety mechanism in adaptive strategies
    /// 
    /// The default value of 5 means:
    /// - The algorithm will always examine at least 5 neighboring solutions in each iteration
    /// - This ensures some exploration even when focusing on promising areas
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 10): More guaranteed exploration, less chance of getting stuck
    /// - Lower values (e.g., 3): More potential for focused exploitation, but higher risk of getting stuck
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems where continued exploration is important
    /// - Decrease it for simpler problems or when computational efficiency is critical
    /// - Scale it with the complexity of your problem
    /// 
    /// For example, in a complex optimization problem with many local optima,
    /// you might increase this to 10 to ensure sufficient exploration throughout the search.
    /// </para>
    /// </remarks>
    public int MinNeighborhoodSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum size of the neighborhood.
    /// </summary>
    /// <value>A positive integer, defaulting to 50.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the upper bound for the neighborhood size when using adaptive neighborhood sizing. 
    /// It ensures that the neighborhood doesn't become too large, which would be computationally expensive and 
    /// might slow down the search excessively. The default value of 50 provides a relatively large maximum 
    /// neighborhood size suitable for many applications, allowing for significant exploration when needed but 
    /// preventing excessive computation. The optimal value depends on the maximum level of exploration desired 
    /// even in the most exploratory phases of the search and the computational resources available.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit on how large the neighborhood can become.
    /// 
    /// The maximum neighborhood size:
    /// - Ensures the neighborhood doesn't become too large during adaptive adjustments
    /// - Prevents excessive computational overhead
    /// - Balances exploration with efficiency
    /// 
    /// The default value of 50 means:
    /// - The algorithm will never examine more than 50 neighboring solutions in each iteration
    /// - This provides substantial exploration for complex problems while avoiding excess
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 100): Allows more thorough exploration, better for complex problems, but slower
    /// - Lower values (e.g., 30): More computationally efficient, faster iterations, but less thorough
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems where thorough exploration is critical
    /// - Decrease it when computational resources are limited or for simpler problems
    /// - Scale it with the complexity of your problem and available computational resources
    /// 
    /// For example, in a very complex combinatorial optimization problem,
    /// you might increase this to 100 to allow for more thorough exploration when needed.
    /// </para>
    /// </remarks>
    public int MaxNeighborhoodSize { get; set; } = 50;
}
