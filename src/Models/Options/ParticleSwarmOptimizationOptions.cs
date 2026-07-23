namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Particle Swarm Optimization (PSO), a population-based stochastic optimization
/// technique inspired by social behavior of bird flocking or fish schooling.
/// </summary>
/// <remarks>
/// <para>
/// Particle Swarm Optimization is a computational method that optimizes a problem by iteratively
/// improving candidate solutions (particles) with regard to a given measure of quality. The algorithm
/// maintains a population of particles, where each particle represents a potential solution to the
/// optimization problem. Particles move through the solution space guided by their own best known position
/// and the swarm's best known position. This social interaction leads to emergent intelligence that
/// efficiently explores complex solution spaces. PSO is particularly effective for continuous optimization
/// problems and has advantages in terms of simplicity, flexibility, and minimal parameter tuning requirements.
/// </para>
/// <para><b>For Beginners:</b> Particle Swarm Optimization is a way to find the best solution by mimicking how birds flock or fish swim in groups.
/// 
/// Imagine you and your friends are searching for the highest point in a hilly landscape while blindfolded:
/// - Each person (particle) can only feel the height at their current position
/// - Everyone remembers the highest point they personally have found so far
/// - The group also shares information about the highest point anyone has found
/// 
/// When deciding where to step next:
/// - You consider continuing in your current direction (inertia)
/// - You're pulled toward the best spot you've personally found (cognitive component)
/// - You're also pulled toward the best spot anyone in the group has found (social component)
/// - You combine these influences to decide your next move
/// 
/// This creates a smart search pattern because:
/// - People explore different areas (maintaining diversity in the search)
/// - The group gradually converges on promising regions
/// - The method balances exploration (finding new areas) with exploitation (refining good solutions)
/// 
/// This class provides extensive options to fine-tune how the particles move and interact,
/// allowing you to customize the algorithm for different types of optimization problems.
/// </para>
/// </remarks>
public class ParticleSwarmOptimizationOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the number of particles in the swarm.
    /// </summary>
    /// <value>The swarm size, defaulting to 50.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the population size of the swarm. Each particle represents a candidate
    /// solution that explores the search space. Larger swarms provide better coverage of the search space
    /// and are less likely to converge prematurely to local optima, but require more computational resources.
    /// Smaller swarms converge faster but may miss the global optimum. The optimal swarm size depends on
    /// the dimensionality and complexity of the problem being solved. For many problems, a swarm size of
    /// 10-50 particles provides a good balance between exploration and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many "searchers" (particles) 
    /// are looking for the solution simultaneously.
    /// 
    /// The default value of 50 means:
    /// - 50 different particles will explore the solution space
    /// - Each particle moves independently but shares information with others
    /// 
    /// Think of it like a search party:
    /// - More people can cover more ground, but require more coordination
    /// - Fewer people are easier to organize, but might miss areas
    /// 
    /// You might want more particles (like 100 or 200) if:
    /// - Your problem is complex with many dimensions
    /// - You suspect there are many local optima to avoid
    /// - You have the computational resources to spare
    /// 
    /// You might want fewer particles (like 20 or 30) if:
    /// - You need faster computation
    /// - Your problem is relatively simple
    /// - You're doing preliminary exploration
    /// 
    /// Finding the right swarm size balances thorough exploration with computational efficiency.
    /// </para>
    /// </remarks>
    public int SwarmSize { get; set; } = 50;

    /// <summary>
    /// Gets or sets the inertia weight that controls the influence of the particle's previous velocity.
    /// </summary>
    /// <value>The inertia weight, defaulting to 0.729.</value>
    /// <remarks>
    /// <para>
    /// The inertia weight controls how much of the particle's previous velocity is retained when updating
    /// its position. Higher values (near 1.0) favor exploration by maintaining momentum, allowing particles
    /// to overshoot optima and continue exploring new regions. Lower values prioritize exploitation of known
    /// good regions. The default value of 0.729 is commonly used in conjunction with acceleration coefficients
    /// of 1.49445 (cognitive and social parameters) as it provides guaranteed convergence in simplified systems.
    /// This parameterization is known as Clerc's constriction coefficient approach.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much a particle tends to
    /// continue moving in its current direction.
    /// 
    /// The default value of 0.729 means:
    /// - About 73% of the particle's previous movement direction is maintained
    /// - This creates a balance between exploring new areas and refining current good spots
    /// 
    /// Think of it like momentum when walking:
    /// - High inertia (like 0.9): You tend to keep going in the same direction, even if you see something interesting nearby
    /// - Low inertia (like 0.4): You can quickly change direction toward interesting areas
    /// 
    /// You might want higher inertia if:
    /// - Your problem has a complex landscape where thorough exploration is needed
    /// - The algorithm seems to get stuck in poor solutions
    /// - You want to prioritize exploration over exploitation
    /// 
    /// You might want lower inertia if:
    /// - You want faster convergence to good solutions
    /// - The algorithm seems to bounce around too much without settling
    /// - You want to prioritize refinement of promising solutions
    /// 
    /// The default value (0.729) is mathematically derived to provide good convergence properties
    /// when used with the default cognitive and social parameters (1.49445).
    /// </para>
    /// </remarks>
    public double InertiaWeight { get; set; } = 0.729;

    /// <summary>
    /// Gets or sets the cognitive parameter that controls the influence of the particle's personal best position.
    /// </summary>
    /// <value>The cognitive parameter, defaulting to 1.49445.</value>
    /// <remarks>
    /// <para>
    /// The cognitive parameter, also known as c1 or the personal acceleration coefficient, determines how much
    /// the particle is influenced by its own past best position. This coefficient controls the "nostalgia" of
    /// the particle, encouraging it to return to regions where it personally found good solutions. Higher values
    /// increase the attraction toward the particle's personal best position. The default value of 1.49445, when
    /// used with the corresponding social parameter and inertia weight, comes from Clerc's constriction
    /// coefficient approach, which provides theoretically backed convergence properties.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly a particle is attracted
    /// to the best position it has personally discovered.
    /// 
    /// The default value of 1.49445 means:
    /// - The particle is strongly influenced by its memory of the best spot it has found
    /// - This creates a balance between exploring new areas and revisiting proven good spots
    /// 
    /// Think of it like your own experience when searching:
    /// - High cognitive value: You strongly trust your own experience and return to places you found valuable
    /// - Low cognitive value: You're less attached to your past discoveries and more willing to try new directions
    /// 
    /// You might want a higher value if:
    /// - You want particles to more thoroughly explore areas they've found promising
    /// - You want to emphasize independent exploration over group consensus
    /// - The problem has many local optima where personal experience is valuable
    /// 
    /// You might want a lower value if:
    /// - You want particles to be less fixated on their own discoveries
    /// - You prefer faster convergence toward the group's best find
    /// - You have many particles and want more coordinated movement
    /// 
    /// The specific default value (1.49445) works in harmony with the default inertia and
    /// social parameter to provide mathematically proven convergence properties.
    /// </para>
    /// </remarks>
    public double CognitiveParameter { get; set; } = 1.49445;

    /// <summary>
    /// Gets or sets the social parameter that controls the influence of the swarm's global best position.
    /// </summary>
    /// <value>The social parameter, defaulting to 1.49445.</value>
    /// <remarks>
    /// <para>
    /// The social parameter, also known as c2 or the social acceleration coefficient, determines how much
    /// the particle is influenced by the best position found by any particle in the swarm. This coefficient
    /// controls the "social knowledge" of the particle, encouraging it to move toward the globally best known
    /// region. Higher values increase the attraction toward the swarm's best position, promoting faster
    /// convergence but potentially at the cost of diversity. The default value of 1.49445 is derived from
    /// Clerc's constriction coefficient approach.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly a particle is attracted
    /// to the best position found by any particle in the entire swarm.
    /// 
    /// The default value of 1.49445 means:
    /// - The particle is strongly influenced by the group's collective knowledge
    /// - This creates a balance between exploring independently and following the group's best find
    /// 
    /// Think of it like social influence:
    /// - High social value: You strongly trust the group's findings and head where others have found success
    /// - Low social value: You're more independent and less influenced by what others have found
    /// 
    /// You might want a higher value if:
    /// - You want faster convergence toward promising solutions
    /// - You believe the global best is likely near the true optimum
    /// - You want the swarm to focus on refining the best known solution
    /// 
    /// You might want a lower value if:
    /// - You want to maintain more diversity in the search
    /// - You're concerned about premature convergence to local optima
    /// - You want more thorough exploration before committing to a region
    /// 
    /// Like the cognitive parameter, the specific default value (1.49445) is part of a
    /// mathematically derived set that provides guaranteed convergence properties.
    /// </para>
    /// </remarks>
    public double SocialParameter { get; set; } = 1.49445;

    /// <summary>
    /// Gets or sets whether to use adaptive inertia weight that changes throughout the optimization process.
    /// </summary>
    /// <value>Whether to use adaptive inertia, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When set to true, this option enables dynamic adjustment of the inertia weight during the optimization
    /// process. Typically, a larger inertia weight is beneficial during early iterations to promote exploration,
    /// while a smaller inertia is more suitable in later iterations to refine solutions (exploitation). Adaptive
    /// inertia schemes automatically decrease the inertia weight over time or based on the swarm's performance,
    /// facilitating a natural transition from exploration to exploitation. This can improve performance on many
    /// problems compared to a fixed inertia weight.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the inertia weight should
    /// automatically change during the optimization process.
    /// 
    /// The default value of false means:
    /// - The inertia weight stays constant throughout the search
    /// - This provides consistent, predictable particle behavior
    /// 
    /// Setting this to true means:
    /// - The inertia weight will typically start higher (favoring exploration)
    /// - And gradually decrease (shifting toward exploitation)
    /// 
    /// Think of it like a search strategy that evolves over time:
    /// - First phase: Broad, wide-ranging exploration with high momentum
    /// - Later phases: More careful, focused refinement with less momentum
    /// 
    /// You might want to set this to true if:
    /// - You want the algorithm to automatically shift from exploration to exploitation
    /// - You're dealing with complex problems that need thorough initial exploration
    /// - You want to avoid manually tuning the inertia parameter
    /// 
    /// You might want to keep it false if:
    /// - You prefer consistent, predictable behavior throughout the search
    /// - You've already tuned the fixed inertia weight for your specific problem
    /// - You're using other adaptive mechanisms in the algorithm
    /// 
    /// When set to true, the actual inertia values used will be determined by other parameters
    /// like InitialInertia, MinInertia, MaxInertia, and InertiaDecayRate.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveInertia { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use adaptive cognitive and social parameters that change throughout the optimization.
    /// </summary>
    /// <value>Whether to use adaptive weights, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When set to true, this option enables dynamic adjustment of the cognitive and social parameters during
    /// the optimization process. Typically, a higher cognitive parameter (personal influence) and lower social
    /// parameter (swarm influence) are beneficial in early iterations to promote diversity and exploration.
    /// In later iterations, reducing the cognitive parameter and increasing the social parameter helps
    /// the swarm converge on promising solutions. This adaptive approach can improve performance by automatically
    /// balancing exploration and exploitation throughout the search process.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the cognitive and social parameters
    /// should automatically change during the optimization process.
    /// 
    /// The default value of false means:
    /// - The cognitive and social parameters remain constant throughout the search
    /// - Every particle maintains the same balance of personal vs. social influence
    /// 
    /// Setting this to true means:
    /// - Initially, particles will rely more on their personal experience (higher cognitive parameter)
    /// - Later, they'll be more influenced by the group's best find (higher social parameter)
    /// 
    /// Think of it like a team working on a problem:
    /// - Early phase: Everyone explores independently, trusting their own findings
    /// - Later phase: The team gradually coordinates more, focusing on the most promising areas
    /// 
    /// You might want to set this to true if:
    /// - You want to encourage diverse exploration early in the search
    /// - You want automatic convergence behavior later in the search
    /// - You're working on problems that benefit from this exploration-to-exploitation transition
    /// 
    /// You might want to keep it false if:
    /// - You want consistent, predictable behavior throughout
    /// - You've already tuned the fixed parameters for your specific problem
    /// - You prefer simpler algorithm behavior that's easier to analyze
    /// 
    /// When set to true, the actual parameter values will be determined by other settings such as
    /// InitialCognitiveWeight, InitialSocialWeight, and their min/max/adaptation rate values.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveWeights { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial inertia weight when using adaptive inertia.
    /// </summary>
    /// <value>The initial inertia weight, defaulting to 0.7.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the starting inertia weight when UseAdaptiveInertia is enabled. The inertia
    /// weight will gradually change from this initial value during the optimization process, typically
    /// decreasing to favor exploitation in later iterations. This parameter is only relevant when
    /// UseAdaptiveInertia is set to true. A moderate initial value allows particles to begin with a
    /// balance between exploration and exploitation, with the balance shifting as the optimization progresses.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the starting value for the inertia weight
    /// when adaptive inertia is enabled.
    /// 
    /// The default value of 0.7 means:
    /// - Particles start with 70% of their previous velocity maintained
    /// - This provides a moderate level of momentum for initial exploration
    /// 
    /// Think of it like setting the initial momentum for your search team:
    /// - Higher values (like 0.9) create more initial momentum, favoring broad exploration
    /// - Lower values (like 0.5) create less initial momentum, favoring more directed movement
    /// 
    /// You might want a higher value if:
    /// - Your problem requires thorough exploration of the search space
    /// - You want to avoid premature convergence to local optima
    /// - Your landscape has many hills and valleys to navigate
    /// 
    /// You might want a lower value if:
    /// - You have a good initial estimate of where the solution lies
    /// - You want faster initial convergence toward promising regions
    /// - Your problem is relatively well-behaved with fewer local optima
    /// 
    /// This setting has no effect unless UseAdaptiveInertia is set to true.
    /// </para>
    /// </remarks>
    public double InitialInertia { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the minimum inertia weight when using adaptive inertia.
    /// </summary>
    /// <value>The minimum inertia weight, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the lower bound for the inertia weight when UseAdaptiveInertia is enabled.
    /// The adaptive inertia scheme will not reduce the inertia below this value, ensuring that particles
    /// maintain at least some momentum even in later stages of optimization. This prevents the swarm from
    /// becoming completely static and allows for continued refinement of solutions. This parameter is only
    /// relevant when UseAdaptiveInertia is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit for how much the inertia weight
    /// can decrease when adaptive inertia is enabled.
    /// 
    /// The default value of 0.1 means:
    /// - The inertia weight will never drop below 10% of previous velocity
    /// - This ensures particles always maintain some degree of momentum
    /// 
    /// Think of it like setting a minimum speed for your search:
    /// - Even in late stages, searchers won't come to a complete stop
    /// - This prevents getting permanently stuck in a single location
    /// 
    /// You might want a lower value (like 0.05) if:
    /// - You want very fine-grained refinement of solutions in later stages
    /// - Your problem benefits from very small, careful steps near the optimum
    /// - You have many iterations to allow for gradual convergence
    /// 
    /// You might want a higher value (like 0.3) if:
    /// - You want to maintain more exploration capability throughout
    /// - You're concerned about getting stuck in local optima
    /// - Your problem landscape has many closely spaced local optima
    /// 
    /// This setting has no effect unless UseAdaptiveInertia is set to true.
    /// </para>
    /// </remarks>
    public double MinInertia { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum inertia weight when using adaptive inertia.
    /// </summary>
    /// <value>The maximum inertia weight, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the upper bound for the inertia weight when UseAdaptiveInertia is enabled.
    /// The adaptive inertia scheme will not increase the inertia above this value, preventing particles
    /// from gaining excessive momentum that might cause them to overshoot promising regions too drastically.
    /// This helps maintain stable exploration behavior. This parameter is only relevant when UseAdaptiveInertia
    /// is set to true and is particularly important in schemes that might dynamically increase inertia based
    /// on swarm performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit for how high the inertia weight
    /// can increase when adaptive inertia is enabled.
    /// 
    /// The default value of 0.9 means:
    /// - The inertia weight will never exceed 90% of previous velocity
    /// - This prevents particles from gaining too much momentum
    /// 
    /// Think of it like setting a speed limit for your search:
    /// - Prevents searchers from moving so quickly they miss important details
    /// - Keeps the search behavior stable and controllable
    /// 
    /// You might want a higher value (like 0.95) if:
    /// - Your search space is very large and requires extensive exploration
    /// - You want particles to more quickly traverse large regions
    /// - Your problem has wide, flat areas that need to be crossed efficiently
    /// 
    /// You might want a lower value (like 0.8) if:
    /// - You want more controlled, cautious exploration
    /// - Your problem has closely spaced features that require careful examination
    /// - You're concerned about numerical stability with high momentum
    /// 
    /// This setting has no effect unless UseAdaptiveInertia is set to true.
    /// </para>
    /// </remarks>
    public double MaxInertia { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the initial cognitive weight when using adaptive weights.
    /// </summary>
    /// <value>The initial cognitive weight, defaulting to 1.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the starting value for the cognitive parameter when UseAdaptiveWeights is enabled.
    /// The cognitive parameter controls how much a particle is influenced by its own past best position.
    /// In adaptive schemes, this weight typically starts higher to promote exploration based on personal
    /// experience, and then decreases to favor social learning in later stages. This parameter is only
    /// relevant when UseAdaptiveWeights is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the starting value for how strongly
    /// particles are attracted to their own personal best positions when adaptive weights are enabled.
    /// 
    /// The default value of 1.5 means:
    /// - Particles start with a relatively strong attraction to their personal best findings
    /// - This encourages initial independent exploration based on individual experience
    /// 
    /// Think of it like setting initial confidence in one's own discoveries:
    /// - Higher values make searchers more likely to revisit and explore around their own best finds
    /// - Lower values make searchers less attached to their previous discoveries
    /// 
    /// You might want a higher value (like 2.0) if:
    /// - You want more diverse, independent exploration early in the search
    /// - You suspect multiple promising regions that should be investigated thoroughly
    /// - You want to delay convergence to maintain diversity
    /// 
    /// You might want a lower value (like 1.0) if:
    /// - You want more coordinated exploration from the beginning
    /// - You have many particles and want to avoid too much redundant searching
    /// - You prefer faster initial convergence toward promising areas
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double InitialCognitiveWeight { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the initial social weight when using adaptive weights.
    /// </summary>
    /// <value>The initial social weight, defaulting to 1.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the starting value for the social parameter when UseAdaptiveWeights is enabled.
    /// The social parameter controls how much a particle is influenced by the best position found by any
    /// particle in the swarm. In adaptive schemes, this weight typically starts lower to reduce premature
    /// convergence, and then increases to promote convergence in later stages. This parameter is only
    /// relevant when UseAdaptiveWeights is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the starting value for how strongly
    /// particles are attracted to the swarm's global best position when adaptive weights are enabled.
    /// 
    /// The default value of 1.5 means:
    /// - Particles start with a moderate attraction to the group's best find
    /// - This balances independent exploration with some group coordination
    /// 
    /// Think of it like setting initial trust in the group's collective knowledge:
    /// - Higher values make searchers more likely to focus on the best area found by anyone
    /// - Lower values make searchers more independent from the group consensus
    /// 
    /// You might want a higher value (like 2.0) if:
    /// - You want faster initial convergence toward promising areas
    /// - You have confidence in the early discoveries being near the true optimum
    /// - You want more coordinated group behavior from the start
    /// 
    /// You might want a lower value (like 1.0) if:
    /// - You want to prevent premature convergence to potentially suboptimal solutions
    /// - You want to ensure thorough exploration before the swarm begins to converge
    /// - You're dealing with a problem with many local optima that could mislead the search
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double InitialSocialWeight { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the minimum cognitive weight when using adaptive weights.
    /// </summary>
    /// <value>The minimum cognitive weight, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the lower bound for the cognitive parameter when UseAdaptiveWeights is enabled.
    /// The adaptive weight scheme will not reduce the cognitive parameter below this value, ensuring that
    /// particles maintain some level of attraction to their personal best positions even in later stages.
    /// This helps prevent complete abandonment of personally discovered promising regions. This parameter
    /// is only relevant when UseAdaptiveWeights is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit for how much the cognitive weight
    /// can decrease when adaptive weights are enabled.
    /// 
    /// The default value of 0.5 means:
    /// - The attraction to personal best positions will never fall below this level
    /// - This ensures particles always maintain some memory of their own discoveries
    /// 
    /// Think of it like maintaining a minimum level of confidence in your own findings:
    /// - Even when the group has found good solutions, you still value your own experience
    /// - This prevents completely abandoning your own discoveries
    /// 
    /// You might want a lower value (like 0.2) if:
    /// - You want stronger convergence behavior in later stages
    /// - You believe the global best is likely to be near the true optimum
    /// - You want particles to eventually focus almost exclusively on refinement
    /// 
    /// You might want a higher value (like 0.8) if:
    /// - You want to maintain diversity throughout the search
    /// - You're concerned about premature convergence to suboptimal solutions
    /// - Your problem has many promising regions that should be continuously explored
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double MinCognitiveWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum cognitive weight when using adaptive weights.
    /// </summary>
    /// <value>The maximum cognitive weight, defaulting to 2.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the upper bound for the cognitive parameter when UseAdaptiveWeights is enabled.
    /// The adaptive weight scheme will not increase the cognitive parameter above this value, preventing
    /// particles from becoming excessively attracted to their personal best positions. This helps maintain
    /// stable search behavior by limiting the cognitive component's influence. This parameter is only relevant
    /// when UseAdaptiveWeights is set to true and is particularly important in schemes that might dynamically
    /// adjust weights based on swarm performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit for how high the cognitive weight
    /// can increase when adaptive weights are enabled.
    /// 
    /// The default value of 2.5 means:
    /// - The attraction to personal best positions will never exceed this level
    /// - This prevents particles from becoming overly fixated on their own discoveries
    /// 
    /// Think of it like setting a maximum level of confidence in your own findings:
    /// - Prevents searchers from being so attached to their own discoveries that they ignore others
    /// - Keeps the balance between personal experience and group knowledge
    /// 
    /// You might want a higher value (like 3.0) if:
    /// - You want to strongly emphasize individual exploration
    /// - You want particles to thoroughly investigate their personal best regions
    /// - Your problem benefits from very diverse, independent search behavior
    /// 
    /// You might want a lower value (like 2.0) if:
    /// - You want more moderate, balanced attraction forces
    /// - You're concerned about numerical stability with high acceleration
    /// - You prefer more coordination between particles
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double MaxCognitiveWeight { get; set; } = 2.5;

    /// <summary>
    /// Gets or sets the minimum social weight when using adaptive weights.
    /// </summary>
    /// <value>The minimum social weight, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the lower bound for the social parameter when UseAdaptiveWeights is enabled.
    /// The adaptive weight scheme will not reduce the social parameter below this value, ensuring that
    /// particles maintain some level of attraction to the global best position even in early stages of
    /// optimization. This helps prevent complete isolation of particles and enables some level of coordination
    /// within the swarm. This parameter is only relevant when UseAdaptiveWeights is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit for how much the social weight
    /// can decrease when adaptive weights are enabled.
    /// 
    /// The default value of 0.5 means:
    /// - The attraction to the swarm's best position will never fall below this level
    /// - This ensures particles always maintain some connection to the group's findings
    /// 
    /// Think of it like maintaining a minimum level of communication among searchers:
    /// - Even when focusing on independent exploration, searchers still consider the group's best find
    /// - This prevents completely isolated search behavior
    /// 
    /// You might want a lower value (like 0.2) if:
    /// - You want more independent exploration, especially early in the search
    /// - You're concerned about premature convergence
    /// - Your problem has many distinct regions that should be explored separately
    /// 
    /// You might want a higher value (like 0.8) if:
    /// - You want to maintain more coordinated search behavior throughout
    /// - You believe early discovered promising regions should be thoroughly exploited
    /// - You want faster convergence toward promising solutions
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double MinSocialWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum social weight when using adaptive weights.
    /// </summary>
    /// <value>The maximum social weight, defaulting to 2.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the upper bound for the social parameter when UseAdaptiveWeights is enabled.
    /// The adaptive weight scheme will not increase the social parameter above this value, preventing
    /// particles from becoming excessively attracted to the global best position. This helps avoid premature
    /// convergence and maintains diversity in the swarm. This parameter is only relevant when UseAdaptiveWeights
    /// is set to true and is particularly important in schemes that increase social influence in later stages.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit for how high the social weight
    /// can increase when adaptive weights are enabled.
    /// 
    /// The default value of 2.5 means:
    /// - The attraction to the swarm's best position will never exceed this level
    /// - This prevents excessive focus on a single promising solution
    /// 
    /// Think of it like setting a maximum level of group influence:
    /// - Prevents searchers from blindly rushing to the current best location
    /// - Maintains some individuality even when converging toward good solutions
    /// 
    /// You might want a higher value (like 3.0) if:
    /// - You want stronger convergence in later stages
    /// - You want to more aggressively refine the best found solution
    /// - Your problem benefits from focused exploitation once good regions are identified
    /// 
    /// You might want a lower value (like 2.0) if:
    /// - You want more balanced attraction forces
    /// - You're concerned about premature convergence
    /// - You prefer more gradual convergence behavior
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double MaxSocialWeight { get; set; } = 2.5;

    /// <summary>
    /// Gets or sets the rate at which inertia weight decays when using adaptive inertia.
    /// </summary>
    /// <value>The inertia decay rate, defaulting to 0.99.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how rapidly the inertia weight changes when UseAdaptiveInertia is enabled.
    /// In typical implementations, this represents a multiplicative factor applied to the inertia weight
    /// after each iteration. Values closer to 1.0 result in slower decay and more gradual transition from
    /// exploration to exploitation. Values further from 1.0 (smaller) cause faster decay and more rapid
    /// transition. This parameter is only relevant when UseAdaptiveInertia is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the inertia weight changes
    /// over time when adaptive inertia is enabled.
    /// 
    /// The default value of 0.99 means:
    /// - After each iteration, the inertia weight is multiplied by 0.99
    /// - This creates a very gradual reduction in momentum over time
    /// 
    /// Think of it like gradually slowing down as the search progresses:
    /// - A value near 1.0 (like 0.99) creates a very slow, gradual transition
    /// - A smaller value (like 0.95) creates a much faster transition from exploration to exploitation
    /// 
    /// You might want a value closer to 1.0 (like 0.995) if:
    /// - You want a very gradual transition from exploration to exploitation
    /// - You have many iterations available for the search
    /// - You want to maintain exploration capability for longer
    /// 
    /// You might want a smaller value (like 0.95) if:
    /// - You want a quicker transition to focused refinement
    /// - You have limited iterations available
    /// - You want the algorithm to settle faster
    /// 
    /// This setting has no effect unless UseAdaptiveInertia is set to true.
    /// </para>
    /// </remarks>
    public double InertiaDecayRate { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the rate at which the cognitive weight adapts when using adaptive weights.
    /// </summary>
    /// <value>The cognitive weight adaptation rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how rapidly the cognitive parameter changes when UseAdaptiveWeights is enabled.
    /// Higher values lead to faster adaptation, while lower values result in more gradual changes.
    /// The exact effect depends on the specific adaptive weight scheme implemented, but generally this
    /// acts as a multiplier on the amount of change applied to the cognitive parameter in each iteration.
    /// This parameter is only relevant when UseAdaptiveWeights is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the cognitive weight (attraction
    /// to personal best) changes over time when adaptive weights are enabled.
    /// 
    /// The default value of 1.0 means:
    /// - The cognitive weight changes at a standard rate
    /// - Neither accelerated nor slowed down
    /// 
    /// Think of it like setting the speed for changing from individual exploration to group consensus:
    /// - Higher values (like 1.5) make this transition happen faster
    /// - Lower values (like 0.5) make this transition happen more gradually
    /// 
    /// You might want a higher value if:
    /// - You want faster transition from exploration to exploitation
    /// - You have limited iterations available for the search
    /// - You want more responsive adaptation to the optimization progress
    /// 
    /// You might want a lower value if:
    /// - You want more gradual, stable changes in search behavior
    /// - You have many iterations available
    /// - You want the exploration phase to last longer
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double CognitiveWeightAdaptationRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the rate at which the social weight adapts when using adaptive weights.
    /// </summary>
    /// <value>The social weight adaptation rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how rapidly the social parameter changes when UseAdaptiveWeights is enabled.
    /// Higher values lead to faster adaptation, while lower values result in more gradual changes.
    /// The exact effect depends on the specific adaptive weight scheme implemented, but generally this
    /// acts as a multiplier on the amount of change applied to the social parameter in each iteration.
    /// This parameter is only relevant when UseAdaptiveWeights is set to true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the social weight (attraction
    /// to group's best position) changes over time when adaptive weights are enabled.
    /// 
    /// The default value of 1.0 means:
    /// - The social weight changes at a standard rate
    /// - Neither accelerated nor slowed down
    /// 
    /// Think of it like setting the speed for changing the level of group influence:
    /// - Higher values (like 1.5) make group consensus become influential more quickly
    /// - Lower values (like 0.5) delay the transition to strong group influence
    /// 
    /// You might want a higher value if:
    /// - You want faster convergence in later stages
    /// - You have limited iterations available for the search
    /// - You want more aggressive refinement once good regions are identified
    /// 
    /// You might want a lower value if:
    /// - You want to maintain diversity longer
    /// - You're concerned about premature convergence
    /// - You prefer more gradual transitions in search behavior
    /// 
    /// This setting has no effect unless UseAdaptiveWeights is set to true.
    /// </para>
    /// </remarks>
    public double SocialWeightAdaptationRate { get; set; } = 1.0;
}
