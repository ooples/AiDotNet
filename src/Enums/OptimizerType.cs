namespace AiDotNet.Enums;

/// <summary>
/// Defines different optimization algorithms used to train machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// For Beginners: Optimizers are like the "learning strategy" for AI models. When an AI model is learning, 
/// it needs to find the best values for its internal settings (called parameters or weights). Optimizers 
/// are different methods for finding these best values efficiently. Think of them as different strategies 
/// for climbing a mountain to reach the peak - some take small careful steps, others take bigger leaps, 
/// and some use special techniques to avoid getting stuck on small hills before reaching the highest peak.
/// </para>
/// </remarks>
public enum OptimizerType
{
    /// <summary>
    /// Adaptive Moment Estimation - combines the benefits of momentum and RMSProp optimizers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Adam is currently one of the most popular optimizers because it works well for many 
    /// different types of problems. It's like a smart hiker who remembers both the general direction they've 
    /// been moving (momentum) and adjusts their step size based on the terrain (adaptive learning rates). 
    /// This makes it good at navigating both steep and flat areas efficiently. Adam is often a good default 
    /// choice if you're not sure which optimizer to use.
    /// </para>
    /// </remarks>
    Adam,

    /// <summary>
    /// Classic optimization algorithm that updates parameters in the direction of steepest descent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Gradient Descent is like always walking directly downhill to find the lowest point in a valley. 
    /// It looks at all your training data at once before taking each step. This makes it very precise but can be slow 
    /// with large datasets. Imagine calculating the average direction from thousands of compass readings before taking 
    /// a single step - accurate but time-consuming.
    /// </para>
    /// </remarks>
    GradientDescent,

    /// <summary>
    /// Variant of gradient descent that uses random subsets of data for each update.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Stochastic Gradient Descent (SGD) is like taking quick steps based on limited information. 
    /// Instead of looking at all your training data before each step (which is slow), SGD looks at just one example 
    /// or a small batch. This makes it much faster but a bit less precise - like quickly changing direction based on 
    /// whatever you see right in front of you. It's faster but can take a more zigzag path to the solution.
    /// </para>
    /// </remarks>
    StochasticGradientDescent,

    /// <summary>
    /// Nature-inspired algorithm based on the foraging behavior of ant colonies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Ant Colony optimization mimics how ants find the shortest path to food. Ants leave chemical 
    /// trails (pheromones) that other ants follow, with stronger trails on shorter paths. In AI, this translates to 
    /// having multiple "agents" explore possible solutions and leave "markers" about how good each path is. Over time, 
    /// the algorithm converges on efficient solutions by following the strongest markers. This works well for problems 
    /// like finding optimal routes or sequences.
    /// </para>
    /// </remarks>
    AntColony,

    /// <summary>
    /// Evolutionary algorithm inspired by natural selection and genetics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Genetic Algorithms work like evolution in nature. They start with multiple random solutions 
    /// (the "population") and then:
    /// 1. Evaluate how good each solution is (survival fitness)
    /// 2. Select the best solutions to "reproduce" (selection)
    /// 3. Create new solutions by mixing parts of good solutions (crossover)
    /// 4. Occasionally make random changes (mutation)
    /// 
    /// Over many generations, good solutions emerge and get better. This is useful when you have complex problems 
    /// with many possible solutions and no clear mathematical approach.
    /// </para>
    /// </remarks>
    GeneticAlgorithm,

    /// <summary>
    /// Probabilistic technique inspired by the annealing process in metallurgy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Simulated Annealing is inspired by how metals cool. When hot, atoms move around freely; 
    /// as they cool, they settle into a stable structure. This optimizer starts with big, sometimes random changes 
    /// (high temperature) and gradually makes more careful, refined changes (cooling down). This helps it avoid 
    /// getting stuck in suboptimal solutions early on. It's like first exploring broadly across a landscape, then 
    /// gradually focusing your search in promising areas as you learn more.
    /// </para>
    /// </remarks>
    SimulatedAnnealing,

    /// <summary>
    /// Swarm intelligence algorithm inspired by the social behavior of birds or fish.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Particle Swarm optimization mimics how birds flock or fish school. Imagine multiple explorers 
    /// searching for treasure, each remembering the best spot they've personally found and also knowing the best spot 
    /// anyone in the group has found. Each explorer moves based on both their own experience and the group's knowledge. 
    /// This balance between individual exploration and group knowledge helps find good solutions efficiently, especially 
    /// for problems with many variables to optimize.
    /// </para>
    /// </remarks>
    ParticleSwarm,

    /// <summary>
    /// Standard optimization approach with fixed learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Beginners: Normal optimization refers to a basic approach with a fixed learning rate (step size). 
    /// It's like walking with the same stride length regardless of the terrain. This simplicity makes it easy 
    /// to understand and implement, but it may not adapt well to different parts of the learning process. It's 
    /// often used as a baseline or in simpler problems where adaptive techniques aren't necessary.
    /// </para>
    /// </remarks>
    Normal
}