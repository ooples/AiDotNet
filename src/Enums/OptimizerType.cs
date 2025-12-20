namespace AiDotNet.Enums;

/// <summary>
/// Defines different optimization algorithms used to train machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Optimizers are like the "learning strategy" for AI models. When an AI model is learning, 
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
    /// <b>For Beginners:</b> Adam is currently one of the most popular optimizers because it works well for many 
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
    /// <b>For Beginners:</b> Gradient Descent is like always walking directly downhill to find the lowest point in a valley. 
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
    /// <b>For Beginners:</b> Stochastic Gradient Descent (SGD) is like taking quick steps based on limited information. 
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
    /// <b>For Beginners:</b> Ant Colony optimization mimics how ants find the shortest path to food. Ants leave chemical 
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
    /// <b>For Beginners:</b> Genetic Algorithms work like evolution in nature. They start with multiple random solutions 
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
    /// <b>For Beginners:</b> Simulated Annealing is inspired by how metals cool. When hot, atoms move around freely; 
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
    /// <b>For Beginners:</b> Particle Swarm optimization mimics how birds flock or fish school. Imagine multiple explorers 
    /// searching for treasure, each remembering the best spot they've personally found and also knowing the best spot 
    /// anyone in the group has found. Each explorer moves based on both their own experience and the group's knowledge. 
    /// This balance between individual exploration and group knowledge helps find good solutions efficiently, especially 
    /// for problems with many variables to optimize.
    /// </para>
    /// </remarks>
    ParticleSwarm,

    /// <summary>
    /// Gradient descent variant that adds a fraction of the previous update to the current one.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Momentum is like rolling a ball downhill. Once it starts moving in a direction, 
    /// it tends to keep going that way. This helps the optimizer move faster in consistent directions and 
    /// push through small bumps or plateaus that might otherwise slow it down. Think of it as having some 
    /// inertia that helps you maintain progress even when the path briefly levels off or gets noisy.
    /// </para>
    /// </remarks>
    Momentum,

    /// <summary>
    /// Advanced variant of Momentum that looks ahead to where the current momentum would take it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Nesterov Accelerated Gradient is like a runner who looks ahead to see what's coming. 
    /// Regular momentum is good but can overshoot targets. Nesterov first takes a step in the direction of the 
    /// previous momentum, then checks the gradient at that "peeked" position to make a correction. This 
    /// "look-before-you-leap" approach gives better control when approaching the optimal solution, like slowing 
    /// down when you see you're getting close to your destination.
    /// </para>
    /// </remarks>
    NesterovAcceleratedGradient,

    /// <summary>
    /// Adaptive learning rate method that scales learning rates individually for each parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adagrad adjusts the learning rate for each parameter based on how frequently
    /// it's been updated. Imagine having different step sizes for different terrains - taking smaller steps 
    /// on well-explored paths and larger steps in new areas. This works well for sparse data (where many 
    /// features are rarely seen) but can cause the learning rate to become too small over time as it 
    /// continuously shrinks, eventually making learning too slow.
    /// </para>
    /// </remarks>
    Adagrad,

    /// <summary>
    /// Extension of Adagrad that addresses the diminishing learning rates problem.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RMSprop (Root Mean Square Propagation) is an improvement on Adagrad that solves 
    /// the problem of the continuously shrinking learning rate. Instead of accumulating all past gradients, 
    /// it only considers recent ones using a moving average. It's like focusing on your recent navigation 
    /// history rather than every step you've ever taken. This prevents the step size from becoming too tiny 
    /// and helps maintain a reasonable pace throughout the learning process.
    /// </para>
    /// </remarks>
    RMSProp,

    /// <summary>
    /// Extension of Adagrad that uses a different approach to address the diminishing learning rates problem.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adadelta is another solution to Adagrad's diminishing learning rates, but it 
    /// goes a step further than RMSprop. It not only tracks a moving average of past squared gradients but 
    /// also maintains a moving average of past parameter updates. This allows it to continue learning even
    /// when the gradients become very small. Adadelta is unique because it doesn't even require setting an
    /// initial learning rate - it's like a hiker who can naturally adjust their pace based on both the 
    /// terrain and their own recent energy expenditure.
    /// </para>
    /// </remarks>
    Adadelta,

    /// <summary>
    /// Variant of Adam that uses the infinity norm instead of the L2 norm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adamax is a variation of Adam that uses a different way to track past gradients. 
    /// While Adam uses the average squared magnitude of past gradients, Adamax uses the maximum magnitude. 
    /// This makes it more robust to extreme gradient values (outliers) and can be especially useful for 
    /// problems where occasional large gradient spikes might throw off other optimizers. It's like focusing 
    /// on the biggest obstacle you've encountered rather than the average difficulty of the terrain.
    /// </para>
    /// </remarks>
    Adamax,

    /// <summary>
    /// Combination of Adam and Nesterov momentum for improved performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Nadam (Nesterov-accelerated Adam) combines the benefits of Adam with those of 
    /// Nesterov momentum. It takes Adam's ability to adapt learning rates individually for each parameter
    /// and adds Nesterov's "look-ahead" approach. This gives you both adaptive step sizes and better
    /// directional awareness - like a hiker who not only adjusts their stride based on the terrain but
    /// also scouts ahead before committing to a direction.
    /// </para>
    /// </remarks>
    Nadam,

    /// <summary>
    /// Variant of Adam that implements a decoupled weight decay for better generalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdamW improves on Adam by handling weight decay (a technique to prevent overfitting) 
    /// in a more effective way. Regular Adam applies weight decay to the already-adapted gradients, which can
    /// make it less effective. AdamW applies weight decay directly to the weights instead. This seemingly small
    /// change leads to better generalization - like making sure your backpack stays light throughout your journey,
    /// rather than only thinking about its weight when deciding how fast to walk. This helps the model perform 
    /// better on new, unseen examples.
    /// </para>
    /// </remarks>
    AdamW,

    /// <summary>
    /// Variant of Adam that rectifies the variance of the adaptive learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RAdam (Rectified Adam) addresses a technical problem with Adam in the early stages 
    /// of training when there isn't enough data to reliably estimate the adaptive learning rates. RAdam starts 
    /// with a more conservative approach and gradually increases its adaptivity as training progresses. It's like 
    /// a cautious explorer who starts with careful, measured steps in unfamiliar territory but becomes more 
    /// confident and efficient as they gather more information about the landscape.
    /// </para>
    /// </remarks>
    RAdam,

    /// <summary>
    /// Second-order optimization method that approximates the Hessian matrix to accelerate convergence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is an advanced optimizer 
    /// that uses information about the curvature of the error surface (not just the slope). While first-order 
    /// methods like SGD only know which way is downhill, LBFGS also has an idea of how quickly the slope is 
    /// changing in different directions. This is like having not just a compass but also a detailed topographic 
    /// map. This makes it incredibly efficient for problems where evaluating the model is expensive, though it 
    /// requires more memory and computation per step. It's typically used for smaller models or when high 
    /// precision is needed.
    /// </para>
    /// </remarks>
    LBFGS,

    /// <summary>
    /// Online learning algorithm that adapts to the data characteristics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FTRL (Follow The Regularized Leader) is specialized for online learning scenarios, 
    /// where data comes in a continuous stream rather than as a fixed dataset. It's particularly good at creating 
    /// sparse models (models where many parameters become exactly zero), which is useful when you have many 
    /// potentially relevant features but suspect only a few matter. FTRL is like a traveler who not only adapts 
    /// their path based on new information but also knows when to completely ignore certain routes as irrelevant, 
    /// creating a simpler, more efficient journey map.
    /// </para>
    /// </remarks>
    FTRL,

    /// <summary>
    /// Variant of Adam that maintains the maximum of past squared gradients for better convergence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AMSGrad is a variation of Adam designed to fix a potential convergence issue in the 
    /// original algorithm. It keeps track of the maximum of all past squared gradients rather than just their 
    /// exponential average. This ensures the learning rates never increase unexpectedly, which can help reach 
    /// the optimal solution more reliably. It's like a hiker who remembers the steepest slope they've ever 
    /// encountered in each direction and uses that information to make sure they never take too large a step, 
    /// even if the current terrain appears gentle.
    /// </para>
    /// </remarks>
    AMSGrad,

    /// <summary>
    /// Optimization algorithm that combines the Lion optimizer with advanced activation techniques.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lion (Evolved Sign Momentum) is a newer optimizer that works differently from 
    /// traditional approaches. Instead of using the exact gradient values, it only looks at their signs (positive 
    /// or negative) combined with a momentum-like update rule. This makes it computationally efficient and often 
    /// provides good performance with less fine-tuning required. It's like a hiker who only cares about the 
    /// general direction (up, down, left, right) rather than precisely how steep each part is, saving energy 
    /// while still making good progress overall.
    /// </para>
    /// </remarks>
    Lion,

    /// <summary>
    /// Standard optimization approach with fixed learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Normal optimization refers to a basic approach with a fixed learning rate (step size). 
    /// It's like walking with the same stride length regardless of the terrain. This simplicity makes it easy 
    /// to understand and implement, but it may not adapt well to different parts of the learning process. It's 
    /// often used as a baseline or in simpler problems where adaptive techniques aren't necessary.
    /// </para>
    /// </remarks>
    Normal,

    /// <summary>
    /// Direct search method that doesn't require gradient information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Nelder-Mead is like exploring a landscape by setting up camp in multiple spots 
    /// and then moving your camps based on where the terrain looks most promising. It doesn't need to calculate 
    /// slopes (gradients) which makes it useful when those calculations are difficult or impossible. Instead, 
    /// it compares the values at different points and gradually moves toward better areas by expanding, 
    /// contracting, or reflecting its search pattern. It's particularly good for problems where calculating 
    /// derivatives is expensive or not possible.
    /// </para>
    /// </remarks>
    NelderMead,

    /// <summary>
    /// Optimization method that updates one parameter at a time while keeping others fixed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Coordinate Descent is like adjusting the knobs on a complex machine one at a time. 
    /// It focuses on a single parameter, finds the best value for it while keeping all other parameters fixed, 
    /// then moves on to the next parameter. This cycle repeats until the solution stops improving. It's 
    /// particularly effective for problems where different parameters don't strongly interact with each other, 
    /// and it can be easier to understand and implement than more complex methods.
    /// </para>
    /// </remarks>
    CoordinateDescent,

    /// <summary>
    /// Optimization approach that builds a probabilistic model of the objective function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Optimization is like a smart explorer who keeps a mental map of where 
    /// treasure might be hidden. After each exploration, it updates its beliefs about where the best solutions 
    /// are likely to be found. This makes it particularly efficient for expensive-to-evaluate functions, as it 
    /// carefully chooses each evaluation point to maximize information gain. It's like deciding where to dig 
    /// for gold based on all your previous findings, rather than digging randomly or following a fixed pattern.
    /// </para>
    /// </remarks>
    BayesianOptimization,

    /// <summary>
    /// Evolutionary algorithm that creates new solutions by combining existing ones.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Differential Evolution is like breeding plants to get better varieties. It maintains 
    /// a population of solutions and creates new candidates by combining existing ones in clever ways. For each 
    /// solution, it creates a "child" by mixing parts of other solutions, and keeps the better one for the next 
    /// generation. This approach is particularly good at exploring complex landscapes with many local optima, 
    /// as it can jump to entirely different regions of the solution space.
    /// </para>
    /// </remarks>
    DifferentialEvolution,

    /// <summary>
    /// Optimization method that uses a simplified model within a trusted region.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trust Region methods are like exploring with a map that's only accurate within a 
    /// certain distance of your current position. The optimizer creates a simplified model of the landscape 
    /// around the current point, solves that simpler problem, then decides how much to trust that model based 
    /// on how well its predictions match reality. If the model works well, the "trusted region" expands; if not, 
    /// it contracts. This cautious approach helps navigate difficult optimization landscapes reliably.
    /// </para>
    /// </remarks>
    TrustRegion,

    /// <summary>
    /// Optimization method that approximates second-order information for better search directions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quasi-Newton methods are like having a partial map of the terrain that gets better 
    /// as you explore. They approximate information about how the slope changes (curvature) without explicitly 
    /// calculating it, which would be expensive. This gives them many of the benefits of second-order methods 
    /// like Newton's method, but with much less computation. It's like learning the shape of a valley as you 
    /// walk through it, helping you take more intelligent steps toward the lowest point.
    /// </para>
    /// </remarks>
    QuasiNewton,

    /// <summary>
    /// Optimization method that generates search directions that don't interfere with previous progress.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Conjugate Gradient is like finding paths through a valley that don't undo your 
    /// previous progress. Regular gradient descent might zigzag inefficiently, but conjugate gradient ensures 
    /// that each new direction doesn't interfere with optimization along previous directions. This makes it 
    /// much more efficient, especially for problems with many parameters. It's like planning a hike where each 
    /// leg of the journey makes steady progress toward the destination without backtracking.
    /// </para>
    /// </remarks>
    ConjugateGradient,

    /// <summary>
    /// Simple optimization strategy that always moves in the direction of immediate improvement.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hill Climbing is one of the simplest optimization strategies - it always moves in 
    /// the direction that immediately improves the solution. It's like hiking uphill by always taking a step in 
    /// the steepest direction. This works well for simple problems with a single peak, but it can easily get 
    /// stuck on small hills (local maxima) instead of finding the highest mountain (global maximum). Despite 
    /// its limitations, it's easy to understand and implement, making it a good starting point.
    /// </para>
    /// </remarks>
    HillClimbing,

    /// <summary>
    /// Optimization method that generates random samples and selects the best performing ones.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-Entropy Method works by generating random solutions, evaluating them, and 
    /// then adjusting the probability distribution to make good solutions more likely in the next round. It's 
    /// like a treasure hunt where you start by looking randomly, but gradually focus your search on areas where 
    /// you've found valuable items. This method is particularly useful for problems with complex constraints or 
    /// discrete variables, and it can handle noisy evaluations well.
    /// </para>
    /// </remarks>
    CrossEntropy,

    /// <summary>
    /// Optimization method that performs sequential line minimizations along conjugate directions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Powell's Method searches along carefully chosen directions one at a time, finding 
    /// the best point along each direction before moving to the next. After each complete cycle, it updates its 
    /// search directions based on the progress made. This approach doesn't require calculating gradients, making 
    /// it useful when those calculations are difficult. It's like exploring a maze by following one corridor at 
    /// a time to its best point before trying a different corridor.
    /// </para>
    /// </remarks>
    PowellMethod,

    /// <summary>
    /// Adaptive gradient algorithm that maintains per-parameter learning rates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adaptive Gradient methods adjust the learning rate separately for each parameter 
    /// based on historical gradient information. Parameters that receive consistent gradient updates get smaller 
    /// learning rates, while rarely updated parameters get larger ones. This is like a hiker who takes careful, 
    /// small steps on well-trodden paths but larger, exploratory steps in unfamiliar territory. This approach 
    /// helps balance learning across all parameters, especially in problems with sparse features.
    /// </para>
    /// </remarks>
    AdaptiveGradient,

    /// <summary>
    /// Evolutionary algorithm that uses principles of natural selection to optimize solutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Evolutionary Algorithms mimic biological evolution to find optimal solutions. 
    /// They maintain a population of candidate solutions, select the fittest ones based on performance, 
    /// create new solutions through crossover (combining parts of existing solutions) and mutation (random changes), 
    /// and repeat this process over generations. It's like breeding animals for desired traits - over time, 
    /// the population evolves toward better solutions. This approach works well for complex problems where 
    /// traditional methods might get stuck.
    /// </para>
    /// </remarks>
    EvolutionaryAlgorithm,

    /// <summary>
    /// Combination of adaptive learning rates and momentum for efficient optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Adam Optimizer (Adaptive Moment Estimation) combines ideas from several other 
    /// optimizers to achieve good results across many different problems. It keeps track of both the average 
    /// gradient (first moment) and the average squared gradient (second moment) for each parameter, using these 
    /// to adapt the learning rate individually. It's like a hiker who remembers both the general direction they've 
    /// been moving and how variable or steep the terrain has been, adjusting their pace accordingly for each part 
    /// of the journey.
    /// </para>
    /// </remarks>
    AdamOptimizer,

    /// <summary>
    /// Optimizer that combines random search with adaptive parameter tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Normal Optimizer uses a combination of random exploration and adaptive learning 
    /// to find good solutions efficiently. It starts with random guesses, learns from each attempt, and gradually 
    /// refines its approach. It's like a treasure hunter who begins by searching in random locations but becomes 
    /// more strategic as they gather clues about where the treasure might be. This balanced approach makes it 
    /// versatile for many different types of problems.
    /// </para>
    /// </remarks>
    NormalOptimizer,

    /// <summary>
    /// Adaptive gradient algorithm that accumulates squared gradients to scale the learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdaGrad adapts the learning rate individually for each parameter based on its 
    /// historical gradient values. Parameters that receive large or frequent updates get smaller learning rates, 
    /// while parameters with small or infrequent updates get larger learning rates. It's like a hiker who takes 
    /// smaller steps on well-trodden, steep paths and larger steps on rarely visited, flat terrain. This makes 
    /// AdaGrad particularly useful for dealing with sparse data, but its continually decreasing learning rates 
    /// can sometimes cause learning to stop too early on complex problems.
    /// </para>
    /// </remarks>
    AdaGrad,

    /// <summary>
    /// Variant of Adam that uses the infinity norm for more stable updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdaMax is a variation of the Adam optimizer that uses the maximum (infinity norm) 
    /// of past gradients rather than their squared values. This makes it more robust to gradient outliers and 
    /// extreme values. It's like a hiker who pays attention to the steepest slope they've ever encountered in 
    /// each direction, rather than the average steepness. This can make AdaMax more stable in some situations, 
    /// particularly when gradients can vary widely in magnitude, though it may be less responsive to recent 
    /// changes compared to Adam.
    /// </para>
    /// </remarks>
    AdaMax,

    /// <summary>
    /// Extension of Adagrad that uses a moving average of squared gradients to adapt the learning rate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdaDelta is an advanced optimizer that improves upon AdaGrad by addressing its
    /// diminishing learning rates problem. Instead of accumulating all past squared gradients, AdaDelta uses
    /// a moving window of gradients, remembering only recent history. What makes AdaDelta special is that it
    /// doesn't even require setting an initial learning rate - it adapts automatically based on the relationship
    /// between parameter updates and gradients. It's like a hiker who adjusts their pace not just based on the
    /// steepness of the terrain, but also on how efficiently they've been covering ground recently. This makes
    /// AdaDelta particularly robust across different types of problems without requiring manual tuning.
    /// </para>
    /// </remarks>
    AdaDelta,

    /// <summary>
    /// Nested Learning optimizer - a multi-level optimization paradigm for continual learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Nested Learning is a new paradigm from Google Research that treats ML models as
    /// interconnected, multi-level learning problems optimized simultaneously. Unlike traditional optimizers
    /// that update all parameters at the same rate, Nested Learning operates at multiple timescales - some
    /// parameters update quickly (learning from immediate feedback) while others update slowly (learning
    /// general patterns). It uses a Continuum Memory System (CMS) that maintains memories at different
    /// frequencies, mimicking how the human brain has both short-term and long-term memory. This makes it
    /// particularly good at continual learning - learning new tasks without forgetting old ones. It's like
    /// having multiple learning strategies working together: one that quickly adapts to new situations,
    /// another that slowly builds general knowledge, and others in between, all coordinating to prevent
    /// "catastrophic forgetting" where learning new tasks destroys knowledge of old tasks.
    /// </para>
    /// </remarks>
    NestedLearning,
}
