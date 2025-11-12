namespace AiDotNet.ReinforcementLearning.Enums;

/// <summary>
/// Defines different exploration strategies for reinforcement learning.
/// </summary>
/// <remarks>
/// <para>
/// Exploration is the process of trying new actions to discover better strategies, while exploitation
/// uses known good actions. The exploration-exploitation tradeoff is fundamental to RL, and different
/// strategies provide different balances.
/// </para>
/// <para><b>For Beginners:</b> These are different ways an AI agent decides when to try new things vs. use what it knows.
///
/// The exploration-exploitation dilemma:
/// - Exploration: Try new things to potentially discover better strategies
/// - Exploitation: Use what you know works to get good results now
///
/// Too much exploration: Waste time on bad strategies, never settle on good ones
/// Too little exploration: Miss better strategies, get stuck in local optimum
///
/// Think of it like choosing a restaurant:
/// - Exploration: Try a new restaurant you've never been to
/// - Exploitation: Go to your favorite restaurant
/// - You need both to find great places and enjoy consistent good meals
///
/// Different strategies provide different balances and are suited to different problems.
/// </para>
/// </remarks>
public enum ExplorationStrategyType
{
    /// <summary>
    /// Epsilon-Greedy: Randomly explore with probability epsilon, otherwise exploit.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Flip a weighted coin - heads try something random, tails use what works best.
    ///
    /// How it works:
    /// - With probability ε (epsilon): Take a completely random action
    /// - With probability 1-ε: Take the best known action
    ///
    /// Parameters:
    /// - ε = 0: Never explore (always greedy)
    /// - ε = 0.1: Explore 10% of the time
    /// - ε = 1.0: Always explore (completely random)
    ///
    /// Typical schedule:
    /// - Start: ε = 1.0 (explore a lot)
    /// - Gradually decay to ε = 0.01 (explore rarely)
    ///
    /// Pros: Simple, well-understood, works well in practice
    /// Cons: Doesn't consider how good alternatives are (treats all equally)
    ///
    /// Best for: Most problems, especially discrete actions
    /// </para>
    /// </remarks>
    EpsilonGreedy,

    /// <summary>
    /// Softmax (Boltzmann): Sample actions proportional to their estimated values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pick randomly but favor better options.
    ///
    /// How it works:
    /// - Convert Q-values to probabilities
    /// - Higher Q-values get higher probabilities
    /// - Still some chance of picking any action
    ///
    /// Temperature parameter:
    /// - High temperature: Nearly uniform random
    /// - Low temperature: Strongly favor best actions
    /// - Temperature → 0: Becomes greedy
    ///
    /// Pros: More intelligent exploration (rarely picks clearly bad actions)
    /// Cons: Requires Q-values, more computationally expensive
    ///
    /// Best for: When you want smooth, probability-based exploration
    /// </para>
    /// </remarks>
    Softmax,

    /// <summary>
    /// Greedy: Always select the action with highest estimated value (no exploration).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Always pick the best known option, never try new things.
    ///
    /// When to use:
    /// - Evaluation/testing after training
    /// - Production deployment
    /// - When exploration is not desired
    ///
    /// Warning: If used during training, agent won't discover new strategies!
    ///
    /// Think of it like: Always going to your favorite restaurant (never trying new ones).
    /// Great when you know what's best, bad when still learning.
    /// </para>
    /// </remarks>
    Greedy,

    /// <summary>
    /// UCB (Upper Confidence Bound): Optimistically explore uncertain actions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Favor actions you're uncertain about - they might be better than you think!
    ///
    /// Principle: "Optimism in the face of uncertainty"
    /// - Actions you've tried less get a bonus
    /// - Actions you're uncertain about get explored
    /// - Balances exploration and exploitation mathematically
    ///
    /// Pros: Theoretically optimal for multi-armed bandits
    /// Cons: Requires tracking uncertainty, less common in deep RL
    ///
    /// Best for: Bandit problems, when you can track visit counts
    /// </para>
    /// </remarks>
    UCB,

    /// <summary>
    /// Thompson Sampling: Bayesian approach to exploration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sample actions based on probability they're optimal.
    ///
    /// How it works:
    /// - Maintain a probability distribution over value estimates
    /// - Sample from these distributions
    /// - Actions that might be optimal (even if uncertain) get tried
    ///
    /// Pros: Theoretically elegant, naturally balances exploration/exploitation
    /// Cons: Requires maintaining distributions, computationally expensive
    ///
    /// Best for: Bandit problems, Bayesian RL
    /// </para>
    /// </remarks>
    ThompsonSampling,

    /// <summary>
    /// Noisy Networks: Add learned noise to network parameters for exploration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The neural network itself has randomness built in.
    ///
    /// How it works:
    /// - Add noise to neural network weights
    /// - Network learns how much noise to use
    /// - State-dependent exploration (different states get different exploration)
    ///
    /// Pros: No hyperparameters (like epsilon), state-dependent exploration
    /// Cons: More complex, requires special network architecture
    ///
    /// Best for: Deep RL with complex state spaces
    /// Used in: Rainbow DQN
    /// </para>
    /// </remarks>
    NoisyNetworks,

    /// <summary>
    /// Entropy Regularization: Encourage policy to be stochastic (maximize entropy).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Add "entropy" (randomness) to the policy as a goal.
    ///
    /// How it works:
    /// - Add entropy term to the reward
    /// - Agent is rewarded for being unpredictable
    /// - Prevents premature convergence to deterministic policy
    ///
    /// Benefits:
    /// - More robust policies
    /// - Better exploration
    /// - Smoother learning
    ///
    /// Used in: SAC (Soft Actor-Critic), maximum entropy RL
    ///
    /// Think of it like: Being rewarded for trying variety, not just getting results.
    /// </para>
    /// </remarks>
    EntropyRegularization,

    /// <summary>
    /// Parameter Space Noise: Add noise to policy parameters instead of actions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Make the policy itself noisy, not just the actions.
    ///
    /// Difference from action noise:
    /// - Action noise: Add randomness to selected actions
    /// - Parameter noise: Add randomness to policy weights
    ///
    /// Benefit: Exploration is more consistent (policy changes smoothly)
    ///
    /// Pros: Can lead to better exploration, especially in continuous control
    /// Cons: More complex to implement
    ///
    /// Best for: Continuous control tasks
    /// </para>
    /// </remarks>
    ParameterNoise,

    /// <summary>
    /// Curiosity-Driven: Explore states that are novel or surprising.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Agent is curious - seeks out new and surprising experiences.
    ///
    /// How it works:
    /// - Add "intrinsic" reward for visiting novel states
    /// - Agent explores to satisfy curiosity
    /// - Helps in sparse reward environments
    ///
    /// Intrinsic motivation:
    /// - Normal reward: From the task (extrinsic)
    /// - Curiosity reward: From exploring new things (intrinsic)
    ///
    /// Pros: Great for sparse reward environments, discovers interesting behaviors
    /// Cons: Can be distracted by irrelevant novelty
    ///
    /// Examples: ICM (Intrinsic Curiosity Module), RND (Random Network Distillation)
    /// </para>
    /// </remarks>
    CuriosityDriven,

    /// <summary>
    /// Count-Based: Explore less-visited states more.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keep track of where you've been, visit new places.
    ///
    /// How it works:
    /// - Count how many times each state has been visited
    /// - Give bonus reward for rarely-visited states
    /// - Encourages systematic exploration
    ///
    /// Pros: Simple concept, guarantees coverage
    /// Cons: Hard to implement in continuous/large state spaces
    ///
    /// Best for: Discrete, small-to-medium state spaces
    /// </para>
    /// </remarks>
    CountBased
}
