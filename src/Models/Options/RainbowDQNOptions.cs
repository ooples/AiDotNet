using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Rainbow DQN agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Rainbow DQN combines six extensions to DQN:
/// 1. Double Q-learning: Reduces overestimation bias
/// 2. Dueling networks: Separates value and advantage streams
/// 3. Prioritized replay: Samples important experiences more frequently
/// 4. Multi-step learning: Uses n-step returns for better credit assignment
/// 5. Distributional RL: Learns full distribution of returns (C51)
/// 6. Noisy networks: Parameter noise for exploration
/// </remarks>
public class RainbowDQNOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Whether to explore with noisy linear layers instead of epsilon-greedy.</summary>
    public bool UseNoisyNetworks { get; init; } = true;

    // Dueling network architecture
    /// <summary>Hidden layer sizes of the shared trunk that feeds both streams.</summary>
    public List<int> SharedLayers { get; init; } = new List<int> { 128 };
    /// <summary>Hidden layer sizes of the dueling value stream.</summary>
    public List<int> ValueStreamLayers { get; init; } = new List<int> { 128 };
    /// <summary>Hidden layer sizes of the dueling advantage stream.</summary>
    public List<int> AdvantageStreamLayers { get; init; } = new List<int> { 128 };

    // Prioritized experience replay parameters (base has UsePrioritizedReplay)
    /// <summary>Prioritization exponent of the replay buffer: how strongly TD error shapes sampling (0 samples uniformly).</summary>
    public double PriorityAlpha { get; init; } = 0.6;
    /// <summary>Initial importance-sampling exponent that corrects the bias of prioritized replay.</summary>
    public double PriorityBeta { get; init; } = 0.4;
    /// <summary>Amount added to <see cref="PriorityBeta"/> after each sample, annealing it toward 1.</summary>
    public double PriorityBetaIncrement { get; init; } = 0.001;
    /// <summary>Small constant added to each priority so no transition has zero probability of being replayed.</summary>
    public double PriorityEpsilon { get; init; } = 1e-6;

    // Multi-step learning parameters
    /// <summary>Number of rewards (n) accumulated in the multi-step return before bootstrapping.</summary>
    public int NSteps { get; init; } = 3;

    // Distributional RL (C51) parameters
    /// <summary>Whether to learn the full return distribution (C51) instead of its mean.</summary>
    public bool UseDistributional { get; init; } = true;
    /// <summary>Number of atoms of the categorical return distribution (51 in C51 and Rainbow).</summary>
    public int NumAtoms { get; init; } = 51;
    /// <summary>Smallest return represented by the distributional support.</summary>
    public double VMin { get; init; } = -10.0;
    /// <summary>Largest return represented by the distributional support.</summary>
    public double VMax { get; init; } = 10.0;

    // Noisy networks parameters
    /// <summary>Initial noise scale (sigma_0) of the noisy linear layers.</summary>
    public double NoisyNetSigma { get; init; } = 0.5;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
