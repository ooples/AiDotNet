using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for QMIX agents (Rashid et al. 2018).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The defaults are the paper's (§5 and Appendix C): a shared GRU agent network of 64 units, a mixing network of 32
/// units whose weights single-layer hypernetworks generate, RMSprop at 5e-4, batches of 32 episodes from a buffer of
/// the 5000 most recent, the target networks updated every 200 episodes, ε annealed linearly from 1 to 0.05 over
/// 50000 steps, γ = 0.99, and the summed squared TD error of eq. 6 against the target network's max. Where the paper is
/// silent, pymarl (the authors' code) fills in: RMSprop's α = 0.99 and ε = 1e-5 and the gradient norm clipped at 10.
/// pymarl's own configuration differs in three places, each an option: <see cref="HypernetLayers"/> = 2,
/// <see cref="DoubleQ"/> and <see cref="MeanLoss"/>.
/// </para>
/// <para><b>For Beginners:</b>
/// QMIX learns a team of agents that each choose their own action from what they see, while training them together:
/// a mixing network turns the agents' values into one team value that can only rise when any agent's value rises, so
/// each agent picking its own best action also picks the team's best joint action.
/// </para>
/// </remarks>
public class QMIXOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Creates the options with the paper's defaults.</summary>
    public QMIXOptions()
    {
        BatchSize = 32;
        ReplayBufferSize = 5000;
        TargetUpdateFrequency = 200;
        EpsilonStart = 1.0;
        EpsilonEnd = 0.05;
        MaxGradientNorm = 10.0;
    }

    /// <summary>The number of agents.</summary>
    public int NumAgents { get; init; }

    /// <summary>The size of each agent's own observation.</summary>
    public int StateSize { get; init; }

    /// <summary>The number of discrete actions each agent chooses from.</summary>
    public int ActionSize { get; init; }

    /// <summary>The size of the global state the mixing network's hypernetworks read.</summary>
    public int GlobalStateSize { get; init; }

    /// <summary>The GRU's width in the agent network (the paper and pymarl: 64).</summary>
    public int RnnHiddenDim { get; init; } = 64;

    /// <summary>The mixing network's hidden width (the paper and pymarl: 32).</summary>
    public int MixingEmbedDim { get; init; } = 32;

    /// <summary>
    /// The layers in each weight hypernetwork: 1 in the paper (a linear layer), 2 in pymarl's qmix.yaml (a ReLU layer of
    /// <see cref="HypernetEmbed"/> units, then the linear one).
    /// </summary>
    public int HypernetLayers { get; init; } = 1;

    /// <summary>The hidden width of two-layer hypernetworks (pymarl: 64). Read only when <see cref="HypernetLayers"/> is 2.</summary>
    public int HypernetEmbed { get; init; } = 64;

    /// <summary>Whether each agent's input includes its previous action, one-hot (the paper and pymarl: yes).</summary>
    public bool IncludeLastAction { get; init; } = true;

    /// <summary>Whether each agent's input includes its id, one-hot, so the shared network can tell agents apart (yes).</summary>
    public bool IncludeAgentId { get; init; } = true;

    /// <summary>
    /// Double Q-learning targets: the target networks evaluate the live agent network's greedy action rather than their
    /// own max. Off in the paper (eq. 6); on in pymarl's qmix.yaml.
    /// </summary>
    public bool DoubleQ { get; init; }

    /// <summary>
    /// Averages the squared TD error over the batch's filled steps rather than summing it (the paper's eq. 6 sums;
    /// pymarl averages).
    /// </summary>
    public bool MeanLoss { get; init; }

    /// <summary>The environment steps over which ε anneals linearly from <c>EpsilonStart</c> to <c>EpsilonEnd</c> (50000).</summary>
    public int EpsilonAnnealSteps { get; init; } = 50000;

    /// <summary>RMSprop's smoothing constant α (pymarl: 0.99).</summary>
    public double RmsPropAlpha { get; init; } = 0.99;

    /// <summary>RMSprop's ε (pymarl: 1e-5).</summary>
    public double RmsPropEpsilon { get; init; } = 1e-5;

    /// <summary>
    /// Replaces the paper's RMSprop. It steps the agent and mixing networks' parameters together, from the gradient
    /// already clipped to <c>MaxGradientNorm</c>.
    /// </summary>
    public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; init; }
}
