using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MuZero agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MuZero combines tree search (like AlphaZero) with learned models.
/// It learns dynamics, rewards, and values without knowing environment rules.
/// </para>
/// <para><b>For Beginners:</b>
/// MuZero is DeepMind's breakthrough that mastered Atari, Go, Chess, and Shogi
/// without being told the rules. It learns its own "internal model" of the game
/// and uses tree search to plan ahead.
///
/// Key innovations:
/// - **Learned Model**: No need for game rules, learns environment dynamics
/// - **MCTS**: Uses Monte Carlo Tree Search for planning
/// - **Three Networks**: Representation, dynamics, and prediction
/// - **Planning**: Searches through imagined futures
///
/// Think of it like: Learning to play chess by watching games, figuring out
/// the rules yourself, then planning moves by mentally simulating the game.
///
/// Famous for: Superhuman performance across Atari, board games, without rules
/// </para>
/// </remarks>
public class MuZeroOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>
    /// Default constructor required for object-initializer syntax.
    /// </summary>
    public MuZeroOptions()
    {
    }

    /// <summary>
    /// Copy constructor required by the Options golden pattern so Clone()
    /// faithfully preserves every property.
    /// </summary>
    public MuZeroOptions(MuZeroOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        LearningRate = other.LearningRate;
        DiscountFactor = other.DiscountFactor;
        LossFunction = other.LossFunction;
        BatchSize = other.BatchSize;
        ReplayBufferSize = other.ReplayBufferSize;
        TargetUpdateFrequency = other.TargetUpdateFrequency;
        UsePrioritizedReplay = other.UsePrioritizedReplay;
        EpsilonStart = other.EpsilonStart;
        EpsilonEnd = other.EpsilonEnd;
        EpsilonDecay = other.EpsilonDecay;
        WarmupSteps = other.WarmupSteps;
        MaxGradientNorm = other.MaxGradientNorm;
        Seed = other.Seed;

        ObservationSize = other.ObservationSize;
        ActionSize = other.ActionSize;
        LatentStateSize = other.LatentStateSize;
        RepresentationLayers = new List<int>(other.RepresentationLayers);
        DynamicsLayers = new List<int>(other.DynamicsLayers);
        PredictionLayers = new List<int>(other.PredictionLayers);
        NumSimulations = other.NumSimulations;
        PUCTConstant = other.PUCTConstant;
        RootDirichletAlpha = other.RootDirichletAlpha;
        RootExplorationFraction = other.RootExplorationFraction;
        UnrollSteps = other.UnrollSteps;
        TDSteps = other.TDSteps;
        PriorityAlpha = other.PriorityAlpha;
        UseValuePrefix = other.UseValuePrefix;
        Optimizer = other.Optimizer;
    }

    /// <summary>
    /// Dimensionality of the environment's observation vector.
    /// </summary>
    /// <value>
    /// Default 4: the canonical CartPole observation, which is the smallest
    /// non-trivial RL benchmark shape used by the repository invariant tests.
    /// </value>
    /// <remarks>
    /// Override for any other environment: Atari frame stacks, Go board state,
    /// MuJoCo joint states, etc. The representation network's input layer reads
    /// this dimension, so an incorrect value causes a first-forward shape
    /// mismatch.
    /// </remarks>
    public int ObservationSize { get; init; } = 4;

    /// <summary>
    /// Number of discrete actions the agent can choose from.
    /// </summary>
    /// <value>
    /// Default 2: the CartPole action set (push-left, push-right) and the
    /// smallest non-degenerate discrete action space.
    /// </value>
    /// <remarks>
    /// Override for environments with more actions. The prediction network's
    /// policy head reads this dimension, and the MCTS tree branches on this
    /// many children per node.
    /// </remarks>
    public int ActionSize { get; init; } = 2;

    // Network architecture
    public int LatentStateSize { get; init; } = 256;
    public List<int> RepresentationLayers { get; init; } = new List<int> { 256, 256 };
    public List<int> DynamicsLayers { get; init; } = new List<int> { 256, 256 };
    public List<int> PredictionLayers { get; init; } = new List<int> { 256, 256 };

    // MCTS parameters
    public int NumSimulations { get; init; } = 50;
    public double PUCTConstant { get; init; } = 1.25;
    public double RootDirichletAlpha { get; init; } = 0.3;
    public double RootExplorationFraction { get; init; } = 0.25;

    // Training parameters
    // Number of recurrent unroll steps K used to build training targets
    // (Schrittwieser et al. 2020 use K=5). MuZeroAgent.Train currently implements
    // the one-step (K=1) unrolled targets and fail-fasts for K>1 (which needs
    // sequence replay + per-unroll-step targets — tracked in #1752). The default
    // must therefore reflect the implemented capability so the out-of-box agent
    // actually trains; a caller wanting the paper's K=5 sets it explicitly and
    // gets a clear "not yet implemented" exception rather than a silent no-op.
    public int UnrollSteps { get; init; } = 1;
    public int TDSteps { get; init; } = 10;
    public double PriorityAlpha { get; init; } = 1.0;

    // Value/Policy targets
    public bool UseValuePrefix { get; init; } = false;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam is used.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
