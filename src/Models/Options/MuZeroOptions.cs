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
    /// <summary>Size of the latent state the model plans in.</summary>
    public int LatentStateSize { get; init; } = 256;
    /// <summary>Hidden layer sizes of the representation network (observation to latent state).</summary>
    public List<int> RepresentationLayers { get; init; } = new List<int> { 256, 256 };
    /// <summary>Hidden layer sizes of the dynamics network (next latent state and reward).</summary>
    public List<int> DynamicsLayers { get; init; } = new List<int> { 256, 256 };
    /// <summary>Hidden layer sizes of the prediction network (policy and value).</summary>
    public List<int> PredictionLayers { get; init; } = new List<int> { 256, 256 };

    // MCTS parameters
    /// <summary>Number of Monte Carlo tree search simulations per action.</summary>
    public int NumSimulations { get; init; } = 50;
    /// <summary>Exploration constant of the pUCT rule that balances prior and value during tree search.</summary>
    public double PUCTConstant { get; init; } = 1.25;
    /// <summary>Concentration of the Dirichlet noise added to the root prior.</summary>
    public double RootDirichletAlpha { get; init; } = 0.3;
    /// <summary>Fraction of the root prior replaced by Dirichlet noise during search, for exploration.</summary>
    public double RootExplorationFraction { get; init; } = 0.25;

    // Training parameters
    // Number of recurrent unroll steps K used to build training targets (Schrittwieser et al. 2020
    // use K=5). MuZeroAgent.Train unrolls the learned model K steps from a real trajectory window and
    // backpropagates one joint loss through the recurrence (#1756), so the paper default is honored.
    // (Supersedes the interim K=1 default from #1755, which existed only while K>1 was unimplemented.)
    /// <summary>Number of steps (K) the dynamics network is unrolled during training.</summary>
    public int UnrollSteps { get; init; } = 5;
    /// <summary>Number of steps (n) in the n-step bootstrapped value target.</summary>
    public int TDSteps { get; init; } = 10;
    /// <summary>Prioritization exponent of the replay buffer: how strongly TD error shapes sampling (0 samples uniformly).</summary>
    public double PriorityAlpha { get; init; } = 1.0;

    // Value/Policy targets
    /// <summary>Whether the dynamics network predicts the cumulative reward (value prefix) of an unrolled segment instead of each step's reward, as in EfficientZero.</summary>
    public bool UseValuePrefix { get; init; } = false;

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam is used.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
