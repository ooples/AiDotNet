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
    /// Default constructor — required for object-initializer syntax.
    /// </summary>
    public MuZeroOptions()
    {
    }

    /// <summary>
    /// Copy constructor — required by the Options golden pattern so
    /// Clone() faithfully preserves every property. Mirrors every base
    /// <see cref="ReinforcementLearningOptions{T}"/> field plus every
    /// MuZero-specific field. Without it, a Cloned agent silently
    /// re-runs the default constructor and loses any customised
    /// hyperparameters.
    /// </summary>
    public MuZeroOptions(MuZeroOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        // Base ReinforcementLearningOptions<T> fields
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

        // ModelOptions base field — reproducibility seed. Missing this
        // would silently change rng behaviour between original and clone.
        Seed = other.Seed;

        // MuZero-specific fields
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
    /// <value>Default 4 — the canonical CartPole observation
    /// (cart position, cart velocity, pole angle, pole angular velocity),
    /// the smallest non-trivial RL benchmark and the size assumed by
    /// <see cref="AiDotNet.Tests.ModelFamilyTests.Base.ReinforcementLearningTestBase"/>
    /// for invariant testing.</value>
    /// <remarks>
    /// <para>Override for any other environment — Atari (96×96×128
    /// framestack flattened, or use the unflattened framestack with a
    /// CNN representation network), Go (19×19 = 361 board), MuJoCo
    /// (17-dim joint states for Walker2d), etc. The representation
    /// network's input layer reads this dimension; getting it wrong
    /// produces a shape mismatch in the very first forward pass.</para>
    /// <para><b>For Beginners:</b> How many numbers describe the state of
    /// your environment at one timestep. For CartPole that's 4 (cart
    /// position, cart velocity, pole angle, pole angular velocity).</para>
    /// </remarks>
    public int ObservationSize { get; init; } = 4;

    /// <summary>
    /// Number of discrete actions the agent can choose from.
    /// </summary>
    /// <value>Default 2 — the CartPole action set (push-left, push-right)
    /// and the smallest non-degenerate discrete action space.</value>
    /// <remarks>
    /// <para>Override for any environment with more actions — Atari
    /// typically 18, Go 362 (361 board positions + pass), chess ~4672.
    /// The prediction network's policy head reads this dimension; the
    /// MCTS tree branches on this many children per node.</para>
    /// <para><b>For Beginners:</b> How many different moves your agent
    /// can pick from at each step. For CartPole that's 2 (left or
    /// right).</para>
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
    public int UnrollSteps { get; init; } = 5;  // Number of steps to unroll for training
    public int TDSteps { get; init; } = 10;  // TD bootstrap steps
    public double PriorityAlpha { get; init; } = 1.0;

    // Value/Policy targets
    public bool UseValuePrefix { get; init; } = false;  // Value prefix for long-horizon tasks

    /// <summary>
    /// The optimizer used for updating network parameters. If null, Adam optimizer will be used by default.
    /// </summary>
    public IOptimizer<T, Vector<T>, Vector<T>>? Optimizer { get; init; }
}
