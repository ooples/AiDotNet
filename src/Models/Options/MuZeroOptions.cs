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
    /// Dimensionality of the environment's observation vector. Defaults to
    /// 4 — the canonical CartPole / cart-pole-balance observation
    /// (cart position, cart velocity, pole angle, pole angular velocity),
    /// which is the smallest non-trivial RL benchmark and the size assumed
    /// by <see cref="AiDotNet.Tests.ModelFamilyTests.Base.ReinforcementLearningTestBase"/>
    /// for invariant testing. Override for any other environment — Atari
    /// (96×96×128 framestack), Go (19×19 board), MuJoCo (17-dim joint
    /// states), etc.
    /// </summary>
    public int ObservationSize { get; init; } = 4;

    /// <summary>
    /// Number of discrete actions the agent can choose from. Defaults to
    /// 2 — the canonical CartPole action set (push-left, push-right) and
    /// the smallest non-degenerate discrete action space. Override for
    /// any environment with more actions — Atari typically 18, Go 362,
    /// chess ~4672.
    /// </summary>
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
