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
        VisitTemperature = other.VisitTemperature;
        UnrollSteps = other.UnrollSteps;
        TDSteps = other.TDSteps;
        PriorityAlpha = other.PriorityAlpha;
        UseValuePrefix = other.UseValuePrefix;
        OptimizerFactory = other.OptimizerFactory;
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

    /// <summary>
    /// Gets or sets the temperature T of the visit-count distribution that training-time actions are sampled from.
    /// </summary>
    /// <remarks>
    /// <para>
    /// While acting in training, MuZero samples the action from p(a) = N(a)^(1/T) / sum_b N(b)^(1/T) over the
    /// root's visit counts N (Schrittwieser et al. 2020, Appendix D). The paper's Atari schedule uses T = 1, then
    /// 0.5, then 0.25 as training proceeds; set this between calls to follow such a schedule. T = 0 picks the most
    /// visited action. Inference (<c>training: false</c>) always takes the policy's most likely action.
    /// </para>
    /// <para><b>For Beginners:</b> After the search, the agent knows how often it explored each action. A
    /// temperature of 1 picks actions in proportion to those counts, which keeps exploring; smaller values favour the
    /// most-explored action more strongly, and 0 always takes it.
    /// </para>
    /// </remarks>
    /// <value>A non-negative temperature. Default: 1.0, the paper's initial value.</value>
    public double VisitTemperature { get; init; } = 1.0;

    // Training parameters
    // Number of recurrent unroll steps K used to build training targets (Schrittwieser et al. 2020
    // use K=5). MuZeroAgent.Train unrolls the learned model K steps from a real trajectory window and
    // backpropagates one joint loss through the recurrence (#1756), so the paper default is honored.
    // (Supersedes the interim K=1 default from #1755, which existed only while K>1 was unimplemented.)
    public int UnrollSteps { get; init; } = 5;
    public int TDSteps { get; init; } = 10;
    public double PriorityAlpha { get; init; } = 1.0;

    // Value/Policy targets
    public bool UseValuePrefix { get; init; } = false;

    /// <summary>
    /// Gets or sets a factory for the optimizer of the agent's joint training step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// MuZero trains its representation, dynamics and prediction networks with one loss in one step
    /// (Schrittwieser et al. 2020), so the agent invokes the factory once and steps all three networks with
    /// that optimizer. When this is <c>null</c>, the step uses Adam at the agent's learning rate.
    /// </para>
    /// <para>
    /// The factory must create a new optimizer instance on each call, so agents built from copies of these
    /// options never share moment buffers or a step counter.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is the rule that turns the training loss's gradients into weight
    /// changes for all three networks at once. Leave this unset to get Adam at the agent's learning rate; supply a
    /// factory to choose a different optimizer or its settings. Return a new optimizer from every call.
    /// </para>
    /// </remarks>
    /// <value>A delegate that returns a new optimizer on each call, or <c>null</c> (the default) for Adam at the
    /// agent's learning rate.</value>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
