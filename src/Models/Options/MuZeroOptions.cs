using AiDotNet.LossFunctions;

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
public class MuZeroOptions<T>
{
    public int ObservationSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }

    // Network architecture
    public int LatentStateSize { get; set; } = 256;
    public List<int> RepresentationLayers { get; set; } = [256, 256];
    public List<int> DynamicsLayers { get; set; } = [256, 256];
    public List<int> PredictionLayers { get; set; } = [256, 256];

    // MCTS parameters
    public int NumSimulations { get; set; } = 50;
    public double PUCTConstant { get; set; } = 1.25;
    public T DiscountFactor { get; set; }
    public double RootDirichletAlpha { get; set; } = 0.3;
    public double RootExplorationFraction { get; set; } = 0.25;

    // Training parameters
    public int UnrollSteps { get; set; } = 5;  // Number of steps to unroll for training
    public int TDSteps { get; set; } = 10;  // TD bootstrap steps
    public int BatchSize { get; set; } = 256;
    public int ReplayBufferSize { get; set; } = 1000000;
    public double PriorityAlpha { get; set; } = 1.0;

    // Value/Policy targets
    public bool UseValuePrefix { get; set; } = false;  // Value prefix for long-horizon tasks
    public int? Seed { get; set; }

    public MuZeroOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.0001);
        DiscountFactor = numOps.FromDouble(0.997);
    }
}
