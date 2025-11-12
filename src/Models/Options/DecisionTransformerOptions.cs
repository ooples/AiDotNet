using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Decision Transformer agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Decision Transformer treats RL as sequence modeling, using transformer architecture
/// to model trajectories conditioned on desired returns.
/// </para>
/// <para><b>For Beginners:</b>
/// Decision Transformer is a radically different approach to RL. Instead of learning
/// "what action is best", it learns "what action was taken when the outcome was X".
/// Then at test time, you tell it "I want outcome X" and it generates actions.
///
/// Key innovation:
/// - **Sequence Modeling**: Uses transformers (like GPT) instead of RL algorithms
/// - **Return Conditioning**: Specify desired return, get action sequence
/// - **Offline-Friendly**: Works excellently with fixed datasets
/// - **No Value Functions**: No Q-networks or critics needed
///
/// Think of it like: "Show me examples of successful chess games, and I'll learn
/// to play moves that lead to success."
///
/// Famous for: Berkeley/Meta research showing transformers can replace RL algorithms
/// </para>
/// </remarks>
public class DecisionTransformerOptions<T>
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T LearningRate { get; set; }

    // Transformer architecture parameters
    public int EmbeddingDim { get; set; } = 128;
    public int NumLayers { get; set; } = 3;
    public int NumHeads { get; set; } = 1;
    public int ContextLength { get; set; } = 20;  // Number of timesteps to condition on
    public double DropoutRate { get; set; } = 0.1;

    // Training parameters
    public int BatchSize { get; set; } = 64;
    public int BufferSize { get; set; } = 1000000;
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredError<T>();
    public int? Seed { get; set; }

    public DecisionTransformerOptions()
    {
        var numOps = NumericOperations<T>.Instance;
        LearningRate = numOps.FromDouble(0.0001);
    }
}
