using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.ReinforcementLearning.Agents;

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
public class DecisionTransformerOptions<T> : ReinforcementLearningOptions<T>
{
    public int StateSize { get; init; } = 16;
    public int ActionSize { get; init; } = 4;

    // Transformer architecture parameters
    public int EmbeddingDim { get; init; } = 128;
    public int NumLayers { get; init; } = 3;
    public int NumHeads { get; init; } = 1;
    public int ContextLength { get; init; } = 20;  // Number of timesteps to condition on
    public double DropoutRate { get; init; } = 0.1;

    // Training parameters
    public int BufferSize { get; init; } = 1000000;

    /// <summary>
    /// Gets or sets a factory for the optimizer each of this agent's networks trains with.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When this is <c>null</c>, the network trains with its default optimizer.
    /// </para>
    /// <para>
    /// The agent invokes the factory once for every network it builds, so the factory must create a new
    /// optimizer instance on each call. A shared instance would share its step counter, and Adam's bias
    /// correction would then advance on every network's step instead of only on that network's own.
    /// </para>
    /// </remarks>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; init; }
}
