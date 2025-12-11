using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

/// <summary>
/// Context window for sequence modeling in Decision Transformer.
/// Maintains recent states, actions, and returns-to-go for transformer input.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SequenceContext<T>
{
    public List<Vector<T>> States { get; set; } = new();
    public List<Vector<T>> Actions { get; set; } = new();
    public List<T> ReturnsToGo { get; set; } = new();
    public int Length => States.Count;
}
