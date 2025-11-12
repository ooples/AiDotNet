using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Agents.MuZero;

/// <summary>
/// Monte Carlo Tree Search (MCTS) node for MuZero agent.
/// Represents a state in the search tree with visit counts and Q-values for actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MCTSNode<T>
{
    public Vector<T> HiddenState { get; set; } = null!;
    public Dictionary<int, MCTSNode<T>> Children { get; set; } = new();
    public Dictionary<int, int> VisitCounts { get; set; } = new();
    public Dictionary<int, T> QValues { get; set; } = new();
    public T Value { get; set; } = default!;
    public int TotalVisits { get; set; }
}
