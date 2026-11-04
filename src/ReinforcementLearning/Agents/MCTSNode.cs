using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Agents.MuZero;

/// <summary>
/// Monte Carlo Tree Search (MCTS) node for MuZero agent.
/// Represents a state in the search tree with visit counts and Q-values for actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MCTSNode<T>
{
    public Vector<T>? HiddenState { get; set; }
    public Dictionary<int, MCTSNode<T>> Children { get; set; } = new();
    public Dictionary<int, int> VisitCounts { get; set; } = new();
    public Dictionary<int, T> QValues { get; set; } = new();
    public Dictionary<int, T> Rewards { get; set; } = new();
    public required T Value { get; set; }
    public int TotalVisits { get; set; }

    /// <summary>
    /// Search priors that replace the prediction network's policy at this node, or null to use the policy.
    /// The root carries the policy mixed with Dirichlet exploration noise while acting in training.
    /// </summary>
    internal double[]? Priors { get; set; }
}
