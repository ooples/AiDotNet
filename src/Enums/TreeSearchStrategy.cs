namespace AiDotNet.Enums
{
    /// <summary>
    /// Defines search strategies for tree exploration in Tree-of-Thought reasoning.
    /// </summary>
    public enum TreeSearchStrategy
    {
        /// <summary>
        /// Beam search: Maintains top-k nodes at each level
        /// </summary>
        BeamSearch,

        /// <summary>
        /// Breadth-first search: Explores level by level
        /// </summary>
        BreadthFirst,

        /// <summary>
        /// Depth-first search: Explores deeply before backtracking
        /// </summary>
        DepthFirst,

        /// <summary>
        /// Monte Carlo tree search: Balances exploration and exploitation
        /// </summary>
        MonteCarlo,

        /// <summary>
        /// A* search: Uses heuristic to guide exploration
        /// </summary>
        AStar
    }
}