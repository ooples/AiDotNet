namespace AiDotNet.Enums
{
    /// <summary>
    /// Represents different strategies for reasoning that can be employed by reasoning models.
    /// </summary>
    public enum ReasoningStrategy
    {
        /// <summary>
        /// Forward chaining: Start from known facts and derive conclusions
        /// </summary>
        ForwardChaining,

        /// <summary>
        /// Backward chaining: Start from the goal and work backwards to find supporting facts
        /// </summary>
        BackwardChaining,

        /// <summary>
        /// Bidirectional: Combine forward and backward chaining for more robust reasoning
        /// </summary>
        Bidirectional,

        /// <summary>
        /// Heuristic-guided: Use domain-specific heuristics to guide the reasoning process
        /// </summary>
        HeuristicGuided,

        /// <summary>
        /// Monte Carlo: Use probabilistic sampling to explore multiple reasoning paths
        /// </summary>
        MonteCarlo,

        /// <summary>
        /// Beam search: Maintain multiple promising reasoning paths simultaneously
        /// </summary>
        BeamSearch
    }
}