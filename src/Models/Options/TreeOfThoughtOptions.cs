using AiDotNet.Enums;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration options specific to Tree-of-Thought reasoning models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how the Tree-of-Thought model explores
    /// different reasoning paths:
    /// - BranchingFactor is like how many different moves you consider in chess
    /// - MaxTreeDepth is how many moves ahead you think
    /// - SearchStrategy determines how you explore the possibilities
    /// - BeamWidth (for beam search) keeps only the best options at each step
    /// 
    /// Higher values generally lead to better reasoning but take more time.
    /// </para>
    /// </remarks>
    public class TreeOfThoughtOptions<T> : ReasoningModelOptions<T>
    {
        /// <summary>
        /// Gets or sets the shape of state representations.
        /// </summary>
        /// <value>Default is [256].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This determines how the model represents each "thought" or
        /// reasoning state. Larger values allow more complex thoughts but use more memory.
        /// </para>
        /// </remarks>
        public int[] StateShape { get; set; } = new[] { 256 };

        /// <summary>
        /// Gets or sets the branching factor for tree generation.
        /// </summary>
        /// <value>Default is 3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is how many different "next thoughts" the model considers
        /// at each step. More branches mean more thorough exploration but exponentially more
        /// computation. Think of it like considering 3 different moves in a game at each turn.
        /// </para>
        /// </remarks>
        public int BranchingFactor { get; set; } = 3;

        /// <summary>
        /// Gets or sets the maximum depth of the reasoning tree.
        /// </summary>
        /// <value>Default is 10.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is how many steps deep the model can think. Deeper trees
        /// can solve more complex problems but take much longer. It's like limiting how many
        /// moves ahead you can plan in chess.
        /// </para>
        /// </remarks>
        public int MaxTreeDepth { get; set; } = 10;

        /// <summary>
        /// Gets or sets the hidden layer size.
        /// </summary>
        /// <value>Default is 256.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls the model's internal processing power. Larger
        /// values make the model more capable but slower and more memory-intensive.
        /// </para>
        /// </remarks>
        public int HiddenSize { get; set; } = 256;

        /// <summary>
        /// Gets or sets the number of attention heads for thought generation.
        /// </summary>
        /// <value>Default is 8.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Attention heads help the model consider different aspects of
        /// the problem when generating new thoughts. More heads mean more comprehensive
        /// consideration but slower processing.
        /// </para>
        /// </remarks>
        public int AttentionHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets the search strategy for tree exploration.
        /// </summary>
        /// <value>Default is BeamSearch.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This determines how the model explores the tree of thoughts:
        /// - BeamSearch: Keeps only the best few options at each level (fast, good quality)
        /// - BreadthFirst: Explores all options level by level (thorough but slow)
        /// - DepthFirst: Follows one path deeply before trying others (fast but can miss good solutions)
        /// - MonteCarlo: Balances exploration and exploitation (good for complex problems)
        /// - AStar: Uses estimates to guide search (efficient for goal-oriented problems)
        /// </para>
        /// </remarks>
        public TreeSearchStrategy SearchStrategy { get; set; } = TreeSearchStrategy.BeamSearch;

        /// <summary>
        /// Gets or sets the selection threshold for thoughts.
        /// </summary>
        /// <value>Default is 0.3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Generated thoughts must score above this threshold to be
        /// included in the tree. Lower values allow more diverse (but potentially worse)
        /// thoughts, while higher values are more selective.
        /// </para>
        /// </remarks>
        public double SelectionThreshold { get; set; } = 0.3;

        /// <summary>
        /// Gets or sets the terminal value threshold.
        /// </summary>
        /// <value>Default is 0.9.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When a thought scores above this value, it's considered a
        /// "solution" and the search can stop early. Lower values find solutions faster but
        /// they might not be the best possible.
        /// </para>
        /// </remarks>
        public double TerminalValueThreshold { get; set; } = 0.9;

        /// <summary>
        /// Gets or sets the number of Monte Carlo iterations.
        /// </summary>
        /// <value>Default is 100.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When using Monte Carlo search, this is how many random
        /// explorations to perform. More iterations generally find better solutions but take
        /// longer. It's like running more simulations to be more confident.
        /// </para>
        /// </remarks>
        public int MonteCarloIterations { get; set; } = 100;

        /// <summary>
        /// Gets or sets the target proximity threshold for training.
        /// </summary>
        /// <value>Default is 0.1.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> During training, thoughts this close to the target answer are
        /// considered "correct." Smaller values require more precise solutions but might be
        /// harder to achieve.
        /// </para>
        /// </remarks>
        public double TargetProximityThreshold { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets the exploration depth for explanations.
        /// </summary>
        /// <value>Default is 3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When generating explanations, the model shows this many levels
        /// of the reasoning tree. Deeper explanations are more detailed but can be overwhelming.
        /// </para>
        /// </remarks>
        public int ExplanationDepth { get; set; } = 3;

        /// <summary>
        /// Gets or sets the number of branches to explore during refinement.
        /// </summary>
        /// <value>Default is 3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When refining an answer, the model explores this many alternative
        /// branches to find improvements. More branches mean better refinement but slower processing.
        /// </para>
        /// </remarks>
        public int RefinementBranches { get; set; } = 3;

        /// <summary>
        /// Gets or sets whether to cache tree nodes.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Caching remembers previously explored thoughts, making the model
        /// faster when encountering similar problems. Turn off if memory is limited.
        /// </para>
        /// </remarks>
        public bool EnableNodeCache { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to use parallel exploration.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Parallel exploration evaluates multiple branches simultaneously,
        /// making the search faster on multi-core systems. Disable if you need deterministic
        /// results or have limited CPU resources.
        /// </para>
        /// </remarks>
        public bool UseParallelExploration { get; set; } = true;
    }
}