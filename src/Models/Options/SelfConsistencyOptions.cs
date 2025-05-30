namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration options specific to Self-Consistency reasoning models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how the Self-Consistency model explores
    /// multiple reasoning paths:
    /// - DefaultPathCount determines how many different ways the model tries to solve each problem
    /// - PathDiversityDropout adds randomness to make each path different
    /// - ConsistencyHeads help the model compare different paths
    /// - SimilarityThreshold determines when two answers are considered "the same"
    /// 
    /// More paths generally mean more reliable answers but take longer to compute.
    /// </para>
    /// </remarks>
    public class SelfConsistencyOptions<T> : ReasoningModelOptions<T>
    {
        /// <summary>
        /// Gets or sets the default number of reasoning paths to generate.
        /// </summary>
        /// <value>Default is 5.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is how many different ways the model will try to solve
        /// the problem. More paths mean more confidence in the answer but take more time.
        /// Think of it like asking 5 different experts - if they all agree, you can be more
        /// confident in the answer.
        /// </para>
        /// </remarks>
        public int DefaultPathCount { get; set; } = 5;

        /// <summary>
        /// Gets or sets the shape of path representations.
        /// </summary>
        /// <value>Default is [256].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This determines how the model internally represents each
        /// reasoning path. Larger values allow more complex paths but use more memory.
        /// </para>
        /// </remarks>
        public int[] PathRepresentationShape { get; set; } = new[] { 256 };

        /// <summary>
        /// Gets or sets the output shape for results.
        /// </summary>
        /// <value>Default is [128].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is the size of the final answer representation. It should
        /// match the complexity of the problems you're solving.
        /// </para>
        /// </remarks>
        public int[] OutputShape { get; set; } = new[] { 128 };

        /// <summary>
        /// Gets or sets the hidden layer size.
        /// </summary>
        /// <value>Default is 256.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls the model's internal processing capacity. Larger
        /// values make the model more powerful but slower.
        /// </para>
        /// </remarks>
        public int HiddenSize { get; set; } = 256;

        /// <summary>
        /// Gets or sets the number of consistency checking attention heads.
        /// </summary>
        /// <value>Default is 4.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> These heads help the model compare different reasoning paths
        /// from multiple perspectives. More heads mean more thorough comparison but slower
        /// processing.
        /// </para>
        /// </remarks>
        public int ConsistencyHeads { get; set; } = 4;

        /// <summary>
        /// Gets or sets the maximum steps per reasoning path.
        /// </summary>
        /// <value>Default is 15.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This limits how many steps each reasoning path can take.
        /// Longer paths can solve more complex problems but take more time.
        /// </para>
        /// </remarks>
        public int MaxStepsPerPath { get; set; } = 15;

        /// <summary>
        /// Gets or sets the dropout rate for path diversity.
        /// </summary>
        /// <value>Default is 0.2.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This adds randomness to make each path different. Higher values
        /// create more diverse paths (which can catch different errors) but might also create
        /// less accurate individual paths.
        /// </para>
        /// </remarks>
        public double PathDiversityDropout { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets the minimum consistency threshold.
        /// </summary>
        /// <value>Default is 0.7.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Paths must agree this much to be considered consistent. Higher
        /// values mean stricter agreement requirements, which can improve reliability but might
        /// reject valid answers if paths express them differently.
        /// </para>
        /// </remarks>
        public double MinConsistencyThreshold { get; set; } = 0.7;

        /// <summary>
        /// Gets or sets the similarity threshold for answer comparison.
        /// </summary>
        /// <value>Default is 0.85.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When comparing answers from different paths, they must be this
        /// similar to be considered "the same answer." Lower values allow more variation in how
        /// answers are expressed.
        /// </para>
        /// </remarks>
        public double SimilarityThreshold { get; set; } = 0.85;

        /// <summary>
        /// Gets or sets whether to use sampling for path diversity.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Sampling adds controlled randomness to create different paths.
        /// This is usually good because it helps explore different solutions, but you might
        /// disable it for deterministic (always same) results.
        /// </para>
        /// </remarks>
        public bool UseSamplingForDiversity { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of paths to generate for explanations.
        /// </summary>
        /// <value>Default is 3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When explaining an answer, the model generates this many paths
        /// to show different ways of reaching the conclusion. More paths give richer explanations
        /// but take longer.
        /// </para>
        /// </remarks>
        public int ExplanationPathCount { get; set; } = 3;

        /// <summary>
        /// Gets or sets the terminal state threshold.
        /// </summary>
        /// <value>Default is 0.01.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This helps the model know when a reasoning path is "done."
        /// When the changes become smaller than this threshold, the path stops.
        /// </para>
        /// </remarks>
        public double TerminalThreshold { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the number of refinement candidates.
        /// </summary>
        /// <value>Default is 3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When refining an answer, the model generates this many
        /// improved versions and picks the best one. More candidates mean better refinement
        /// but slower processing.
        /// </para>
        /// </remarks>
        public int RefinementCandidates { get; set; } = 3;

        /// <summary>
        /// Gets or sets whether to cache generated paths.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Caching remembers paths for repeated problems, making the
        /// model faster. Turn this off if you're working with many unique problems and want
        /// to save memory.
        /// </para>
        /// </remarks>
        public bool EnablePathCache { get; set; } = true;
    }
}