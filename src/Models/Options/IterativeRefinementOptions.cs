namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration options specific to Iterative Refinement reasoning models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how the Iterative Refinement model improves
    /// its answers through multiple revisions:
    /// - MaxRefinementIterations is like how many drafts you write before the final version
    /// - ConvergenceThreshold determines when the answer is "good enough" to stop refining
    /// - CritiqueDecay makes the model less critical over time (like becoming satisfied with improvements)
    /// - ResidualWeight controls how much of the original answer to keep in each revision
    /// 
    /// The model works like an editor who keeps improving a document until it's perfect.
    /// </para>
    /// </remarks>
    public class IterativeRefinementOptions<T> : ReasoningModelOptions<T>
    {
        /// <summary>
        /// Gets or sets the input shape for the model.
        /// </summary>
        /// <value>Default is [512].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This defines how much information the model can process at once.
        /// Larger values allow more complex inputs but require more memory.
        /// </para>
        /// </remarks>
        public int[] InputShape { get; set; } = new[] { 512 };

        /// <summary>
        /// Gets or sets the shape of reasoning representations.
        /// </summary>
        /// <value>Default is [256].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is the size of the model's "thoughts" during reasoning.
        /// Larger values allow more complex reasoning but slow down processing.
        /// </para>
        /// </remarks>
        public int[] ReasoningShape { get; set; } = new[] { 256 };

        /// <summary>
        /// Gets or sets the shape of critique representations.
        /// </summary>
        /// <value>Default is [128].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This determines how detailed the model's self-criticism can be.
        /// The critique identifies what needs improvement in each iteration.
        /// </para>
        /// </remarks>
        public int[] CritiqueShape { get; set; } = new[] { 128 };

        /// <summary>
        /// Gets or sets the hidden layer size.
        /// </summary>
        /// <value>Default is 512.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls the model's internal processing capacity. Larger
        /// values make the model more powerful but slower and more memory-intensive.
        /// </para>
        /// </remarks>
        public int HiddenSize { get; set; } = 512;

        /// <summary>
        /// Gets or sets the number of attention heads for the critic.
        /// </summary>
        /// <value>Default is 8.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Attention heads help the critic analyze different aspects of
        /// the reasoning simultaneously. More heads mean more thorough critique but slower
        /// processing.
        /// </para>
        /// </remarks>
        public int AttentionHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets the maximum number of refinement iterations.
        /// </summary>
        /// <value>Default is 5.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is the maximum number of times the model can revise its
        /// answer. More iterations can produce better results but take longer. It's like limiting
        /// how many drafts you can write.
        /// </para>
        /// </remarks>
        public int MaxRefinementIterations { get; set; } = 5;

        /// <summary>
        /// Gets or sets the default number of refinement iterations.
        /// </summary>
        /// <value>Default is 3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is the usual number of revisions the model makes. It can
        /// stop early if the answer converges, or continue up to MaxRefinementIterations if needed.
        /// </para>
        /// </remarks>
        public int DefaultRefinementIterations { get; set; } = 3;

        /// <summary>
        /// Gets or sets the convergence threshold.
        /// </summary>
        /// <value>Default is 0.95.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When the model thinks its answer is this good (95% confident),
        /// it stops refining. Higher values mean more perfectionism but might never converge.
        /// </para>
        /// </remarks>
        public double ConvergenceThreshold { get; set; } = 0.95;

        /// <summary>
        /// Gets or sets the critique decay rate.
        /// </summary>
        /// <value>Default is 0.1.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This makes the model less critical with each iteration, like
        /// becoming satisfied with smaller improvements over time. Higher values mean faster
        /// decay (less perfectionism in later iterations).
        /// </para>
        /// </remarks>
        public double CritiqueDecay { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets whether to use residual connections.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Residual connections keep some of the original answer in each
        /// revision, preventing the model from completely changing its mind. This usually leads
        /// to more stable refinement.
        /// </para>
        /// </remarks>
        public bool UseResidualConnections { get; set; } = true;

        /// <summary>
        /// Gets or sets the residual connection weight.
        /// </summary>
        /// <value>Default is 0.7.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls how much of the previous answer to keep (70%)
        /// versus how much to change (30%) in each revision. Higher values mean more conservative
        /// changes.
        /// </para>
        /// </remarks>
        public double ResidualWeight { get; set; } = 0.7;

        /// <summary>
        /// Gets or sets whether to use momentum in refinement.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Momentum helps the refinement process continue in the same
        /// direction rather than zigzagging. It's like having inertia in the improvement process.
        /// </para>
        /// </remarks>
        public bool UseMomentum { get; set; } = true;

        /// <summary>
        /// Gets or sets the momentum coefficient.
        /// </summary>
        /// <value>Default is 0.9.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls how much the previous improvement direction
        /// influences the next one. Higher values mean stronger momentum (more consistent
        /// improvement direction).
        /// </para>
        /// </remarks>
        public double MomentumCoefficient { get; set; } = 0.9;

        /// <summary>
        /// Gets or sets the terminal quality threshold.
        /// </summary>
        /// <value>Default is 0.9.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When the model judges its answer quality to be above this
        /// threshold, it considers the reasoning complete. It's like setting a quality bar
        /// for acceptable answers.
        /// </para>
        /// </remarks>
        public double TerminalQualityThreshold { get; set; } = 0.9;

        /// <summary>
        /// Gets or sets the number of refinement steps during training.
        /// </summary>
        /// <value>Default is 4.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> During training, the model learns by seeing sequences of
        /// improvements. This controls how many improvement steps to show it.
        /// </para>
        /// </remarks>
        public int TrainingRefinementSteps { get; set; } = 4;

        /// <summary>
        /// Gets or sets the number of iterations for explanations.
        /// </summary>
        /// <value>Default is 2.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When explaining its reasoning, the model shows this many
        /// refinement steps. Fewer steps make explanations simpler but less detailed.
        /// </para>
        /// </remarks>
        public int ExplanationIterations { get; set; } = 2;

        /// <summary>
        /// Gets or sets whether to store refinement history.
        /// </summary>
        /// <value>Default is false.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Storing history lets you analyze how the model improved its
        /// answers, but uses more memory. Enable this for debugging or analysis.
        /// </para>
        /// </remarks>
        public bool StoreRefinementHistory { get; set; } = false;
    }
}