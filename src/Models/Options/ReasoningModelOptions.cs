using AiDotNet.Enums;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration options for reasoning models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how reasoning models work. You can think of them as
    /// settings that determine how the AI "thinks through" problems. For example:
    /// - DefaultMaxSteps controls how many reasoning steps the model can take (like limiting how many
    ///   steps you can use to solve a math problem)
    /// - DefaultStrategy determines the approach (like deciding whether to work forward from what you
    ///   know or backward from what you want to find)
    /// - Temperature affects randomness (higher values make the reasoning more creative but less predictable)
    /// 
    /// Start with the default values and adjust based on your specific needs.
    /// </para>
    /// </remarks>
    public class ReasoningModelOptions<T> : ModelOptions
    {
        /// <summary>
        /// Gets or sets the default maximum number of reasoning steps.
        /// </summary>
        /// <value>Default is 10.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This limits how many steps the model can take when solving a problem.
        /// More steps allow for more complex reasoning but take longer. Think of it like limiting the
        /// number of moves in a chess game - sometimes you need more moves for complex positions.
        /// </para>
        /// </remarks>
        public int DefaultMaxSteps { get; set; } = 10;

        /// <summary>
        /// Gets or sets the default reasoning strategy.
        /// </summary>
        /// <value>Default is ForwardChaining.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> The reasoning strategy determines how the model approaches problems:
        /// - ForwardChaining: Start from what you know and work toward the goal
        /// - BackwardChaining: Start from the goal and work backward to find supporting facts
        /// - Bidirectional: Use both approaches together
        /// - BeamSearch: Explore multiple promising paths simultaneously
        /// 
        /// Forward chaining is usually the simplest and most intuitive approach.
        /// </para>
        /// </remarks>
        public ReasoningStrategy DefaultStrategy { get; set; } = ReasoningStrategy.ForwardChaining;

        /// <summary>
        /// Gets or sets the temperature for stochastic reasoning.
        /// </summary>
        /// <value>Default is 0.7.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Temperature controls randomness in reasoning:
        /// - Low values (0.1-0.5): More deterministic, conservative reasoning
        /// - Medium values (0.6-0.8): Balanced creativity and consistency
        /// - High values (0.9-1.5): More creative, exploratory reasoning
        /// 
        /// Lower temperatures are better for tasks requiring precision (like math),
        /// while higher temperatures work well for creative tasks.
        /// </para>
        /// </remarks>
        public double Temperature { get; set; } = 0.7;

        /// <summary>
        /// Gets or sets whether to vary strategies during self-consistency checking.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When the model checks its work by solving the problem multiple ways,
        /// this setting determines whether it should try different approaches each time. Varying
        /// strategies can catch errors that a single approach might miss, like checking math by
        /// solving it different ways.
        /// </para>
        /// </remarks>
        public bool VaryStrategyInSelfConsistency { get; set; } = true;

        /// <summary>
        /// Gets or sets the beam width for beam search strategy.
        /// </summary>
        /// <value>Default is 5.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When using beam search, this controls how many different reasoning
        /// paths the model explores at each step. Higher values explore more possibilities but take
        /// more time. Think of it like considering the top 5 best moves in chess instead of just
        /// the single best move.
        /// </para>
        /// </remarks>
        public int BeamWidth { get; set; } = 5;

        /// <summary>
        /// Gets or sets whether to enable reasoning chain validation.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This makes the model check whether each reasoning step logically
        /// follows from the previous ones. It's like having a built-in fact-checker that ensures
        /// the model's "thought process" makes sense. Turning this off can speed up reasoning but
        /// might allow logical errors.
        /// </para>
        /// </remarks>
        public bool EnableChainValidation { get; set; } = true;

        /// <summary>
        /// Gets or sets the minimum confidence threshold for accepting reasoning steps.
        /// </summary>
        /// <value>Default is 0.6.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> The model assigns confidence scores to each reasoning step. Steps
        /// below this threshold might be reconsidered or refined. Higher thresholds mean the model
        /// is more cautious and might take longer to reach conclusions. Lower thresholds allow
        /// faster but potentially less reliable reasoning.
        /// </para>
        /// </remarks>
        public double MinConfidenceThreshold { get; set; } = 0.6;

        /// <summary>
        /// Gets or sets whether to store detailed diagnostics during reasoning.
        /// </summary>
        /// <value>Default is false.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Enabling this makes the model keep track of detailed information
        /// about its reasoning process (timing, memory usage, decision points). This is helpful
        /// for debugging and understanding how the model works, but uses more memory and slightly
        /// slows down reasoning.
        /// </para>
        /// </remarks>
        public bool EnableDetailedDiagnostics { get; set; } = false;

        /// <summary>
        /// Gets or sets the maximum depth for recursive reasoning.
        /// </summary>
        /// <value>Default is 5.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Some problems require the model to "dive deeper" into sub-problems.
        /// This limits how deep the model can go to prevent infinite loops and excessive computation.
        /// It's like limiting how many "why?" questions a child can ask in a row - at some point,
        /// you need to stop and work with what you have.
        /// </para>
        /// </remarks>
        public int MaxRecursionDepth { get; set; } = 5;

        /// <summary>
        /// Gets or sets whether to use caching for repeated reasoning patterns.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> When the model encounters similar reasoning patterns, it can remember
        /// and reuse previous solutions instead of solving them again. This is like remembering that
        /// 2+2=4 instead of counting on your fingers every time. Caching speeds up reasoning but uses
        /// more memory.
        /// </para>
        /// </remarks>
        public bool EnableReasoningCache { get; set; } = true;

        /// <summary>
        /// Gets or sets the timeout for individual reasoning steps in milliseconds.
        /// </summary>
        /// <value>Default is 5000 (5 seconds).</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This prevents the model from getting stuck on difficult reasoning
        /// steps. If a step takes longer than this timeout, the model will move on or try a different
        /// approach. Set this based on your performance requirements and problem complexity.
        /// </para>
        /// </remarks>
        public int StepTimeoutMs { get; set; } = 5000;
    }
}