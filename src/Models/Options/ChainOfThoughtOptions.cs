namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration options specific to Chain-of-Thought reasoning models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These options control how the Chain-of-Thought model works:
    /// - InputShape and HiddenShape determine the size of data the model can process
    /// - HiddenSize controls the model's capacity (bigger = more powerful but slower)
    /// - AttentionHeads help the model focus on important parts of the reasoning
    /// - MaxChainLength limits how many steps the model can take to solve a problem
    /// 
    /// Start with the defaults and adjust based on your problem complexity.
    /// </para>
    /// </remarks>
    public class ChainOfThoughtOptions<T> : ReasoningModelOptions<T>
    {
        /// <summary>
        /// Gets or sets the input shape for the reasoning network.
        /// </summary>
        /// <value>Default is [512].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This defines how much information the model can take in at once.
        /// Larger values allow more complex inputs but require more memory. Think of it like the
        /// size of a worksheet - bigger sheets can hold more information but take more space.
        /// </para>
        /// </remarks>
        public int[] InputShape { get; set; } = new[] { 512 };

        /// <summary>
        /// Gets or sets the hidden representation shape.
        /// </summary>
        /// <value>Default is [256].</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is the size of the model's "working memory" for each reasoning
        /// step. Larger values allow more complex thoughts but slow down processing. It's like having
        /// more scratch paper to work out a problem.
        /// </para>
        /// </remarks>
        public int[] HiddenShape { get; set; } = new[] { 256 };

        /// <summary>
        /// Gets or sets the hidden layer size for the reasoning network.
        /// </summary>
        /// <value>Default is 128.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls the internal processing power of the model. Higher
        /// values make the model more capable but also slower and more memory-intensive. It's a
        /// balance between capability and efficiency.
        /// </para>
        /// </remarks>
        public int HiddenSize { get; set; } = 128;

        /// <summary>
        /// Gets or sets the number of attention heads.
        /// </summary>
        /// <value>Default is 8.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Attention heads let the model focus on different aspects of the
        /// problem simultaneously. More heads mean the model can consider more perspectives at once,
        /// like having multiple experts looking at different parts of a problem.
        /// </para>
        /// </remarks>
        public int AttentionHeads { get; set; } = 8;

        /// <summary>
        /// Gets or sets the maximum chain length for reasoning.
        /// </summary>
        /// <value>Default is 20.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is the maximum number of "thinking steps" the model can take.
        /// Longer chains allow solving more complex problems but take more time. It's like limiting
        /// how many steps you can show when solving a math problem.
        /// </para>
        /// </remarks>
        public int MaxChainLength { get; set; } = 20;

        /// <summary>
        /// Gets or sets the threshold for determining terminal steps.
        /// </summary>
        /// <value>Default is 0.01.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This helps the model know when it's "done thinking." When the
        /// changes between steps become smaller than this threshold, the model stops. Lower values
        /// mean more thorough reasoning but might lead to unnecessary steps.
        /// </para>
        /// </remarks>
        public double TerminalThreshold { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets whether to use bidirectional reasoning.
        /// </summary>
        /// <value>Default is false.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Bidirectional reasoning means the model can think both forward
        /// (from problem to solution) and backward (from desired outcome to requirements). This
        /// can help solve certain types of problems more effectively but doubles the computation.
        /// </para>
        /// </remarks>
        public bool UseBidirectionalReasoning { get; set; } = false;

        /// <summary>
        /// Gets or sets the dropout rate for reasoning steps.
        /// </summary>
        /// <value>Default is 0.1.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Dropout randomly "turns off" some parts of the model during training
        /// to prevent overfitting (memorizing instead of learning). Higher values make the model more
        /// robust but might reduce accuracy. It's like practicing with handicaps to become stronger.
        /// </para>
        /// </remarks>
        public double DropoutRate { get; set; } = 0.1;

        /// <summary>
        /// Gets or sets whether to use gradient checkpointing for memory efficiency.
        /// </summary>
        /// <value>Default is true.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Gradient checkpointing trades computation time for memory. When
        /// enabled, the model uses less memory but takes slightly longer to train. This is useful
        /// for running larger models on limited hardware.
        /// </para>
        /// </remarks>
        public bool UseGradientCheckpointing { get; set; } = true;

        /// <summary>
        /// Gets or sets the learning rate for the reasoning network.
        /// </summary>
        /// <value>Default is 0.001.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> The learning rate controls how quickly the model learns from mistakes.
        /// Too high and it might overshoot good solutions; too low and it learns very slowly. It's
        /// like adjusting how much you correct your aim after missing a target.
        /// </para>
        /// </remarks>
        public double LearningRate { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the weight decay for regularization.
        /// </summary>
        /// <value>Default is 0.0001.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Weight decay prevents the model from becoming too confident in its
        /// reasoning by slightly reducing all weights over time. This helps the model generalize
        /// better to new problems instead of just memorizing training examples.
        /// </para>
        /// </remarks>
        public double WeightDecay { get; set; } = 0.0001;

        /// <summary>
        /// Gets or sets the convergence threshold for iterative reasoning.
        /// </summary>
        /// <value>Default is 0.001.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This determines when the model should stop refining its reasoning
        /// because the improvements are too small to matter. Lower values mean more precise reasoning
        /// but take longer to converge.
        /// </para>
        /// </remarks>
        public double ConvergenceThreshold { get; set; } = 0.001;

        /// <summary>
        /// Gets or sets the exploration factor for reasoning diversity.
        /// </summary>
        /// <value>Default is 0.3.</value>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This controls how much the model explores different reasoning paths
        /// versus sticking to the most promising one. Higher values encourage more exploration,
        /// which can find better solutions but takes more time.
        /// </para>
        /// </remarks>
        public double ExplorationFactor { get; set; } = 0.3;
    }
}