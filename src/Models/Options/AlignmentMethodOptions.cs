namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for AI alignment methods.
/// </summary>
/// <remarks>
/// <para>
/// These options control how models are aligned with human values and intentions through
/// techniques like RLHF, constitutional AI, and red teaming.
/// </para>
/// <para><b>For Beginners:</b> These settings control how your AI learns to behave according
/// to human values. You can adjust how much to weight human feedback, what principles to follow,
/// and how thoroughly to test for problems.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class AlignmentMethodOptions<T>
{
    /// <summary>
    /// Gets or sets the learning rate for alignment training.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.00001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how quickly the model adapts to human feedback.
    /// Smaller values make training more stable but slower.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the number of training iterations for RLHF.
    /// </summary>
    /// <value>The number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More iterations allow the model to learn better from feedback
    /// but take longer to train.</para>
    /// </remarks>
    public int TrainingIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the discount factor for reward modeling.
    /// </summary>
    /// <value>The gamma value (0-1), defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how much the model values long-term rewards
    /// vs. immediate rewards. Higher values make it consider future consequences more.</para>
    /// </remarks>
    public double Gamma { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the KL divergence penalty coefficient.
    /// </summary>
    /// <value>The KL coefficient, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the model from changing too much during alignment.
    /// It's like a leash that keeps the aligned model close to the original model's behavior.</para>
    /// </remarks>
    public double KLCoefficient { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use constitutional AI principles.
    /// </summary>
    /// <value>True to use constitutional AI, false otherwise (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Constitutional AI teaches the model explicit principles
    /// to guide its behavior, like a set of rules to follow.</para>
    /// </remarks>
    public bool UseConstitutionalAI { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of critique iterations for constitutional AI.
    /// </summary>
    /// <value>The number of critique iterations, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model critiques and improves its own outputs this many times
    /// using the constitutional principles.</para>
    /// </remarks>
    public int CritiqueIterations { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to perform red teaming.
    /// </summary>
    /// <value>True to enable red teaming, false otherwise (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Red teaming tries to find ways to make the model misbehave,
    /// helping you discover and fix problems before deployment.</para>
    /// </remarks>
    public bool EnableRedTeaming { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of red teaming attempts.
    /// </summary>
    /// <value>The number of attempts, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More attempts mean more thorough testing but take longer.</para>
    /// </remarks>
    public int RedTeamingAttempts { get; set; } = 100;

    /// <summary>
    /// Gets or sets the reward model architecture.
    /// </summary>
    /// <value>The model architecture name, defaulting to "Transformer".</value>
    public string RewardModelArchitecture { get; set; } = "Transformer";
}
