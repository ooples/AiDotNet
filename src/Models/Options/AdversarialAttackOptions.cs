namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for adversarial attack algorithms.
/// </summary>
/// <remarks>
/// <para>
/// These options control how adversarial examples are generated, including the strength
/// of perturbations, attack iterations, and norm constraints.
/// </para>
/// <para><b>For Beginners:</b> These settings control how the "stress test" for your AI works.
/// You can adjust how strong the attacks are, how many attempts they make to fool your model,
/// and what type of changes they're allowed to make to inputs.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class AdversarialAttackOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the maximum perturbation budget (epsilon).
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon controls how much the attack can change the input.
    /// For images, 0.1 means changing pixel values by up to 10%. Smaller values make attacks
    /// harder to detect but less powerful, while larger values make attacks more effective
    /// but easier to notice.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the step size for iterative attacks.
    /// </summary>
    /// <value>The step size, defaulting to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For iterative attacks (like PGD), this controls how big
    /// each step is when searching for adversarial examples. Smaller steps are more precise
    /// but take more iterations, while larger steps are faster but might miss good attacks.</para>
    /// </remarks>
    public double StepSize { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the number of iterations for iterative attacks.
    /// </summary>
    /// <value>The number of iterations, defaulting to 40.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many attempts the attack makes to fool the model.
    /// More iterations generally create stronger attacks but take longer to compute.</para>
    /// </remarks>
    public int Iterations { get; set; } = 40;

    /// <summary>
    /// Gets or sets the norm type for perturbation constraints.
    /// </summary>
    /// <value>The norm type, defaulting to <see cref="AdversarialNormType.LInfinity"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This defines what "small change" means:
    /// - L-infinity: No single value changes by more than epsilon
    /// - L2: The total change size is limited by epsilon</para>
    /// </remarks>
    public AdversarialNormType NormType { get; set; } = AdversarialNormType.LInfinity;

    /// <summary>
    /// Gets or sets whether to use targeted or untargeted attacks.
    /// </summary>
    /// <value>True for targeted attacks, false for untargeted (default).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// - Untargeted: Just try to make the model wrong (easier)
    /// - Targeted: Try to make the model predict a specific wrong class (harder)</para>
    /// </remarks>
    public bool IsTargeted { get; set; } = false;

    /// <summary>
    /// Gets or sets the target class for targeted attacks.
    /// </summary>
    /// <value>The target class index, defaulting to 0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If using targeted attacks, this is the class you want
    /// to trick the model into predicting.</para>
    /// </remarks>
    public int TargetClass { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use random initialization.
    /// </summary>
    /// <value>True to use random initialization (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Starting from a random point within the allowed budget
    /// can make attacks stronger by exploring different starting positions.</para>
    /// </remarks>
    public bool UseRandomStart { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>The random seed, defaulting to 42.</value>
    public int RandomSeed { get; set; } = 42;

    /// <summary>
    /// Gets or sets the Carlini-Wagner trade-off constant c between distortion and the attack loss.
    /// </summary>
    /// <value>Defaults to 1, the torchattacks CW default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger values push harder to fool the model at the cost of a
    /// bigger change to the input.</para>
    /// </remarks>
    public double CWConstant { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the Carlini-Wagner confidence margin kappa.
    /// </summary>
    /// <value>Defaults to 0, as in the paper's L2 attack.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The attack keeps going until the wrong class wins by at least this
    /// margin in the model's logits; higher values give adversarial examples that transfer better.</para>
    /// </remarks>
    public double CWConfidence { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the Adam learning rate the Carlini-Wagner attack uses in tanh space.
    /// </summary>
    /// <value>Defaults to 0.01, the paper's and torchattacks' value.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets how big a step the attack's optimizer takes each iteration.
    /// Larger values move faster but can overshoot and bounce around a good solution; smaller values are
    /// steadier but need more iterations to converge.</para>
    /// </remarks>
    public double CWLearningRate { get; set; } = 0.01;
}
