namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// SEAL is a meta-learning algorithm that combines gradient-based meta-learning with
/// efficient adaptation strategies. It learns initial parameters that can be quickly
/// adapted to new tasks with just a few gradient steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> SEAL is an algorithm that learns how to learn quickly.
/// It trains on many different tasks, learning starting points (initial parameters)
/// that make it easy to adapt to new tasks with just a few examples.
///
/// Think of it like learning to cook:
/// - Instead of learning just one recipe, you learn cooking principles
/// - When you see a new recipe, you can adapt quickly because you understand the basics
/// - SEAL does the same with machine learning - it learns principles that help it
///   quickly adapt to new tasks
/// </para>
/// </remarks>
public class SEALAlgorithmOptions<T, TInput, TOutput> : MetaLearningAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the temperature parameter for SEAL's adaptation strategy.
    /// </summary>
    /// <value>The temperature value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Temperature controls how "confident" the model is during adaptation.
    /// - Higher values (>1.0) make the model more exploratory, considering more possibilities
    /// - Lower values (<1.0) make the model more focused on the most likely predictions
    /// - 1.0 is neutral (no temperature scaling)
    ///
    /// This is particularly useful when adapting to very few examples, where you want to
    /// avoid being overconfident based on limited data.
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use adaptive inner learning rate.
    /// </summary>
    /// <value>True to adapt the inner learning rate during meta-training, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When enabled, SEAL learns the best learning rate to use
    /// for each task adaptation, rather than using a fixed learning rate. This can improve
    /// performance but makes training slightly more complex.
    ///
    /// Think of it like learning not just what to learn, but also how fast to learn it.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveInnerLR { get; set; } = false;

    /// <summary>
    /// Gets or sets the entropy regularization coefficient.
    /// </summary>
    /// <value>The entropy coefficient, defaulting to 0.0 (no regularization).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Entropy regularization encourages the model to maintain
    /// some uncertainty in its predictions, which can help prevent overfitting to the
    /// few examples in the support set.
    ///
    /// - 0.0 means no entropy regularization
    /// - Higher values (e.g., 0.01-0.1) encourage more diverse predictions
    ///
    /// This is like telling the model "don't be too sure of yourself based on just
    /// a few examples."
    /// </para>
    /// </remarks>
    public double EntropyCoefficient { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use context-dependent adaptation.
    /// </summary>
    /// <value>True to use context-dependent adaptation, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Context-dependent adaptation allows SEAL to adjust its
    /// adaptation strategy based on the characteristics of each task. Different tasks
    /// might need different adaptation approaches.
    ///
    /// For example, some tasks might need more aggressive adaptation while others
    /// need more conservative updates. This feature lets SEAL learn which approach
    /// works best for each situation.
    /// </para>
    /// </remarks>
    public bool UseContextDependentAdaptation { get; set; } = false;

    /// <summary>
    /// Gets or sets the gradient clipping threshold.
    /// </summary>
    /// <value>The gradient clipping value, or null for no clipping.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient clipping prevents very large gradient updates
    /// that can destabilize training. If gradients exceed this threshold, they are
    /// scaled down to this maximum value.
    ///
    /// This is like having a speed limit for how much the model can change in one step.
    /// A typical value is 10.0, or null to disable clipping.
    /// </para>
    /// </remarks>
    public double? GradientClipThreshold { get; set; } = null;

    /// <summary>
    /// Gets or sets the weight decay (L2 regularization) coefficient.
    /// </summary>
    /// <value>The weight decay coefficient, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weight decay prevents the model parameters from becoming
    /// too large, which helps prevent overfitting. It adds a small penalty for large
    /// parameter values.
    ///
    /// - 0.0 means no weight decay
    /// - Small values like 0.0001-0.001 are typical
    ///
    /// Think of it as encouraging the model to keep things simple.
    /// </para>
    /// </remarks>
    public double WeightDecay { get; set; } = 0.0;
}
