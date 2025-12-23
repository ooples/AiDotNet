namespace AiDotNet.Models;

/// <summary>
/// Represents adversarial robustness diagnostics aggregated over a dataset.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This container is designed to integrate with the existing AiDotNet evaluation pipeline by living alongside
/// <see cref="ErrorStats{T}"/> and <see cref="PredictionStats{T}"/> within <see cref="DataSetStats{T, TInput, TOutput}"/>.
/// It stores metrics related to model robustness against adversarial attacks and certified defenses.
/// </para>
/// <para><b>For Beginners:</b> This stores summary robustness metrics (like accuracy under attack)
/// for an entire dataset, helping you understand how well your model resists adversarial perturbations.
///
/// Key concepts:
/// - **Clean Accuracy**: How accurate the model is on unmodified inputs
/// - **Adversarial Accuracy**: How accurate the model is when inputs are perturbed by attacks
/// - **Certified Accuracy**: The fraction of samples with provably correct predictions within a perturbation radius
/// - **Attack Success Rate**: How often an attacker can fool the model
/// - **Average Perturbation Size**: How much inputs need to be changed to fool the model
/// </para>
/// </remarks>
public sealed class RobustnessStats<T>
{
    /// <summary>
    /// Gets the accuracy of the model on clean (unperturbed) inputs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the normal accuracy you would measure without any attacks.
    /// It serves as a baseline to compare against adversarial accuracy.</para>
    /// </remarks>
    public double CleanAccuracy { get; set; }

    /// <summary>
    /// Gets the accuracy of the model on adversarially perturbed inputs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how often the model still makes correct predictions
    /// when inputs have been modified by an adversarial attack. Lower values mean the model is
    /// more vulnerable to attacks.</para>
    /// </remarks>
    public double AdversarialAccuracy { get; set; }

    /// <summary>
    /// Gets the fraction of inputs for which the attack successfully fooled the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is 1 - AdversarialAccuracy for correctly classified clean inputs.
    /// A high attack success rate means the model is easy to fool.</para>
    /// </remarks>
    public double AttackSuccessRate { get; set; }

    /// <summary>
    /// Gets the average size of perturbations needed to create successful adversarial examples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how much the input needs to be changed on average
    /// to fool the model. Larger values suggest the model is more robust (harder to fool with
    /// small changes).</para>
    /// </remarks>
    public double AveragePerturbationSize { get; set; }

    /// <summary>
    /// Gets the certified accuracy at the specified perturbation radius.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the fraction of predictions that are mathematically
    /// guaranteed to be correct even if the input is perturbed within a certain radius.
    /// Unlike adversarial accuracy (which tests specific attacks), certified accuracy provides
    /// provable guarantees against ALL possible perturbations within the radius.</para>
    /// </remarks>
    public double CertifiedAccuracy { get; set; }

    /// <summary>
    /// Gets the average certified robustness radius across samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the average "safety zone" around inputs where predictions
    /// are guaranteed to stay the same. Larger radii mean stronger certified robustness.</para>
    /// </remarks>
    public double AverageCertifiedRadius { get; set; }

    /// <summary>
    /// Gets the perturbation radius (epsilon) used for robustness evaluation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the maximum allowed perturbation size used when
    /// evaluating robustness. For image data, this is often around 8/255 ≈ 0.031 for L-infinity
    /// attacks (imperceptible pixel changes).</para>
    /// </remarks>
    public double EvaluationEpsilon { get; set; }

    /// <summary>
    /// Gets the type of attack used for adversarial robustness evaluation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different attacks have different strengths:
    /// - FGSM: Fast but weak
    /// - PGD: Slower but stronger
    /// - C&amp;W: Slowest but often finds smallest perturbations
    /// - AutoAttack: Ensemble of attacks for reliable evaluation
    /// </para>
    /// </remarks>
    public string AttackType { get; set; } = string.Empty;

    /// <summary>
    /// Gets the norm type used for measuring perturbation size (e.g., "L2", "Linf").
    /// </summary>
    public string NormType { get; set; } = "Linf";

    /// <summary>
    /// Gets a combined robustness score (0-1) that balances clean and adversarial performance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a single number that summarizes overall robustness.
    /// Higher values indicate better robustness. The default formula is:
    /// (CleanAccuracy + AdversarialAccuracy) / 2</para>
    /// </remarks>
    public double RobustnessScore { get; set; }

    /// <summary>
    /// Gets a dictionary of additional robustness metrics.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This stores any extra metrics that don't fit the standard
    /// properties, allowing for extensibility without changing the class structure.</para>
    /// </remarks>
    public Dictionary<string, T> AdditionalMetrics { get; } = new();

    /// <summary>
    /// Gets or sets whether robustness evaluation has been performed.
    /// </summary>
    public bool IsEvaluated { get; set; }

    /// <summary>
    /// Creates an empty <see cref="RobustnessStats{T}"/> instance.
    /// </summary>
    public static RobustnessStats<T> Empty()
    {
        return new RobustnessStats<T>();
    }
}
