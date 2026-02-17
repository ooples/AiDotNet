namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for GDPR-compliant entity unlearning in vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Under GDPR and similar privacy regulations, individuals have the
/// "right to be forgotten". When a person requests deletion, not only must their data be removed
/// from storage, but the model must also be updated to remove any influence of their data.
/// This is called "machine unlearning".</para>
///
/// <para>In VFL, this is more complex than in standard ML because data is spread across
/// multiple parties, and the model is split across parties. Each party must participate
/// in the unlearning process.</para>
///
/// <para>Example:
/// <code>
/// var unlearningOptions = new VflUnlearningOptions
/// {
///     Enabled = true,
///     Method = VflUnlearningMethod.Certified,
///     MaxUnlearnBatchSize = 100
/// };
/// </code>
/// </para>
/// </remarks>
public class VflUnlearningOptions
{
    /// <summary>
    /// Gets or sets whether unlearning support is enabled for this VFL training run.
    /// When enabled, the trainer stores information needed for efficient unlearning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Enable this if you may need to remove individuals from the
    /// model later. This adds some overhead during training but makes unlearning possible
    /// without full retraining.</para>
    /// </remarks>
    public bool Enabled { get; set; }

    /// <summary>
    /// Gets or sets the unlearning method to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GradientAscent is fastest but least thorough.
    /// Certified provides the strongest guarantees but may reduce accuracy slightly.
    /// Retraining is the gold standard but requires redoing all training.</para>
    /// </remarks>
    public VflUnlearningMethod Method { get; set; } = VflUnlearningMethod.GradientAscent;

    /// <summary>
    /// Gets or sets the maximum number of entities to unlearn in a single batch.
    /// Larger batches are more efficient but may cause larger accuracy drops.
    /// </summary>
    public int MaxUnlearnBatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of gradient ascent steps for the GradientAscent method.
    /// More steps remove more influence but may overshoot and degrade model quality.
    /// </summary>
    public int GradientAscentSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the learning rate for gradient ascent unlearning.
    /// </summary>
    public double UnlearningLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the privacy budget (epsilon) for certified unlearning.
    /// Smaller values provide stronger privacy guarantees.
    /// </summary>
    public double CertificationEpsilon { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to verify unlearning effectiveness by checking that
    /// the unlearned model cannot distinguish removed entities from unseen entities.
    /// </summary>
    public bool VerifyUnlearning { get; set; }
}
