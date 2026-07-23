namespace AiDotNet.Models.Options;

/// <summary>
/// Top-level configuration for vertical federated learning (VFL).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Vertical Federated Learning allows multiple parties that hold
/// different features for the same entities to jointly train a model without sharing their
/// raw data. For example, a bank (income, credit score) and a hospital (diagnoses, prescriptions)
/// can jointly predict loan default risk.</para>
///
/// <para>This is fundamentally different from horizontal FL (where each party has the same
/// features for different samples). VFL requires:</para>
/// <list type="bullet">
/// <item><description>Entity alignment (finding shared entities via PSI)</description></item>
/// <item><description>Split neural networks (each party runs part of the model)</description></item>
/// <item><description>Secure gradient exchange (protecting intermediate activations)</description></item>
/// <item><description>Missing feature handling (not all parties have all entities)</description></item>
/// </list>
///
/// <para>Example:
/// <code>
/// var vflOptions = new VerticalFederatedLearningOptions
/// {
///     EntityAlignment = new PsiOptions { Protocol = PsiProtocol.ObliviousTransfer },
///     SplitModel = new SplitModelOptions { EmbeddingDimension = 64 },
///     MissingFeatures = new MissingFeatureOptions { Strategy = MissingFeatureStrategy.Mean },
///     LearningRate = 0.001,
///     NumberOfEpochs = 50
/// };
/// </code>
/// </para>
/// </remarks>
public class VerticalFederatedLearningOptions
{
    /// <summary>
    /// Gets or sets options for Private Set Intersection used to align entities across parties.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before training can begin, the parties must discover which
    /// entities (patients, customers, etc.) they share in common. PSI does this privately.</para>
    /// </remarks>
    public PsiOptions EntityAlignment { get; set; } = new PsiOptions();

    /// <summary>
    /// Gets or sets options for the split neural network architecture.
    /// </summary>
    public SplitModelOptions SplitModel { get; set; } = new SplitModelOptions();

    /// <summary>
    /// Gets or sets options for handling missing features across parties.
    /// </summary>
    public MissingFeatureOptions MissingFeatures { get; set; } = new MissingFeatureOptions();

    /// <summary>
    /// Gets or sets options for GDPR-compliant entity unlearning.
    /// </summary>
    public VflUnlearningOptions Unlearning { get; set; } = new VflUnlearningOptions();

    /// <summary>
    /// Gets or sets the learning rate for model training.
    /// </summary>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    public int NumberOfEpochs { get; set; } = 50;

    /// <summary>
    /// Gets or sets the batch size for mini-batch training.
    /// </summary>
    public int BatchSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of parties participating in VFL.
    /// </summary>
    public int NumberOfParties { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to encrypt gradients exchanged between parties.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Enabling encryption protects gradient values from being
    /// intercepted or analyzed. This adds computational overhead but is recommended
    /// for production deployments handling sensitive data.</para>
    /// </remarks>
    public bool EncryptGradients { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to apply differential privacy noise to the label holder's gradients.
    /// This prevents feature-holding parties from inferring labels from gradient values.
    /// </summary>
    public bool EnableLabelDifferentialPrivacy { get; set; }

    /// <summary>
    /// Gets or sets the differential privacy epsilon for label protection.
    /// Smaller values provide stronger privacy but may reduce model accuracy.
    /// </summary>
    public double LabelDpEpsilon { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the differential privacy delta for label protection.
    /// </summary>
    public double LabelDpDelta { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the random seed for reproducible training.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets whether to log detailed training metrics (loss, alignment stats, etc.)
    /// at each epoch.
    /// </summary>
    public bool VerboseLogging { get; set; }
}
