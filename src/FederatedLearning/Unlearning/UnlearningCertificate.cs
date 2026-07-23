using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Certificate proving that a client's data has been unlearned from the federated model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a client requests data removal, the system needs to prove it
/// actually happened. This certificate contains verifiable metrics showing the client's contribution
/// has been removed. Think of it as a "receipt of forgetting" — proof for GDPR auditors that the
/// right to be forgotten was honored.</para>
///
/// <para><b>Key metrics:</b></para>
/// <list type="bullet">
/// <item><description><b>MembershipInferenceScore:</b> After unlearning, how well can an attack distinguish
/// whether the client's data was used? Lower = better unlearning.</description></item>
/// <item><description><b>ModelDivergence:</b> How different is the unlearned model from the original?
/// Should be noticeable but not catastrophic.</description></item>
/// <item><description><b>RetainedAccuracy:</b> How much accuracy remains on non-target clients?
/// Should stay high (unlearning shouldn't hurt others).</description></item>
/// </list>
/// </remarks>
public class UnlearningCertificate
{
    /// <summary>
    /// Gets or sets the client ID whose data was unlearned.
    /// </summary>
    public int TargetClientId { get; set; }

    /// <summary>
    /// Gets or sets the unlearning method used.
    /// </summary>
    public UnlearningMethod MethodUsed { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when unlearning was performed.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets whether the unlearning was verified as correct.
    /// </summary>
    public bool Verified { get; set; }

    /// <summary>
    /// Gets or sets the membership inference attack score after unlearning.
    /// Range [0, 1]: 0.5 = no memorization detected (ideal), close to 1.0 = still memorized (bad).
    /// </summary>
    public double MembershipInferenceScore { get; set; }

    /// <summary>
    /// Gets or sets the L2 distance between the original and unlearned model parameters.
    /// </summary>
    public double ModelDivergence { get; set; }

    /// <summary>
    /// Gets or sets the retained accuracy on non-target clients after unlearning.
    /// Range [0, 1]: 1.0 = no accuracy loss (ideal).
    /// </summary>
    public double RetainedAccuracy { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of training rounds the target client participated in.
    /// </summary>
    public int ClientRoundsParticipated { get; set; }

    /// <summary>
    /// Gets or sets the wall-clock time in milliseconds the unlearning took.
    /// </summary>
    public long UnlearningTimeMs { get; set; }

    /// <summary>
    /// Gets or sets a hash of the model state before unlearning (for audit trail).
    /// </summary>
    public string PreUnlearningModelHash { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets a hash of the model state after unlearning.
    /// </summary>
    public string PostUnlearningModelHash { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets an optional human-readable summary of the unlearning result.
    /// </summary>
    public string Summary { get; set; } = string.Empty;
}
