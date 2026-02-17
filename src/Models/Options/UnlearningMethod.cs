namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated unlearning method to use when a client requests data removal.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When someone exercises their "right to be forgotten" (GDPR Article 17),
/// their contribution to the model must be removed. These methods differ in speed vs. guarantee strength:</para>
/// <list type="bullet">
/// <item><description><b>ExactRetraining:</b> Retrain from scratch without the target client. Provably correct
/// but very expensive (days of compute). Gold standard for audits.</description></item>
/// <item><description><b>GradientAscent:</b> Run training in reverse (ascending the loss) on the target client's
/// data to undo their learning. Fast (minutes), approximate guarantee.</description></item>
/// <item><description><b>InfluenceFunction:</b> Mathematically estimate each client's contribution and subtract it
/// using a Newton step. Efficient for small removals, degrades for large ones.</description></item>
/// <item><description><b>DiffusiveNoise:</b> Inject structured noise targeting memorized samples from the target
/// client, then fine-tune to recover global performance. Novel approach from 2025 research.</description></item>
/// </list>
/// </remarks>
public enum UnlearningMethod
{
    /// <summary>Retrain from scratch excluding the target client (provably correct, expensive).</summary>
    ExactRetraining,

    /// <summary>Gradient ascent on target client data to reverse learning (fast, approximate).</summary>
    GradientAscent,

    /// <summary>Influence function-based removal (Newton step, efficient for small removals).</summary>
    InfluenceFunction,

    /// <summary>Structured noise injection targeting memorized samples (2025 research).</summary>
    DiffusiveNoise
}
