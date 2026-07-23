namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the adversary model assumed by the MPC protocol.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The security model describes how much we trust the participants:</para>
/// <list type="bullet">
/// <item><description><b>SemiHonest:</b> Parties follow the protocol correctly but try to learn extra
/// information from the messages they see. This is the cheapest and most common model.</description></item>
/// <item><description><b>Malicious:</b> Parties may deviate arbitrarily from the protocol. Requires
/// zero-knowledge proofs or MACs for every operation, making it much slower.</description></item>
/// <item><description><b>CovertSecurity:</b> A middle ground — cheating is possible but will be detected
/// with a specified probability (deterrence factor). Cheaper than full malicious security.</description></item>
/// </list>
/// </remarks>
public enum MpcSecurityModel
{
    /// <summary>
    /// Semi-honest (honest-but-curious) — parties follow the protocol but try to infer extra info.
    /// </summary>
    SemiHonest,

    /// <summary>
    /// Malicious — parties may deviate arbitrarily. Requires integrity checks on every operation.
    /// </summary>
    Malicious,

    /// <summary>
    /// Covert — cheating is detected with a configurable probability. Cheaper than malicious.
    /// </summary>
    CovertSecurity
}
