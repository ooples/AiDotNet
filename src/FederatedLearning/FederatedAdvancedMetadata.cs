using System.Collections.Generic;

namespace AiDotNet.FederatedLearning;

/// <summary>
/// Results from the advanced federated-learning capabilities configured via
/// <c>ConfigureFederatedLearning(...)</c>: client contribution, fairness, drift, private-set-intersection
/// overlap, TEE attestation, zero-knowledge proofs, secure multi-party computation, and unlearning readiness.
/// </summary>
/// <remarks>
/// <para>
/// Each block is populated only when the corresponding capability was configured; unconfigured capabilities
/// leave their <c>*Enabled</c>/<c>*Evaluated</c> flag <c>false</c>. This is surfaced on
/// <see cref="AiDotNet.Models.FederatedLearningMetadata"/>.
/// </para>
/// </remarks>
public sealed class FederatedAdvancedMetadata
{
    // --- Client contribution (Shapley-style value assessment) ---

    /// <summary>Whether client contributions were evaluated.</summary>
    public bool ContributionEvaluated { get; set; }

    /// <summary>The contribution method's name.</summary>
    public string ContributionMethod { get; set; } = "None";

    /// <summary>Per-client contribution scores (higher = more valuable to the global model).</summary>
    public Dictionary<int, double> ClientContributions { get; set; } = new();

    /// <summary>Clients identified as free-riders (contribution below the method's threshold).</summary>
    public HashSet<int> FreeRiders { get; set; } = new();

    // --- Fairness ---

    /// <summary>Whether a fairness constraint was evaluated.</summary>
    public bool FairnessEvaluated { get; set; }

    /// <summary>The fairness constraint's name.</summary>
    public string FairnessConstraint { get; set; } = "None";

    /// <summary>The fairness violation score (0 = perfectly fair, higher = more unfair).</summary>
    public double FairnessViolation { get; set; }

    /// <summary>Whether the final model satisfies the fairness constraint.</summary>
    public bool FairnessSatisfied { get; set; }

    // --- Concept drift ---

    /// <summary>Whether drift detection ran.</summary>
    public bool DriftDetectionEnabled { get; set; }

    /// <summary>The number of rounds drift was checked.</summary>
    public int DriftRoundsChecked { get; set; }

    /// <summary>The number of rounds in which drift was detected.</summary>
    public int DriftDetectedCount { get; set; }

    /// <summary>Whether drift was detected in any round.</summary>
    public bool AnyDriftDetected { get; set; }

    // --- Private set intersection (cross-client sample overlap) ---

    /// <summary>Whether PSI ran.</summary>
    public bool PsiEnabled { get; set; }

    /// <summary>The total number of overlapping samples found across clients (data-leakage / dedup signal).</summary>
    public int PsiTotalOverlap { get; set; }

    // --- Trusted execution environment attestation ---

    /// <summary>Whether TEE attestation ran.</summary>
    public bool TeeEnabled { get; set; }

    /// <summary>The number of attestation quotes produced over the aggregated model.</summary>
    public int TeeAttestationCount { get; set; }

    /// <summary>The enclave measurement (code-identity) hash.</summary>
    public string TeeMeasurementHash { get; set; } = string.Empty;

    // --- Zero-knowledge verifiable aggregation ---

    /// <summary>Whether zero-knowledge proofs were produced.</summary>
    public bool ZkEnabled { get; set; }

    /// <summary>The zero-knowledge system's name.</summary>
    public string ZkSystem { get; set; } = "None";

    /// <summary>The number of range proofs produced (one per round, bounding the update norm).</summary>
    public int ZkProofCount { get; set; }

    /// <summary>Whether every produced proof verified.</summary>
    public bool ZkAllVerified { get; set; }

    // --- Secure multi-party computation ---

    /// <summary>Whether secure MPC aggregation was exercised.</summary>
    public bool McpEnabled { get; set; }

    /// <summary>The MPC protocol's name.</summary>
    public string McpProtocol { get; set; } = "None";

    /// <summary>Whether the secure secret-shared sum reconstructed to the plaintext sum (correctness check).</summary>
    public bool McpSecureSumVerified { get; set; }

    // --- Unlearning readiness ---

    /// <summary>Whether an unlearner is configured (GDPR right-to-be-forgotten available on the result).</summary>
    public bool UnlearningAvailable { get; set; }

    /// <summary>The unlearning method's name.</summary>
    public string UnlearningMethod { get; set; } = "None";
}
