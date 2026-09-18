using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;

namespace AiDotNet.Interfaces;

/// <summary>Retains and verifies raw program-fitness evidence independently of reusable summary metadata.</summary>
/// <remarks>
/// Implementations must be thread-safe and bind the exact candidate, measurement values and original sample
/// identities to the raw observations. A hash of the summary alone is not raw evidence. No source code is
/// executed by this contract. Bound and meter physical I/O, elapsed time and external services separately;
/// the reuse decorator accounts only for logical evidence-store invocations. Do not include secrets in identities.
/// </remarks>
public interface IProgramMeasurementEvidenceStore
{
    /// <summary>Gets the immutable evidence schema and verification-policy fingerprint.</summary>
    string VersionHash { get; }

    /// <summary>Retains genuine raw observations and returns their SHA256, or null when unavailable.</summary>
    ValueTask<string?> RetainAsync(ProgramGenome candidate, EvolutionTaskResult measurement,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default);

    /// <summary>Verifies retrievable raw evidence against its digest, exact candidate and original measurement.</summary>
    ValueTask<bool> VerifyAsync(ProgramGenome candidate, EvolutionTaskResult measurement, string evidenceSha256,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default);
}
