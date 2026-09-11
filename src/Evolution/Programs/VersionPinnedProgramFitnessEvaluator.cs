using AiDotNet.Interfaces;

namespace AiDotNet.Evolution.Programs;

/// <summary>Preserves a caller-owned backend's declared identity across a program run.</summary>
internal sealed class VersionPinnedProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private readonly IProgramFitnessEvaluator _inner;
    private readonly string _declaredVersion;

    internal VersionPinnedProgramFitnessEvaluator(IProgramFitnessEvaluator inner)
    {
        if (inner is null) throw new ArgumentNullException(nameof(inner));
        _inner = inner;
        Id = inner.Id;
        _declaredVersion = inner.VersionHash;
        ValidateIdentity(Id, nameof(inner.Id));
        ValidateIdentity(_declaredVersion, nameof(inner.VersionHash));
        VersionHash = EvolutionHash.Combine(new[] { "version-pinned-program-fitness-v1", Id, _declaredVersion });
    }

    public string Id { get; }
    public string VersionHash { get; }

    public async ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        CheckIdentity();
        var result = await _inner.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        // Even a successful result is not reusable when the backend changes its declared semantics mid-call.
        CheckIdentity();
        return result;
    }

    private void CheckIdentity()
    {
        if (!string.Equals(Id, _inner.Id, StringComparison.Ordinal) ||
            !string.Equals(_declaredVersion, _inner.VersionHash, StringComparison.Ordinal))
            throw new InvalidOperationException("The custom program fitness evaluator changed identity during the run.");
    }

    internal static void ValidateIdentity(string value, string name)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > 256 || value.Any(char.IsControl))
            throw new ArgumentException("Declare a bounded printable evaluator identity.", name);
        new System.Text.UTF8Encoding(false, true).GetByteCount(value);
    }
}
