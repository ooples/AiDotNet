using AiDotNet.Interfaces;

namespace AiDotNet.Evolution.Programs;

/// <summary>Turns a public-test pass fraction into a hard gate without executing those tests twice.</summary>
internal sealed class AllPassProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private readonly IProgramFitnessEvaluator _inner;
    internal AllPassProgramFitnessEvaluator(IProgramFitnessEvaluator inner)
    {
        _inner = new VersionPinnedProgramFitnessEvaluator(inner);
        VersionHash = EvolutionHash.Combine(new[] { "all-pass-program-v1", _inner.Id, _inner.VersionHash });
    }
    public string Id => "all-pass-program";
    public string VersionHash { get; }
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        var result = await _inner.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Public correctness returned no result or resource receipt.");
        return result.Status != EvolutionEvaluationStatus.Completed || CorrectnessGatedProgramFitnessEvaluator.PassesCorrectness(result)
            ? result : CorrectnessGatedProgramFitnessEvaluator.Copy(result, EvolutionEvaluationStatus.Rejected, result.CostUnits);
    }
}
