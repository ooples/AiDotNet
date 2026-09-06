using System.Globalization;
using AiDotNet.Evolution;

namespace AiDotNetTests.UnitTests.Evolution;

internal sealed class TestGenome : IImmutableEvolutionGenome
{
    public TestGenome(int value) => Value = value;

    public int Value { get; }
}

internal sealed class TestGenomeCodec : IEvolutionGenomeCodec<TestGenome>
{
    public string Id => "int";
    public string VersionHash => "int-v1";

    public string Serialize(TestGenome genome) => genome.Value.ToString(CultureInfo.InvariantCulture);

    public TestGenome Deserialize(string payload) => new(int.Parse(payload, CultureInfo.InvariantCulture));
}

internal sealed class IncrementVariation : IVariationOperator<TestGenome>
{
    public string Id => "increment";
    public string VersionHash => "increment-v1";

    public ValueTask<TestGenome> ProposeAsync(
        EvolutionVariationContext<TestGenome> context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<TestGenome>(
            new TestGenome(context.Parent.Candidate.CanonicalGenome.Genome.Value + 1));
    }
}

internal sealed class SyntheticEvolutionTask : IEvolutionTask<TestGenome>
{
    private readonly CancellationTokenSource? _cancelOnEvaluation;

    public SyntheticEvolutionTask(CancellationTokenSource? cancelOnEvaluation = null)
    {
        _cancelOnEvaluation = cancelOnEvaluation;
    }

    public string Id => "synthetic";
    public string VersionHash => "synthetic-task-v1";
    public string EvaluatorVersionHash => "synthetic-evaluator-v1";

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(
        TestGenome genome,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<EvolutionCanonicalGenome<TestGenome>>(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));
    }

    public ValueTask<EvolutionTaskResult> EvaluateAsync(
        EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        if (_cancelOnEvaluation is not null && candidate.EvaluationId >= 4)
        {
            _cancelOnEvaluation.Cancel();
            cancellationToken.ThrowIfCancellationRequested();
        }

        int value = candidate.CanonicalGenome.Genome.Value;
        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(value,
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["x"] = Math.Max(0, Math.Min(100, value))
            }));
    }
}

internal sealed class TemporaryDirectory : IDisposable
{
    public TemporaryDirectory()
    {
        Path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            "aidotnet-evolution-tests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public void Dispose() => Directory.Delete(Path, recursive: true);
}
