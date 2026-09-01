using System.Globalization;
using AiDotNet.Evolution;

namespace AiDotNetTests.UnitTests.Evolution;

internal sealed class TestGenome
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

    public ValueTask<TestGenome> ProposeAsync(EvolutionVariationContext<TestGenome> context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<TestGenome>(new TestGenome(context.Parent.Candidate.CanonicalGenome.Genome.Value + 1));
    }
}

internal sealed class SyntheticEvolutionTask : IEvolutionTask<TestGenome>
{
    private readonly int _delayScale;
    private readonly int? _throwOnValue;
    private readonly CancellationTokenSource? _cancelOnEvaluation;
    private int _calls;
    private int _concurrency;
    private int _maxConcurrency;

    public SyntheticEvolutionTask(int delayScale = 0, int? throwOnValue = null,
        CancellationTokenSource? cancelOnEvaluation = null)
    {
        _delayScale = delayScale;
        _throwOnValue = throwOnValue;
        _cancelOnEvaluation = cancelOnEvaluation;
    }

    public string Id => "synthetic";
    public string VersionHash => "synthetic-task-v1";
    public string EvaluatorVersionHash => "synthetic-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);
    public int MaxConcurrency => Volatile.Read(ref _maxConcurrency);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<EvolutionCanonicalGenome<TestGenome>>(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));
    }

    public async ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        int current = Interlocked.Increment(ref _concurrency);
        UpdateMaximum(current);
        try
        {
            if (_cancelOnEvaluation is not null && candidate.EvaluationId >= 4)
            {
                _cancelOnEvaluation.Cancel();
                cancellationToken.ThrowIfCancellationRequested();
            }
            int value = candidate.CanonicalGenome.Genome.Value;
            if (_delayScale > 0)
                await Task.Delay((Math.Abs(value * 17) % 5 + 1) * _delayScale, cancellationToken);
            if (_throwOnValue == value) throw new InvalidOperationException("synthetic");
            return EvolutionTaskResult.Completed(value,
                new Dictionary<string, double> { ["x"] = Math.Max(0, Math.Min(100, value)) });
        }
        finally
        {
            Interlocked.Decrement(ref _concurrency);
        }
    }

    private void UpdateMaximum(int value)
    {
        while (true)
        {
            int current = Volatile.Read(ref _maxConcurrency);
            if (value <= current || Interlocked.CompareExchange(ref _maxConcurrency, value, current) == current) return;
        }
    }
}

internal sealed class FailOnceEvolutionTask : IEvolutionTask<TestGenome>
{
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _attempts = new();
    private int _calls;

    public string Id => "fail-once";
    public string VersionHash => "fail-once-task-v1";
    public string EvaluatorVersionHash => "fail-once-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        int attempt = _attempts.AddOrUpdate(candidate.CanonicalGenome.Id, 1, (_, current) => current + 1);
        return attempt == 1
            ? new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                EvolutionEvaluationStatus.Failed,
                costUnits: 1,
                diagnostics: new[] { new EvolutionDiagnostic("first_attempt", "synthetic") }))
            : new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(candidate.CanonicalGenome.Genome.Value,
                new Dictionary<string, double> { ["x"] = candidate.CanonicalGenome.Genome.Value }, costUnits: 2));
    }
}

internal sealed class CancelOnceEvolutionTask : IEvolutionTask<TestGenome>
{
    private int _calls;

    public string Id => "cancel-once";
    public string VersionHash => "cancel-once-task-v1";
    public string EvaluatorVersionHash => "cancel-once-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        int call = Interlocked.Increment(ref _calls);
        if (call == 1)
        {
            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                EvolutionEvaluationStatus.Canceled,
                costUnits: 1,
                diagnostics: new[] { new EvolutionDiagnostic("cooperative_cancel", "retry requested") }));
        }

        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
            candidate.CanonicalGenome.Genome.Value,
            new Dictionary<string, double> { ["x"] = candidate.CanonicalGenome.Genome.Value },
            costUnits: 2));
    }
}

internal sealed class ThrowingEvolutionObserver : IEvolutionObserver<TestGenome>
{
    public ValueTask OnEventAsync(EvolutionEvent<TestGenome> evolutionEvent,
        CancellationToken cancellationToken = default) => throw new InvalidOperationException("observer failure");
}

internal sealed class NullEvolutionSelectionPolicy : ISelectionPolicy<TestGenome>
{
    public string Id => "null-selection";
    public string VersionHash => "null-selection-v1";
    public EvolutionSelection<TestGenome>? Select(IEvolutionArchive<TestGenome> archive, StableRandom random,
        int inspirationCount) => null;
}

internal sealed class VerboseFailOnceEvolutionTask : IEvolutionTask<TestGenome>
{
    private int _calls;

    public string Id => "verbose-fail-once";
    public string VersionHash => "verbose-fail-once-task-v1";
    public string EvaluatorVersionHash => "verbose-fail-once-evaluator-v1";

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        int call = Interlocked.Increment(ref _calls);
        if (call == 1)
        {
            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                EvolutionEvaluationStatus.Failed,
                diagnostics: Enumerable.Range(0, 64).Select(index =>
                    new EvolutionDiagnostic($"retry_{index}", "synthetic"))));
        }
        return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed,
            candidate.CanonicalGenome.Genome.Value,
            descriptors: new Dictionary<string, double> { ["x"] = candidate.CanonicalGenome.Genome.Value },
            diagnostics: new[] { new EvolutionDiagnostic("final", "synthetic") }));
    }
}

internal sealed class SaturatingCostEvolutionTask : IEvolutionTask<TestGenome>
{
    private int _calls;

    public string Id => "saturating-cost";
    public string VersionHash => "saturating-cost-task-v1";
    public string EvaluatorVersionHash => "saturating-cost-evaluator-v1";

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        int call = Interlocked.Increment(ref _calls);
        return call == 1
            ? new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                EvolutionEvaluationStatus.Failed, costUnits: double.MaxValue,
                diagnostics: new[] { new EvolutionDiagnostic("retry", "synthetic") }))
            : new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
                candidate.CanonicalGenome.Genome.Value,
                new Dictionary<string, double> { ["x"] = candidate.CanonicalGenome.Genome.Value },
                costUnits: double.MaxValue));
    }
}

internal sealed class CooperativeBlockingEvolutionTask : IEvolutionTask<TestGenome>
{
    private int _calls;

    public string Id => "cooperative-blocking";
    public string VersionHash => "cooperative-blocking-task-v1";
    public string EvaluatorVersionHash => "cooperative-blocking-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public async ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        await Task.Delay(Timeout.Infinite, cancellationToken);
        throw new InvalidOperationException("Unreachable.");
    }
}
