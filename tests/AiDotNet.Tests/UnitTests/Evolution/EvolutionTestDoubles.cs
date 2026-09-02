using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;

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

internal sealed class AbsoluteGenomeDistance : IGenomeDistance<TestGenome>
{
    public string Id => "absolute";
    public string VersionHash => "absolute-v1";
    public double Distance(TestGenome first, TestGenome second) => Math.Abs((double)first.Value - second.Value);
}

internal sealed class RecordingEvolutionObserver : IEvolutionObserver<TestGenome>
{
    private readonly List<EvolutionEventKind> _kinds = new();

    public IReadOnlyList<EvolutionEventKind> Kinds
    {
        get { lock (_kinds) return _kinds.ToArray(); }
    }

    public int CountOf(EvolutionEventKind kind) => Kinds.Count(item => item == kind);

    public ValueTask OnEventAsync(EvolutionEvent<TestGenome> evolutionEvent, CancellationToken cancellationToken = default)
    {
        lock (_kinds) _kinds.Add(evolutionEvent.Kind);
        return default;
    }
}

internal sealed class StagedEvolutionTask : ICascadeEvolutionTask<TestGenome>
{
    private readonly int[] _stageCalls;
    private readonly double _stageCost;
    private readonly EvolutionOptimizationDirection _direction;
    private int _directCalls;

    public StagedEvolutionTask(int stageCount = 2, double stageCost = 1,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
    {
        StageCount = stageCount;
        _stageCalls = new int[stageCount];
        _stageCost = stageCost;
        _direction = direction;
    }

    public string Id => "staged";
    public string VersionHash => "staged-task-v1";
    public string EvaluatorVersionHash => "staged-evaluator-v1";
    public int StageCount { get; }
    public int DirectCalls => Volatile.Read(ref _directCalls);
    public int StageCalls(int stage) => Volatile.Read(ref _stageCalls[stage]);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _directCalls);
        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
            candidate.CanonicalGenome.Genome.Value, Descriptors(candidate), _direction));
    }

    public ValueTask<EvolutionTaskResult> EvaluateStageAsync(int stage, EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _stageCalls[stage]);
        double value = candidate.CanonicalGenome.Genome.Value;
        bool last = stage == StageCount - 1;
        return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed, value, _direction,
            last ? Descriptors(candidate) : new Dictionary<string, double>(),
            costUnits: _stageCost * (stage + 1),
            metrics: new Dictionary<string, double>
            {
                ["stage" + stage.ToString(CultureInfo.InvariantCulture)] = value
            }));
    }

    private static Dictionary<string, double> Descriptors(EvolutionCandidate<TestGenome> candidate) => new()
    {
        ["x"] = Math.Max(0, Math.Min(100, candidate.CanonicalGenome.Genome.Value))
    };
}

internal sealed class PlateauEvolutionTask : IEvolutionTask<TestGenome>
{
    private int _calls;

    public string Id => "plateau";
    public string VersionHash => "plateau-task-v1";
    public string EvaluatorVersionHash => "plateau-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(1.0,
            new Dictionary<string, double>
            {
                ["x"] = Math.Max(0, Math.Min(100, candidate.CanonicalGenome.Genome.Value * 10))
            }));
    }
}

internal sealed class ArtifactEvolutionTask : IEvolutionTask<TestGenome>
{
    private readonly string _text;
    private readonly CancellationTokenSource? _cancelOnEvaluation;

    public ArtifactEvolutionTask(string text, CancellationTokenSource? cancelOnEvaluation = null)
    {
        _text = text;
        _cancelOnEvaluation = cancelOnEvaluation;
    }

    public string Id => "artifact";
    public string VersionHash => "artifact-task-v1";
    public string EvaluatorVersionHash => "artifact-evaluator-v1";

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        if (_cancelOnEvaluation is not null && candidate.EvaluationId >= 4)
        {
            _cancelOnEvaluation.Cancel();
            cancellationToken.ThrowIfCancellationRequested();
        }
        int value = candidate.CanonicalGenome.Genome.Value;
        return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed, value,
            descriptors: new Dictionary<string, double> { ["x"] = Math.Max(0, Math.Min(100, value)) },
            artifacts: new[]
            {
                new EvolutionArtifact("stderr", _text),
                new EvolutionArtifact("stdout", "value " + value.ToString(CultureInfo.InvariantCulture))
            }));
    }
}

internal sealed class FailingChildArtifactTask : IEvolutionTask<TestGenome>
{
    public string Id => "failing-child-artifact";
    public string VersionHash => "failing-child-artifact-task-v1";
    public string EvaluatorVersionHash => "failing-child-artifact-evaluator-v1";

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        int value = candidate.CanonicalGenome.Genome.Value;
        if (value == 1)
        {
            return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(value,
                new Dictionary<string, double> { ["x"] = value }));
        }
        return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(EvolutionEvaluationStatus.Failed,
            diagnostics: new[] { new EvolutionDiagnostic("synthetic_failure", "child failed") },
            artifacts: new[] { new EvolutionArtifact("stderr", "child " + value.ToString(CultureInfo.InvariantCulture) + " failed") }));
    }
}

internal sealed class SequentialVariation : IVariationOperator<TestGenome>
{
    private int _next = 1;

    public string Id => "sequential";
    public string VersionHash => "sequential-v1";

    public ValueTask<TestGenome> ProposeAsync(EvolutionVariationContext<TestGenome> context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<TestGenome>(new TestGenome(Interlocked.Increment(ref _next)));
    }
}

internal sealed class ArtifactRecordingVariation : IVariationOperator<TestGenome>
{
    private readonly List<string> _received = new();

    public string Id => "artifact-recording";
    public string VersionHash => "artifact-recording-v1";

    public IReadOnlyList<string> Received
    {
        get { lock (_received) return _received.ToArray(); }
    }

    public ValueTask<TestGenome> ProposeAsync(EvolutionVariationContext<TestGenome> context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        lock (_received)
        {
            foreach (EvolutionArtifact artifact in context.ParentArtifacts)
                _received.Add(artifact.Key + "=" + artifact.Text);
        }
        return new ValueTask<TestGenome>(new TestGenome(context.Parent.Candidate.CanonicalGenome.Genome.Value + 1));
    }
}

internal sealed class TimeoutOnceEvolutionTask : IEvolutionTask<TestGenome>
{
    private int _calls;

    public string Id => "timeout-once";
    public string VersionHash => "timeout-once-task-v1";
    public string EvaluatorVersionHash => "timeout-once-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        int call = Interlocked.Increment(ref _calls);
        return call == 1
            ? new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(EvolutionEvaluationStatus.TimedOut))
            : new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
                candidate.CanonicalGenome.Genome.Value,
                new Dictionary<string, double> { ["x"] = candidate.CanonicalGenome.Genome.Value }));
    }
}

internal sealed class NonCooperativeEvolutionTask : IEvolutionTask<TestGenome>
{
    private readonly int _blockMilliseconds;
    private int _calls;
    private int _finished;

    public NonCooperativeEvolutionTask(int blockMilliseconds) => _blockMilliseconds = blockMilliseconds;

    public string Id => "non-cooperative";
    public string VersionHash => "non-cooperative-task-v1";
    public string EvaluatorVersionHash => "non-cooperative-evaluator-v1";
    public int Calls => Volatile.Read(ref _calls);
    public bool HasFinished => Volatile.Read(ref _finished) != 0;

    public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
        CancellationToken cancellationToken = default) => new(new EvolutionCanonicalGenome<TestGenome>(
            new TestGenome(genome.Value), genome.Value.ToString(CultureInfo.InvariantCulture)));

    public async ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        int block = _blockMilliseconds;
        await Task.Run(() => Thread.Sleep(block)).ConfigureAwait(false);
        Interlocked.Exchange(ref _finished, 1);
        return EvolutionTaskResult.Completed(candidate.CanonicalGenome.Genome.Value,
            new Dictionary<string, double> { ["x"] = candidate.CanonicalGenome.Genome.Value });
    }
}

internal sealed class StopRequestingObserver : IEvolutionObserver<TestGenome>
{
    private readonly Action _requestStop;
    private readonly int _afterEvaluations;
    private int _evaluations;

    public StopRequestingObserver(Action requestStop, int afterEvaluations)
    {
        _requestStop = requestStop;
        _afterEvaluations = afterEvaluations;
    }

    public ValueTask OnEventAsync(EvolutionEvent<TestGenome> evolutionEvent, CancellationToken cancellationToken = default)
    {
        if (evolutionEvent.Kind == EvolutionEventKind.Evaluated &&
            Interlocked.Increment(ref _evaluations) == _afterEvaluations)
        {
            _requestStop();
        }
        return default;
    }
}

internal sealed class EvaluationRecordingObserver : IEvolutionObserver<TestGenome>
{
    private readonly List<EvolutionEvaluation> _evaluations = new();

    public IReadOnlyList<EvolutionEvaluation> Evaluations
    {
        get { lock (_evaluations) return _evaluations.ToArray(); }
    }

    public ValueTask OnEventAsync(EvolutionEvent<TestGenome> evolutionEvent, CancellationToken cancellationToken = default)
    {
        if (evolutionEvent.Kind == EvolutionEventKind.Evaluated && evolutionEvent.Evaluation is not null)
            lock (_evaluations) _evaluations.Add(evolutionEvent.Evaluation);
        return default;
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

internal sealed class TraceProbeObserver : IEvolutionObserver<TestGenome>
{
    private readonly EvolutionTraceObserver<TestGenome> _inner;
    private readonly string _tracePath;
    private int _checkpoints;

    public TraceProbeObserver(EvolutionTraceObserver<TestGenome> inner, string tracePath)
    {
        _inner = inner;
        _tracePath = tracePath;
    }

    public int RecordsVisibleAtFirstCheckpoint { get; private set; } = -1;
    public int EvaluationsBeforeFirstCheckpoint { get; private set; }
    public int TotalEvaluations { get; private set; }

    public async ValueTask OnEventAsync(EvolutionEvent<TestGenome> evolutionEvent,
        CancellationToken cancellationToken = default)
    {
        if (evolutionEvent.Kind == EvolutionEventKind.Evaluated)
        {
            TotalEvaluations++;
            if (_checkpoints == 0) EvaluationsBeforeFirstCheckpoint++;
        }

        await _inner.OnEventAsync(evolutionEvent, cancellationToken);

        if (evolutionEvent.Kind == EvolutionEventKind.Checkpointed && _checkpoints++ == 0)
            RecordsVisibleAtFirstCheckpoint = EvolutionTraceFile.Read(_tracePath).Records.Count;
    }
}
