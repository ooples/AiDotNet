using AiDotNet.Agentic.Embeddings;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;

namespace AiDotNetTests.UnitTests.Evolution.Programs.Novelty;

/// <summary>An evaluator that records how many candidates actually reached it.</summary>
internal sealed class RecordingProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    private readonly double _quality;
    private int _calls;

    public RecordingProgramFitnessEvaluator(double quality = 0.5) => _quality = quality;

    public string Id => "recording-evaluator";

    public string VersionHash => "recording-evaluator-v1";

    public int Calls => _calls;

    public List<string> SeenIds { get; } = new();

    public ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        _calls++;
        SeenIds.Add(candidate.Id);
        return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
            EvolutionEvaluationStatus.Completed, _quality));
    }
}

/// <summary>A judge that answers from a fixed script and counts how often it was consulted.</summary>
internal sealed class ScriptedNoveltyJudge : IProgramNoveltyJudge
{
    private readonly ProgramNoveltyVerdict[] _verdicts;
    private int _calls;

    public ScriptedNoveltyJudge(params ProgramNoveltyVerdict[] verdicts) => _verdicts = verdicts;

    public string Id => "scripted-novelty-judge";

    public int Calls => _calls;

    public ValueTask<ProgramNoveltyVerdict> JudgeAsync(
        ProgramGenome candidate,
        ProgramGenome incumbent,
        CancellationToken cancellationToken = default)
    {
        int index = _calls;
        _calls++;
        ProgramNoveltyVerdict verdict = _verdicts.Length == 0
            ? ProgramNoveltyVerdict.Unavailable
            : _verdicts[Math.Min(index, _verdicts.Length - 1)];
        return new ValueTask<ProgramNoveltyVerdict>(verdict);
    }
}

/// <summary>An embedding client that always reports the provider as unreachable.</summary>
internal sealed class UnavailableEmbeddingClient : IEmbeddingClient
{
    private int _calls;

    public string ModelId => "unavailable-embedding";

    public int Calls => _calls;

    public ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        _calls++;
        return new ValueTask<EmbeddingBatch>(EmbeddingBatch.Failure("the provider is unreachable in this test"));
    }
}

/// <summary>An embedding client that fails a configured number of times and then succeeds.</summary>
internal sealed class FlakyEmbeddingClient : IEmbeddingClient
{
    private readonly DeterministicEmbeddingClient _inner = new(dimensions: 16);
    private int _remainingFailures;
    private int _calls;

    public FlakyEmbeddingClient(int failures) => _remainingFailures = failures;

    public string ModelId => "flaky-embedding";

    public int Calls => _calls;

    public ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        _calls++;
        if (_remainingFailures > 0)
        {
            _remainingFailures--;
            return new ValueTask<EmbeddingBatch>(EmbeddingBatch.Failure("transient"));
        }

        return _inner.EmbedAsync(texts, cancellationToken);
    }
}

/// <summary>An embedding client that returns a fixed vector for every input, making everything identical.</summary>
internal sealed class ConstantEmbeddingClient : IEmbeddingClient
{
    private int _calls;

    public string ModelId => "constant-embedding";

    public int Calls => _calls;

    public ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        _calls++;
        var vectors = new List<EmbeddingVector>(texts.Count);
        for (int index = 0; index < texts.Count; index++)
        {
            vectors.Add(new EmbeddingVector(new[] { 1.0, 0.0, 0.0, 0.0 }));
        }

        return new ValueTask<EmbeddingBatch>(EmbeddingBatch.Success(vectors));
    }
}

/// <summary>An embedding client whose vectors depend only on the input's position, so nothing is ever similar.</summary>
internal sealed class OrthogonalEmbeddingClient : IEmbeddingClient
{
    private int _nextAxis;
    private int _calls;
    private readonly Dictionary<string, EmbeddingVector> _assigned = new(StringComparer.Ordinal);

    public string ModelId => "orthogonal-embedding";

    public int Calls => _calls;

    public ValueTask<EmbeddingBatch> EmbedAsync(
        IReadOnlyList<string> texts,
        CancellationToken cancellationToken = default)
    {
        _calls++;
        var vectors = new List<EmbeddingVector>(texts.Count);
        foreach (string text in texts)
        {
            if (!_assigned.TryGetValue(text, out EmbeddingVector? vector))
            {
                var components = new double[32];
                components[_nextAxis % components.Length] = 1.0;
                _nextAxis++;
                vector = new EmbeddingVector(components);
                _assigned[text] = vector;
            }

            vectors.Add(vector);
        }

        return new ValueTask<EmbeddingBatch>(EmbeddingBatch.Success(vectors));
    }
}
