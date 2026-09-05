using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>The three-rung novelty gate: a free structural check, then an optional embedding check, then a judge.</summary>
/// <remarks>
/// <para>
/// The policy answers one question — is this candidate different enough from what we already have to be worth
/// evaluating? — and answers it as cheaply as the configuration allows. Rung one computes a structural distance
/// against every known genome; a candidate whose nearest neighbour is at least
/// <see cref="EmbeddingNoveltyOptions.StructuralNoveltyThreshold"/> away is novel there and then, having made no
/// request of any kind. Only a structural near-duplicate descends: rung two embeds it together with its nearest
/// neighbours in one batched request and compares cosine similarity against
/// <see cref="EmbeddingNoveltyOptions.EmbeddingSimilarityThreshold"/>, and rung three asks an
/// <see cref="IProgramNoveltyJudge"/> about the single closest incumbent.
/// </para>
/// <para>
/// Both optional rungs are genuinely optional. With no embedding client and no judge the policy is a pure
/// structural gate that never touches the network — which is the shipped default, and the difference between this
/// and the reference implementation, whose gate begins by embedding every single candidate and is therefore
/// unusable without a provider and a key. When a provider is supplied, wrapping it in a
/// <see cref="AiDotNet.Agentic.Embeddings.CachingEmbeddingClient"/> means a re-proposed candidate costs nothing on
/// the second encounter as well.
/// </para>
/// <para>
/// Failure handling is explicit rather than accidental. An unreachable provider or an unusable judgement resolves
/// through <see cref="EmbeddingNoveltyOptions.FailOpenOnEmbeddingFailure"/> and
/// <see cref="EmbeddingNoveltyOptions.FailOpenOnJudgeFailure"/>, and the returned
/// <see cref="ProgramNoveltyDecision"/> records which rung answered and what it spent. Nothing here executes program
/// text, and no program text reaches a reason string or a log.
/// </para>
/// <para>
/// This is a policy, not an engine change: it is a service a task, an evaluator, or a caller consults, and the
/// caller supplies the set of known genomes. It cannot see the engine's archive on its own.
/// </para>
/// <para><b>For Beginners:</b> Evolutionary search wastes most of its budget re-testing candidates that are almost
/// the same as ones it already tested. This checks a new candidate against the ones you have, using a free text
/// comparison first and only paying for a smarter comparison when the free one cannot tell. Construct it with no
/// arguments and it is free to run; add an embedding client and a judge when you want it to be cleverer.</para>
/// </remarks>
public sealed class ProgramNoveltyPolicy
{
    private readonly EmbeddingNoveltyOptions _options;
    private readonly IGenomeDistance<ProgramGenome> _structuralDistance;
    private readonly EmbeddingCosineGenomeDistance? _embeddingDistance;
    private readonly IProgramNoveltyJudge? _judge;

    private long _decisions;
    private long _structuralComparisons;
    private long _embeddingRequests;
    private long _judgeRequests;
    private long _freeDecisions;

    /// <summary>Initializes a novelty policy.</summary>
    /// <param name="options">The thresholds; <c>null</c> uses the validated defaults.</param>
    /// <param name="structuralDistance">
    /// The free structural metric; <c>null</c> uses a <see cref="ProgramTokenSetDistance"/>.
    /// </param>
    /// <param name="embeddingClient">
    /// The embedding provider for rung two, or <c>null</c> to disable it. Wrap it in a
    /// <see cref="AiDotNet.Agentic.Embeddings.CachingEmbeddingClient"/> to make repeated candidates free.
    /// </param>
    /// <param name="judge">The judge for rung three, or <c>null</c> to disable it.</param>
    /// <remarks>Every argument is optional: <c>null</c> selects the default metric or disables that rung.</remarks>
    public ProgramNoveltyPolicy(
        EmbeddingNoveltyOptions? options = null,
        IGenomeDistance<ProgramGenome>? structuralDistance = null,
        IEmbeddingClient? embeddingClient = null,
        IProgramNoveltyJudge? judge = null)
    {
        _options = options ?? new EmbeddingNoveltyOptions();
        _structuralDistance = structuralDistance ?? new ProgramTokenSetDistance();
        _embeddingDistance = embeddingClient is null
            ? null
            : new EmbeddingCosineGenomeDistance(embeddingClient, _structuralDistance);
        _judge = judge;
        VersionHash = "program-novelty-policy-v1-" + EvolutionHash.Combine(new[]
        {
            _options.ToVersionString(),
            _structuralDistance.Id,
            _structuralDistance.VersionHash,
            _embeddingDistance is null ? "no-embedding" : _embeddingDistance.VersionHash,
            _judge is null ? "no-judge" : _judge.Id
        });
    }

    /// <summary>Gets the thresholds this policy applies.</summary>
    public EmbeddingNoveltyOptions Options => _options;

    /// <summary>Gets the free structural metric rung one uses.</summary>
    public IGenomeDistance<ProgramGenome> StructuralDistance => _structuralDistance;

    /// <summary>Gets whether an embedding provider was supplied, enabling rung two.</summary>
    public bool HasEmbeddingStage => _embeddingDistance is not null;

    /// <summary>Gets whether a judge was supplied, enabling rung three.</summary>
    public bool HasJudgeStage => _judge is not null;

    /// <summary>Gets how many decisions this policy has produced.</summary>
    public long Decisions => Interlocked.Read(ref _decisions);

    /// <summary>Gets how many structural distances this policy has computed in total.</summary>
    public long StructuralComparisons => Interlocked.Read(ref _structuralComparisons);

    /// <summary>Gets how many embedding requests this policy has issued in total.</summary>
    public long EmbeddingRequests => Interlocked.Read(ref _embeddingRequests);

    /// <summary>Gets how many judging requests this policy has issued in total.</summary>
    public long JudgeRequests => Interlocked.Read(ref _judgeRequests);

    /// <summary>Gets how many decisions reached neither a provider nor a model.</summary>
    public long FreeDecisions => Interlocked.Read(ref _freeDecisions);

    /// <summary>Gets a stable version string covering the thresholds and every configured rung.</summary>
    /// <remarks>Computed once during construction, so reading it in a hot path costs nothing.</remarks>
    public string VersionHash { get; }

    /// <summary>Decides whether a candidate is novel against a set of known genomes.</summary>
    /// <param name="candidate">The proposed genome.</param>
    /// <param name="known">The genomes already accepted; an empty set makes any candidate novel.</param>
    /// <param name="cancellationToken">A token that cancels the decision.</param>
    /// <returns>The verdict together with the rung that produced it and what that rung cost.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="candidate"/> or <paramref name="known"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="known"/> contains a <c>null</c> entry.</exception>
    public async ValueTask<ProgramNoveltyDecision> EvaluateAsync(
        ProgramGenome candidate,
        IReadOnlyList<ProgramGenome> known,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(known);
        cancellationToken.ThrowIfCancellationRequested();
        Interlocked.Increment(ref _decisions);

        if (known.Count == 0)
        {
            Interlocked.Increment(ref _freeDecisions);
            return new ProgramNoveltyDecision(
                isNovel: true,
                decidedBy: ProgramNoveltyStage.None,
                reason: "nothing to compare against");
        }

        var ranked = new List<Neighbour>(known.Count);
        for (int index = 0; index < known.Count; index++)
        {
            ProgramGenome other = known[index];
            if (other is null)
            {
                throw new ArgumentException("The known set cannot contain a null genome.", nameof(known));
            }

            ranked.Add(new Neighbour(other, _structuralDistance.Distance(candidate, other), index));
        }

        Interlocked.Add(ref _structuralComparisons, known.Count);

        // Ordering by distance and then by arrival index keeps the choice of nearest neighbour deterministic when
        // several are equidistant, which a replayed run depends on.
        ranked.Sort(static (left, right) =>
        {
            int byDistance = left.Distance.CompareTo(right.Distance);
            return byDistance != 0 ? byDistance : left.Order.CompareTo(right.Order);
        });

        Neighbour nearest = ranked[0];
        if (nearest.Distance >= _options.StructuralNoveltyThreshold)
        {
            Interlocked.Increment(ref _freeDecisions);
            return new ProgramNoveltyDecision(
                isNovel: true,
                decidedBy: ProgramNoveltyStage.Structural,
                reason: "nearest structural distance " + Format(nearest.Distance) + " reached the threshold " +
                    Format(_options.StructuralNoveltyThreshold),
                nearestGenomeId: nearest.Genome.Id,
                nearestStructuralDistance: nearest.Distance,
                structuralComparisons: known.Count);
        }

        if (_embeddingDistance is null && _judge is null)
        {
            Interlocked.Increment(ref _freeDecisions);
            return new ProgramNoveltyDecision(
                isNovel: false,
                decidedBy: ProgramNoveltyStage.Structural,
                reason: "nearest structural distance " + Format(nearest.Distance) + " is below the threshold " +
                    Format(_options.StructuralNoveltyThreshold),
                nearestGenomeId: nearest.Genome.Id,
                nearestStructuralDistance: nearest.Distance,
                structuralComparisons: known.Count);
        }

        int embeddingRequests = 0;
        double? bestSimilarity = null;
        Neighbour mostSimilar = nearest;

        if (_embeddingDistance is not null)
        {
            var batch = new List<ProgramGenome> { candidate };
            int comparisons = Math.Min(_options.MaxEmbeddingComparisons, ranked.Count);
            for (int index = 0; index < comparisons; index++) batch.Add(ranked[index].Genome);

            embeddingRequests = 1;
            Interlocked.Increment(ref _embeddingRequests);
            bool primed = await _embeddingDistance
                .PrimeAsync(batch, cancellationToken)
                .ConfigureAwait(false);

            if (!primed)
            {
                return new ProgramNoveltyDecision(
                    isNovel: _options.FailOpenOnEmbeddingFailure,
                    decidedBy: ProgramNoveltyStage.Embedding,
                    reason: _options.FailOpenOnEmbeddingFailure
                        ? "the embedding provider was unavailable and the policy fails open"
                        : "the embedding provider was unavailable and the policy fails closed",
                    nearestGenomeId: nearest.Genome.Id,
                    nearestStructuralDistance: nearest.Distance,
                    structuralComparisons: known.Count,
                    embeddingRequests: embeddingRequests);
            }

            for (int index = 0; index < comparisons; index++)
            {
                double? similarity = _embeddingDistance.Similarity(candidate, ranked[index].Genome);
                if (similarity is not { } value) continue;
                if (bestSimilarity is null || value > bestSimilarity.Value)
                {
                    bestSimilarity = value;
                    mostSimilar = ranked[index];
                }
            }

            if (bestSimilarity is { } best && best < _options.EmbeddingSimilarityThreshold)
            {
                return new ProgramNoveltyDecision(
                    isNovel: true,
                    decidedBy: ProgramNoveltyStage.Embedding,
                    reason: "highest cosine similarity " + Format(best) + " is below the threshold " +
                        Format(_options.EmbeddingSimilarityThreshold),
                    nearestGenomeId: mostSimilar.Genome.Id,
                    nearestStructuralDistance: nearest.Distance,
                    embeddingSimilarity: best,
                    structuralComparisons: known.Count,
                    embeddingRequests: embeddingRequests);
            }
        }

        if (_judge is null)
        {
            return new ProgramNoveltyDecision(
                isNovel: false,
                decidedBy: ProgramNoveltyStage.Embedding,
                reason: bestSimilarity is { } similarity
                    ? "highest cosine similarity " + Format(similarity) + " reached the threshold " +
                        Format(_options.EmbeddingSimilarityThreshold)
                    : "no embedding comparison was possible and no judge is configured",
                nearestGenomeId: mostSimilar.Genome.Id,
                nearestStructuralDistance: nearest.Distance,
                embeddingSimilarity: bestSimilarity,
                structuralComparisons: known.Count,
                embeddingRequests: embeddingRequests);
        }

        Interlocked.Increment(ref _judgeRequests);
        ProgramNoveltyVerdict verdict = await _judge
            .JudgeAsync(candidate, mostSimilar.Genome, cancellationToken)
            .ConfigureAwait(false);

        bool isNovel = verdict switch
        {
            ProgramNoveltyVerdict.Novel => true,
            ProgramNoveltyVerdict.NotNovel => false,
            _ => _options.FailOpenOnJudgeFailure
        };

        string judgeReason = verdict switch
        {
            ProgramNoveltyVerdict.Novel => "the judge found the change meaningful",
            ProgramNoveltyVerdict.NotNovel => "the judge found the change trivial",
            _ => _options.FailOpenOnJudgeFailure
                ? "the judge gave no usable answer and the policy fails open"
                : "the judge gave no usable answer and the policy fails closed"
        };

        return new ProgramNoveltyDecision(
            isNovel: isNovel,
            decidedBy: ProgramNoveltyStage.LanguageModel,
            reason: judgeReason,
            nearestGenomeId: mostSimilar.Genome.Id,
            nearestStructuralDistance: nearest.Distance,
            embeddingSimilarity: bestSimilarity,
            structuralComparisons: known.Count,
            embeddingRequests: embeddingRequests,
            judgeRequests: 1);
    }

    private static string Format(double value) => value.ToString("0.####", CultureInfo.InvariantCulture);

    private readonly struct Neighbour
    {
        internal Neighbour(ProgramGenome genome, double distance, int order)
        {
            Genome = genome;
            Distance = distance;
            Order = order;
        }

        internal ProgramGenome Genome { get; }

        internal double Distance { get; }

        internal int Order { get; }
    }
}
