using AiDotNet.LinearAlgebra;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Selects the next token id from a logits vector according to <see cref="LocalSamplingOptions"/>:
/// greedy (argmax) when temperature is zero, otherwise temperature-scaled softmax sampling restricted by
/// optional top-k and top-p filters.
/// </summary>
/// <typeparam name="T">The logits element type.</typeparam>
/// <remarks>
/// <para>
/// The sampler owns a single <see cref="Random"/> (seeded from <see cref="LocalSamplingOptions.Seed"/> when
/// provided), so a fixed seed yields a reproducible token stream. Logits are read through
/// <see cref="Convert.ToDouble(object)"/>, so the same code path serves <see cref="float"/> and
/// <see cref="double"/> models.
/// </para>
/// <para><b>For Beginners:</b> This is the "dice roll" step. With temperature 0 it isn't a roll at all — it
/// just takes the single most likely token. Otherwise it turns the scores into probabilities, optionally
/// throws away the unlikely options (top-k / top-p), and then picks one at random in proportion to how
/// likely each is.
/// </para>
/// </remarks>
public sealed class TokenSampler<T>
{
    private readonly Random _random;

    /// <summary>
    /// Initializes a new sampler.
    /// </summary>
    /// <param name="seed">Optional RNG seed for reproducibility. <c>null</c> uses a non-deterministic seed.</param>
    public TokenSampler(int? seed = null)
    {
        _random = seed is { } value ? new Random(value) : new Random();
    }

    /// <summary>
    /// Chooses the next token id from the supplied logits, optionally restricted to an allowed set
    /// (constrained decoding).
    /// </summary>
    /// <param name="logits">The next-token logits (length = vocabulary size). Must be non-empty.</param>
    /// <param name="options">The sampling settings. <c>null</c> uses defaults (temperature 1.0, no filters).</param>
    /// <param name="allowedTokenIds">
    /// When non-null, sampling is restricted to these token ids (others are excluded). Must be non-empty when
    /// provided. <c>null</c> means all tokens are eligible.
    /// </param>
    /// <returns>The chosen token id (an index into <paramref name="logits"/>).</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="logits"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="logits"/> or <paramref name="allowedTokenIds"/> is empty.</exception>
    public int Sample(Vector<T> logits, LocalSamplingOptions? options = null, IReadOnlyCollection<int>? allowedTokenIds = null)
    {
        Guard.NotNull(logits);
        var count = logits.Length;
        if (count == 0)
        {
            throw new ArgumentException("Logits must be non-empty.", nameof(logits));
        }

        if (allowedTokenIds is not null && allowedTokenIds.Count == 0)
        {
            throw new ArgumentException("The allowed token set must be non-empty when provided.", nameof(allowedTokenIds));
        }

        var settings = options ?? new LocalSamplingOptions();

        var scores = new double[count];
        for (var i = 0; i < count; i++)
        {
            scores[i] = Convert.ToDouble(logits[i]);
        }

        var allowed = BuildAllowedMask(count, allowedTokenIds);

        var temperature = settings.Temperature ?? 1.0;
        if (temperature <= 0)
        {
            return ArgMax(scores, allowed);
        }

        for (var i = 0; i < count; i++)
        {
            scores[i] /= temperature;
        }

        // Numerically stable softmax.
        var max = double.NegativeInfinity;
        for (var i = 0; i < count; i++)
        {
            if (scores[i] > max)
            {
                max = scores[i];
            }
        }

        var probabilities = new double[count];
        var sum = 0.0;
        for (var i = 0; i < count; i++)
        {
            var p = Math.Exp(scores[i] - max);
            probabilities[i] = p;
            sum += p;
        }

        for (var i = 0; i < count; i++)
        {
            probabilities[i] /= sum;
        }

        // Candidate token ids (allowed only), ordered by descending probability so top-k / top-p slice
        // from the front.
        var candidates = new List<int>(count);
        for (var i = 0; i < count; i++)
        {
            if (allowed is null || allowed[i])
            {
                candidates.Add(i);
            }
        }

        candidates.Sort((a, b) => probabilities[b].CompareTo(probabilities[a]));

        if (settings.TopK is { } topK && topK > 0 && topK < candidates.Count)
        {
            candidates = candidates.GetRange(0, topK);
        }

        if (settings.TopP is { } topP && topP > 0 && topP < 1)
        {
            var kept = new List<int>(candidates.Count);
            var cumulative = 0.0;
            foreach (var id in candidates)
            {
                kept.Add(id);
                cumulative += probabilities[id];
                if (cumulative >= topP)
                {
                    break;
                }
            }

            candidates = kept;
        }

        var total = 0.0;
        foreach (var id in candidates)
        {
            total += probabilities[id];
        }

        var threshold = _random.NextDouble() * total;
        var accumulated = 0.0;
        foreach (var id in candidates)
        {
            accumulated += probabilities[id];
            if (threshold <= accumulated)
            {
                return id;
            }
        }

        return candidates[candidates.Count - 1];
    }

    private static bool[]? BuildAllowedMask(int count, IReadOnlyCollection<int>? allowedTokenIds)
    {
        if (allowedTokenIds is null)
        {
            return null;
        }

        var mask = new bool[count];
        foreach (var id in allowedTokenIds)
        {
            if (id >= 0 && id < count)
            {
                mask[id] = true;
            }
        }

        return mask;
    }

    private static int ArgMax(double[] scores, bool[]? allowed)
    {
        var bestIndex = -1;
        var best = double.NegativeInfinity;
        for (var i = 0; i < scores.Length; i++)
        {
            if (allowed is not null && !allowed[i])
            {
                continue;
            }

            if (bestIndex < 0 || scores[i] > best)
            {
                best = scores[i];
                bestIndex = i;
            }
        }

        // If the allowed set referenced only out-of-range ids, fall back to the global argmax.
        return bestIndex >= 0 ? bestIndex : 0;
    }
}
