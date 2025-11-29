namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Interface for draft models used in speculative decoding.
/// </summary>
/// <remarks>
/// <para>
/// Draft models are small, fast models that generate candidate tokens
/// for verification by the larger target model. They should be lightweight
/// enough to generate multiple tokens with minimal latency.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
public interface IDraftModel<T>
{
    /// <summary>
    /// Gets the maximum number of tokens this draft model can generate in one call.
    /// </summary>
    int MaxDraftTokens { get; }

    /// <summary>
    /// Generates draft tokens autoregressively.
    /// </summary>
    /// <param name="inputTokens">The input token sequence.</param>
    /// <param name="numDraftTokens">Number of draft tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <returns>Draft generation result with tokens and probabilities.</returns>
    DraftResult<T> GenerateDraft(
        ReadOnlySpan<int> inputTokens,
        int numDraftTokens,
        float temperature = 1.0f);

    /// <summary>
    /// Gets the vocabulary size of this model.
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Resets any internal state (e.g., KV cache).
    /// </summary>
    void Reset();
}

/// <summary>
/// Result of draft token generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DraftResult<T>
{
    /// <summary>
    /// Gets the generated draft tokens.
    /// </summary>
    public int[] Tokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets the probability distributions for each draft position.
    /// Shape: [num_draft_tokens, vocab_size]
    /// </summary>
    public T[,] Probabilities { get; set; } = new T[0, 0];

    /// <summary>
    /// Gets the sampled token probabilities (p(token) for each drafted token).
    /// </summary>
    public float[] TokenProbabilities { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets the number of draft tokens generated.
    /// </summary>
    public int NumTokens => Tokens.Length;
}

/// <summary>
/// A simple n-gram based draft model for testing and baselines.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a very simple model that predicts
/// the next word based on what typically follows the previous words.
///
/// For example, if "the" is often followed by "quick" in training data,
/// then when we see "the", this model might predict "quick".
///
/// It's not as good as a neural network, but it's very fast!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class NGramDraftModel<T> : IDraftModel<T>
{
    private readonly Dictionary<string, Dictionary<int, int>> _ngrams;
    private readonly int _n;
    private readonly int _vocabSize;
    private readonly Random _random;

    /// <inheritdoc/>
    public int MaxDraftTokens => 8;

    /// <inheritdoc/>
    public int VocabSize => _vocabSize;

    /// <summary>
    /// Creates an n-gram draft model.
    /// </summary>
    /// <param name="n">The n-gram order (e.g., 3 for trigrams).</param>
    /// <param name="vocabSize">Vocabulary size.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public NGramDraftModel(int n = 3, int vocabSize = 50000, int? seed = null)
    {
        _n = n;
        _vocabSize = vocabSize;
        _ngrams = new Dictionary<string, Dictionary<int, int>>();
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Trains the n-gram model on a corpus.
    /// </summary>
    /// <param name="corpus">Token sequences to train on.</param>
    public void Train(IEnumerable<int[]> corpus)
    {
        foreach (var sequence in corpus)
        {
            for (int i = _n - 1; i < sequence.Length; i++)
            {
                var context = GetContext(sequence, i);
                int nextToken = sequence[i];

                if (!_ngrams.TryGetValue(context, out var counts))
                {
                    counts = new Dictionary<int, int>();
                    _ngrams[context] = counts;
                }

                counts[nextToken] = counts.GetValueOrDefault(nextToken, 0) + 1;
            }
        }
    }

    /// <inheritdoc/>
    public DraftResult<T> GenerateDraft(
        ReadOnlySpan<int> inputTokens,
        int numDraftTokens,
        float temperature = 1.0f)
    {
        var tokens = new List<int>();
        var probs = new List<float[]>();
        var tokenProbs = new List<float>();

        var context = new List<int>(inputTokens.ToArray());

        for (int i = 0; i < numDraftTokens; i++)
        {
            var distribution = GetDistribution(context, temperature);
            int token = SampleFromDistribution(distribution);

            tokens.Add(token);
            probs.Add(distribution);
            tokenProbs.Add(distribution[token]);

            context.Add(token);
            if (context.Count > _n - 1)
            {
                context.RemoveAt(0);
            }
        }

        // Convert to result
        var result = new DraftResult<T>
        {
            Tokens = tokens.ToArray(),
            TokenProbabilities = tokenProbs.ToArray(),
            Probabilities = new T[numDraftTokens, _vocabSize]
        };

        for (int i = 0; i < probs.Count; i++)
        {
            for (int v = 0; v < _vocabSize; v++)
            {
                result.Probabilities[i, v] = FromFloat(probs[i][v]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // No state to reset for n-gram model
    }

    private string GetContext(int[] sequence, int position)
    {
        var contextTokens = new int[_n - 1];
        int start = Math.Max(0, position - _n + 1);
        int len = position - start;

        Array.Copy(sequence, start, contextTokens, _n - 1 - len, len);
        return string.Join(",", contextTokens);
    }

    private string GetContext(List<int> tokens)
    {
        var contextTokens = tokens.Skip(Math.Max(0, tokens.Count - _n + 1)).Take(_n - 1);
        return string.Join(",", contextTokens);
    }

    private float[] GetDistribution(List<int> context, float temperature)
    {
        var distribution = new float[_vocabSize];
        var contextKey = GetContext(context);

        if (_ngrams.TryGetValue(contextKey, out var counts))
        {
            int total = counts.Values.Sum();
            foreach (var (token, count) in counts)
            {
                distribution[token] = (float)count / total;
            }
        }
        else
        {
            // Uniform distribution if unseen context
            float uniform = 1.0f / _vocabSize;
            for (int i = 0; i < _vocabSize; i++)
            {
                distribution[i] = uniform;
            }
        }

        // Apply temperature
        if (Math.Abs(temperature - 1.0f) > 0.001f)
        {
            float sum = 0;
            for (int i = 0; i < distribution.Length; i++)
            {
                distribution[i] = MathF.Pow(distribution[i], 1.0f / temperature);
                sum += distribution[i];
            }
            for (int i = 0; i < distribution.Length; i++)
            {
                distribution[i] /= sum;
            }
        }

        return distribution;
    }

    private int SampleFromDistribution(float[] distribution)
    {
        float r = (float)_random.NextDouble();
        float cumulative = 0;

        for (int i = 0; i < distribution.Length; i++)
        {
            cumulative += distribution[i];
            if (r <= cumulative)
                return i;
        }

        return distribution.Length - 1;
    }

    private static T FromFloat(float value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)value;
        if (typeof(T) == typeof(double))
            return (T)(object)(double)value;

        return (T)Convert.ChangeType(value, typeof(T));
    }
}

/// <summary>
/// Wrapper for using a small neural network as a draft model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NeuralDraftModel<T> : IDraftModel<T>
{
    private readonly Func<ReadOnlySpan<int>, float[]> _forwardFunc;
    private readonly int _vocabSize;
    private readonly int _maxDraftTokens;
    private readonly Random _random;

    /// <inheritdoc/>
    public int MaxDraftTokens => _maxDraftTokens;

    /// <inheritdoc/>
    public int VocabSize => _vocabSize;

    /// <summary>
    /// Creates a neural draft model wrapper.
    /// </summary>
    /// <param name="forwardFunc">Function that takes input tokens and returns logits.</param>
    /// <param name="vocabSize">Vocabulary size.</param>
    /// <param name="maxDraftTokens">Maximum draft tokens to generate.</param>
    /// <param name="seed">Random seed.</param>
    public NeuralDraftModel(
        Func<ReadOnlySpan<int>, float[]> forwardFunc,
        int vocabSize,
        int maxDraftTokens = 5,
        int? seed = null)
    {
        _forwardFunc = forwardFunc ?? throw new ArgumentNullException(nameof(forwardFunc));
        _vocabSize = vocabSize;
        _maxDraftTokens = maxDraftTokens;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc/>
    public DraftResult<T> GenerateDraft(
        ReadOnlySpan<int> inputTokens,
        int numDraftTokens,
        float temperature = 1.0f)
    {
        numDraftTokens = Math.Min(numDraftTokens, _maxDraftTokens);

        var tokens = new List<int>();
        var probs = new List<float[]>();
        var tokenProbs = new List<float>();

        var currentTokens = new List<int>(inputTokens.ToArray());

        for (int i = 0; i < numDraftTokens; i++)
        {
            // Forward pass
            var logits = _forwardFunc(currentTokens.ToArray());

            // Convert to probabilities with temperature
            var distribution = Softmax(logits, temperature);

            // Sample
            int token = SampleFromDistribution(distribution);

            tokens.Add(token);
            probs.Add(distribution);
            tokenProbs.Add(distribution[token]);

            currentTokens.Add(token);
        }

        var result = new DraftResult<T>
        {
            Tokens = tokens.ToArray(),
            TokenProbabilities = tokenProbs.ToArray(),
            Probabilities = new T[numDraftTokens, _vocabSize]
        };

        for (int i = 0; i < probs.Count; i++)
        {
            for (int v = 0; v < _vocabSize; v++)
            {
                result.Probabilities[i, v] = FromFloat(probs[i][v]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // Neural models may need KV cache reset - handled externally
    }

    private float[] Softmax(float[] logits, float temperature)
    {
        var result = new float[logits.Length];

        // Apply temperature
        float maxLogit = logits.Max();
        float sum = 0;

        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp((logits[i] - maxLogit) / temperature);
            sum += result[i];
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    private int SampleFromDistribution(float[] distribution)
    {
        float r = (float)_random.NextDouble();
        float cumulative = 0;

        for (int i = 0; i < distribution.Length; i++)
        {
            cumulative += distribution[i];
            if (r <= cumulative)
                return i;
        }

        return distribution.Length - 1;
    }

    private static T FromFloat(float value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)value;
        if (typeof(T) == typeof(double))
            return (T)(object)(double)value;

        return (T)Convert.ChangeType(value, typeof(T));
    }
}
