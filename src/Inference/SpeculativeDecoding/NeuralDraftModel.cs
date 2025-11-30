namespace AiDotNet.Inference.SpeculativeDecoding;

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
