using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

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
internal class NGramDraftModel<T> : IDraftModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Dictionary<string, Dictionary<int, int>> _ngrams;
    private readonly int _ngramSize;
    private readonly int _vocabSize;
    private readonly Random _random;

    /// <inheritdoc/>
    public int MaxDraftTokens => 8;

    /// <inheritdoc/>
    public int VocabSize => _vocabSize;

    /// <summary>
    /// Creates an n-gram draft model.
    /// </summary>
    /// <param name="ngramSize">The n-gram order (e.g., 3 for trigrams).</param>
    /// <param name="vocabSize">Vocabulary size.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public NGramDraftModel(int ngramSize = 3, int vocabSize = 50000, int? seed = null)
    {
        _ngramSize = ngramSize;
        _vocabSize = vocabSize;
        _ngrams = new Dictionary<string, Dictionary<int, int>>();
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the n-gram model on a corpus.
    /// </summary>
    /// <param name="corpus">Token sequences to train on.</param>
    public void Train(IEnumerable<Vector<int>> corpus)
    {
        foreach (var sequence in corpus)
        {
            for (int i = _ngramSize - 1; i < sequence.Length; i++)
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
        Vector<int> inputTokens,
        int numDraftTokens,
        T temperature)
    {
        var tokens = new List<int>();
        var probs = new List<Vector<T>>();
        var tokenProbs = new List<T>();

        var context = new List<int>();
        for (int i = 0; i < inputTokens.Length; i++)
        {
            context.Add(inputTokens[i]);
        }

        for (int i = 0; i < numDraftTokens; i++)
        {
            var distribution = GetDistribution(context, temperature);
            int token = SampleFromDistribution(distribution);

            tokens.Add(token);
            probs.Add(distribution);
            tokenProbs.Add(distribution[token]);

            context.Add(token);
            if (context.Count > _ngramSize - 1)
            {
                context.RemoveAt(0);
            }
        }

        // Convert to result
        var resultTokens = new Vector<int>(tokens.ToArray());
        var resultTokenProbs = new Vector<T>(tokenProbs.ToArray());
        var resultProbs = new Matrix<T>(numDraftTokens, _vocabSize);

        for (int i = 0; i < probs.Count; i++)
        {
            for (int v = 0; v < _vocabSize && v < probs[i].Length; v++)
            {
                resultProbs[i, v] = probs[i][v];
            }
        }

        return new DraftResult<T>
        {
            Tokens = resultTokens,
            TokenProbabilities = resultTokenProbs,
            Probabilities = resultProbs
        };
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // No state to reset for n-gram model
    }

    private string GetContext(Vector<int> sequence, int position)
    {
        var contextTokens = new int[_ngramSize - 1];
        int start = Math.Max(0, position - _ngramSize + 1);
        int len = position - start;

        for (int i = 0; i < len; i++)
        {
            contextTokens[_ngramSize - 1 - len + i] = sequence[start + i];
        }
        return string.Join(",", contextTokens);
    }

    private string GetContext(List<int> tokens)
    {
        var contextTokens = tokens.Skip(Math.Max(0, tokens.Count - _ngramSize + 1)).Take(_ngramSize - 1);
        return string.Join(",", contextTokens);
    }

    private Vector<T> GetDistribution(List<int> context, T temperature)
    {
        var distribution = new Vector<T>(_vocabSize);
        var contextKey = GetContext(context);

        if (_ngrams.TryGetValue(contextKey, out var counts))
        {
            int total = counts.Values.Sum();
            foreach (var kvp in counts)
            {
                int token = kvp.Key;
                int count = kvp.Value;
                distribution[token] = NumOps.FromDouble((double)count / total);
            }
        }
        else
        {
            // Uniform distribution if unseen context
            T uniform = NumOps.FromDouble(1.0 / _vocabSize);
            for (int i = 0; i < _vocabSize; i++)
            {
                distribution[i] = uniform;
            }
        }

        // Apply temperature
        T one = NumOps.One;
        if (!NumOps.Equals(temperature, one))
        {
            T sum = NumOps.Zero;
            T invTemp = NumOps.Divide(one, temperature);
            for (int i = 0; i < distribution.Length; i++)
            {
                distribution[i] = NumOps.Power(distribution[i], invTemp);
                sum = NumOps.Add(sum, distribution[i]);
            }
            if (NumOps.GreaterThan(sum, NumOps.Zero))
            {
                for (int i = 0; i < distribution.Length; i++)
                {
                    distribution[i] = NumOps.Divide(distribution[i], sum);
                }
            }
        }

        return distribution;
    }

    private int SampleFromDistribution(Vector<T> distribution)
    {
        T r = NumOps.FromDouble(_random.NextDouble());
        T cumulative = NumOps.Zero;

        for (int i = 0; i < distribution.Length; i++)
        {
            cumulative = NumOps.Add(cumulative, distribution[i]);
            if (NumOps.LessThanOrEquals(r, cumulative))
                return i;
        }

        return distribution.Length - 1;
    }
}
