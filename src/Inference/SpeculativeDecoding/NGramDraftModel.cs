using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// A simple n-gram based draft model for speculative decoding. It is the zero-cost default "guesser": it
/// predicts the next token from the statistics of which tokens typically follow the recent ones, needing no
/// second neural network and no GPU.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a very simple model that predicts
/// the next word based on what typically follows the previous words.
///
/// For example, if "the" is often followed by "quick" in training data,
/// then when we see "the", this model might predict "quick".
///
/// It's not as good as a neural network, but it's very fast! Speculative decoding uses it as the default
/// "guesser" — you can pass one to the builder's <c>ConfigureSpeculativeDecoding</c> if you want to control its
/// order/seed, or supply your own <see cref="IDraftModel{T}"/> (for example a
/// <see cref="NeuralDraftModel{T}"/> wrapping a smaller model) for smarter guesses.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class NGramDraftModel<T> : IDraftModel<T>
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
            var tokens = sequence.AsSpan();
            for (int i = _ngramSize - 1; i < tokens.Length; i++)
            {
                var context = GetContext(sequence, i);
                int nextToken = tokens[i];

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

        var context = new List<int>(inputTokens.ToArray());

        for (int i = 0; i < numDraftTokens; i++)
        {
            var distribution = GetDistribution(context, temperature);
            int token = TemperatureSampling<T>.Sample(distribution, _random);

            tokens.Add(token);
            probs.Add(distribution);
            tokenProbs.Add(distribution.AsSpan()[token]);

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
            resultProbs.SetRow(i, probs[i]);
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

        sequence.AsSpan().Slice(start, len).CopyTo(contextTokens.AsSpan(_ngramSize - 1 - len));
        return string.Join(",", contextTokens);
    }

    private string GetContext(List<int> tokens)
    {
        var contextTokens = tokens.Skip(Math.Max(0, tokens.Count - _ngramSize + 1)).Take(_ngramSize - 1);
        return string.Join(",", contextTokens);
    }

    private Vector<T> GetDistribution(List<int> context, T temperature)
    {
        // Relative frequencies of the tokens seen after this context, uniform for an unseen context.
        var frequencies = new double[_vocabSize];
        if (_ngrams.TryGetValue(GetContext(context), out var counts))
        {
            double total = counts.Values.Sum();
            foreach (var kvp in counts)
            {
                frequencies[kvp.Key] = kvp.Value / total;
            }
        }
        else
        {
            frequencies.AsSpan().Fill(1.0 / _vocabSize);
        }

        var distribution = Vector<double>.Wrap(frequencies).Transform(NumOps.FromDouble);

        // Temperature 0 collapses to the greedy one-hot rather than dividing by zero.
        return TemperatureSampling<T>.Apply(distribution, temperature);
    }
}
