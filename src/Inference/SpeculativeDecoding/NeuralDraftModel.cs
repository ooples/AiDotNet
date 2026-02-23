using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Wrapper for using a small neural network as a draft model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class wraps a neural network (like a small transformer)
/// to use as the "fast guesser" in speculative decoding. The neural network should
/// be much smaller and faster than the main model you're trying to accelerate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal class NeuralDraftModel<T> : IDraftModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<int>, Vector<T>> _forwardFunc;
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
        Func<Vector<int>, Vector<T>> forwardFunc,
        int vocabSize,
        int maxDraftTokens = 5,
        int? seed = null)
    {
        Guard.NotNull(forwardFunc);
        _forwardFunc = forwardFunc;
        _vocabSize = vocabSize;
        _maxDraftTokens = maxDraftTokens;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public DraftResult<T> GenerateDraft(
        Vector<int> inputTokens,
        int numDraftTokens,
        T temperature)
    {
        numDraftTokens = Math.Min(numDraftTokens, _maxDraftTokens);

        var tokens = new List<int>();
        var probs = new List<Vector<T>>();
        var tokenProbs = new List<T>();

        var currentTokens = new List<int>();
        for (int i = 0; i < inputTokens.Length; i++)
        {
            currentTokens.Add(inputTokens[i]);
        }

        for (int i = 0; i < numDraftTokens; i++)
        {
            // Forward pass
            var currentVector = new Vector<int>(currentTokens.ToArray());
            var logits = _forwardFunc(currentVector);

            // Convert to probabilities with temperature
            var distribution = Softmax(logits, temperature);

            // Sample
            int token = SampleFromDistribution(distribution);

            tokens.Add(token);
            probs.Add(distribution);
            tokenProbs.Add(distribution[token]);

            currentTokens.Add(token);
        }

        // Build result
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
        // Neural models may need KV cache reset - handled externally
    }

    /// <summary>
    /// Applies softmax with temperature to logits.
    /// </summary>
    private Vector<T> Softmax(Vector<T> logits, T temperature)
    {
        var result = new Vector<T>(logits.Length);

        // Find max logit for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.GreaterThan(logits[i], maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        // Apply temperature and compute exp
        T sum = NumOps.Zero;
        T one = NumOps.One;

        for (int i = 0; i < logits.Length; i++)
        {
            T scaled = NumOps.Divide(NumOps.Subtract(logits[i], maxLogit), temperature);
            result[i] = NumOps.Exp(scaled);
            // Note: No Pow/Power needed here - we use exp((logit - max) / temp) for softmax
            sum = NumOps.Add(sum, result[i]);
        }

        // Normalize
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Divide(result[i], sum);
            }
        }

        return result;
    }

    /// <summary>
    /// Samples a token index from a probability distribution.
    /// </summary>
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
