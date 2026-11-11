using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Ready-to-use <see cref="IDraftModel{T}"/> that wraps any next-token-logits function as the draft model for
/// speculative decoding. Supply a smaller/faster model's forward pass (tokens -&gt; logits) and this handles the
/// autoregressive drafting and sampling for you.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class wraps a neural network (like a small transformer)
/// to use as the "fast guesser" in speculative decoding. The neural network should
/// be much smaller and faster than the main model you're trying to accelerate. You give it a function that
/// turns tokens into next-token scores; it does the rest.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class NeuralDraftModel<T> : IDraftModel<T>
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
        Guard.Positive(vocabSize);
        Guard.Positive(maxDraftTokens);
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
        Guard.NotNull(inputTokens);
        Guard.NonNegative(numDraftTokens);
        if (!NumOps.GreaterThan(temperature, NumOps.Zero))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "temperature must be positive.");
        }
        numDraftTokens = Math.Min(numDraftTokens, _maxDraftTokens);

        var tokens = new List<int>();
        var probs = new List<Vector<T>>();
        var tokenProbs = new List<T>();

        var currentTokens = new List<int>(inputTokens.ToArray());

        for (int i = 0; i < numDraftTokens; i++)
        {
            // Forward pass
            var currentVector = new Vector<int>(currentTokens.ToArray());
            var logits = _forwardFunc(currentVector);
            if (logits is null || logits.Length != _vocabSize)
            {
                throw new InvalidOperationException(
                    $"The draft model's forward function must return exactly {_vocabSize} logits (VocabSize); got {(logits is null ? "null" : logits.Length.ToString())}.");
            }

            // Convert to probabilities with temperature
            var distribution = Softmax(logits, temperature);

            // Sample
            int token = TemperatureSampling<T>.Sample(distribution, _random);

            tokens.Add(token);
            probs.Add(distribution);
            tokenProbs.Add(distribution.AsSpan()[token]);

            currentTokens.Add(token);
        }

        // Build result
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
        // Neural models may need KV cache reset - handled externally
    }

    /// <summary>
    /// Applies softmax with temperature to logits.
    /// </summary>
    private static Vector<T> Softmax(Vector<T> logits, T temperature)
    {
        var engine = AiDotNetEngine.Current;
        return (Vector<T>)engine.Softmax(engine.Divide(logits, temperature));
    }
}
