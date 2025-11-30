using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Implements speculative decoding for faster LLM inference.
/// </summary>
/// <remarks>
/// <para>
/// Speculative decoding uses a small, fast "draft" model to generate candidate tokens,
/// which are then verified in parallel by the larger "target" model. Accepted tokens
/// get free speedup; rejected tokens are resampled from the target distribution.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're a slow but accurate writer (target model)
/// working with a fast but sometimes wrong assistant (draft model).
///
/// Normal writing: You write each word yourself, one at a time. Slow but correct.
///
/// Speculative decoding:
/// 1. Assistant quickly suggests 5 words
/// 2. You check all 5 at once (parallel verification)
/// 3. If words 1-3 are good, keep them! You wrote 3 words in the time of 1
/// 4. If word 4 is wrong, fix it and restart
///
/// Benefits:
/// - 2-3x faster generation when draft model is good
/// - EXACT same output distribution as using target model alone
/// - No accuracy loss - just faster!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
public class SpeculativeDecoder<T>
{
    private readonly IDraftModel<T> _draftModel;
    private readonly Func<ReadOnlySpan<int>, float[][]> _targetForward;
    private readonly SpeculativeDecodingConfig _config;
    private readonly Random _random;

    // Statistics
    private long _totalTokensGenerated;
    private long _totalDraftTokens;
    private long _acceptedDraftTokens;
    private long _totalVerificationCalls;

    /// <summary>
    /// Gets the configuration.
    /// </summary>
    public SpeculativeDecodingConfig Config => _config;

    /// <summary>
    /// Gets the draft acceptance rate.
    /// </summary>
    public double AcceptanceRate => _totalDraftTokens > 0
        ? (double)_acceptedDraftTokens / _totalDraftTokens
        : 0;

    /// <summary>
    /// Gets the average tokens generated per verification call.
    /// </summary>
    public double TokensPerVerification => _totalVerificationCalls > 0
        ? (double)_totalTokensGenerated / _totalVerificationCalls
        : 0;

    /// <summary>
    /// Creates a speculative decoder.
    /// </summary>
    /// <param name="draftModel">The small, fast draft model.</param>
    /// <param name="targetForward">Function that runs the target model on a sequence
    /// and returns probabilities for all positions. Shape: [seq_len, vocab_size]</param>
    /// <param name="config">Configuration options.</param>
    public SpeculativeDecoder(
        IDraftModel<T> draftModel,
        Func<ReadOnlySpan<int>, float[][]> targetForward,
        SpeculativeDecodingConfig? config = null)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
        _targetForward = targetForward ?? throw new ArgumentNullException(nameof(targetForward));
        _config = config ?? new SpeculativeDecodingConfig();
        _random = _config.Seed.HasValue ? new Random(_config.Seed.Value) : new Random();
    }

    /// <summary>
    /// Generates tokens using speculative decoding.
    /// </summary>
    /// <param name="inputTokens">Initial input tokens.</param>
    /// <param name="maxNewTokens">Maximum number of new tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <param name="eosToken">End-of-sequence token ID (optional).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Generation result with all tokens and statistics.</returns>
    public async Task<SpeculativeResult> GenerateAsync(
        int[] inputTokens,
        int maxNewTokens,
        float temperature = 1.0f,
        int? eosToken = null,
        CancellationToken cancellationToken = default)
    {
        var tokens = new List<int>(inputTokens);
        int generated = 0;
        var stepStats = new List<StepStatistics>();

        while (generated < maxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Determine how many draft tokens to generate
            int numDraft = Math.Min(_config.NumDraftTokens, maxNewTokens - generated);

            // Generate draft tokens
            var draft = _draftModel.GenerateDraft(
                tokens.ToArray(),
                numDraft,
                temperature);

            _totalDraftTokens += draft.NumTokens;

            // Verify with target model
            var verifyTokens = new int[tokens.Count + draft.NumTokens];
            tokens.CopyTo(verifyTokens, 0);
            Array.Copy(draft.Tokens, 0, verifyTokens, tokens.Count, draft.NumTokens);

            var targetProbs = await Task.Run(() => _targetForward(verifyTokens), cancellationToken);

            _totalVerificationCalls++;

            // Accept/reject loop
            int accepted = 0;
            for (int i = 0; i < draft.NumTokens; i++)
            {
                int draftToken = draft.Tokens[i];
                int targetPos = tokens.Count + i - 1; // Position in target output

                if (targetPos < 0 || targetPos >= targetProbs.Length)
                    break;

                float pTarget = targetProbs[targetPos][draftToken];
                float pDraft = draft.TokenProbabilities[i];

                // Speculative acceptance: accept with probability min(1, p_target / p_draft)
                bool accept;
                if (pDraft <= 0)
                {
                    accept = pTarget > 0;
                }
                else
                {
                    float acceptProb = Math.Min(1.0f, pTarget / pDraft);
                    accept = (float)_random.NextDouble() < acceptProb;
                }

                if (accept)
                {
                    tokens.Add(draftToken);
                    accepted++;
                    generated++;

                    if (eosToken.HasValue && draftToken == eosToken.Value)
                    {
                        stepStats.Add(new StepStatistics
                        {
                            DraftTokens = i + 1,
                            AcceptedTokens = accepted,
                            ResampledToken = false
                        });
                        goto done;
                    }
                }
                else
                {
                    // Rejection: sample from adjusted distribution
                    var adjustedDist = ComputeAdjustedDistribution(
                        targetProbs[targetPos],
                        GetDraftDistribution(draft, i),
                        temperature);

                    int resampledToken = SampleFromDistribution(adjustedDist);
                    tokens.Add(resampledToken);
                    generated++;

                    stepStats.Add(new StepStatistics
                    {
                        DraftTokens = i + 1,
                        AcceptedTokens = accepted,
                        ResampledToken = true
                    });

                    if (eosToken.HasValue && resampledToken == eosToken.Value)
                        goto done;

                    break; // Stop accepting after rejection
                }
            }

            _acceptedDraftTokens += accepted;

            // If all draft tokens accepted, sample one more from target
            if (accepted == draft.NumTokens && generated < maxNewTokens)
            {
                int lastPos = tokens.Count - 1;
                if (lastPos < targetProbs.Length)
                {
                    int bonusToken = SampleFromDistribution(
                        ApplyTemperature(targetProbs[lastPos], temperature));
                    tokens.Add(bonusToken);
                    generated++;

                    if (eosToken.HasValue && bonusToken == eosToken.Value)
                    {
                        stepStats.Add(new StepStatistics
                        {
                            DraftTokens = draft.NumTokens,
                            AcceptedTokens = accepted,
                            ResampledToken = false,
                            BonusToken = true
                        });
                        goto done;
                    }
                }

                stepStats.Add(new StepStatistics
                {
                    DraftTokens = draft.NumTokens,
                    AcceptedTokens = accepted,
                    ResampledToken = false,
                    BonusToken = true
                });
            }
        }

        done:
        _totalTokensGenerated += generated;

        return new SpeculativeResult
        {
            Tokens = tokens.ToArray(),
            NewTokens = tokens.Skip(inputTokens.Length).ToArray(),
            NumGenerated = generated,
            AcceptanceRate = AcceptanceRate,
            TokensPerVerification = TokensPerVerification,
            StepStatistics = stepStats
        };
    }

    /// <summary>
    /// Synchronous generation method.
    /// </summary>
    public SpeculativeResult Generate(
        int[] inputTokens,
        int maxNewTokens,
        float temperature = 1.0f,
        int? eosToken = null)
    {
        return GenerateAsync(inputTokens, maxNewTokens, temperature, eosToken).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Resets generation statistics.
    /// </summary>
    public void ResetStatistics()
    {
        _totalTokensGenerated = 0;
        _totalDraftTokens = 0;
        _acceptedDraftTokens = 0;
        _totalVerificationCalls = 0;
        _draftModel.Reset();
    }

    /// <summary>
    /// Gets current statistics.
    /// </summary>
    public SpeculativeDecodingStats GetStatistics()
    {
        return new SpeculativeDecodingStats
        {
            TotalTokensGenerated = _totalTokensGenerated,
            TotalDraftTokens = _totalDraftTokens,
            AcceptedDraftTokens = _acceptedDraftTokens,
            TotalVerificationCalls = _totalVerificationCalls,
            AcceptanceRate = AcceptanceRate,
            TokensPerVerification = TokensPerVerification,
            SpeedupEstimate = TokensPerVerification // Approximate speedup
        };
    }

    private float[] ComputeAdjustedDistribution(float[] targetDist, float[] draftDist, float temperature)
    {
        // Compute p_target - p_draft, clipped to [0, inf), then normalize
        var adjusted = new float[targetDist.Length];
        float sum = 0;

        for (int i = 0; i < targetDist.Length; i++)
        {
            adjusted[i] = Math.Max(0, targetDist[i] - draftDist[i]);
            sum += adjusted[i];
        }

        // Normalize
        if (sum > 0)
        {
            for (int i = 0; i < adjusted.Length; i++)
            {
                adjusted[i] /= sum;
            }
        }
        else
        {
            // Fallback to target distribution
            Array.Copy(targetDist, adjusted, targetDist.Length);
        }

        return adjusted;
    }

    private float[] GetDraftDistribution(DraftResult<T> draft, int position)
    {
        int vocabSize = draft.Probabilities.GetLength(1);
        var dist = new float[vocabSize];

        for (int v = 0; v < vocabSize; v++)
        {
            dist[v] = ToFloat(draft.Probabilities[position, v]);
        }

        return dist;
    }

    private float[] ApplyTemperature(float[] distribution, float temperature)
    {
        if (Math.Abs(temperature - 1.0f) < 0.001f)
            return distribution;

        var result = new float[distribution.Length];
        float sum = 0;

        for (int i = 0; i < distribution.Length; i++)
        {
            result[i] = MathF.Pow(distribution[i], 1.0f / temperature);
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

    private static float ToFloat(T value)
    {
        return MathHelper.GetNumericOperations<T>().ToFloat(value);
    }
}

/// <summary>
/// Configuration for speculative decoding.
/// </summary>
public class SpeculativeDecodingConfig
{
    /// <summary>
    /// Number of draft tokens to generate per verification.
    /// </summary>
    public int NumDraftTokens { get; set; } = 5;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Whether to use tree-based speculation (multiple draft continuations).
    /// </summary>
    public bool UseTreeSpeculation { get; set; } = false;

    /// <summary>
    /// Branching factor for tree speculation.
    /// </summary>
    public int TreeBranchFactor { get; set; } = 2;

    /// <summary>
    /// Maximum tree depth for tree speculation.
    /// </summary>
    public int MaxTreeDepth { get; set; } = 4;

    /// <summary>
    /// Minimum acceptance rate before reducing draft length.
    /// </summary>
    public float MinAcceptanceRate { get; set; } = 0.5f;

    /// <summary>
    /// Whether to dynamically adjust draft length based on acceptance rate.
    /// </summary>
    public bool AdaptiveDraftLength { get; set; } = false;
}

/// <summary>
/// Result of speculative decoding generation.
/// </summary>
public class SpeculativeResult
{
    /// <summary>
    /// All tokens (input + generated).
    /// </summary>
    public int[] Tokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Only the newly generated tokens.
    /// </summary>
    public int[] NewTokens { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Number of tokens generated.
    /// </summary>
    public int NumGenerated { get; set; }

    /// <summary>
    /// Overall draft acceptance rate.
    /// </summary>
    public double AcceptanceRate { get; set; }

    /// <summary>
    /// Average tokens generated per verification call.
    /// </summary>
    public double TokensPerVerification { get; set; }

    /// <summary>
    /// Statistics for each decoding step.
    /// </summary>
    public List<StepStatistics> StepStatistics { get; set; } = new();
}

/// <summary>
/// Statistics for a single decoding step.
/// </summary>
public class StepStatistics
{
    /// <summary>Number of draft tokens generated.</summary>
    public int DraftTokens { get; set; }

    /// <summary>Number of draft tokens accepted.</summary>
    public int AcceptedTokens { get; set; }

    /// <summary>Whether a token was resampled due to rejection.</summary>
    public bool ResampledToken { get; set; }

    /// <summary>Whether a bonus token was sampled after full acceptance.</summary>
    public bool BonusToken { get; set; }
}

/// <summary>
/// Overall statistics for speculative decoding.
/// </summary>
public class SpeculativeDecodingStats
{
    /// <summary>Total tokens generated.</summary>
    public long TotalTokensGenerated { get; set; }

    /// <summary>Total draft tokens proposed.</summary>
    public long TotalDraftTokens { get; set; }

    /// <summary>Draft tokens that were accepted.</summary>
    public long AcceptedDraftTokens { get; set; }

    /// <summary>Total verification calls to target model.</summary>
    public long TotalVerificationCalls { get; set; }

    /// <summary>Draft acceptance rate.</summary>
    public double AcceptanceRate { get; set; }

    /// <summary>Average tokens per verification.</summary>
    public double TokensPerVerification { get; set; }

    /// <summary>Estimated speedup factor.</summary>
    public double SpeedupEstimate { get; set; }
}
