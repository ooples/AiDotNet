using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IDraftModel<T> _draftModel;
    private readonly Func<Vector<int>, Matrix<T>> _targetForward;
    private readonly SpeculativeDecodingConfig<T> _config;
    private readonly Random _random;

    // Statistics
    private long _totalTokensGenerated;
    private long _totalDraftTokens;
    private long _acceptedDraftTokens;
    private long _totalVerificationCalls;

    /// <summary>
    /// Gets the configuration.
    /// </summary>
    public SpeculativeDecodingConfig<T> Config => _config;

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
        Func<Vector<int>, Matrix<T>> targetForward,
        SpeculativeDecodingConfig<T>? config = null)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
        _targetForward = targetForward ?? throw new ArgumentNullException(nameof(targetForward));
        _config = config ?? new SpeculativeDecodingConfig<T>();
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
        Vector<int> inputTokens,
        int maxNewTokens,
        T temperature,
        int? eosToken = null,
        CancellationToken cancellationToken = default)
    {
        var tokens = new List<int>(inputTokens.Length + maxNewTokens);
        for (int i = 0; i < inputTokens.Length; i++)
        {
            tokens.Add(inputTokens[i]);
        }

        int generated = 0;
        var stepStats = new List<StepStatistics>();

        while (generated < maxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Determine how many draft tokens to generate
            int numDraft = Math.Min(_config.NumDraftTokens, maxNewTokens - generated);

            // Generate draft tokens
            var currentTokens = new Vector<int>(tokens.ToArray());
            var draft = _draftModel.GenerateDraft(currentTokens, numDraft, temperature);

            _totalDraftTokens += draft.NumTokens;

            // Verify with target model - create combined sequence
            var verifyTokens = new Vector<int>(tokens.Count + draft.NumTokens);
            for (int i = 0; i < tokens.Count; i++)
            {
                verifyTokens[i] = tokens[i];
            }
            for (int i = 0; i < draft.NumTokens; i++)
            {
                verifyTokens[tokens.Count + i] = draft.Tokens[i];
            }

            var targetProbs = await Task.Run(() => _targetForward(verifyTokens), cancellationToken);

            _totalVerificationCalls++;

            // Accept/reject loop
            int accepted = 0;
            for (int i = 0; i < draft.NumTokens; i++)
            {
                int draftToken = draft.Tokens[i];
                int targetPos = tokens.Count + i - 1; // Position in target output

                if (targetPos < 0 || targetPos >= targetProbs.Rows)
                    break;

                T pTarget = targetProbs[targetPos, draftToken];
                T pDraft = draft.TokenProbabilities[i];

                // Speculative acceptance: accept with probability min(1, p_target / p_draft)
                bool accept;
                if (NumOps.LessThanOrEquals(pDraft, NumOps.Zero))
                {
                    accept = NumOps.GreaterThan(pTarget, NumOps.Zero);
                }
                else
                {
                    T ratio = NumOps.Divide(pTarget, pDraft);
                    T acceptProb = NumOps.LessThan(ratio, NumOps.One) ? ratio : NumOps.One;
                    accept = _random.NextDouble() < NumOps.ToDouble(acceptProb);
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
                    var targetDist = targetProbs.GetRow(targetPos);
                    var draftDist = draft.Probabilities.GetRow(i);
                    var adjustedDist = ComputeAdjustedDistribution(targetDist, draftDist, temperature);

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
                if (lastPos < targetProbs.Rows)
                {
                    var targetDist = targetProbs.GetRow(lastPos);
                    var temperedDist = ApplyTemperature(targetDist, temperature);
                    int bonusToken = SampleFromDistribution(temperedDist);
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

        var resultTokens = new Vector<int>(tokens.ToArray());
        var newTokens = new Vector<int>(generated);
        for (int i = 0; i < generated; i++)
        {
            newTokens[i] = tokens[inputTokens.Length + i];
        }

        return new SpeculativeResult
        {
            Tokens = resultTokens,
            NewTokens = newTokens,
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
        Vector<int> inputTokens,
        int maxNewTokens,
        T temperature,
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
            SpeedupEstimate = TokensPerVerification
        };
    }

    /// <summary>
    /// Computes the adjusted distribution for rejection sampling.
    /// </summary>
    private Vector<T> ComputeAdjustedDistribution(Vector<T> targetDist, Vector<T> draftDist, T temperature)
    {
        // Compute max(0, p_target - p_draft), then normalize
        var adjusted = new Vector<T>(targetDist.Length);
        T sum = NumOps.Zero;

        for (int i = 0; i < targetDist.Length; i++)
        {
            T diff = NumOps.Subtract(targetDist[i], draftDist[i]);
            adjusted[i] = NumOps.GreaterThan(diff, NumOps.Zero) ? diff : NumOps.Zero;
            sum = NumOps.Add(sum, adjusted[i]);
        }

        // Normalize
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int i = 0; i < adjusted.Length; i++)
            {
                adjusted[i] = NumOps.Divide(adjusted[i], sum);
            }
        }
        else
        {
            // Fallback to target distribution
            for (int i = 0; i < adjusted.Length; i++)
            {
                adjusted[i] = targetDist[i];
            }
        }

        return adjusted;
    }

    /// <summary>
    /// Applies temperature scaling to a probability distribution.
    /// </summary>
    private Vector<T> ApplyTemperature(Vector<T> distribution, T temperature)
    {
        T one = NumOps.One;
        if (NumOps.Equals(temperature, one))
            return distribution;

        var result = new Vector<T>(distribution.Length);
        T sum = NumOps.Zero;
        T invTemp = NumOps.Divide(one, temperature);

        for (int i = 0; i < distribution.Length; i++)
        {
            result[i] = NumOps.Power(distribution[i], invTemp);
            sum = NumOps.Add(sum, result[i]);
        }

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
