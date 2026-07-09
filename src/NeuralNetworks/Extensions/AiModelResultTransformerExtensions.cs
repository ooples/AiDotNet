using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Transformers.Extensions;

/// <summary>
/// Model-specific inference extensions on <see cref="AiModelResult{T, TInput, TOutput}"/> for the
/// transformer / language-model family. Part of #1836.
/// </summary>
/// <remarks>
/// <para>
/// See <c>AiModelResultRadianceFieldExtensions</c> for the full design rationale — extension
/// methods live in the same assembly as <see cref="AiModelResult{T, TInput, TOutput}"/> so they
/// access internal <c>Model</c> directly without exposing it.
/// </para>
/// <para>
/// Autoregressive generation is a common transformer inference pattern — the model outputs
/// next-token logits per position, we sample/argmax the last position, append, and repeat.
/// These extensions ship the two most common decoding strategies (greedy argmax + temperature
/// sampling); more sophisticated strategies (beam search, top-k / top-p nucleus) can be layered
/// on the same primitive.
/// </para>
/// </remarks>
public static class AiModelResultTransformerExtensions
{
    /// <summary>
    /// Runs greedy autoregressive generation — at each step, pick the token with the highest
    /// logit. Deterministic given the same starting sequence. Useful for tasks where you want
    /// the model's single best continuation (translation, extractive summarization).
    /// </summary>
    /// <param name="result">The trained model result.</param>
    /// <param name="startTokens">Prompt token IDs [B, T_prompt].</param>
    /// <param name="maxNewTokens">Number of tokens to append (generation stops early on an EOS token if <paramref name="eosTokenId"/> is set).</param>
    /// <param name="eosTokenId">Optional end-of-sequence token; when the model emits it, generation stops early.</param>
    /// <returns>Concatenated prompt + generated tokens [B, T_prompt + N_new].</returns>
    public static Tensor<T> GenerateGreedy<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> startTokens,
        int maxNewTokens,
        int? eosTokenId = null)
    {
        RequireTransformerCapability(result, nameof(GenerateGreedy));
        return GenerateInternal(result, startTokens, maxNewTokens, argmax: true, temperature: default!, eosTokenId, seed: null);
    }

    /// <summary>
    /// Runs temperature-sampled autoregressive generation. Higher temperature → more diverse
    /// outputs; lower temperature → closer to greedy. temperature=0 falls back to
    /// <see cref="GenerateGreedy"/>.
    /// </summary>
    /// <param name="temperature">Softmax temperature (t=1 sample from the raw distribution; t&lt;1 sharpens; t&gt;1 flattens).</param>
    /// <param name="seed">Optional RNG seed for reproducibility.</param>
    public static Tensor<T> GenerateSampled<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> startTokens,
        int maxNewTokens,
        T temperature,
        int? eosTokenId = null,
        int? seed = null)
    {
        RequireTransformerCapability(result, nameof(GenerateSampled));
        return GenerateInternal(result, startTokens, maxNewTokens, argmax: false, temperature, eosTokenId, seed);
    }

    private static Tensor<T> GenerateInternal<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        Tensor<T> startTokens,
        int maxNewTokens,
        bool argmax,
        T temperature,
        int? eosTokenId,
        int? seed)
    {
        if (startTokens is null) throw new ArgumentNullException(nameof(startTokens));
        if (maxNewTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "Must be positive.");
        if (startTokens.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"startTokens must be [batch, seqLen]; got shape [{string.Join(",", startTokens.Shape)}].",
                nameof(startTokens));
        }

        // Extensions run against the internal model directly (Model is internal but visible in-assembly).
        // The transformer returns next-token logits of shape [B, T, V] from Predict — we sample the
        // LAST position's distribution to get the next token, append, and repeat.
        var model = result.Model
            ?? throw new InvalidOperationException($"AiModelResult.{nameof(GenerateInternal)}: no model — result not built.");

        int batch = startTokens.Shape[0];
        var current = startTokens;
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var numOps = AiDotNet.Helpers.MathHelper.GetNumericOperations<T>();

        for (int step = 0; step < maxNewTokens; step++)
        {
            var logits = (Tensor<T>)(object)model.Predict((TInput)(object)current)!;
            if (logits.Shape.Length != 3)
            {
                throw new InvalidOperationException(
                    $"Transformer inference extensions expect logits of shape [B, T, V]; got " +
                    $"[{string.Join(",", logits.Shape)}]. If this model produces a different " +
                    $"logits shape, use Predict directly and implement custom sampling.");
            }

            int seqLen = logits.Shape[1];
            int vocab  = logits.Shape[2];

            var nextIds = new float[batch];
            for (int b = 0; b < batch; b++)
            {
                if (argmax)
                {
                    int bestIdx = 0;
                    T bestVal = logits[b, seqLen - 1, 0];
                    for (int v = 1; v < vocab; v++)
                    {
                        if (numOps.GreaterThan(logits[b, seqLen - 1, v], bestVal))
                        {
                            bestVal = logits[b, seqLen - 1, v];
                            bestIdx = v;
                        }
                    }
                    nextIds[b] = bestIdx;
                }
                else
                {
                    var probs = new double[vocab];
                    double invT = 1.0 / Math.Max(1e-6, Convert.ToDouble(temperature));
                    double maxLogit = double.NegativeInfinity;
                    for (int v = 0; v < vocab; v++)
                    {
                        double l = Convert.ToDouble(logits[b, seqLen - 1, v]) * invT;
                        probs[v] = l;
                        if (l > maxLogit) maxLogit = l;
                    }
                    double sum = 0;
                    for (int v = 0; v < vocab; v++)
                    {
                        probs[v] = Math.Exp(probs[v] - maxLogit);
                        sum += probs[v];
                    }
                    double r = rng.NextDouble() * sum;
                    double cum = 0;
                    int picked = vocab - 1;
                    for (int v = 0; v < vocab; v++)
                    {
                        cum += probs[v];
                        if (r <= cum) { picked = v; break; }
                    }
                    nextIds[b] = picked;
                }
            }

            var appended = new float[batch * (current.Shape[1] + 1)];
            for (int b = 0; b < batch; b++)
            {
                for (int t = 0; t < current.Shape[1]; t++)
                {
                    appended[b * (current.Shape[1] + 1) + t] = Convert.ToSingle(current[b, t]);
                }
                appended[b * (current.Shape[1] + 1) + current.Shape[1]] = nextIds[b];
            }
            var appendedTyped = new T[batch * (current.Shape[1] + 1)];
            for (int i = 0; i < appendedTyped.Length; i++)
            {
                appendedTyped[i] = numOps.FromDouble(appended[i]);
            }
            current = new Tensor<T>(new[] { batch, current.Shape[1] + 1 }, new Vector<T>(appendedTyped));

            if (eosTokenId.HasValue && Array.TrueForAll(nextIds, id => (int)id == eosTokenId.Value))
            {
                break;
            }
        }

        return current;
    }

    private static void RequireTransformerCapability<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        string extensionName)
    {
        if (result is null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        // Transformer generation targets any model whose Predict returns [B, T, V] logits from
        // a [B, T] token input — the paper-standard next-token-prediction shape. We can't cleanly
        // gate on a single interface without inventing one (see #1836 follow-up), so gate on
        // TInput/TOutput being Tensor<T> and leave the [B, T, V] check for GenerateInternal.
        if (result.Model is null)
        {
            throw new InvalidOperationException(
                $"AiModelResult.{extensionName}: no model — result not built yet.");
        }

        if (typeof(TInput) != typeof(Tensor<T>) || typeof(TOutput) != typeof(Tensor<T>))
        {
            throw new InvalidOperationException(
                $"AiModelResult.{extensionName} requires the result to be typed as " +
                $"AiModelResult<{typeof(T).Name}, Tensor<{typeof(T).Name}>, Tensor<{typeof(T).Name}>> " +
                $"(the paper-standard token-tensor in / logits-tensor out shape). This result is " +
                $"typed as <{typeof(T).Name}, {typeof(TInput).Name}, {typeof(TOutput).Name}>. " +
                $"For matrix-in / vector-out regression/classification models, use " +
                $"AiModelResult.Predict directly.");
        }
    }
}
