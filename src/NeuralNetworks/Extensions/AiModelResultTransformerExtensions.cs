using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

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
        return GenerateInternal(result, startTokens, maxNewTokens, argmax: true, temperature: default!, eosTokenId, seed: null, onToken: null, cancellationToken: default);
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
        return GenerateInternal(result, startTokens, maxNewTokens, argmax: false, temperature, eosTokenId, seed, onToken: null, cancellationToken: default);
    }

    private static Tensor<T> GenerateInternal<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        Tensor<T> startTokens,
        int maxNewTokens,
        bool argmax,
        T temperature,
        int? eosTokenId,
        int? seed,
        IProgress<int>? onToken,
        CancellationToken cancellationToken)
    {
        if (startTokens is null) throw new ArgumentNullException(nameof(startTokens));
        if (maxNewTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "Must be positive.");
        if (startTokens.Shape.Length != 2)
        {
            throw new ArgumentException(
                $"startTokens must be [batch, seqLen]; got shape [{string.Join(",", startTokens.Shape)}].",
                nameof(startTokens));
        }

        // Use the ILanguageModel<T> interface (Transformer implements it) so custom
        // transformer subclasses can plug in — no more shape-heuristic gating.
        var lm = result.Model as AiDotNet.Interfaces.ILanguageModel<T>
            ?? throw new InvalidOperationException(
                $"AiModelResult.{nameof(GenerateInternal)}: model does not implement " +
                $"AiDotNet.Interfaces.ILanguageModel<{typeof(T).Name}>. Attach one via " +
                $"a Transformer<T> or a custom subclass that implements it.");

        int batch = startTokens.Shape[0];
        var current = startTokens;
        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();

        for (int step = 0; step < maxNewTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var logits = lm.ForwardLogits(current);
            if (logits.Shape.Length != 3)
            {
                throw new InvalidOperationException(
                    $"Transformer inference extensions expect logits of shape [B, T, V]; got " +
                    $"[{string.Join(",", logits.Shape)}]. If this model produces a different " +
                    $"logits shape, use Predict directly and implement custom sampling.");
            }

            int seqLen = logits.Shape[1];
            int vocab  = logits.Shape[2];

            var nextIds = new int[batch];
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

            // Append per-batch nextIds directly into a T[] — token IDs stay T all the way,
            // no float32 round-trip (avoids the 2^24 precision-loss risk on double T if
            // vocab > 16M) and eliminates two extra array allocations per step.
            int newLen = current.Shape[1] + 1;
            var appendedTyped = new T[batch * newLen];
            for (int b = 0; b < batch; b++)
            {
                for (int t = 0; t < current.Shape[1]; t++)
                {
                    appendedTyped[b * newLen + t] = current[b, t];
                }
                appendedTyped[b * newLen + current.Shape[1]] = numOps.FromDouble(nextIds[b]);
            }
            current = new Tensor<T>(new[] { batch, newLen }, new Vector<T>(appendedTyped));

            // Streaming: report each generated token id as it emits (first batch's token for
            // multi-batch generation; per-batch callback would require IProgress<int[]>).
            onToken?.Report(nextIds[0]);

            if (eosTokenId.HasValue && Array.TrueForAll(nextIds, id => id == eosTokenId.Value))
            {
                break;
            }
        }

        return current;
    }

    /// <summary>
    /// Async generation with per-token streaming callback. Reference impls (llama.cpp, HF
    /// transformers) require callers to implement their own streaming loop; here it's a
    /// one-liner. The callback receives each generated token id as it's emitted so callers
    /// can print progressively / short-circuit on stop sequences / stream to a WebSocket.
    /// </summary>
    public static Task<Tensor<T>> GenerateGreedyAsync<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> startTokens,
        int maxNewTokens,
        int? eosTokenId = null,
        IProgress<int>? onToken = null,
        CancellationToken cancellationToken = default)
    {
        RequireTransformerCapability(result, nameof(GenerateGreedyAsync));
        return Task.Run(
            () => GenerateInternal(result, startTokens, maxNewTokens, argmax: true, default!, eosTokenId,
                seed: null, onToken: onToken, cancellationToken: cancellationToken),
            cancellationToken);
    }

    /// <summary>
    /// Batched greedy generation — runs N independent prompt sequences through the model in
    /// parallel batches (per-prompt independence). Reference impls loop one prompt at a
    /// time; batched generation is the paper-standard way to amortize KV-cache setup.
    /// </summary>
    public static Tensor<T>[] GenerateGreedyBatch<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T>[] startTokensPerPrompt,
        int maxNewTokens,
        int? eosTokenId = null)
    {
        RequireTransformerCapability(result, nameof(GenerateGreedyBatch));
        if (startTokensPerPrompt is null) throw new ArgumentNullException(nameof(startTokensPerPrompt));
        var outputs = new Tensor<T>[startTokensPerPrompt.Length];
        for (int i = 0; i < startTokensPerPrompt.Length; i++)
        {
            outputs[i] = GenerateInternal(result, startTokensPerPrompt[i], maxNewTokens, argmax: true, default!, eosTokenId, seed: null, onToken: null, cancellationToken: default);
        }
        return outputs;
    }

    private static void RequireTransformerCapability<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        string extensionName)
        => AiDotNet.Extensions.Capability.AiModelResultExtensionsCapabilityGate.Require<
            T, TInput, TOutput, AiDotNet.Interfaces.ILanguageModel<T>>(
            result,
            extensionName,
            $"AiDotNet.Interfaces.ILanguageModel<{typeof(T).Name}>",
            hint: "(Transformer<T> or any custom decoder-only/encoder-decoder model implementing it).");
}
