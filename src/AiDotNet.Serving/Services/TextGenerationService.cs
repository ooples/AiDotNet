using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Serving.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Default <see cref="ITextGenerationService"/> implementation backed by the continuous-batching
/// engine (<see cref="ContinuousBatcher{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// For a single REST request the engine is driven synchronously: the request is enqueued and
/// <see cref="ContinuousBatcher{T}.Step"/> is invoked until the sequence completes (EOS, a stop
/// token, or <see cref="SpeculativeDecodingRequest.MaxNewTokens"/>). This reuses the full
/// vLLM-style scheduling, KV-cache, and speculative-decoding pipeline without spinning a
/// background loop per request.
/// </para>
/// </remarks>
public sealed class TextGenerationService : ITextGenerationService
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<TextGenerationService> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextGenerationService"/> class.
    /// </summary>
    /// <param name="modelRepository">The repository used to resolve loaded models.</param>
    /// <param name="logger">Logger for diagnostics.</param>
    public TextGenerationService(IModelRepository modelRepository, ILogger<TextGenerationService> logger)
    {
        Guard.NotNull(modelRepository);
        _modelRepository = modelRepository;
        Guard.NotNull(logger);
        _logger = logger;
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    public Task<SpeculativeDecodingResponse> GenerateAsync(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);

        return numericType switch
        {
            NumericType.Double => GenerateTypedAsync<double>(modelName, request, cancellationToken),
            NumericType.Float => GenerateTypedAsync<float>(modelName, request, cancellationToken),
            NumericType.Decimal => GenerateTypedAsync<decimal>(modelName, request, cancellationToken),
            _ => Task.FromResult(new SpeculativeDecodingResponse
            {
                Error = $"Unsupported numeric type '{numericType}' for text generation.",
                RequestId = request.RequestId
            })
        };
    }

    /// <summary>
    /// Synchronous wrapper over <see cref="GenerateAsync"/> for non-request-thread callers. Request-serving
    /// controllers must call <see cref="GenerateAsync"/> so the request thread is not blocked for the whole
    /// completion (which would starve the thread pool under serving concurrency).
    /// </summary>
    public SpeculativeDecodingResponse Generate(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default)
        => GenerateAsync(modelName, numericType, request, cancellationToken).GetAwaiter().GetResult();

    /// <inheritdoc/>
    public bool SupportsGeneration(string modelName, NumericType numericType) => numericType switch
    {
        NumericType.Double => SupportsGenerationTyped<double>(modelName),
        NumericType.Float => SupportsGenerationTyped<float>(modelName),
        NumericType.Decimal => SupportsGenerationTyped<decimal>(modelName),
        _ => false
    };

    private bool SupportsGenerationTyped<T>(string modelName)
        => _modelRepository.GetModel<T>(modelName) is IServableGenerativeModel<T> gm && gm.SupportsGeneration;

    /// <inheritdoc/>
    public IEnumerable<int> GenerateStream(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);
        return numericType switch
        {
            NumericType.Double => GenerateStreamTyped<double>(modelName, request, cancellationToken),
            NumericType.Float => GenerateStreamTyped<float>(modelName, request, cancellationToken),
            NumericType.Decimal => GenerateStreamTyped<decimal>(modelName, request, cancellationToken),
            _ => Enumerable.Empty<int>()
        };
    }

    /// <summary>
    /// Incremental decode with sampling, yielding one token id per step. Uses the KV-cached session
    /// path when the model supports it (each step forwards only the new token), otherwise falls back
    /// to stateless full-context decode. Speculative decoding is intentionally not used here so token
    /// emission cadence matches wall-clock decode steps (accurate TTFT / inter-token latency).
    /// </summary>
    private IEnumerable<int> GenerateStreamTyped<T>(string modelName, SpeculativeDecodingRequest request, CancellationToken ct)
    {
        var model = _modelRepository.GetModel<T>(modelName);
        if (model is not IServableGenerativeModel<T> gm || !gm.SupportsGeneration)
        {
            yield break;
        }
        if (request.InputTokens is null || request.InputTokens.Length == 0)
        {
            yield break;
        }

        int eosTokenId = request.EosTokenId ?? ContinuousBatcherConfig.DefaultEosTokenId;

        // Preferred path: stream tokens off the ONE shared continuous-batching engine as they are
        // produced (paged KV, prefix sharing, per-request sampling). EOS terminates the stream and is
        // not yielded. The engine honors MaxNewTokens; a defensive cap is applied here as well.
        if (model is ServableModelWrapper<T> wrapper && wrapper.SupportsIncrementalGeneration)
        {
            int emitted = 0;
            foreach (var token in wrapper.StreamGeneration(BuildGenerationRequest<T>(request, disableSpeculation: true), eosTokenId, ct))
            {
                if (ct.IsCancellationRequested || emitted >= request.MaxNewTokens) yield break;
                emitted++;
                yield return token;
            }
            yield break;
        }

        // Stateless fallback (models without the shared incremental engine): drive a per-request
        // continuous-batching engine and stream its tokens as they are produced. Routing through the
        // engine means structured-output constraints, logit_bias, frequency/presence penalties, and stop
        // handling are applied by the SAME sampler as the incremental path — the previous hand-written loop
        // silently ignored all of those, so identical requests behaved differently by model capability.
        var fallbackConfig = new ContinuousBatcherConfig
        {
            AutoStart = false,
            EosTokenId = eosTokenId,
            EnableSpeculativeDecoding = false, // streaming: speculation stays off (accurate inter-token latency)
        };
        var streamRequest = BuildGenerationRequest<T>(request, disableSpeculation: true);
        var produced = new Queue<int>();
        streamRequest.OnTokenGenerated = tok => produced.Enqueue(tok);

        using var streamBatcher = new ContinuousBatcher<T>(fallbackConfig, tokens => gm.Forward(tokens));
        var streamTask = streamBatcher.GenerateAsync(streamRequest);

        int streamed = 0;
        // Each Step emits at least one token for the running sequence; bound the loop defensively.
        int stepBudget = request.MaxNewTokens + request.InputTokens.Length + 8;
        while (!streamTask.IsCompleted && stepBudget-- > 0)
        {
            if (ct.IsCancellationRequested) yield break;
            streamBatcher.Step();
            while (produced.Count > 0)
            {
                int tok = produced.Dequeue();
                // EOS terminates the stream and is not yielded; honor MaxNewTokens defensively.
                if (tok == eosTokenId || streamed >= request.MaxNewTokens) yield break;
                streamed++;
                yield return tok;
            }
        }

        // Drain any tokens produced on the final completing step.
        while (produced.Count > 0)
        {
            int tok = produced.Dequeue();
            if (tok == eosTokenId || streamed >= request.MaxNewTokens) yield break;
            streamed++;
            yield return tok;
        }
    }

    /// <summary>
    /// Maps a serving-layer <see cref="SpeculativeDecodingRequest"/> to the engine's
    /// <see cref="GenerationRequest{T}"/>. Sampling parameters travel with the request (a shared engine
    /// serves many requests): temperature (0 =&gt; greedy), top-p/top-k/min-p, per-request EOS, and the
    /// speculation draft depth (<see cref="SpeculativeDecodingRequest.NumDraftTokens"/>; 0 disables it).
    /// The RNG seed is derived from the request id when present (reproducible), else null (a seedless
    /// request at temperature &gt; 0 samples non-deterministically, matching industry serving defaults).
    /// </summary>
    private static GenerationRequest<T> BuildGenerationRequest<T>(SpeculativeDecodingRequest request, bool disableSpeculation = false)
    {
        return new GenerationRequest<T>
        {
            PromptTokenIds = new List<int>(request.InputTokens),
            MaxNewTokens = request.MaxNewTokens,
            Temperature = (float)request.Temperature,
            TopP = (float)request.TopP,
            TopK = request.TopK,
            MinP = (float)request.MinP,
            EosTokenId = request.EosTokenId,
            // Streaming forces speculation off: accepted drafts arrive in bursts that distort inter-token
            // latency (TPOT) measurements, and the docs state streaming intentionally avoids speculation.
            SpeculationDepth = disableSpeculation ? 0 : request.NumDraftTokens,
            // An explicit request seed (OpenAI `seed`) wins for reproducibility; otherwise derive a stable
            // per-request seed from the request id so greedy stays deterministic and sampling is reproducible.
            Seed = request.Seed ?? (request.RequestId is { } id ? id.GetHashCode() : (int?)null),
            Constraint = request.Constraint,
            LogitBias = request.LogitBias,
            FrequencyPenalty = (float)request.FrequencyPenalty,
            PresencePenalty = (float)request.PresencePenalty,
            IncludeLogProbs = request.Logprobs,
            TopLogProbs = request.TopLogprobs,
            AdapterId = request.AdapterId
        };
    }

    private async Task<SpeculativeDecodingResponse> GenerateTypedAsync<T>(string modelName, SpeculativeDecodingRequest request, CancellationToken cancellationToken)
    {
        var model = _modelRepository.GetModel<T>(modelName);
        if (model is null)
        {
            // Distinguish "not loaded / does not exist" from "loaded but not generative" — the
            // `is not IServableGenerativeModel<T>` check below also succeeds for null, but its
            // message would misreport a missing model as an unsupported one.
            return new SpeculativeDecodingResponse
            {
                Error = $"Model '{modelName}' is not loaded or does not exist.",
                RequestId = request.RequestId
            };
        }

        if (model is not IServableGenerativeModel<T> generativeModel || !generativeModel.SupportsGeneration)
        {
            return new SpeculativeDecodingResponse
            {
                Error = $"Model '{modelName}' does not support text generation. " +
                        "Text generation requires a tensor-based (token-to-logits) model such as a transformer language model.",
                RequestId = request.RequestId
            };
        }

        // Preferred path: the ONE shared continuous-batching engine (paged KV cache, RadixAttention
        // prefix sharing, greedy-exact prompt-lookup speculation, per-request sampling). All requests to
        // this model co-batch on the same engine; each gets an isolated paged-cache sequence and only
        // pays for the new token per decode step. On any failure, fall through to the proven full-context
        // path below so a model whose incremental path misbehaves still serves correct (if slower) results.
        //
        // A request that opts into TREE speculation (UseTreeSpeculation) is routed to the per-request
        // full-context batcher below instead: tree branch-factor / max-depth are batcher-CONFIG settings,
        // and the shared engine has a single fixed config, so it cannot honor per-request tree knobs. The
        // fallback builds a batcher per request and applies them (see config below), so those requests get
        // the behavior they asked for rather than having the knobs silently dropped.
        if (model is ServableModelWrapper<T> wrapper && wrapper.SupportsIncrementalGeneration && !request.UseTreeSpeculation)
        {
            try
            {
                int eos = request.EosTokenId ?? ContinuousBatcherConfig.DefaultEosTokenId;
                var genResult = await wrapper.RunGenerationAsync(BuildGenerationRequest<T>(request), cancellationToken)
                    .ConfigureAwait(false);

                // Remove a trailing EOS, cap to MaxNewTokens, and trim log-probs to the SAME final count so
                // both generation paths honor the one-logprob-per-generated-token contract identically.
                var (genTokens, genLogProbs) = NormalizeGeneration(genResult.GeneratedTokens, genResult.LogProbs, eos, request.MaxNewTokens);

                var all = new List<int>(request.InputTokens);
                all.AddRange(genTokens);
                return new SpeculativeDecodingResponse
                {
                    AllTokens = all.ToArray(),
                    GeneratedTokens = genTokens.ToArray(),
                    NumGenerated = genTokens.Count,
                    AcceptanceRate = wrapper.SpeculationAcceptanceRate ?? 0.0,
                    RequestId = request.RequestId,
                    LogProbs = genLogProbs
                };
            }
            catch (OperationCanceledException)
            {
                return new SpeculativeDecodingResponse
                {
                    Error = "Text generation was cancelled.",
                    RequestId = request.RequestId
                };
            }
            catch (Exception ex)
            {
                // A structured-output constraint is a STATEFUL machine that the failed incremental attempt has
                // already advanced. The full-context fallback below reuses the same request.Constraint, so
                // resuming from its mid-state would emit malformed / truncated structured output. When a
                // constraint is in play, fail instead of retrying with the advanced constraint.
                if (request.Constraint is not null)
                {
                    _logger.LogWarning(ex,
                        "Batched incremental generation failed for model '{ModelName}' with an active structured-output constraint; failing rather than retrying with the advanced constraint.",
                        modelName);
                    return new SpeculativeDecodingResponse
                    {
                        Error = "Structured-output generation failed and cannot be safely retried.",
                        RequestId = request.RequestId
                    };
                }
                _logger.LogWarning(ex,
                    "Batched incremental generation failed for model '{ModelName}'; falling back to full-context decode.",
                    modelName);
            }
        }

        var config = new ContinuousBatcherConfig
        {
            // Drive the engine synchronously via Step() for this single REST request.
            AutoStart = false,
            EosTokenId = request.EosTokenId ?? ContinuousBatcherConfig.DefaultEosTokenId,
            EnableSpeculativeDecoding = true,
            SpeculationDepth = request.NumDraftTokens,
            UseTreeSpeculation = request.UseTreeSpeculation,
            // Tree knobs only take effect when tree speculation is engaged; pass them through so
            // the request's TreeBranchFactor/MaxTreeDepth are honored rather than silently ignored.
            TreeBranchFactor = request.UseTreeSpeculation ? request.TreeBranchFactor : 0,
            MaxTreeDepth = request.UseTreeSpeculation ? request.MaxTreeDepth : 0
        };

        using var batcher = new ContinuousBatcher<T>(config, tokens => generativeModel.Forward(tokens));

        // Build the SAME engine request as the incremental path so every field flows — EOS, top-p/top-k/
        // min-p, seed, speculation depth, and any structured-output Constraint. (Previously this path built
        // the request inline and silently dropped all of those, so e.g. a response_format was ignored on the
        // full-context fallback.)
        var generationRequest = BuildGenerationRequest<T>(request);

        // Enqueue the request (AutoStart=false keeps the background loop off) and drive Step()
        // until the sequence completes.
        var task = batcher.GenerateAsync(generationRequest);

        // Safety bound: each Step produces at least one token for the running sequence, so the
        // request completes within MaxNewTokens iterations plus a small margin for prefill.
        int maxIterations = request.MaxNewTokens + request.InputTokens.Length + 8;
        while (!task.IsCompleted && maxIterations-- > 0)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                // Client disconnected / request aborted — stop driving the engine instead of
                // leaving orphaned generation work running to the iteration budget.
                return new SpeculativeDecodingResponse
                {
                    Error = "Text generation was cancelled.",
                    RequestId = request.RequestId
                };
            }
            batcher.Step();
        }

        if (!task.IsCompleted)
        {
            _logger.LogWarning(
                "Text generation for model '{ModelName}' did not complete within the iteration budget.",
                modelName);
            return new SpeculativeDecodingResponse
            {
                Error = "Text generation did not complete within the allotted budget.",
                RequestId = request.RequestId
            };
        }

        var result = await task.ConfigureAwait(false);

        // Same normalization as the incremental path: strip trailing EOS, cap to MaxNewTokens, and trim
        // log-probs to the final token count so output is identical regardless of which path served it.
        var (generated, fallbackLogProbs) = NormalizeGeneration(
            result.GeneratedTokens, result.LogProbs, request.EosTokenId ?? ContinuousBatcherConfig.DefaultEosTokenId, request.MaxNewTokens);

        var allTokens = new List<int>(request.InputTokens);
        allTokens.AddRange(generated);

        return new SpeculativeDecodingResponse
        {
            AllTokens = allTokens.ToArray(),
            GeneratedTokens = generated.ToArray(),
            NumGenerated = generated.Count,
            AcceptanceRate = batcher.SpeculationAcceptanceRate ?? 0.0,
            RequestId = request.RequestId,
            LogProbs = fallbackLogProbs
        };
    }

    // Normalizes a raw engine result to the serving contract: drops a trailing EOS token, caps to
    // maxNewTokens, and trims per-token log-probs to the SAME final count (one logprob per generated token).
    // Applied on BOTH generation paths so the response never depends on which path served it.
    private static (List<int> Tokens, List<ContinuousBatching.PositionLogProbs>? LogProbs) NormalizeGeneration(
        IReadOnlyList<int> generated, List<ContinuousBatching.PositionLogProbs>? logProbs, int eos, int maxNewTokens)
    {
        var tokens = new List<int>(generated);
        if (tokens.Count > 0 && tokens[tokens.Count - 1] == eos)
        {
            tokens.RemoveAt(tokens.Count - 1);
        }
        if (tokens.Count > maxNewTokens)
        {
            tokens = tokens.Take(maxNewTokens).ToList();
        }
        var lp = logProbs;
        if (lp is not null && lp.Count > tokens.Count)
        {
            lp = lp.Take(tokens.Count).ToList();
        }
        return (tokens, lp);
    }

    /// <summary>Builds a <c>[1, n]</c> token-id tensor from token IDs.</summary>
    private static Tensor<T> TokensToTensor<T>(IReadOnlyList<int> tokens)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(new[] { 1, tokens.Count });
        for (int i = 0; i < tokens.Count; i++)
        {
            tensor[0, i] = numOps.FromDouble(tokens[i]);
        }
        return tensor;
    }

}
