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
public sealed class TextGenerationService : ITextGenerationService, IDisposable
{
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<TextGenerationService> _logger;

    // One shared, continuously-running ContinuousBatcher per (model, numeric type). Concurrent requests to a
    // model are enqueued into the SAME batcher so they batch together in-flight (the actual throughput win),
    // instead of each request spinning up its own single-sequence engine. Keyed by "modelName|T".
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, IDisposable> _batchers = new();
    private volatile bool _disposed;

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

    /// <summary>
    /// Gets (or lazily creates and starts) the shared continuous batcher for a model. The batcher runs a
    /// background loop, so requests submitted from different HTTP calls are scheduled into the same in-flight
    /// batch. Built over the model's stateless token→logits forward.
    /// </summary>
    private ContinuousBatcher<T> GetOrCreateBatcher<T>(string modelName, IServableGenerativeModel<T> model, int eosTokenId)
    {
        string key = modelName + "|" + typeof(T).FullName;
        var batcher = (ContinuousBatcher<T>)_batchers.GetOrAdd(key, _ =>
        {
            var config = new ContinuousBatcherConfig
            {
                AutoStart = true, // background run loop → cross-request in-flight batching
                EosTokenId = eosTokenId,
            };
            var created = new ContinuousBatcher<T>(config, tokens => model.Forward(tokens));
            created.Start();
            return created;
        });
        return batcher;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var batcher in _batchers.Values)
        {
            try { batcher.Dispose(); }
            catch (Exception ex) { _logger.LogWarning(ex, "Failed to dispose a continuous batcher."); }
        }
        _batchers.Clear();
    }

    /// <inheritdoc/>
    public SpeculativeDecodingResponse Generate(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);

        return numericType switch
        {
            NumericType.Double => GenerateTyped<double>(modelName, request, cancellationToken),
            NumericType.Float => GenerateTyped<float>(modelName, request, cancellationToken),
            NumericType.Decimal => GenerateTyped<decimal>(modelName, request, cancellationToken),
            _ => new SpeculativeDecodingResponse
            {
                Error = $"Unsupported numeric type '{numericType}' for text generation.",
                RequestId = request.RequestId
            }
        };
    }

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
        var batcher = GetOrCreateBatcher(modelName, gm, eosTokenId);

        // Bridge the batcher's push-based per-token callback to this pull-based enumerable so streamed HTTP
        // responses drain tokens as the shared batch produces them. The batcher owns sampling
        // (temperature / top-p / top-k / min-p) and scheduling across all concurrent sequences.
        var tokens = new System.Collections.Concurrent.BlockingCollection<int>();
        var genRequest = new GenerationRequest<T>
        {
            PromptTokenIds = new List<int>(request.InputTokens),
            MaxNewTokens = request.MaxNewTokens,
            Temperature = (float)request.Temperature,
            TopP = (float)request.TopP,
            TopK = request.TopK,
            MinP = (float)request.MinP,
            OnTokenGenerated = tok => { try { tokens.Add(tok); } catch (InvalidOperationException) { /* stream closed */ } },
        };

        var task = batcher.GenerateAsync(genRequest, ct);
        // Terminate the stream when generation finishes (completed, cancelled, or faulted).
        _ = task.ContinueWith(_ => { try { tokens.CompleteAdding(); } catch { /* already completed */ } },
            System.Threading.Tasks.TaskScheduler.Default);

        foreach (int tok in tokens.GetConsumingEnumerable(ct))
        {
            yield return tok;
        }
    }

    /// <summary>Extracts the last-position vocabulary row of a logits tensor as doubles.</summary>
    private static double[] LastPositionLogits<T>(Tensor<T> logits)
    {
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];
        int positions = 1;
        for (int d = 0; d < rank - 1; d++)
        {
            positions *= logits.Shape[d];
        }
        int baseOffset = (positions - 1) * vocab;

        var flat = logits.AsSpan();
        var row = new double[vocab];
        for (int v = 0; v < vocab; v++)
        {
            row[v] = Convert.ToDouble(flat[baseOffset + v]);
        }
        return row;
    }

    /// <summary>
    /// Samples a token id from a logits row. Greedy (argmax) when <paramref name="temperature"/> ≤ 0,
    /// otherwise temperature-scaled softmax with optional top-k and nucleus (top-p) filtering.
    /// </summary>
    private static int SampleToken(double[] logits, double temperature, double topP, int topK, double minP, Random rng)
    {
        int n = logits.Length;
        if (n == 0) return 0;
        if (temperature <= 0.0)
        {
            return ArgMax(logits);
        }

        // Order indices by descending logit for top-k / top-p truncation.
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => logits[b].CompareTo(logits[a]));

        int limit = topK > 0 ? Math.Min(topK, n) : n;

        // Softmax over the retained candidates (numerically stable).
        double max = logits[idx[0]];
        var probs = new double[limit];
        double sum = 0.0;
        for (int j = 0; j < limit; j++)
        {
            double p = Math.Exp((logits[idx[j]] - max) / temperature);
            probs[j] = p;
            sum += p;
        }
        for (int j = 0; j < limit; j++) probs[j] /= sum;

        // Min-p: drop tokens below minP × the top token's probability (probs are sorted descending), then
        // renormalize. Applied before top-p, matching the common vLLM/HF ordering.
        if (minP > 0.0 && limit > 1)
        {
            double threshold = minP * probs[0];
            int cut = limit;
            for (int j = 1; j < limit; j++)
            {
                if (probs[j] < threshold) { cut = j; break; }
            }
            if (cut < limit)
            {
                limit = cut;
                double s = 0.0;
                for (int j = 0; j < limit; j++) s += probs[j];
                if (s > 0) for (int j = 0; j < limit; j++) probs[j] /= s;
            }
        }

        // Nucleus (top-p): keep the smallest prefix whose cumulative prob ≥ topP, then renormalize.
        if (topP > 0.0 && topP < 1.0)
        {
            double cum = 0.0;
            int cut = limit;
            for (int j = 0; j < limit; j++)
            {
                cum += probs[j];
                if (cum >= topP) { cut = j + 1; break; }
            }
            limit = cut;
            double s2 = 0.0;
            for (int j = 0; j < limit; j++) s2 += probs[j];
            if (s2 > 0) for (int j = 0; j < limit; j++) probs[j] /= s2;
        }

        double r = rng.NextDouble();
        double acc = 0.0;
        for (int j = 0; j < limit; j++)
        {
            acc += probs[j];
            if (r <= acc) return idx[j];
        }
        return idx[limit - 1];
    }

    /// <summary>Argmax over a logits row.</summary>
    private static int ArgMax(double[] logits)
    {
        int best = 0;
        double bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best;
    }

    private SpeculativeDecodingResponse GenerateTyped<T>(string modelName, SpeculativeDecodingRequest request, CancellationToken cancellationToken)
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

        // Preferred path: KV-cached incremental decode over a per-request session (each request gets
        // its own paged-cache sequence id, so concurrent requests to the same model are isolated and
        // each decode step only pays for the new token instead of recomputing the full context).
        // On any failure, fall through to the proven full-context path below so a model whose
        // incremental path misbehaves still serves correct (if slower) results.
        if (generativeModel.SupportsIncrementalGeneration)
        {
            try
            {
                return GenerateIncremental(modelName, generativeModel, request, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Incremental generation failed for model '{ModelName}'; falling back to full-context decode.",
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

        var generationRequest = new GenerationRequest<T>
        {
            PromptTokenIds = new List<int>(request.InputTokens),
            MaxNewTokens = request.MaxNewTokens,
            Temperature = (float)request.Temperature
        };

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

        var result = task.GetAwaiter().GetResult();

        var generated = result.GeneratedTokens;
        // Honor the MaxNewTokens contract exactly at the serving boundary (defensive cap).
        if (generated.Count > request.MaxNewTokens)
        {
            generated = generated.Take(request.MaxNewTokens).ToList();
        }

        var allTokens = new List<int>(request.InputTokens);
        allTokens.AddRange(generated);

        return new SpeculativeDecodingResponse
        {
            AllTokens = allTokens.ToArray(),
            GeneratedTokens = generated.ToArray(),
            NumGenerated = generated.Count,
            AcceptanceRate = batcher.SpeculationAcceptanceRate ?? 0.0,
            RequestId = request.RequestId
        };
    }

    /// <summary>
    /// KV-cached incremental decode: prefill the prompt, then decode one token at a time over the
    /// session's paged KV cache (each step forwards only the new token). The session owns a distinct
    /// cache sequence id, so concurrent requests to the same model are isolated.
    /// </summary>
    private SpeculativeDecodingResponse GenerateIncremental<T>(
        string modelName,
        IServableGenerativeModel<T> model,
        SpeculativeDecodingRequest request,
        CancellationToken cancellationToken)
    {
        int eosTokenId = request.EosTokenId ?? ContinuousBatcherConfig.DefaultEosTokenId;
        var generated = new List<int>(request.MaxNewTokens);

        // Prefix sharing (RadixAttention): the session may start with a registered prompt prefix
        // already in its KV cache (forked copy-on-write); we then forward only the remaining suffix.
        using var session = model.BeginGeneration(request.InputTokens);
        int prefillStart = session.CachedPromptTokens; // 0..InputTokens.Length-1 (strict prefix)

        // Prefill the (remaining) prompt. When the model accepts a multi-token forward, do it in a
        // SINGLE batched pass (per-position logits, take the last); otherwise one token at a time
        // (universally compatible with fixed single-token-step models). There is always >= 1 suffix
        // token because any shared prefix is strict. Exceptions propagate to the caller's fallback
        // (full-context decode); cancellation returns a cancelled response directly.
        int suffixLength = request.InputTokens.Length - prefillStart;
        Tensor<T> logits;
        if (model.SupportsBatchedPrefill && suffixLength > 1)
        {
            var suffix = new int[suffixLength];
            for (int i = 0; i < suffixLength; i++)
            {
                suffix[i] = request.InputTokens[prefillStart + i];
            }
            logits = session.Forward(TokensToTensor<T>(suffix));
        }
        else
        {
            logits = session.Forward(TokensToTensor<T>(new[] { request.InputTokens[prefillStart] }));
            for (int i = prefillStart + 1; i < request.InputTokens.Length; i++)
            {
                logits = session.Forward(TokensToTensor<T>(new[] { request.InputTokens[i] }));
            }
        }

        // Register this prompt as a reusable prefix so later requests that extend it can fork its KV.
        session.RegisterPromptPrefix(request.InputTokens);

        // Running token stream (prompt + generated) — used both for the response and as the corpus for
        // prompt-lookup speculative drafting.
        var allTokens = new List<int>(request.InputTokens.Length + request.MaxNewTokens);
        allTokens.AddRange(request.InputTokens);

        // Speculative decoding composes with the paged KV cache only when the model produces per-position
        // logits (a multi-token verify forward is the whole point) — the same capability batched prefill
        // requires. We use draft-model-free PROMPT-LOOKUP speculation (n-gram match against the running
        // stream): no second model, exact-greedy output, and a real speed-up on repetitive text.
        double acceptanceRate = 0.0;
        bool speculative = request.NumDraftTokens > 0 && model.SupportsBatchedPrefill;
        if (speculative)
        {
            acceptanceRate = SpeculativeDecodeLoop(
                session, logits, request, eosTokenId, generated, allTokens, cancellationToken,
                out bool cancelled);
            if (cancelled)
            {
                return new SpeculativeDecodingResponse { Error = "Text generation was cancelled.", RequestId = request.RequestId };
            }
        }
        else
        {
            for (int step = 0; step < request.MaxNewTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    return new SpeculativeDecodingResponse { Error = "Text generation was cancelled.", RequestId = request.RequestId };
                }

                int next = ArgMaxLastPosition(logits);
                if (next == eosTokenId)
                {
                    break;
                }

                generated.Add(next);
                allTokens.Add(next);

                // Decode: forward only the new token; the KV cache supplies the prior context.
                logits = session.Forward(TokensToTensor<T>(new[] { next }));
            }
        }

        return new SpeculativeDecodingResponse
        {
            AllTokens = allTokens.ToArray(),
            GeneratedTokens = generated.ToArray(),
            NumGenerated = generated.Count,
            AcceptanceRate = acceptanceRate,
            RequestId = request.RequestId
        };
    }

    /// <summary>
    /// Exact-greedy speculative decode over the per-sequence paged cache, using prompt-lookup (n-gram)
    /// drafting. Each round: draft up to <c>NumDraftTokens</c> tokens, verify them in ONE multi-token
    /// forward (per-position logits), accept the greedy-matching prefix, and on rejection roll the KV
    /// cache back so the corrected token overwrites the rejected drafts. The emitted tokens are
    /// byte-for-byte identical to plain greedy decode; speculation only changes how many forwards it takes.
    /// </summary>
    /// <returns>The fraction of drafted tokens that were accepted (0 when nothing was ever drafted).</returns>
    private static double SpeculativeDecodeLoop<T>(
        IGenerationSession<T> session,
        Tensor<T> prefillLogits,
        SpeculativeDecodingRequest request,
        int eosTokenId,
        List<int> generated,
        List<int> allTokens,
        CancellationToken cancellationToken,
        out bool cancelled)
    {
        cancelled = false;
        int maxNew = request.MaxNewTokens;
        int k = Math.Max(1, request.NumDraftTokens);
        var nextLogits = prefillLogits; // predicts the next position to emit
        long drafted = 0, accepted = 0;

        while (generated.Count < maxNew)
        {
            if (cancellationToken.IsCancellationRequested) { cancelled = true; return Ratio(accepted, drafted); }

            int greedy = ArgMaxLastPosition(nextLogits);
            var draft = PromptLookupDraft(allTokens, k);

            if (draft.Count == 0)
            {
                // No n-gram match this round: take a single ordinary greedy step.
                if (greedy == eosTokenId) break;
                generated.Add(greedy); allTokens.Add(greedy);
                if (generated.Count >= maxNew) break;
                nextLogits = session.Forward(TokensToTensor<T>(new[] { greedy }));
                continue;
            }

            // Verify the K draft tokens in ONE forward. They occupy positions pos..pos+K-1; row j of the
            // returned logits predicts position pos+j+1 given the prefix + draft[0..j].
            int posBefore = session.Position;
            var verifyLogits = session.Forward(TokensToTensor<T>(draft));
            drafted += draft.Count;

            // Accept the longest greedy-matching prefix: expected_0 = greedy(prefill);
            // expected_{j+1} = argmax(verify row j).
            int expected = greedy;
            int nAccept = 0;
            for (int j = 0; j < draft.Count; j++)
            {
                if (draft[j] != expected) break;
                nAccept++;
                expected = ArgMaxAtPosition(verifyLogits, j);
            }
            accepted += nAccept;

            // Emit accepted drafts (their KV at pos..pos+nAccept-1 is correct — real tokens as context).
            bool stop = false;
            for (int j = 0; j < nAccept; j++)
            {
                if (draft[j] == eosTokenId) { stop = true; break; }
                generated.Add(draft[j]); allTokens.Add(draft[j]);
                if (generated.Count >= maxNew) { stop = true; break; }
            }
            if (stop) break;

            // Drop the rejected drafts' KV (positions pos+nAccept..), then emit + commit the correction
            // (target's own greedy at the divergence, or the bonus token after a full accept).
            session.Truncate(posBefore + nAccept);
            if (expected == eosTokenId) break;
            generated.Add(expected); allTokens.Add(expected);
            if (generated.Count >= maxNew) break;
            nextLogits = session.Forward(TokensToTensor<T>(new[] { expected }));
        }

        return Ratio(accepted, drafted);

        static double Ratio(long a, long d) => d > 0 ? (double)a / d : 0.0;
    }

    /// <summary>
    /// Prompt-lookup draft: find the most recent earlier occurrence of the last <c>ngram</c> tokens in
    /// the running stream and propose the up-to-<paramref name="k"/> tokens that followed it. Returns an
    /// empty list when no match exists (the caller then takes a plain greedy step). No draft model needed.
    /// </summary>
    private static List<int> PromptLookupDraft(IReadOnlyList<int> tokens, int k, int ngram = 2)
    {
        int n = tokens.Count;
        var draft = new List<int>(k);
        if (n < ngram + 1)
        {
            return draft;
        }

        // Scan backwards for the most recent earlier match of the trailing ngram.
        for (int start = n - ngram - 1; start >= 0; start--)
        {
            bool match = true;
            for (int g = 0; g < ngram; g++)
            {
                if (tokens[start + g] != tokens[n - ngram + g]) { match = false; break; }
            }
            if (!match) continue;

            int src = start + ngram;
            for (int i = 0; i < k && src + i < n; i++)
            {
                draft.Add(tokens[src + i]);
            }
            return draft;
        }
        return draft;
    }

    /// <summary>Argmax over vocab at output position <paramref name="posIndex"/> of a per-position logits
    /// tensor (<c>[1, positions, vocab]</c> or <c>[positions, vocab]</c>).</summary>
    private static int ArgMaxAtPosition<T>(Tensor<T> logits, int posIndex)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];
        int baseOffset = posIndex * vocab;

        var flat = logits.AsSpan();
        int best = 0;
        T bestVal = flat[baseOffset];
        for (int v = 1; v < vocab; v++)
        {
            if (numOps.GreaterThan(flat[baseOffset + v], bestVal))
            {
                bestVal = flat[baseOffset + v];
                best = v;
            }
        }
        return best;
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

    /// <summary>
    /// Returns the argmax token id of the last position of a logits tensor, handling
    /// <c>[1, seq, vocab]</c>, <c>[seq, vocab]</c>, <c>[1, vocab]</c>, and <c>[vocab]</c> shapes.
    /// </summary>
    private static int ArgMaxLastPosition<T>(Tensor<T> logits)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];

        // Flat offset of the last position's vocab row.
        int positions = 1;
        for (int d = 0; d < rank - 1; d++)
        {
            positions *= logits.Shape[d];
        }
        int baseOffset = (positions - 1) * vocab;

        var flat = logits.AsSpan();
        int best = 0;
        T bestVal = flat[baseOffset];
        for (int v = 1; v < vocab; v++)
        {
            if (numOps.GreaterThan(flat[baseOffset + v], bestVal))
            {
                bestVal = flat[baseOffset + v];
                best = v;
            }
        }
        return best;
    }
}
