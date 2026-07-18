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

        // Preferred path: stream tokens off the ONE shared continuous-batching engine as they are
        // produced (paged KV, prefix sharing, per-request sampling). EOS terminates the stream and is
        // not yielded. The engine honors MaxNewTokens; a defensive cap is applied here as well.
        if (model is ServableModelWrapper<T> wrapper && wrapper.SupportsIncrementalGeneration)
        {
            int emitted = 0;
            foreach (var token in wrapper.StreamGeneration(BuildGenerationRequest<T>(request), eosTokenId, ct))
            {
                if (ct.IsCancellationRequested || emitted >= request.MaxNewTokens) yield break;
                emitted++;
                yield return token;
            }
            yield break;
        }

        // Stateless fallback (models without the incremental path): re-forward the full growing context
        // each step and sample.
        double temperature = request.Temperature;
        double topP = request.TopP;
        int topK = request.TopK;
        double minP = request.MinP;
        // Seeded per request id when present so greedy stays deterministic and sampling is reproducible.
        var rng = new Random(request.RequestId?.GetHashCode() ?? 0);
        var context = new List<int>(request.InputTokens);
        for (int step = 0; step < request.MaxNewTokens; step++)
        {
            if (ct.IsCancellationRequested) yield break;
            var logits = gm.Forward(TokensToTensor<T>(context));
            int next = SampleToken(LastPositionLogits(logits), temperature, topP, topK, minP, rng);
            if (next == eosTokenId) yield break;
            yield return next;
            context.Add(next);
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
    private static GenerationRequest<T> BuildGenerationRequest<T>(SpeculativeDecodingRequest request)
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
            SpeculationDepth = request.NumDraftTokens,
            Seed = request.RequestId is { } id ? id.GetHashCode() : (int?)null,
            Constraint = request.Constraint,
            LogitBias = request.LogitBias,
            FrequencyPenalty = (float)request.FrequencyPenalty,
            PresencePenalty = (float)request.PresencePenalty
        };
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

        // Preferred path: the ONE shared continuous-batching engine (paged KV cache, RadixAttention
        // prefix sharing, greedy-exact prompt-lookup speculation, per-request sampling). All requests to
        // this model co-batch on the same engine; each gets an isolated paged-cache sequence and only
        // pays for the new token per decode step. On any failure, fall through to the proven full-context
        // path below so a model whose incremental path misbehaves still serves correct (if slower) results.
        if (model is ServableModelWrapper<T> wrapper && wrapper.SupportsIncrementalGeneration)
        {
            try
            {
                int eos = request.EosTokenId ?? ContinuousBatcherConfig.DefaultEosTokenId;
                var genResult = wrapper.RunGeneration(BuildGenerationRequest<T>(request), cancellationToken);

                var genTokens = new List<int>(genResult.GeneratedTokens);
                // The engine emits EOS as a terminating token; it is not part of the completion content.
                if (genTokens.Count > 0 && genTokens[genTokens.Count - 1] == eos)
                {
                    genTokens.RemoveAt(genTokens.Count - 1);
                }
                // Honor the MaxNewTokens contract exactly at the serving boundary (defensive cap).
                if (genTokens.Count > request.MaxNewTokens)
                {
                    genTokens = genTokens.Take(request.MaxNewTokens).ToList();
                }

                var all = new List<int>(request.InputTokens);
                all.AddRange(genTokens);
                return new SpeculativeDecodingResponse
                {
                    AllTokens = all.ToArray(),
                    GeneratedTokens = genTokens.ToArray(),
                    NumGenerated = genTokens.Count,
                    AcceptanceRate = wrapper.SpeculationAcceptanceRate ?? 0.0,
                    RequestId = request.RequestId
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
