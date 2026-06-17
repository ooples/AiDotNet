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

        using var session = model.BeginGeneration();

        // Prefill: forward the whole prompt, producing the first next-token logits.
        // Exceptions propagate to the caller's fallback (full-context decode); cancellation returns
        // a cancelled response directly (no fallback).
        var logits = session.Forward(TokensToTensor<T>(request.InputTokens));

        for (int step = 0; step < request.MaxNewTokens; step++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                return new SpeculativeDecodingResponse
                {
                    Error = "Text generation was cancelled.",
                    RequestId = request.RequestId
                };
            }

            int next = ArgMaxLastPosition(logits);
            if (next == eosTokenId)
            {
                break;
            }

            generated.Add(next);

            // Decode: forward only the new token; the KV cache supplies the prior context.
            logits = session.Forward(TokensToTensor<T>(new[] { next }));
        }

        var allTokens = new List<int>(request.InputTokens);
        allTokens.AddRange(generated);

        return new SpeculativeDecodingResponse
        {
            AllTokens = allTokens.ToArray(),
            GeneratedTokens = generated.ToArray(),
            NumGenerated = generated.Count,
            AcceptanceRate = 0.0, // speculative decoding composes with incremental in a later stage
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
