using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Serving.Models;
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
    public SpeculativeDecodingResponse Generate(string modelName, NumericType numericType, SpeculativeDecodingRequest request)
    {
        Guard.NotNull(request);

        return numericType switch
        {
            NumericType.Double => GenerateTyped<double>(modelName, request),
            NumericType.Float => GenerateTyped<float>(modelName, request),
            NumericType.Decimal => GenerateTyped<decimal>(modelName, request),
            _ => new SpeculativeDecodingResponse
            {
                Error = $"Unsupported numeric type '{numericType}' for text generation.",
                RequestId = request.RequestId
            }
        };
    }

    private SpeculativeDecodingResponse GenerateTyped<T>(string modelName, SpeculativeDecodingRequest request)
    {
        var model = _modelRepository.GetModel<T>(modelName);
        if (model is not IServableGenerativeModel<T> generativeModel || !generativeModel.SupportsGeneration)
        {
            return new SpeculativeDecodingResponse
            {
                Error = $"Model '{modelName}' does not support text generation. " +
                        "Text generation requires a tensor-based (token-to-logits) model such as a transformer language model.",
                RequestId = request.RequestId
            };
        }

        var config = new ContinuousBatcherConfig
        {
            // Drive the engine synchronously via Step() for this single REST request.
            AutoStart = false,
            EosTokenId = request.EosTokenId ?? new ContinuousBatcherConfig().EosTokenId,
            EnableSpeculativeDecoding = true,
            SpeculationDepth = request.NumDraftTokens,
            UseTreeSpeculation = request.UseTreeSpeculation
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
}
