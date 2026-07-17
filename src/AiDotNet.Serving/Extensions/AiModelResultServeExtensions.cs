using System;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Models.Results;
using AiDotNet.Serving.Engine;
using AiDotNet.Serving.Engine.Http;

namespace AiDotNet.Serving.Extensions;

/// <summary>
/// The one-line serving surface on a built model: <c>using var server = model.Serve(tokenizer);</c> starts a
/// running, OpenAI-compatible HTTP endpoint backed by the paged-KV, continuously-batched engine. No engine
/// arguments, KV math, or server plumbing — the vLLM-class deployment none of the expert-only frameworks make
/// this easy.
/// </summary>
public static class AiModelResultServeExtensions
{
    /// <summary>
    /// Starts an OpenAI-compatible inference server for the model and returns it running. Dispose the returned
    /// server to stop it and free resources.
    /// </summary>
    /// <param name="result">The built model.</param>
    /// <param name="tokenizer">Tokenizer for the HTTP text API. May be null when the model carries its own
    /// via <see cref="IProvidesGenerationTokenizer"/>.</param>
    /// <param name="options">Optional serving options (URLs, model name, engine tuning). Sensible defaults.</param>
    public static InferenceServer<T> Serve<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        IGenerationTokenizer? tokenizer = null,
        ServeOptions? options = null)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));

        var model = result.Model
            ?? throw new InvalidOperationException("The model has not been trained/initialized; cannot serve.");

        var resolvedTokenizer = tokenizer ?? (model as IProvidesGenerationTokenizer)?.GetGenerationTokenizer()
            ?? throw new InvalidOperationException(
                "Serve() needs a tokenizer for its HTTP text API: pass one, or have the model implement " +
                "IProvidesGenerationTokenizer.");

        var opts = options ?? new ServeOptions();
        var defaultSampling = opts.DefaultSampling ?? new SamplingParameters { Temperature = 0.0, MaxTokens = 128 };
        defaultSampling.Validate();

        var selection = ServingRunnerFactory.Create<T>(model);
        int? eos = selection.EosTokenId ?? (resolvedTokenizer.EosTokenId >= 0 ? resolvedTokenizer.EosTokenId : (int?)null);

        var engineOptions = opts.EngineOptions ?? ServingConfigMapper.ToEngineOptions(opts.InferenceConfig, eos);
        engineOptions = ApplyEos(engineOptions, eos);
        var engine = new ContinuousBatchingEngine<T>(selection.Runner, engineOptions);
        var host = new AsyncEngineHost<T>(engine);

        return new InferenceServer<T>(host, resolvedTokenizer, opts.ModelName, defaultSampling, opts.Urls);
    }

    private static EngineOptions ApplyEos(EngineOptions o, int? eos)
    {
        if (o.EosTokenId is not null || eos is null) return o;
        return new EngineOptions
        {
            MaxNumSequences = o.MaxNumSequences,
            MaxBatchedTokens = o.MaxBatchedTokens,
            BlockSize = o.BlockSize,
            NumKvBlocks = o.NumKvBlocks,
            EosTokenId = eos,
        };
    }
}
