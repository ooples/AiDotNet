namespace AiDotNet.Serving.Engine.Http;

/// <summary>
/// Options for <c>model.Serve(...)</c>. Every value has a sensible default, so <c>model.Serve(tokenizer)</c>
/// starts a working OpenAI-compatible server with no further configuration.
/// </summary>
public sealed class ServeOptions
{
    /// <summary>URLs to bind. Default binds a local port; use "http://0.0.0.0:8080" to expose externally.</summary>
    public string[] Urls { get; init; } = new[] { "http://127.0.0.1:8080" };

    /// <summary>The model id reported at <c>/v1/models</c> and echoed in responses. Default "aidotnet-model".</summary>
    public string ModelName { get; init; } = "aidotnet-model";

    /// <summary>Engine tuning (KV pool, batch limits). Derived from <see cref="InferenceConfig"/> when null.</summary>
    public EngineOptions? EngineOptions { get; init; }

    /// <summary>Inference-optimization config used to derive <see cref="EngineOptions"/> when it is null.</summary>
    public AiDotNet.Configuration.InferenceOptimizationConfig? InferenceConfig { get; init; }

    /// <summary>Sampling used when a request omits parameters. Defaults to greedy, 128 tokens.</summary>
    public SamplingParameters? DefaultSampling { get; init; }
}
