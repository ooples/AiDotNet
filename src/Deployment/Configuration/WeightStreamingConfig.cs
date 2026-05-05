namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for weight streaming (paging large model weights to disk
/// when they don't fit in RAM). Issue #1222 / weight-streaming v1.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Modern foundation models (GPT-class LLMs,
/// vision-language models like PaLM-E) have so many parameters that the
/// raw weights don't fit in your machine's RAM — a 562B-parameter model
/// is ~2.25 TB at fp32. Weight streaming pages those weights to a fast
/// local disk and only loads the slice the model needs RIGHT NOW into
/// RAM, so the model can run on a laptop instead of a multi-GPU
/// workstation. The trade-off is throughput: cold disk reads add
/// latency. AiDotNet's defaults handle this for you (auto-enabled when
/// the model crosses ~10B parameters); use this config to override.</para>
///
/// <para>The default behavior is "smart on": every neural network you
/// build through <see cref="AiDotNet.IAiModelBuilder{T,TInput,TOutput}"/>
/// runs eagerly (zero overhead) until its parameter count crosses the
/// threshold, at which point streaming auto-engages. To force streaming
/// on / off regardless of size, set <see cref="Enabled"/> explicitly. To
/// override the auto-detect threshold, set
/// <see cref="ThresholdParameters"/>.</para>
/// </remarks>
public class WeightStreamingConfig
{
    /// <summary>
    /// Gets or sets whether weight streaming is enabled.
    /// </summary>
    /// <remarks>
    /// <para>Three states:</para>
    /// <list type="bullet">
    /// <item><c>null</c> (default) — auto-detect by parameter count. Models
    /// below the threshold train eagerly; models above it stream.</item>
    /// <item><c>true</c> — force streaming on regardless of size. Useful for
    /// integration tests that need predictable streaming behavior on
    /// small models.</item>
    /// <item><c>false</c> — force streaming off. The model stays fully
    /// resident in RAM. Use when you know the model fits and want to
    /// avoid the per-layer prefetch/materialize overhead.</item>
    /// </list>
    /// </remarks>
    public bool? Enabled { get; set; }

    /// <summary>
    /// Gets or sets the parameter-count threshold above which auto-detect
    /// engages streaming. Only consulted when <see cref="Enabled"/> is
    /// <c>null</c> (the auto-detect mode).
    /// </summary>
    /// <remarks>
    /// <para>Default is 10 billion parameters (~40 GB at fp32, ~20 GB at
    /// fp16) — the point where consumer GPUs (24 GB max) and most
    /// workstation systems start hitting memory pressure. Values are
    /// also overridable per-process via the
    /// <c>AIDOTNET_STREAMING_THRESHOLD_PARAMS</c> environment variable;
    /// the env var takes precedence over the compiled-in default but is
    /// itself overridden by an explicit value here.</para>
    /// </remarks>
    public long? ThresholdParameters { get; set; }
}
