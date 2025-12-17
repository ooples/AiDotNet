namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// Configuration for the continuous batcher.
/// </summary>
public class ContinuousBatcherConfig
{
    /// <summary>
    /// Scheduler configuration.
    /// </summary>
    public BatchSchedulerConfig SchedulerConfig { get; set; } = new();

    /// <summary>
    /// End-of-sequence token ID.
    /// </summary>
    public int EosTokenId { get; set; } = 2;

    /// <summary>
    /// Milliseconds to sleep when idle.
    /// </summary>
    public int IdleSleepMs { get; set; } = 10;

    /// <summary>
    /// Whether to automatically start the batcher when a request is submitted.
    /// </summary>
    public bool AutoStart { get; set; } = true;

    /// <summary>
    /// Maximum number of tokens in context (prompt + generated).
    /// </summary>
    public int MaxContextLength { get; set; } = 4096;

    /// <summary>
    /// Whether to enable speculative decoding.
    /// </summary>
    public bool EnableSpeculativeDecoding { get; set; } = false;

    /// <summary>
    /// Policy for when speculative decoding should run (default: Auto).
    /// </summary>
    public AiDotNet.Configuration.SpeculationPolicy SpeculationPolicy { get; set; } = AiDotNet.Configuration.SpeculationPolicy.Auto;

    /// <summary>
    /// Number of tokens to draft ahead when speculative decoding is enabled.
    /// </summary>
    public int SpeculationDepth { get; set; } = 4;

    /// <summary>
    /// Speculative decoding method to use (default: Auto).
    /// </summary>
    /// <remarks>
    /// This keeps the public serving surface compact while enabling internal selection of
    /// classic draft-model speculation vs tree-based alternatives (Medusa/EAGLE).
    /// </remarks>
    public AiDotNet.Configuration.SpeculativeMethod SpeculativeMethod { get; set; } = AiDotNet.Configuration.SpeculativeMethod.Auto;

    /// <summary>
    /// Whether to use tree-based speculation (multiple draft continuations).
    /// </summary>
    /// <remarks>
    /// This is an advanced option; when false the batcher uses classic speculative decoding.
    /// Some speculative methods may implicitly enable this internally.
    /// </remarks>
    public bool UseTreeSpeculation { get; set; } = false;

    /// <summary>
    /// Creates config for a specific model.
    /// </summary>
    public static ContinuousBatcherConfig ForModel(string modelName, int maxBatchSize = 8)
    {
        return new ContinuousBatcherConfig
        {
            SchedulerConfig = BatchSchedulerConfig.ForModel(modelName, maxBatchSize),
            MaxContextLength = modelName.ToLowerInvariant() switch
            {
                "llama-7b" or "llama-13b" => 4096,
                "llama-70b" => 4096,
                "gpt2" => 1024,
                _ => 2048
            }
        };
    }
}
