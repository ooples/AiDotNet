using System.Text.Json.Serialization;

namespace AiDotNet.Tools.ModelPerfProbe;

/// <summary>
/// Structured result emitted by a single probe run. One record per probed model,
/// per probe configuration. Serialized to JSON for the CI manifest and human
/// summary; the field names here are stable and tracked by the regression
/// baseline so renaming a field is a breaking change for the baseline format.
/// </summary>
internal sealed class ProbeResult
{
    [JsonPropertyName("model")] public string Model { get; set; } = "";

    /// <summary>Time spent in the parameterless / arch-only ctor + InitializeLayers / Embeddings.</summary>
    [JsonPropertyName("constructMs")] public double ConstructMs { get; set; }

    /// <summary>Time spent in the warm-up forward (first call exercises lazy initialization).</summary>
    [JsonPropertyName("warmupForwardMs")] public double WarmupForwardMs { get; set; }

    /// <summary>Time spent in the warm-up Train step (first backward exercises tape + optimizer init).</summary>
    [JsonPropertyName("warmupTrainMs")] public double WarmupTrainMs { get; set; }

    [JsonPropertyName("stepCount")] public int StepCount { get; set; }
    [JsonPropertyName("totalMs")] public double TotalMs { get; set; }
    [JsonPropertyName("avgStepMs")] public double AvgStepMs { get; set; }

    /// <summary>Total managed-heap bytes allocated across all measured steps (excludes warm-up).</summary>
    [JsonPropertyName("allocBytes")] public long AllocBytes { get; set; }
    [JsonPropertyName("allocMbPerStep")] public double AllocMbPerStep { get; set; }

    [JsonPropertyName("gen0")] public int Gen0 { get; set; }
    [JsonPropertyName("gen1")] public int Gen1 { get; set; }
    [JsonPropertyName("gen2")] public int Gen2 { get; set; }

    /// <summary>
    /// Projected xUnit-budget wall time at the model-family scaffold's default
    /// TrainingIterations*3 = 30 steps. Used to flag models that would time out
    /// the 120 s Training_ShouldReduceLoss invariant.
    /// </summary>
    [JsonPropertyName("projected30iterS")] public double Projected30IterS { get; set; }

    [JsonPropertyName("status")] public string Status { get; set; } = "ok";
    [JsonPropertyName("error")] public string? Error { get; set; }

    /// <summary>Slow-budget tag — true when avg step ms or alloc MB/step exceeds the CLI threshold.</summary>
    [JsonPropertyName("flagged")] public bool Flagged { get; set; }
    [JsonPropertyName("flagReason")] public string? FlagReason { get; set; }
}
