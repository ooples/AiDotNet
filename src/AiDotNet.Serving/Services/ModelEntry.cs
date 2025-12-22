using AiDotNet.Serving.Configuration;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Stores a loaded model instance with basic metadata for serving.
/// </summary>
internal sealed class ModelEntry
{
    public object Model { get; set; } = null!;
    public NumericType NumericType { get; set; } = NumericType.Double;
    public DateTime LoadedAt { get; set; }
    public string? SourcePath { get; set; }
    public bool IsFromRegistry { get; set; }
    public int? RegistryVersion { get; set; }
    public string? RegistryStage { get; set; }
}

