namespace AiDotNet.Serving.Services;

/// <summary>
/// Stores a loaded model instance with basic metadata for serving.
/// </summary>
internal sealed class ModelEntry
{
    public object Model { get; set; } = null!;
    public string NumericType { get; set; } = string.Empty;
    public DateTime LoadedAt { get; set; }
    public string? SourcePath { get; set; }
}

