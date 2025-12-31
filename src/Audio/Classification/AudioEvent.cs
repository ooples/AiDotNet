namespace AiDotNet.Audio.Classification;

/// <summary>
/// Represents a detected audio event.
/// </summary>
public class AudioEvent
{
    /// <summary>Event label/category.</summary>
    public required string Label { get; init; }

    /// <summary>Confidence score (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Event start time in seconds.</summary>
    public double StartTime { get; init; }

    /// <summary>Event end time in seconds.</summary>
    public double EndTime { get; init; }

    /// <summary>Event duration in seconds.</summary>
    public double Duration => EndTime - StartTime;

    public override string ToString() =>
        $"{Label} ({Confidence:P0}): {StartTime:F2}s - {EndTime:F2}s";
}
