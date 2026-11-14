using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Result of key detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> KeyDetectionResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the model that produced the result.</typeparam>
public class KeyDetectionResult<T>
{
    /// <summary>
    /// Gets or sets the key index (0 = C, 1 = C#, etc.).
    /// </summary>
    public int KeyIndex { get; set; }

    /// <summary>
    /// Gets or sets the full key name (e.g., "C major", "A minor").
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the root note name (e.g., "C", "A").
    /// </summary>
    public string RootNote { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the key mode (major or minor).
    /// </summary>
    public KeyMode Mode { get; set; }

    /// <summary>
    /// Gets or sets the Pearson correlation with the key profile (-1 to 1).
    /// </summary>
    public T Correlation { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public T Confidence { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets or sets the relative major/minor key.
    /// </summary>
    public string RelativeKey { get; set; } = string.Empty;
}
