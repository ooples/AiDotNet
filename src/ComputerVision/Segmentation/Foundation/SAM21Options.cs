using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for SAM 2.1 (Segment Anything Model 2.1).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM 2.1 is an improved version of SAM 2 with refined checkpoints
/// for better segmentation accuracy. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class SAM21Options : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SAM21Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAM21Options(SAM21Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MemoryBankSize = other.MemoryBankSize;
    }

    /// <summary>
    /// Maximum number of frames to keep in the memory bank for video segmentation.
    /// When null, defaults to 7 (as in the SAM 2 paper).
    /// </summary>
    public int? MemoryBankSize { get; set; }
}
