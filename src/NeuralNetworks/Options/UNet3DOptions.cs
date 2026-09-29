using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the UNet3D.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UNet3D labels every cell in a 3D volume rather than the volume as a
/// whole — segmenting an organ in a medical scan, say. It shrinks the volume down and then
/// expands it back, keeping detail through skip connections. The values here are the ones it
/// ships with.
/// </para>
/// </remarks>
public class UNet3DOptions : VoxelModelOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public UNet3DOptions()
    {
        VoxelResolution = 32;
        NumEncoderBlocks = 4;
        BaseFilters = 32;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public UNet3DOptions(UNet3DOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        VoxelResolution = other.VoxelResolution;
        NumEncoderBlocks = other.NumEncoderBlocks;
        BaseFilters = other.BaseFilters;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the number of encoder blocks, which is also the number of decoder blocks
    /// mirroring them. Default: 4.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each encoder block halves the volume while the decoder side
    /// builds it back up, so this sets how deep the U shape goes.</para>
    /// </remarks>
    public int NumEncoderBlocks { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    public void Validate()
    {
        ValidateCore();
        Require(NumEncoderBlocks, nameof(NumEncoderBlocks));

        // Cross-field: each block halves the grid, so the resolution must survive them all.
        // Moved here from the constructor along with the two values it compares, and it keeps
        // ArgumentOutOfRangeException — a range relationship, not an unset dimension.
        int minResolution = 1 << NumEncoderBlocks;
        if (VoxelResolution < minResolution)
        {
            throw new ArgumentOutOfRangeException(
                nameof(VoxelResolution),
                $"VoxelResolution must be at least {minResolution} for {NumEncoderBlocks} encoder blocks.");
        }
    }
}
