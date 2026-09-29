using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the VoxelCNN.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> VoxelCNN classifies 3D shapes given as a cube of filled and empty cells
/// — the volumetric equivalent of an image classifier. The values here are the ones it ships
/// with.
/// </para>
/// </remarks>
public class VoxelCNNOptions : VoxelModelOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public VoxelCNNOptions()
    {
        VoxelResolution = 32;
        NumConvBlocks = 3;
        BaseFilters = 32;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public VoxelCNNOptions(VoxelCNNOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        VoxelResolution = other.VoxelResolution;
        NumConvBlocks = other.NumConvBlocks;
        BaseFilters = other.BaseFilters;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the number of 3D convolution blocks. Default: 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Named for convolution blocks rather than encoder blocks, which is what UNet3D counts —
    /// the structures differ, so the two models keep separate names.
    /// </para>
    /// </remarks>
    public int NumConvBlocks { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    public void Validate()
    {
        ValidateCore();
        Require(NumConvBlocks, nameof(NumConvBlocks));

        // Cross-field: each block halves the grid, so the resolution must survive them all.
        // Moved here from the constructor along with the two values it compares, and it keeps
        // ArgumentOutOfRangeException — a range relationship, not an unset dimension.
        int minResolution = 1 << NumConvBlocks;
        if (VoxelResolution < minResolution)
        {
            throw new ArgumentOutOfRangeException(
                nameof(VoxelResolution),
                $"VoxelResolution must be at least {minResolution} for {NumConvBlocks} convolutional blocks.");
        }
    }
}
