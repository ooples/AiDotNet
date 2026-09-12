namespace AiDotNet.Models.Options;

/// <summary>
/// Shared hyperparameters for the volumetric models that work on a 3D voxel grid.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A voxel is a pixel in three dimensions. These models take a cube of
/// voxels as input, so the two settings they share are how finely that cube is divided and how
/// wide the first layer is.
/// </para>
/// <para>
/// The block count is not here: VoxelCNN counts convolution blocks and UNet3D counts encoder
/// blocks, and those names describe genuinely different structures, so each keeps its own.
/// </para>
/// </remarks>
public abstract class VoxelModelOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the number of voxels along each edge of the input cube. Default: 32.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 32 means a 32x32x32 cube — around 33,000 cells. Doubling this
    /// multiplies the work by eight, which is why volumetric models stay coarse.</para>
    /// </remarks>
    public int VoxelResolution { get; set; }

    /// <summary>
    /// Gets or sets the number of filters in the first convolution layer. Default: 32.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different patterns the first layer looks for. Later
    /// layers typically double this as the grid shrinks.</para>
    /// </remarks>
    public int BaseFilters { get; set; }

    /// <summary>
    /// Gets or sets the learning rate used when the model creates its own optimizer.
    /// Default: 1e-3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How big a correction the model makes after each training batch.
    /// Too large and training thrashes; too small and it barely moves.
    /// </para>
    /// <para>
    /// Both volumetric models previously built a BARE optimizer -- `new AdamOptimizer&lt;...&gt;(this)`
    /// -- so they trained at the optimizer's own default with no way for a caller to say otherwise
    /// short of constructing an optimizer by hand. 1e-3 is Adam's default and therefore preserves
    /// the behaviour those models already had; it is a documented library default rather than a
    /// value from either paper, and supplying an optimizer explicitly still bypasses it.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Throws if a dimension every volumetric model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative, which means the derived options
    /// class did not assign its published defaults.
    /// </exception>
    protected void ValidateCore()
    {
        Require(VoxelResolution, nameof(VoxelResolution));
        Require(BaseFilters, nameof(BaseFilters));
        Require(LearningRate, nameof(LearningRate));
    }
}
