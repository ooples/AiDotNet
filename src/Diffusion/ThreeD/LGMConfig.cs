using AiDotNet.Diffusion.ThreeD.Meshing;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// Configuration of the LGM (Large Multi-View Gaussian Model) reconstructor.
/// </summary>
/// <remarks>
/// <para>
/// Every default is the "big" configuration of the official LGM release (3DTopia/LGM, <c>core/options.py</c>), the one
/// the paper's results and the released checkpoint use: four 256x256 input views, an asymmetric U-Net with down
/// channels (64, 128, 256, 512, 1024, 1024) and up channels (1024, 1024, 512, 256, 128), multi-view self-attention at
/// the three coarsest resolutions on both paths and in the middle block, and 128x128 splatter maps, so 4 x 128 x 128 =
/// 65,536 Gaussians per object.
/// </para>
/// <para>
/// Reference: Tang et al., "LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation", ECCV 2024,
/// arXiv 2402.05054.
/// </para>
/// <para><b>For Beginners:</b> leave these alone to get the published model. The test suite shrinks the channel lists
/// and the input size so the network fits in a unit test; the structure stays the same.</para>
/// </remarks>
public class LGMConfig
{
    /// <summary>Number of input views (the paper's four orthogonal views at azimuths 0, 90, 180 and 270 degrees).</summary>
    public int NumInputViews { get; set; } = 4;

    /// <summary>Side length the input views are resized to before the U-Net (<c>input_size</c>, 256).</summary>
    public int InputSize { get; set; } = 256;

    /// <summary>Channels of each U-Net down block (<c>down_channels</c>).</summary>
    public int[] DownChannels { get; set; } = { 64, 128, 256, 512, 1024, 1024 };

    /// <summary>Whether each down block applies multi-view self-attention (<c>down_attention</c>).</summary>
    public bool[] DownAttention { get; set; } = { false, false, false, true, true, true };

    /// <summary>Whether the middle block applies multi-view self-attention (<c>mid_attention</c>).</summary>
    public bool MidAttention { get; set; } = true;

    /// <summary>Channels of each U-Net up block (<c>up_channels</c>). Fewer up blocks than down blocks is what makes the
    /// U-Net asymmetric: the output stays at half the input resolution.</summary>
    public int[] UpChannels { get; set; } = { 1024, 1024, 512, 256, 128 };

    /// <summary>Whether each up block applies multi-view self-attention (<c>up_attention</c>).</summary>
    public bool[] UpAttention { get; set; } = { true, true, true, false, false };

    /// <summary>Residual blocks per down block (<c>layers_per_block</c>); each up block has one more.</summary>
    public int LayersPerBlock { get; set; } = 2;

    /// <summary>Attention heads of every multi-view self-attention block (16 in the official U-Net).</summary>
    public int AttentionHeads { get; set; } = 16;

    /// <summary>GroupNorm group count (32). A channel count it does not divide uses its largest divisor below it.</summary>
    public int NormGroups { get; set; } = 32;

    /// <summary>GroupNorm epsilon (1e-5).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    /// <summary>Scale applied to every residual sum (<c>skip_scale</c>, sqrt(0.5)).</summary>
    public double SkipScale { get; set; } = System.Math.Sqrt(0.5);

    /// <summary>Vertical field of view of the input cameras in degrees (<c>fovy</c>, 49.1).</summary>
    public double FieldOfViewY { get; set; } = 49.1;

    /// <summary>Distance of the input cameras from the object centre (<c>cam_radius</c>, 1.5).</summary>
    public double CameraRadius { get; set; } = 1.5;

    /// <summary>
    /// Elevation of the input cameras in degrees, in the kiui orbit-camera convention the official code uses
    /// (negative looks down from above). The paper renders and generates its four views at elevation 0.
    /// </summary>
    public double InputElevation { get; set; } = 0.0;

    /// <summary>Per-channel RGB mean the views are normalized with (the ImageNet mean, as in the official code).</summary>
    public double[] ImageMean { get; set; } = { 0.485, 0.456, 0.406 };

    /// <summary>Per-channel RGB standard deviation the views are normalized with (the ImageNet std).</summary>
    public double[] ImageStd { get; set; } = { 0.229, 0.224, 0.225 };

    /// <summary>Bound of the position activation: positions are clamped to [-bound, bound] (1).</summary>
    public double PositionBound { get; set; } = 1.0;

    /// <summary>Factor of the scale activation <c>factor * softplus(x)</c> (0.1).</summary>
    public double ScaleFactor { get; set; } = 0.1;

    /// <summary>
    /// Gaussians below this opacity are dropped from point clouds and meshes, as the official <c>save_ply</c> does
    /// (0.005).
    /// </summary>
    public double OpacityThreshold { get; set; } = 0.005;

    /// <summary>
    /// Settings of the Gaussian-to-mesh extraction (density grid and marching cubes, DreamGaussian's defaults: a 128^3
    /// grid in 16^3 blocks, density level 1).
    /// </summary>
    public GaussianMeshExtractionOptions MeshExtraction { get; set; } = new GaussianMeshExtractionOptions();

    /// <summary>Gets a new configuration holding the paper defaults.</summary>
    public static LGMConfig Default => new LGMConfig();
}
