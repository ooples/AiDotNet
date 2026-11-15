namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The One-2-3-45 pipeline configuration (Liu et al. 2023, "One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds
/// without Per-Shape Optimization", arXiv 2306.16928). Every default is the official release's value
/// (One-2-3-45/One-2-3-45: <c>run.py</c>, <c>utils/zero123_utils.py</c>, <c>utils/utils.py</c>,
/// <c>elevation_estimate/</c> and <c>reconstruction/confs/one2345_lod0_val_demo.conf</c>).
/// </summary>
/// <remarks>
/// <para>
/// Angles are in degrees. A polar angle is measured from the +z axis (90 is the horizon), as in Zero123 and the
/// official <c>calc_pose</c>; a positive polar delta moves the camera down.
/// </para>
/// <para><b>For Beginners:</b> the defaults reproduce the published pipeline: 8 first-stage views, 4 nearby views
/// for each of them (32 source views), elevation estimated from the first view's 4 neighbours, and a 256³
/// marching-cubes mesh. Lower the resolutions and step counts for quick experiments.</para>
/// </remarks>
public class One2345Options
{
    /// <summary>DDIM steps for the first-stage Zero123 views (<c>ddim_steps=75</c>).</summary>
    public int Stage1Steps { get; set; } = 75;

    /// <summary>DDIM steps for the second-stage nearby views (<c>stage2_steps=50</c>).</summary>
    public int Stage2Steps { get; set; } = 50;

    /// <summary>Zero123 classifier-free guidance scale (<c>scale=3</c>).</summary>
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>Azimuth deltas of the 4 first-stage views at the input's elevation (<c>0 + 90·i</c>).</summary>
    public double[] Stage1AzimuthDeltas { get; set; } = new[] { 0.0, 90.0, 180.0, 270.0 };

    /// <summary>Azimuth deltas of the 4 first-stage views at the offset elevation (<c>30 + 90·i</c>).</summary>
    public double[] Stage1OffsetAzimuthDeltas { get; set; } = new[] { 30.0, 120.0, 210.0, 300.0 };

    /// <summary>The offset ring's polar delta (30): +30 (below) when the input's polar angle is at most
    /// <see cref="OffsetBelowMaxPolar"/>, −30 (above) otherwise.</summary>
    public double Stage1OffsetPolarDelta { get; set; } = 30.0;

    /// <summary>The largest estimated polar angle for which the offset ring goes below the input (75).</summary>
    public double OffsetBelowMaxPolar { get; set; } = 75.0;

    /// <summary>Polar deltas of the 4 second-stage views around each first-stage view (<c>delta_x_2</c>).</summary>
    public double[] Stage2PolarDeltas { get; set; } = new[] { -10.0, 10.0, 0.0, 0.0 };

    /// <summary>Azimuth deltas of the 4 second-stage views around each first-stage view (<c>delta_y_2</c>).</summary>
    public double[] Stage2AzimuthDeltas { get; set; } = new[] { 0.0, 0.0, -10.0, 10.0 };

    /// <summary>The input view's absolute azimuth in the reconstruction frame (<c>overlook_theta[0]</c> = 30).</summary>
    public double InputAzimuth { get; set; } = 30.0;

    /// <summary>The camera distance from the object center (<c>calc_pose radius</c> = 1.2).</summary>
    public double CameraRadius { get; set; } = 1.2;

    /// <summary>The focal length as a fraction of the image width (280 px at 256 px: <c>focal = 560/2</c>).</summary>
    public double FocalLengthRatio { get; set; } = 280.0 / 256.0;

    /// <summary>The near depth used for the scene bounding box (<c>near_far[0]</c> = 1.2 − 0.7).</summary>
    public double Near { get; set; } = 0.5;

    /// <summary>The far depth used for the scene bounding box (<c>near_far[1]</c> = 1.2 + 0.6).</summary>
    public double Far { get; set; } = 1.8;

    /// <summary>The bounding-sphere enlargement applied when normalizing the scene (<c>cal_scale_mat factor</c> = 1.1).</summary>
    public double BoundingBoxFactor { get; set; } = 1.1;

    /// <summary>Pixel values at or above this (in [0, 1]) are set to white before a second-stage view is predicted
    /// (<c>input_im_init[input_im_init &gt;= 253] = 255</c>).</summary>
    public double WhiteCleanupThreshold { get; set; } = 253.0 / 255.0;

    /// <summary>A pixel whose every channel exceeds this (in [0, 1]) is background for elevation matching
    /// (<c>img &gt; 245</c>).</summary>
    public double BackgroundThreshold { get; set; } = 245.0 / 255.0;

    /// <summary>The smallest polar angle of the coarse elevation search (<c>min_elev</c> = 30).</summary>
    public double ElevationSearchMinPolar { get; set; } = 30.0;

    /// <summary>The coarse search's exclusive upper bound (<c>max_elev</c> = 150).</summary>
    public double ElevationSearchMaxPolar { get; set; } = 150.0;

    /// <summary>The coarse search step (<c>interval</c> = 10).</summary>
    public double ElevationCoarseStep { get; set; } = 10.0;

    /// <summary>The fine search spans the coarse optimum ± this (10), exclusive of the upper end.</summary>
    public double ElevationFineHalfWidth { get; set; } = 10.0;

    /// <summary>The fine search step (1).</summary>
    public double ElevationFineStep { get; set; } = 1.0;

    /// <summary>The polar angle used when the views share no correspondences (<c>polar_angle = 90</c>).</summary>
    public double FallbackPolar { get; set; } = 90.0;

    /// <summary>A correspondence of the reference pair joins a third view's match when their first-image points lie
    /// closer than this, in pixels (<c>dist &lt; 1</c>).</summary>
    public double CorrespondenceTolerance { get; set; } = 1.0;

    /// <summary>The marching-cubes grid resolution over [−1, 1]³ (<c>--mesh_resolution</c> = 256).</summary>
    public int MeshResolution { get; set; } = 256;

    /// <summary>The square image size the text-to-image model renders for text-to-3D (256, Zero123's input).</summary>
    public int TextToImageSize { get; set; } = 256;

    /// <summary>SparseNeuS geometry volume voxels per axis (<c>vol_dims</c> = 96; a multiple of 8).</summary>
    public int VolumeResolution { get; set; } = 96;

    /// <summary>SparseNeuS compressed feature width (<c>d_pyramid_feature_compress</c> = 16).</summary>
    public int CompressedFeatureDim { get; set; } = 16;

    /// <summary>SparseNeuS geometry encoding width (<c>regnet_d_out</c> = 16).</summary>
    public int GeometryFeatureDim { get; set; } = 16;

    /// <summary>SparseNeuS SDF MLP hidden width (<c>hidden_dim</c> = 128).</summary>
    public int SdfHiddenDim { get; set; } = 128;

    /// <summary>SparseNeuS <c>num_sdf_layers</c> (4).</summary>
    public int SdfLayers { get; set; } = 4;

    /// <summary>SparseNeuS point-embedding frequencies (<c>multires</c> = 6).</summary>
    public int SdfFrequencies { get; set; } = 6;

    /// <summary>SparseNeuS initial variance (<c>variance_network.init_val</c> = 0.2).</summary>
    public double InitialVariance { get; set; } = 0.2;

    /// <summary>SparseNeuS rendering-network anti-alias pooling (<c>anti_alias_pooling = True</c>).</summary>
    public bool AntiAliasPooling { get; set; } = true;

    /// <summary>Points per SDF or colour query batch during mesh extraction.</summary>
    public int QueryChunkSize { get; set; } = 32768;

    /// <summary>Checks every value; called by the model's constructor.</summary>
    /// <exception cref="ArgumentException">A value is out of range or an angle list has the wrong length.</exception>
    public virtual void Validate()
    {
        if (Stage1Steps <= 0) throw new ArgumentOutOfRangeException(nameof(Stage1Steps));
        if (Stage2Steps <= 0) throw new ArgumentOutOfRangeException(nameof(Stage2Steps));
        RequireFour(Stage1AzimuthDeltas, nameof(Stage1AzimuthDeltas));
        RequireFour(Stage1OffsetAzimuthDeltas, nameof(Stage1OffsetAzimuthDeltas));
        RequireFour(Stage2PolarDeltas, nameof(Stage2PolarDeltas));
        RequireFour(Stage2AzimuthDeltas, nameof(Stage2AzimuthDeltas));
        if (!(CameraRadius > 0)) throw new ArgumentOutOfRangeException(nameof(CameraRadius));
        if (!(FocalLengthRatio > 0)) throw new ArgumentOutOfRangeException(nameof(FocalLengthRatio));
        if (!(Near > 0) || !(Far > Near)) throw new ArgumentOutOfRangeException(nameof(Far), "Need 0 < Near < Far.");
        if (!(BoundingBoxFactor > 0)) throw new ArgumentOutOfRangeException(nameof(BoundingBoxFactor));
        if (!(ElevationSearchMaxPolar > ElevationSearchMinPolar)) throw new ArgumentOutOfRangeException(nameof(ElevationSearchMaxPolar));
        if (!(ElevationCoarseStep > 0)) throw new ArgumentOutOfRangeException(nameof(ElevationCoarseStep));
        if (!(ElevationFineHalfWidth > 0)) throw new ArgumentOutOfRangeException(nameof(ElevationFineHalfWidth));
        if (!(ElevationFineStep > 0)) throw new ArgumentOutOfRangeException(nameof(ElevationFineStep));
        if (!(CorrespondenceTolerance > 0)) throw new ArgumentOutOfRangeException(nameof(CorrespondenceTolerance));
        if (MeshResolution < 2) throw new ArgumentOutOfRangeException(nameof(MeshResolution));
        if (TextToImageSize <= 0 || TextToImageSize % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(TextToImageSize), "The image size must be a positive multiple of 8.");
        if (VolumeResolution <= 0 || VolumeResolution % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(VolumeResolution), "The volume resolution must be a positive multiple of 8.");
        if (QueryChunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(QueryChunkSize));
    }

    private static void RequireFour(double[] values, string name)
    {
        if (values is null || values.Length != 4) throw new ArgumentException("Exactly four angles are needed.", name);
    }
}
