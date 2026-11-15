namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The LoFTR matcher configuration (Sun et al. 2021, "LoFTR: Detector-Free Local Feature Matching with
/// Transformers", CVPR). Every default is the value One-2-3-45 runs with: the vendored
/// <c>elevation_estimate/loftr/utils/cvpr_ds_config.py</c> <c>default_cfg</c> (dual-softmax, ResNet-FPN 8/2) with
/// <c>temp_bug_fix = True</c>, which is the configuration of the <c>indoor_ds_new.ckpt</c> weights it loads, and the
/// 480 × 480 grayscale input of <c>elev_est_api.py</c>.
/// </summary>
/// <remarks>
/// <para>
/// The coarse transformer width is <c>BlockDims[2]</c> and the fine width <c>BlockDims[0]</c>: the reference
/// requires <c>coarse.d_model == block_dims[2]</c> and <c>fine.d_model == block_dims[0]</c>, so they are not
/// separate settings here.
/// </para>
/// <para><b>For Beginners:</b> keep the defaults to load the official checkpoint. Smaller widths and fewer layers
/// give a fast matcher for experiments, but then the official weights no longer fit.</para>
/// </remarks>
public class LoFTROptions
{
    /// <summary>Output channels of the 7 × 7 stem convolution (<c>RESNETFPN.INITIAL_DIM = 128</c>); must equal
    /// <c>BlockDims[0]</c>, because the first stage has stride 1 and no projection shortcut.</summary>
    public int InitialDim { get; set; } = 128;

    /// <summary>Channels of the three ResNet stages at 1/2, 1/4 and 1/8 (<c>RESNETFPN.BLOCK_DIMS = [128, 196, 256]</c>).</summary>
    public int[] BlockDims { get; set; } = new[] { 128, 196, 256 };

    /// <summary>Attention heads of the coarse transformer (<c>COARSE.NHEAD = 8</c>).</summary>
    public int CoarseHeads { get; set; } = 8;

    /// <summary>Self/cross layer pairs of the coarse transformer (<c>COARSE.LAYER_NAMES = ['self', 'cross'] × 4</c>).</summary>
    public int CoarseLayerPairs { get; set; } = 4;

    /// <summary>Attention heads of the fine transformer (<c>FINE.NHEAD = 8</c>).</summary>
    public int FineHeads { get; set; } = 8;

    /// <summary>Self/cross layer pairs of the fine transformer (<c>FINE.LAYER_NAMES = ['self', 'cross'] × 1</c>).</summary>
    public int FineLayerPairs { get; set; } = 1;

    /// <summary>The fine window side (<c>FINE_WINDOW_SIZE = 5</c>); odd.</summary>
    public int FineWindowSize { get; set; } = 5;

    /// <summary>Whether the fine windows are merged with the projected coarse features
    /// (<c>FINE_CONCAT_COARSE_FEAT = True</c>).</summary>
    public bool FineConcatCoarseFeature { get; set; } = true;

    /// <summary>The coarse confidence threshold (<c>MATCH_COARSE.THR = 0.2</c>).</summary>
    public double MatchThreshold { get; set; } = 0.2;

    /// <summary>Coarse cells removed at every image border (<c>MATCH_COARSE.BORDER_RM = 2</c>).</summary>
    public int BorderRemoval { get; set; } = 2;

    /// <summary>The dual-softmax temperature (<c>MATCH_COARSE.DSMAX_TEMPERATURE = 0.1</c>).</summary>
    public double DualSoftmaxTemperature { get; set; } = 0.1;

    /// <summary>Whether the positional encoding uses the corrected frequencies (<c>COARSE.TEMP_BUG_FIX = True</c>,
    /// the <c>*_new</c> checkpoints); false reproduces the original release's encoding.</summary>
    public bool PositionEncodingBugFix { get; set; } = true;

    /// <summary>The square side images are resized to before matching (480 in One-2-3-45's <c>elev_est_api.py</c>);
    /// keypoints are scaled back to the input size. 0 keeps the input size, which must then be a multiple of 8.</summary>
    public int InputSize { get; set; } = 480;

    /// <summary>Seeds the initialization of an untrained matcher; null draws a secure random seed.</summary>
    public int? Seed { get; set; }

    /// <summary>Checks the configuration.</summary>
    /// <exception cref="ArgumentException">A value is out of range.</exception>
    public virtual void Validate()
    {
        if (BlockDims is null || BlockDims.Length != 3 || BlockDims.Any(d => d <= 0))
            throw new ArgumentException("Exactly three positive stage widths are needed.", nameof(BlockDims));
        if (InitialDim != BlockDims[0])
            throw new ArgumentException("The stem width must equal the first stage width (stride-1 stage, no shortcut projection).", nameof(InitialDim));
        if (CoarseHeads <= 0 || BlockDims[2] % CoarseHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(CoarseHeads), "The coarse width must be divisible by the head count.");
        if (FineHeads <= 0 || BlockDims[0] % FineHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(FineHeads), "The fine width must be divisible by the head count.");
        if (BlockDims[2] % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(BlockDims), "The coarse width must be a multiple of 4 for the sine encoding.");
        if (CoarseLayerPairs < 0) throw new ArgumentOutOfRangeException(nameof(CoarseLayerPairs));
        if (FineLayerPairs < 0) throw new ArgumentOutOfRangeException(nameof(FineLayerPairs));
        if (FineWindowSize < 3 || FineWindowSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(FineWindowSize), "The fine window must be odd and at least 3.");
        if (!(MatchThreshold >= 0) || MatchThreshold >= 1) throw new ArgumentOutOfRangeException(nameof(MatchThreshold));
        if (BorderRemoval < 0) throw new ArgumentOutOfRangeException(nameof(BorderRemoval));
        if (!(DualSoftmaxTemperature > 0)) throw new ArgumentOutOfRangeException(nameof(DualSoftmaxTemperature));
        if (InputSize < 0 || InputSize % 8 != 0)
            throw new ArgumentOutOfRangeException(nameof(InputSize), "The input size must be 0 or a positive multiple of 8.");
    }
}
