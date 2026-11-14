using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// The Point Transformer V3 backbone settings shared by <see cref="PointTransformerV3Options"/> and
/// <see cref="SonataOptions"/>.
/// </summary>
/// <remarks>
/// <para>
/// Each array has one entry per stage. The encoder arrays describe stages from finest to coarsest;
/// the decoder arrays describe stages from finest to coarsest as well, one fewer than the encoder,
/// exactly as Pointcept's <c>dec_*</c> tuples are written.
/// </para>
/// <para>
/// <b>For Beginners:</b> These settings decide how big the network is (how many blocks, how wide
/// each stage is, how many attention heads it has) and how it groups points (the voxel size, the
/// patch size, the space-filling curves used to order the points). The defaults of each derived
/// options class are the values from the paper it reproduces.
/// </para>
/// </remarks>
public abstract class PointTransformerV3EncoderOptions : SegmentationModelOptions
{
    /// <summary>Initializes the settings with the PTv3 m1 base values.</summary>
    protected PointTransformerV3EncoderOptions()
    {
    }

    /// <summary>Copies every backbone setting from <paramref name="other"/>.</summary>
    /// <param name="other">The options to copy.</param>
    protected PointTransformerV3EncoderOptions(PointTransformerV3EncoderOptions other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        InputChannels = other.InputChannels;
        GridSize = other.GridSize;
        Strides = (int[])other.Strides.Clone();
        EncoderDepths = (int[])other.EncoderDepths.Clone();
        EncoderChannels = (int[])other.EncoderChannels.Clone();
        EncoderNumHeads = (int[])other.EncoderNumHeads.Clone();
        EncoderPatchSizes = (int[])other.EncoderPatchSizes.Clone();
        DecoderDepths = (int[])other.DecoderDepths.Clone();
        DecoderChannels = (int[])other.DecoderChannels.Clone();
        DecoderNumHeads = (int[])other.DecoderNumHeads.Clone();
        DecoderPatchSizes = (int[])other.DecoderPatchSizes.Clone();
        MlpRatio = other.MlpRatio;
        DropPathRate = other.DropPathRate;
        EnableConditionalPositionalEncoding = other.EnableConditionalPositionalEncoding;
        PreNorm = other.PreNorm;
        ShuffleOrders = other.ShuffleOrders;
        SerializationOrders = (PointSerializationOrder[])other.SerializationOrders.Clone();
    }

    /// <summary>
    /// Feature channels per point after the xyz coordinates. Default: 6 (colour and normal).
    /// </summary>
    public int InputChannels { get; set; } = 6;

    /// <summary>Voxel edge length in metres used to grid the coordinates. Default: 0.02.</summary>
    public double GridSize { get; set; } = 0.02;

    /// <summary>Grid-pooling stride between consecutive encoder stages. Default: (2, 2, 2, 2).</summary>
    public int[] Strides { get; set; } = { 2, 2, 2, 2 };

    /// <summary>Blocks per encoder stage. Default: (2, 2, 2, 6, 2).</summary>
    public int[] EncoderDepths { get; set; } = { 2, 2, 2, 6, 2 };

    /// <summary>Channels per encoder stage. Default: (32, 64, 128, 256, 512).</summary>
    public int[] EncoderChannels { get; set; } = { 32, 64, 128, 256, 512 };

    /// <summary>Attention heads per encoder stage. Default: (2, 4, 8, 16, 32).</summary>
    public int[] EncoderNumHeads { get; set; } = { 2, 4, 8, 16, 32 };

    /// <summary>Serialized patch size per encoder stage. Default: 1024 for every stage.</summary>
    public int[] EncoderPatchSizes { get; set; } = { 1024, 1024, 1024, 1024, 1024 };

    /// <summary>Blocks per decoder stage, finest first. Default: (2, 2, 2, 2).</summary>
    public int[] DecoderDepths { get; set; } = { 2, 2, 2, 2 };

    /// <summary>Channels per decoder stage, finest first. Default: (64, 64, 128, 256).</summary>
    public int[] DecoderChannels { get; set; } = { 64, 64, 128, 256 };

    /// <summary>Attention heads per decoder stage, finest first. Default: (4, 4, 8, 16).</summary>
    public int[] DecoderNumHeads { get; set; } = { 4, 4, 8, 16 };

    /// <summary>Serialized patch size per decoder stage, finest first. Default: 1024 for every stage.</summary>
    public int[] DecoderPatchSizes { get; set; } = { 1024, 1024, 1024, 1024 };

    /// <summary>Ratio of the MLP hidden width to the block width. Default: 4.</summary>
    public double MlpRatio { get; set; } = 4.0;

    /// <summary>
    /// Stochastic-depth rate of the deepest block; earlier blocks use a linear ramp from 0. Default: 0.3.
    /// </summary>
    public double DropPathRate { get; set; } = 0.3;

    /// <summary>Whether every block begins with the sparse-convolution xCPE. Default: true.</summary>
    public bool EnableConditionalPositionalEncoding { get; set; } = true;

    /// <summary>Whether blocks normalise before attention and MLP (pre-norm). Default: true.</summary>
    public bool PreNorm { get; set; } = true;

    /// <summary>Whether training shuffles the serialization orders after every pooling. Default: true.</summary>
    public bool ShuffleOrders { get; set; } = true;

    /// <summary>
    /// The serialization orders the blocks cycle through. Default: z, z-trans, hilbert, hilbert-trans.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Reorder, repeat or drop entries to change which curves the blocks use.
    /// For a curve of your own, implement <see cref="ISpaceFillingCurve"/>, register it with
    /// <see cref="SpaceFillingCurves.Register(ISpaceFillingCurve)"/> and pass its name to
    /// <see cref="PointTransformerV3Encoder{T}"/>.
    /// </remarks>
    public PointSerializationOrder[] SerializationOrders { get; set; } =
    {
        PointSerializationOrder.ZOrder, PointSerializationOrder.ZOrderTransposed,
        PointSerializationOrder.Hilbert, PointSerializationOrder.HilbertTransposed
    };

    /// <summary>Throws when the backbone settings cannot build a network.</summary>
    /// <param name="encoderOnly">Whether the decoder arrays are ignored.</param>
    protected void ValidateBackbone(bool encoderOnly)
    {
        if (InputChannels <= 0) throw new ArgumentException("InputChannels must be positive.");
        if (!(GridSize > 0) || double.IsInfinity(GridSize)) throw new ArgumentException("GridSize must be positive and finite.");
        if (Strides is null || EncoderDepths is null || EncoderChannels is null || EncoderNumHeads is null
            || EncoderPatchSizes is null || DecoderDepths is null || DecoderChannels is null || DecoderNumHeads is null
            || DecoderPatchSizes is null || SerializationOrders is null)
            throw new ArgumentException("Backbone arrays must not be null.");
        int stages = EncoderDepths.Length;
        if (stages == 0) throw new ArgumentException("EncoderDepths needs at least one stage.");
        if (EncoderChannels.Length != stages || EncoderNumHeads.Length != stages || EncoderPatchSizes.Length != stages)
            throw new ArgumentException("EncoderChannels, EncoderNumHeads and EncoderPatchSizes need one entry per stage.");
        if (Strides.Length != stages - 1) throw new ArgumentException("Strides needs one entry between each pair of stages.");
        for (int s = 0; s < stages; s++)
        {
            if (EncoderDepths[s] < 0 || EncoderChannels[s] <= 0 || EncoderPatchSizes[s] <= 0 || EncoderNumHeads[s] <= 0
                || EncoderChannels[s] % EncoderNumHeads[s] != 0)
                throw new ArgumentException($"Encoder stage {s} needs positive sizes and channels divisible by heads.");
        }

        if (!encoderOnly)
        {
            if (DecoderDepths.Length != stages - 1 || DecoderChannels.Length != stages - 1
                || DecoderNumHeads.Length != stages - 1 || DecoderPatchSizes.Length != stages - 1)
                throw new ArgumentException("Decoder arrays need one entry per encoder stage but the last.");
            for (int s = 0; s < stages - 1; s++)
            {
                if (DecoderDepths[s] < 0 || DecoderChannels[s] <= 0 || DecoderPatchSizes[s] <= 0 || DecoderNumHeads[s] <= 0
                    || DecoderChannels[s] % DecoderNumHeads[s] != 0)
                    throw new ArgumentException($"Decoder stage {s} needs positive sizes and channels divisible by heads.");
            }
        }

        if (!(MlpRatio > 0)) throw new ArgumentException("MlpRatio must be positive.");
        if (DropPathRate < 0 || DropPathRate >= 1) throw new ArgumentException("DropPathRate must be in [0, 1).");
        if (SerializationOrders.Length == 0) throw new ArgumentException("SerializationOrders needs at least one curve.");
        foreach (var order in SerializationOrders) SpaceFillingCurves.Get(order);
        ValidateSegmentationCore();
    }
}
