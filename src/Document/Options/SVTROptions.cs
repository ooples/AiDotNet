using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the SVTR document model.
/// </summary>
public class SVTROptions : DocumentNeuralNetworkOptions
{
    public SVTROptions()
    {
    }

    public SVTROptions(SVTROptions other)
    {
        ArgumentNullException.ThrowIfNull(other);
        InputHeight = other.InputHeight;
        InputWidth = other.InputWidth;
        EmbedDimensions = (int[])other.EmbedDimensions.Clone();
        StageDepths = (int[])other.StageDepths.Clone();
        StageHeads = (int[])other.StageHeads.Clone();
        LocalMixingBlocks = other.LocalMixingBlocks;
        LocalWindowHeight = other.LocalWindowHeight;
        LocalWindowWidth = other.LocalWindowWidth;
        OutputCharacterPositions = other.OutputCharacterPositions;
        OutputChannels = other.OutputChannels;
        LastStageDropout = other.LastStageDropout;
        DropPathRate = other.DropPathRate;
        UseTpsRectification = other.UseTpsRectification;
        TpsInputHeight = other.TpsInputHeight;
        TpsInputWidth = other.TpsInputWidth;
        TpsControlPointCount = other.TpsControlPointCount;
        TpsMarginX = other.TpsMarginX;
        TpsMarginY = other.TpsMarginY;
    }

    /// <summary>Post-TPS input height used by the reference SVTR-Tiny network.</summary>
    public int InputHeight { get; set; } = 32;

    /// <summary>Post-TPS input width used by the reference SVTR-Tiny network.</summary>
    public int InputWidth { get; set; } = 100;

    /// <summary>Stage embedding dimensions.</summary>
    public int[] EmbedDimensions { get; set; } = [64, 128, 256];

    /// <summary>Component-mixing block count in each stage.</summary>
    public int[] StageDepths { get; set; } = [3, 6, 3];

    /// <summary>Attention head count in each stage.</summary>
    public int[] StageHeads { get; set; } = [2, 4, 8];

    /// <summary>Number of local-mixing blocks before global mixing begins.</summary>
    public int LocalMixingBlocks { get; set; } = 6;

    /// <summary>Local component-mixing window height.</summary>
    public int LocalWindowHeight { get; set; } = 7;

    /// <summary>Local component-mixing window width.</summary>
    public int LocalWindowWidth { get; set; } = 11;

    /// <summary>Width of the recognition sequence after height collapse.</summary>
    public int OutputCharacterPositions { get; set; } = 25;

    /// <summary>Channel width presented to the CTC head.</summary>
    public int OutputChannels { get; set; } = 192;

    /// <summary>Dropout applied immediately before the CTC head.</summary>
    public double LastStageDropout { get; set; } = 0.1;

    /// <summary>Maximum stochastic-depth probability, linearly scheduled across 12 blocks.</summary>
    public double DropPathRate { get; set; } = 0.1;

    /// <summary>Whether the training recipe uses its optional TPS rectifier before SVTRNet.</summary>
    public bool UseTpsRectification { get; set; } = true;

    public int TpsInputHeight { get; set; } = 32;
    public int TpsInputWidth { get; set; } = 64;
    public int TpsControlPointCount { get; set; } = 20;
    public double TpsMarginX { get; set; } = 0.05;
    public double TpsMarginY { get; set; } = 0.05;

    /// <summary>Validates that the three-stage SVTR-Tiny topology is internally consistent.</summary>
    public void ValidateReferenceTopology()
    {
        if (EmbedDimensions.Length != 3 || StageDepths.Length != 3 || StageHeads.Length != 3)
            throw new ArgumentException("SVTR requires exactly three stages.");
        if (!EmbedDimensions.SequenceEqual([64, 128, 256]) ||
            !StageDepths.SequenceEqual([3, 6, 3]) ||
            !StageHeads.SequenceEqual([2, 4, 8]))
            throw new ArgumentException("SVTR-Tiny requires stage depths [3,6,3] (12 blocks total).");
        for (int i = 0; i < 3; i++)
        {
            if (EmbedDimensions[i] <= 0 || StageHeads[i] <= 0 || EmbedDimensions[i] % StageHeads[i] != 0)
                throw new ArgumentException($"Stage {i} embedding dimension must be divisible by its head count.");
        }
        if (InputHeight != 32 || InputWidth != 100 || OutputCharacterPositions != 25)
            throw new ArgumentException("SVTR-Tiny reference geometry is 32x100 with 25 output positions.");
        if (LocalMixingBlocks != 6 || LocalWindowHeight != 7 || LocalWindowWidth != 11)
            throw new ArgumentException("SVTR-Tiny requires six 7x11 local-mixing blocks followed by six global blocks.");
        if (DropPathRate < 0 || DropPathRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(DropPathRate));
        if (UseTpsRectification &&
            (TpsInputHeight != 32 || TpsInputWidth != 64 || TpsControlPointCount != 20))
            throw new ArgumentException("SVTR STN_ON requires a 32x64 localization input and 20 TPS control points.");
    }
}
