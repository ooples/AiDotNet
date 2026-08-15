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
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
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

    /// <summary>Height of the TPS localization-network input.</summary>
    public int TpsInputHeight { get; set; } = 32;
    /// <summary>Width of the TPS localization-network input.</summary>
    public int TpsInputWidth { get; set; } = 64;
    /// <summary>Even number of boundary control points predicted by the TPS head.</summary>
    public int TpsControlPointCount { get; set; } = 20;
    /// <summary>Horizontal inset of the target control-point rectangle.</summary>
    public double TpsMarginX { get; set; } = 0.05;
    /// <summary>Vertical inset of the target control-point rectangle.</summary>
    public double TpsMarginY { get; set; } = 0.05;

    /// <summary>Validates that the three-stage SVTR-Tiny topology is internally consistent.</summary>
    public void ValidateReferenceTopology()
    {
        if (EmbedDimensions.Length != 3 || StageDepths.Length != 3 || StageHeads.Length != 3)
            throw new ArgumentException("SVTR requires exactly three stages.");
        for (int i = 0; i < 3; i++)
        {
            if (EmbedDimensions[i] <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(EmbedDimensions), $"Stage {i} embedding dimension must be positive.");
            if (StageDepths[i] <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(StageDepths), $"Stage {i} depth must be positive.");
            if (StageHeads[i] <= 0 || EmbedDimensions[i] % StageHeads[i] != 0)
                throw new ArgumentException($"Stage {i} embedding dimension must be divisible by its head count.");
        }
        if (InputHeight <= 0 || InputHeight % 16 != 0)
            throw new ArgumentOutOfRangeException(
                nameof(InputHeight), "Input height must be positive and divisible by 16.");
        if (InputWidth <= 0 || InputWidth % 4 != 0)
            throw new ArgumentOutOfRangeException(
                nameof(InputWidth), "Input width must be positive and divisible by 4.");
        if (OutputCharacterPositions != InputWidth / 4)
            throw new ArgumentException(
                "OutputCharacterPositions must equal InputWidth / 4.",
                nameof(OutputCharacterPositions));
        int totalBlocks = StageDepths.Sum();
        if (LocalMixingBlocks < 0 || LocalMixingBlocks > totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(LocalMixingBlocks));
        if (LocalWindowHeight <= 0 || LocalWindowWidth <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(LocalWindowHeight), "Local mixing windows must be positive.");
        if (DropPathRate < 0 || DropPathRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(DropPathRate));
        if (LastStageDropout < 0 || LastStageDropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(LastStageDropout));
        if (OutputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(OutputChannels));
        if (UseTpsRectification)
        {
            if (TpsInputHeight <= 0 || TpsInputWidth <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(TpsInputHeight), "TPS localization dimensions must be positive.");
            if (TpsControlPointCount < 4 || TpsControlPointCount % 2 != 0)
                throw new ArgumentOutOfRangeException(
                    nameof(TpsControlPointCount), "TPS requires an even control-point count of at least four.");
            if (TpsMarginX < 0 || TpsMarginX >= 0.5 || TpsMarginY < 0 || TpsMarginY >= 0.5)
                throw new ArgumentOutOfRangeException(
                    nameof(TpsMarginX), "TPS margins must be in [0, 0.5).");
        }
    }
}
