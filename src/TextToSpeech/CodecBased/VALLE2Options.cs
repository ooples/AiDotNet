namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for VALL-E 2 (Chen et al. 2024, "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot
/// Text to Speech Synthesizers").</summary>
/// <remarks>
/// <para>
/// VALL-E 2 keeps VALL-E's Transformers (§4.1.1: "both the AR model and the NAR models employ the same Transformer
/// architecture in VALL-E"; <see cref="VALLEOptions"/>'s sizes), EnCodec at 6 kbps and AdamW warmed up over 32k updates
/// with linear decay (§4.1.1), and adds grouped code modeling (§3.1) and repetition-aware sampling (§3.4.1, Algorithm 1:
/// window K = 10, threshold 0.1). It decodes with Vocos's released EnCodec model.
/// </para>
/// <para>
/// Where the paper searches or leaves a value open: the group size defaults to 2 (it trains 1, 2, 4 and 8; 1 and 2 score
/// best, 2 at half the AR length); top-p defaults to 0 (it searches 0.0–0.8 and finds small values most robust); the
/// NAR's acoustic condition is at most half the utterance, otherwise a random 3–30 seconds (§4.1.1, "the maximum of half
/// of the current utterance with a random value from 3s to 30s"); the text is espeak-style phonemes, as VALL-E's
/// reproduction reads them (the paper's BPE model is not public); learned position tables hold
/// <see cref="MaxCodePositions"/> code groups.
/// </para>
/// <para><b>For Beginners:</b> These options set VALL-E 2's sizes, how many codes it predicts per step, how it samples,
/// and how it is trained. The defaults follow the paper.</para>
/// </remarks>
public class VALLE2Options : VALLEOptions
{
    /// <summary>Creates the paper's options.</summary>
    public VALLE2Options()
    {
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public VALLE2Options(VALLE2Options other)
        : base(other)
    {
        GroupSize = other.GroupSize;
        TopP = other.TopP;
        RepetitionWindow = other.RepetitionWindow;
        RepetitionThreshold = other.RepetitionThreshold;
        MinPromptSeconds = other.MinPromptSeconds;
        MaxPromptSeconds = other.MaxPromptSeconds;
        MaxCodePositions = other.MaxCodePositions;
        DecoderDim = other.DecoderDim;
        DecoderIntermediateDim = other.DecoderIntermediateDim;
        DecoderLayers = other.DecoderLayers;
    }

    /// <summary>First-codebook codes per AR step (2; the paper trains 1, 2, 4 and 8, §4.1.1).</summary>
    public int GroupSize { get; set; } = 2;

    /// <summary>Nucleus-sampling mass (0, keeping the most likely code; the paper searches 0.0–0.8).</summary>
    public double TopP { get; set; }

    /// <summary>Codes over which the repetition ratio is counted (K = 10, §4.1.2).</summary>
    public int RepetitionWindow { get; set; } = 10;

    /// <summary>Repetition ratio above which a code is resampled at random (t_r = 0.1, §4.1.2).</summary>
    public double RepetitionThreshold { get; set; } = 0.1;

    /// <summary>Shortest NAR acoustic condition in training, in seconds (3, §4.1.1).</summary>
    public double MinPromptSeconds { get; set; } = 3.0;

    /// <summary>Longest NAR acoustic condition in training, in seconds (30, §4.1.1).</summary>
    public double MaxPromptSeconds { get; set; } = 30.0;

    /// <summary>Rows of the learned code-position tables (4096 frames, about 55 seconds at 75 Hz; the paper does not say).</summary>
    public int MaxCodePositions { get; set; } = 4096;

    /// <summary>Width of the Vocos decoder's backbone (384, charactr/vocos-encodec-24khz). Its bandwidth classes are 2, 4,
    /// 8 and 16 codebooks (1.5–12 kbps), so <see cref="VALLEOptions.NumCodebooks"/> must be one of them.</summary>
    public int DecoderDim { get; set; } = 384;

    /// <summary>The Vocos decoder's ConvNeXt intermediate width (1152).</summary>
    public int DecoderIntermediateDim { get; set; } = 1152;

    /// <summary>The Vocos decoder's ConvNeXt blocks (8).</summary>
    public int DecoderLayers { get; set; } = 8;
}
