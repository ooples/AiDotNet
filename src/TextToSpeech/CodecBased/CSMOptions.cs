namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for <see cref="CSM{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are CSM 1B (Sesame 2025, "Crossing the uncanny valley of conversational voice"; SesameAILabs/csm
/// <c>models.py</c>, <c>generator.py</c>; <c>sesame/csm-1b</c>): a Llama 3.2 1B backbone (width 2048, 16 layers, 32 heads
/// over 8 key-value heads, SwiGLU 8192) and a Llama 100M audio decoder (width 1024, 4 layers, 8 heads over 2, SwiGLU 8192),
/// both with rotary θ 500000 and Llama 3 scaling ×32, over Llama 3's 128,256 text tokens and Mimi's 32 codebooks of 2048
/// codes (2051 per codebook in the model's tables) at 12.5 Hz and 24 kHz.</para>
/// <para>Generation follows <c>generator.py</c>: top-k 50 at temperature 0.9 for every codebook, up to 90 seconds, a frame
/// of zeros ending the speech, and the output watermarked with SilentCipher under the release's public key at 36 dB.
/// Training follows the blog where it is explicit — codebook 0 learned on every frame, the audio decoder on a random
/// sixteenth of them — and Hugging Face's CSM port, written with Sesame, otherwise: the backbone's next-frame loss on
/// codebook 0 plus the decoder's loss on codebooks 1–31, under the Trainer's default AdamW (no recipe was released).</para>
/// <para><b>For Beginners:</b> These settings size CSM's two Transformers and its audio codec, control how it samples each
/// frame of sound, and set how it trains; the defaults reproduce the released 1B model.</para>
/// </remarks>
public class CSMOptions : TtsModelOptions
{
    /// <summary>Creates the CSM 1B configuration.</summary>
    public CSMOptions()
    {
        SampleRate = 24000;
        HopSize = 1920;
        HiddenDim = 2048;
        VocabSize = 128256;
        MaxTextLength = 2048;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
        DropoutRate = 0.0;
    }

    /// <summary>Copies another instance.</summary>
    public CSMOptions(CSMOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        BackboneLayers = other.BackboneLayers;
        BackboneHeads = other.BackboneHeads;
        BackboneKeyValueHeads = other.BackboneKeyValueHeads;
        BackboneIntermediate = other.BackboneIntermediate;
        DecoderHidden = other.DecoderHidden;
        DecoderLayers = other.DecoderLayers;
        DecoderHeads = other.DecoderHeads;
        DecoderKeyValueHeads = other.DecoderKeyValueHeads;
        DecoderIntermediate = other.DecoderIntermediate;
        RmsEpsilon = other.RmsEpsilon;
        RopeTheta = other.RopeTheta;
        RopeScalingFactor = other.RopeScalingFactor;
        RopeLowFrequencyFactor = other.RopeLowFrequencyFactor;
        RopeHighFrequencyFactor = other.RopeHighFrequencyFactor;
        RopeOriginalMaxPositions = other.RopeOriginalMaxPositions;
        AudioVocabSize = other.AudioVocabSize;
        AudioCodebooks = other.AudioCodebooks;
        MaxSequenceLength = other.MaxSequenceLength;
        Temperature = other.Temperature;
        TopK = other.TopK;
        MaxAudioLengthMs = other.MaxAudioLengthMs;
        ConstrainToCodebook = other.ConstrainToCodebook;
        Watermark = other.Watermark;
        WatermarkModelPath = other.WatermarkModelPath;
        WatermarkKey = (int[])other.WatermarkKey.Clone();
        WatermarkSdr = other.WatermarkSdr;
        MimiDimension = other.MimiDimension;
        MimiFilters = other.MimiFilters;
        MimiRatios = (int[])other.MimiRatios.Clone();
        MimiTransformerLayers = other.MimiTransformerLayers;
        MimiHeads = other.MimiHeads;
        MimiFeedForward = other.MimiFeedForward;
        MimiContext = other.MimiContext;
        MimiCodebookSize = other.MimiCodebookSize;
        MimiCodebookDim = other.MimiCodebookDim;
        DecoderFrameFraction = other.DecoderFrameFraction;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        TrainingSteps = other.TrainingSteps;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>The backbone's layers (16).</summary>
    public int BackboneLayers { get; set; } = 16;

    /// <summary>The backbone's attention heads (32).</summary>
    public int BackboneHeads { get; set; } = 32;

    /// <summary>The backbone's key-value heads (8).</summary>
    public int BackboneKeyValueHeads { get; set; } = 8;

    /// <summary>The backbone's SwiGLU width (8192).</summary>
    public int BackboneIntermediate { get; set; } = 8192;

    /// <summary>The audio decoder's width (1024).</summary>
    public int DecoderHidden { get; set; } = 1024;

    /// <summary>The audio decoder's layers (4).</summary>
    public int DecoderLayers { get; set; } = 4;

    /// <summary>The audio decoder's attention heads (8).</summary>
    public int DecoderHeads { get; set; } = 8;

    /// <summary>The audio decoder's key-value heads (2).</summary>
    public int DecoderKeyValueHeads { get; set; } = 2;

    /// <summary>The audio decoder's SwiGLU width (8192).</summary>
    public int DecoderIntermediate { get; set; } = 8192;

    /// <summary>RMSNorm epsilon (1e-5).</summary>
    public double RmsEpsilon { get; set; } = 1e-5;

    /// <summary>Rotary base θ (500000).</summary>
    public double RopeTheta { get; set; } = 500000;

    /// <summary>Llama 3 rotary scaling factor (32; 0 for none).</summary>
    public double RopeScalingFactor { get; set; } = 32;

    /// <summary>Llama 3 rotary scaling's low-frequency factor (1).</summary>
    public double RopeLowFrequencyFactor { get; set; } = 1;

    /// <summary>Llama 3 rotary scaling's high-frequency factor (4).</summary>
    public double RopeHighFrequencyFactor { get; set; } = 4;

    /// <summary>Llama 3 rotary scaling's original context (8192).</summary>
    public int RopeOriginalMaxPositions { get; set; } = 8192;

    /// <summary>Entries per codebook in the model's audio tables and heads (2051: Mimi's 2048 and three more).</summary>
    public int AudioVocabSize { get; set; } = 2051;

    /// <summary>Codebooks per frame (32).</summary>
    public int AudioCodebooks { get; set; } = 32;

    /// <summary>The backbone's longest sequence of frames, context and generation together (2048).</summary>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Sampling temperature (0.9).</summary>
    public double Temperature { get; set; } = 0.9;

    /// <summary>Top-k sampling's k (50).</summary>
    public int TopK { get; set; } = 50;

    /// <summary>The longest speech generated, in milliseconds (90000: 1125 frames of 80 ms).</summary>
    public double MaxAudioLengthMs { get; set; } = 90000;

    /// <summary>Whether sampling is restricted to codes Mimi can decode (below <see cref="MimiCodebookSize"/>; false). Not
    /// part of the release, whose tables have three codes more than Mimi's codebooks; it keeps a model that has not learned
    /// to avoid them yet decodable.</summary>
    public bool ConstrainToCodebook { get; set; }

    /// <summary>Whether synthesis watermarks its output with SilentCipher, as the release does (true). The watermark model
    /// is read from <see cref="WatermarkModelPath"/>; without it synthesis refuses rather than skip the watermark.</summary>
    public bool Watermark { get; set; } = true;

    /// <summary>The SilentCipher model directory the watermark is read from (<c>sony/silentcipher</c>'s
    /// <c>44_1_khz/73999_iteration</c>: <c>enc_c.ckpt</c>, <c>dec_c.ckpt</c>, <c>dec_m_0.ckpt</c>), loaded on first use.</summary>
    public string? WatermarkModelPath { get; set; }

    /// <summary>The watermark's five bytes (the release's public key, [212, 211, 146, 56, 201], which marks audio as made by
    /// CSM 1B; the release asks other applications to use a private key of their own).</summary>
    public int[] WatermarkKey { get; set; } = { 212, 211, 146, 56, 201 };

    /// <summary>The watermark's level below the audio in dB (36).</summary>
    public double WatermarkSdr { get; set; } = 36;

    /// <summary>Mimi's latent width (512).</summary>
    public int MimiDimension { get; set; } = 512;

    /// <summary>Mimi's SEANet base width (64).</summary>
    public int MimiFilters { get; set; } = 64;

    /// <summary>Mimi's SEANet ratios (8, 6, 5, 4): 960 samples per 25 Hz frame, two of those per code frame.</summary>
    public int[] MimiRatios { get; set; } = { 8, 6, 5, 4 };

    /// <summary>Mimi's Transformer layers on each side (8).</summary>
    public int MimiTransformerLayers { get; set; } = 8;

    /// <summary>Mimi's attention heads (8).</summary>
    public int MimiHeads { get; set; } = 8;

    /// <summary>Mimi's feed-forward width (2048).</summary>
    public int MimiFeedForward { get; set; } = 2048;

    /// <summary>Mimi's attention window in 25 Hz frames (250).</summary>
    public int MimiContext { get; set; } = 250;

    /// <summary>Entries in each of Mimi's codebooks (2048).</summary>
    public int MimiCodebookSize { get; set; } = 2048;

    /// <summary>Mimi's quantizer width (256).</summary>
    public int MimiCodebookDim { get; set; } = 256;

    /// <summary>The share of a sample's audio frames the audio decoder learns from (1/16, the blog's compute amortization);
    /// codebook 0 learns from all of them.</summary>
    public double DecoderFrameFraction { get; set; } = 1.0 / 16;

    /// <summary>AdamW's β₁ (0.9, Hugging Face's default).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW's β₂ (0.999, Hugging Face's default).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW's ε (1e-8, Hugging Face's default).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The gradient-norm clip (1.0, Hugging Face's default; 0 for none).</summary>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>The total optimizer steps the linear decay to 0 spans (0: a constant learning rate).</summary>
    public int TrainingSteps { get; set; }

    /// <summary>The seed of generation's sampling, of the decoder's training frames, and of the native model's
    /// initialization when the architecture has no seed (0).</summary>
    public int SamplingSeed { get; set; }
}
