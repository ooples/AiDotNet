using AiDotNet.Audio.Effects;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for <see cref="Dia{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are Dia 1.6B (Nari Labs 2025; nari-labs/dia <c>dia/config.py</c>, <c>nari-labs/Dia-1.6B-0626</c>): a
/// Transformer encoder over the text's UTF-8 bytes (width 1024, 12 layers, 16 heads of 128, SwiGLU 4096) and a decoder over
/// DAC's nine codebooks (width 2048, 18 layers, 16 query heads over 4 key-value heads of 128, cross-attention of 16 heads of
/// 128, SwiGLU 8192), 1028 tokens per codebook (codes 0–1023, end 1024, padding 1025, start 1026), the codebooks delayed by
/// [0, 8, 9, 10, 11, 12, 13, 14, 15] frames, rotary θ 10000, at 44.1 kHz.</para>
/// <para>Generation follows <c>Dia.generate</c>: classifier-free guidance at scale 3, temperature 1.2, top-p 0.95 and a CFG
/// filter keeping the 45 best guided codes, up to 3072 steps. transformers' example (<c>hf.py</c>) samples at temperature
/// 1.8 and top-p 0.90 instead. Training follows transformers' DiaForConditionalGeneration, written with Nari Labs and the
/// only training code, under the Trainer's default AdamW (no recipe was released).</para>
/// <para><b>For Beginners:</b> These settings size Dia's text encoder and audio decoder, control how it samples each step
/// of sound, and set how it trains; the defaults reproduce the released 1.6B model.</para>
/// </remarks>
public class DiaOptions : TtsModelOptions
{
    /// <summary>Creates the Dia 1.6B configuration.</summary>
    public DiaOptions()
    {
        SampleRate = 44100;
        HopSize = 512;
        HiddenDim = 1024;
        VocabSize = 256;
        MaxTextLength = 1024;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
        DropoutRate = 0.0;
        Codec = new DACOptions();
    }

    /// <summary>Copies another instance.</summary>
    public DiaOptions(DiaOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        EncoderLayers = other.EncoderLayers;
        EncoderHeads = other.EncoderHeads;
        EncoderKeyValueHeads = other.EncoderKeyValueHeads;
        EncoderHeadDim = other.EncoderHeadDim;
        EncoderIntermediate = other.EncoderIntermediate;
        DecoderHidden = other.DecoderHidden;
        DecoderLayers = other.DecoderLayers;
        DecoderHeads = other.DecoderHeads;
        DecoderKeyValueHeads = other.DecoderKeyValueHeads;
        DecoderHeadDim = other.DecoderHeadDim;
        CrossHeads = other.CrossHeads;
        CrossKeyValueHeads = other.CrossKeyValueHeads;
        CrossHeadDim = other.CrossHeadDim;
        DecoderIntermediate = other.DecoderIntermediate;
        NormEpsilon = other.NormEpsilon;
        RopeTheta = other.RopeTheta;
        AudioVocabSize = other.AudioVocabSize;
        Channels = other.Channels;
        DelayPattern = (int[])other.DelayPattern.Clone();
        EndToken = other.EndToken;
        PadToken = other.PadToken;
        BeginToken = other.BeginToken;
        MaxAudioTokens = other.MaxAudioTokens;
        CfgScale = other.CfgScale;
        Temperature = other.Temperature;
        TopP = other.TopP;
        CfgFilterTopK = other.CfgFilterTopK;
        Codec = (DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        TrainingSteps = other.TrainingSteps;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>The encoder's layers (12); its width is <see cref="TtsModelOptions.HiddenDim"/> (1024) and its vocabulary
    /// <see cref="TtsModelOptions.VocabSize"/> (256 bytes).</summary>
    public int EncoderLayers { get; set; } = 12;

    /// <summary>The encoder's attention heads (16).</summary>
    public int EncoderHeads { get; set; } = 16;

    /// <summary>The encoder's key-value heads (16).</summary>
    public int EncoderKeyValueHeads { get; set; } = 16;

    /// <summary>The encoder's head width (128).</summary>
    public int EncoderHeadDim { get; set; } = 128;

    /// <summary>The encoder's feed-forward width (4096).</summary>
    public int EncoderIntermediate { get; set; } = 4096;

    /// <summary>The decoder's width (2048).</summary>
    public int DecoderHidden { get; set; } = 2048;

    /// <summary>The decoder's layers (18).</summary>
    public int DecoderLayers { get; set; } = 18;

    /// <summary>The decoder's self-attention query heads (16).</summary>
    public int DecoderHeads { get; set; } = 16;

    /// <summary>The decoder's self-attention key-value heads (4).</summary>
    public int DecoderKeyValueHeads { get; set; } = 4;

    /// <summary>The decoder's self-attention head width (128).</summary>
    public int DecoderHeadDim { get; set; } = 128;

    /// <summary>The cross-attention's query heads (16).</summary>
    public int CrossHeads { get; set; } = 16;

    /// <summary>The cross-attention's key-value heads (16).</summary>
    public int CrossKeyValueHeads { get; set; } = 16;

    /// <summary>The cross-attention's head width (128).</summary>
    public int CrossHeadDim { get; set; } = 128;

    /// <summary>The decoder's feed-forward width (8192).</summary>
    public int DecoderIntermediate { get; set; } = 8192;

    /// <summary>RMSNorm's ε (1e-5).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    /// <summary>The rotary embedding's θ (10000).</summary>
    public double RopeTheta { get; set; } = 10000;

    /// <summary>Tokens per codebook in the decoder's tables and heads (1028: DAC's 1024 codes, end, padding, start, and one
    /// unused).</summary>
    public int AudioVocabSize { get; set; } = 1028;

    /// <summary>DAC codebooks the decoder predicts (9).</summary>
    public int Channels { get; set; } = 9;

    /// <summary>Each codebook's delay in frames ([0, 8, 9, 10, 11, 12, 13, 14, 15]).</summary>
    public int[] DelayPattern { get; set; } = { 0, 8, 9, 10, 11, 12, 13, 14, 15 };

    /// <summary>The end-of-audio token (1024).</summary>
    public int EndToken { get; set; } = 1024;

    /// <summary>The padding token (1025).</summary>
    public int PadToken { get; set; } = 1025;

    /// <summary>The start-of-audio token (1026).</summary>
    public int BeginToken { get; set; } = 1026;

    /// <summary>The most decoder steps (3072, the decoder's positions; about 35 seconds).</summary>
    public int MaxAudioTokens { get; set; } = 3072;

    /// <summary>Classifier-free guidance's scale (3.0).</summary>
    public double CfgScale { get; set; } = 3.0;

    /// <summary>The sampling temperature (1.2; 0 takes the most likely code).</summary>
    public double Temperature { get; set; } = 1.2;

    /// <summary>Nucleus sampling's probability mass (0.95; 1 keeps every code).</summary>
    public double TopP { get; set; } = 0.95;

    /// <summary>The codes kept by the CFG filter and by sampling's top-k (45).</summary>
    public int CfgFilterTopK { get; set; } = 45;

    /// <summary>The codec (DAC 44.1 kHz, 9 codebooks); load its released weights with
    /// <see cref="DAC{T}.LoadDescriptCheckpoint"/>.</summary>
    public DACOptions Codec { get; set; }

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

    /// <summary>The seed of generation's sampling and of the native model's initialization when the architecture has no
    /// seed (0).</summary>
    public int SamplingSeed { get; set; }
}
