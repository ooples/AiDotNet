using System;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for ChatTTS (2noise/ChatTTS, 2024), as its released configuration (<c>ChatTTS/config/config.py</c>)
/// and inference defaults (<c>RefineTextParams</c>, <c>InferCodeParams</c>) set them.</summary>
/// <remarks>
/// <para>
/// <b>GPT</b>: a Llama decoder (Hugging Face <c>LlamaModel</c>: 20 layers, width 768, 12 heads, feed-forward 3072,
/// RMSNorm ε 1e-6) over ChatTTS's own embeddings: a text table of 21,178 ids (bert-base-chinese's 21,128 plus 50
/// control tokens) and 4 code tables of 626 ids (625 DVAE codes and the end). Weight-normalized heads read a text
/// logit or 4 code logits from each position.
/// </para>
/// <para>
/// <b>DVAE</b>: 24 kHz audio to 100 log-mel bands (n_fft 1024, hop 256), a 2× downsampling convolution pair, a 12-block
/// ConvNeXt encoder to 1024 channels, and grouped residual FSQ (levels 5·5·5·5 = 625, 2 groups, 2 residual stages):
/// 4 codes per frame at 46.875 Hz. Its decoder maps codes back to mels; the "Decoder" (a second stack, 384 wide)
/// maps the GPT's hidden states to mels, which synthesis uses by default. Vocos (8 blocks, width 512) renders the
/// mels.
/// </para>
/// <para>
/// No training code or paper exists; <see cref="TtsModelOptions.LearningRate"/> is the library's default, not
/// ChatTTS's.
/// </para>
/// <para><b>For Beginners:</b> ChatTTS turns text, with optional laughs and pauses, into conversational speech in a
/// voice drawn at random or given by a speaker embedding or a short recording.</para>
/// </remarks>
public class ChatTTSOptions : TtsModelOptions
{
    /// <summary>Creates the released configuration.</summary>
    public ChatTTSOptions()
    {
        SampleRate = 24000;
        HopSize = 256;
        FftSize = 1024;
        MelChannels = 100;
        HiddenDim = 768;
        NumHeads = 12;
        NumDecoderLayers = 20;
        NumEncoderLayers = 0;
        DropoutRate = 0.0;
        LearningRate = 1e-4;
        WeightDecay = 0.0;
        MaxTextLength = 512;
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public ChatTTSOptions(ChatTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        IntermediateSize = other.IntermediateSize;
        RmsNormEpsilon = other.RmsNormEpsilon;
        NumVq = other.NumVq;
        FsqLevels = (int[])other.FsqLevels.Clone();
        FsqGroups = other.FsqGroups;
        FsqResiduals = other.FsqResiduals;
        DvaeDim = other.DvaeDim;
        DvaeEncoderHidden = other.DvaeEncoderHidden;
        DvaeDecoderHidden = other.DvaeDecoderHidden;
        DvaeLayers = other.DvaeLayers;
        DvaeBottleneck = other.DvaeBottleneck;
        DecoderDim = other.DecoderDim;
        DecoderHidden = other.DecoderHidden;
        DecoderLayers = other.DecoderLayers;
        DecoderBottleneck = other.DecoderBottleneck;
        VocosDim = other.VocosDim;
        VocosIntermediate = other.VocosIntermediate;
        VocosLayers = other.VocosLayers;
        RefineText = other.RefineText;
        UseHiddenStateDecoder = other.UseHiddenStateDecoder;
        CodeTemperature = other.CodeTemperature;
        CodeTopP = other.CodeTopP;
        CodeTopK = other.CodeTopK;
        CodeRepetitionPenalty = other.CodeRepetitionPenalty;
        MaxNewCodeTokens = other.MaxNewCodeTokens;
        RefineTemperature = other.RefineTemperature;
        RefineTopP = other.RefineTopP;
        RefineTopK = other.RefineTopK;
        RefineRepetitionPenalty = other.RefineRepetitionPenalty;
        MaxNewRefineTokens = other.MaxNewRefineTokens;
        SamplingSeed = other.SamplingSeed;
    }

    // ---------------------------------------------------------------- GPT (config.GPT)

    /// <summary>The Llama feed-forward width (3072). Width, heads and layers are <see cref="TtsModelOptions.HiddenDim"/>,
    /// <see cref="TtsModelOptions.NumHeads"/> and <see cref="TtsModelOptions.NumDecoderLayers"/>.</summary>
    public int IntermediateSize { get; set; } = 3072;

    /// <summary>RMSNorm ε (Hugging Face <c>LlamaConfig</c>'s default, 1e-6, which ChatTTS keeps).</summary>
    public double RmsNormEpsilon { get; set; } = 1e-6;

    /// <summary>Codes per audio frame, one code table and head each (4).</summary>
    public int NumVq { get; set; } = 4;

    // ---------------------------------------------------------------- DVAE (config.DVAE)

    /// <summary>FSQ levels per dimension ((5, 5, 5, 5): 625 codes).</summary>
    public int[] FsqLevels { get; set; } = { 5, 5, 5, 5 };

    /// <summary>FSQ groups (2).</summary>
    public int FsqGroups { get; set; } = 2;

    /// <summary>Residual FSQ stages per group (2). Groups × stages is <see cref="NumVq"/>.</summary>
    public int FsqResiduals { get; set; } = 2;

    /// <summary>The DVAE's width (512); its encoder emits twice this (1024).</summary>
    public int DvaeDim { get; set; } = 512;

    /// <summary>The DVAE encoder's ConvNeXt width (256).</summary>
    public int DvaeEncoderHidden { get; set; } = 256;

    /// <summary>The DVAE decoder's ConvNeXt width (256).</summary>
    public int DvaeDecoderHidden { get; set; } = 256;

    /// <summary>ConvNeXt blocks in each DVAE stack (12).</summary>
    public int DvaeLayers { get; set; } = 12;

    /// <summary>The DVAE stacks' bottleneck width (128).</summary>
    public int DvaeBottleneck { get; set; } = 128;

    // ---------------------------------------------------------------- hidden-state decoder (config.Decoder)

    /// <summary>The hidden-state decoder's width (384, half the GPT's).</summary>
    public int DecoderDim { get; set; } = 384;

    /// <summary>Its ConvNeXt width (512).</summary>
    public int DecoderHidden { get; set; } = 512;

    /// <summary>Its ConvNeXt blocks (12).</summary>
    public int DecoderLayers { get; set; } = 12;

    /// <summary>Its bottleneck width (128).</summary>
    public int DecoderBottleneck { get; set; } = 128;

    // ---------------------------------------------------------------- Vocos (config.Vocos)

    /// <summary>Vocos's width (512).</summary>
    public int VocosDim { get; set; } = 512;

    /// <summary>Vocos's ConvNeXt intermediate width (1536).</summary>
    public int VocosIntermediate { get; set; } = 1536;

    /// <summary>Vocos's ConvNeXt blocks (8).</summary>
    public int VocosLayers { get; set; } = 8;

    // ---------------------------------------------------------------- inference (core.Chat.infer)

    /// <summary>Whether synthesis first refines the text (adds pauses, laughs and fillers with the text head); the
    /// release does by default (<c>skip_refine_text=False</c>).</summary>
    public bool RefineText { get; set; } = true;

    /// <summary>Whether mels come from the GPT's hidden states through the "Decoder" (the default,
    /// <c>use_decoder=True</c>) or from the codes through the DVAE.</summary>
    public bool UseHiddenStateDecoder { get; set; } = true;

    /// <summary>Code sampling temperature (0.3).</summary>
    public double CodeTemperature { get; set; } = 0.3;

    /// <summary>Code nucleus threshold (0.7, keeping at least 3).</summary>
    public double CodeTopP { get; set; } = 0.7;

    /// <summary>Code top-k (20, keeping at least 3).</summary>
    public int CodeTopK { get; set; } = 20;

    /// <summary>Code repetition penalty over the last 16 generated frames (1.05).</summary>
    public double CodeRepetitionPenalty { get; set; } = 1.05;

    /// <summary>The most code frames per utterance (2048).</summary>
    public int MaxNewCodeTokens { get; set; } = 2048;

    /// <summary>Refinement temperature (0.7).</summary>
    public double RefineTemperature { get; set; } = 0.7;

    /// <summary>Refinement nucleus threshold (0.7).</summary>
    public double RefineTopP { get; set; } = 0.7;

    /// <summary>Refinement top-k (20).</summary>
    public int RefineTopK { get; set; } = 20;

    /// <summary>Refinement repetition penalty (1.0: none).</summary>
    public double RefineRepetitionPenalty { get; set; } = 1.0;

    /// <summary>The most refined text tokens (384).</summary>
    public int MaxNewRefineTokens { get; set; } = 384;

    /// <summary>The seed of synthesis's sampling (fixed, so synthesis is deterministic).</summary>
    public int SamplingSeed { get; set; } = 1234;
}
