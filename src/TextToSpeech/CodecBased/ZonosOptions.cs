using AiDotNet.Audio.Effects;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The backbone a Zonos model uses.</summary>
public enum ZonosBackbone
{
    /// <summary>Zonos-v0.1-transformer: 26 attention layers.</summary>
    Transformer,

    /// <summary>Zonos-v0.1-hybrid: 46 layers, Mamba2 except every fourth (attention).</summary>
    Hybrid,
}

/// <summary>Options for <see cref="Zonos{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are Zonos-v0.1-transformer (Zyphra 2025; Zyphra/Zonos <c>zonos/config.py</c>, the release's
/// config.json): width 2048, 26 attention layers of 16 query heads over 4 key-value heads with rotary positions on adjacent
/// pairs, gated feed-forward 8192, over DAC's nine 1024-code codebooks at 44.1 kHz. <see cref="Hybrid"/> gives
/// Zonos-v0.1-hybrid: 46 layers, attention at 0, 4, …, 44 (rotary on halves) and Mamba2 elsewhere (state 128, head width 64,
/// expansion 2, one group), with four more conditioners.</para>
/// <para>Generation follows <c>Zonos.generate</c> with <c>make_cond_dict</c>'s defaults: classifier-free guidance at 2.0,
/// min-p 0.1, a repetition penalty of 3 over the last 2 frames, up to 30 seconds; English, a neutral-leaning emotion mix,
/// fmax 22050 Hz, pitch deviation 20 Hz and 15 phonemes a second. Training (no code was released) scores each head on its
/// next delayed code, drops each optional conditioner 10% of the time, and uses the Trainer's default AdamW.</para>
/// <para><b>For Beginners:</b> These settings size Zonos, pick its backbone, control how it samples, set the default
/// speaking style, and set how it trains; the defaults reproduce the released transformer model.</para>
/// </remarks>
public class ZonosOptions : TtsModelOptions
{
    /// <summary>Creates the Zonos-v0.1-transformer configuration.</summary>
    public ZonosOptions()
    {
        SampleRate = 44100;
        HopSize = 512;
        HiddenDim = 2048;
        VocabSize = 189;
        MaxTextLength = 2048;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
        DropoutRate = 0.0;
        Codec = new DACOptions();
    }

    /// <summary>The Zonos-v0.1-hybrid configuration.</summary>
    public static ZonosOptions Hybrid() => new()
    {
        Backbone = ZonosBackbone.Hybrid, Layers = 46, AttentionLayers = Enumerable.Range(0, 12).Select(i => 4 * i).ToArray(),
        RotaryInterleaved = false, Conditioners = ZonosConditioner.Hybrid.ToList(),
    };

    /// <summary>Copies another instance.</summary>
    public ZonosOptions(ZonosOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Backbone = other.Backbone;
        Layers = other.Layers;
        AttentionLayers = other.AttentionLayers is null ? null : (int[])other.AttentionLayers.Clone();
        Heads = other.Heads;
        KeyValueHeads = other.KeyValueHeads;
        AttentionIntermediate = other.AttentionIntermediate;
        MambaIntermediate = other.MambaIntermediate;
        RotaryDim = other.RotaryDim;
        RotaryInterleaved = other.RotaryInterleaved;
        NormEpsilon = other.NormEpsilon;
        StateDim = other.StateDim;
        ConvWidth = other.ConvWidth;
        Expand = other.Expand;
        MambaHeadDim = other.MambaHeadDim;
        Groups = other.Groups;
        ChunkSize = other.ChunkSize;
        Conditioners = other.Conditioners.ToList();
        Codebooks = other.Codebooks;
        EndToken = other.EndToken;
        MaskToken = other.MaskToken;
        MaxNewTokens = other.MaxNewTokens;
        CfgScale = other.CfgScale;
        Temperature = other.Temperature;
        TopP = other.TopP;
        TopK = other.TopK;
        MinP = other.MinP;
        UnifiedLinear = other.UnifiedLinear;
        UnifiedConfidence = other.UnifiedConfidence;
        UnifiedQuadratic = other.UnifiedQuadratic;
        RepetitionPenalty = other.RepetitionPenalty;
        RepetitionWindow = other.RepetitionWindow;
        Language = other.Language;
        Emotion = (double[])other.Emotion.Clone();
        Fmax = other.Fmax;
        PitchStd = other.PitchStd;
        SpeakingRate = other.SpeakingRate;
        VqScore = (double[])other.VqScore.Clone();
        CtcLoss = other.CtcLoss;
        DnsmosOverall = other.DnsmosOverall;
        SpeakerNoised = other.SpeakerNoised;
        UnconditionalInputs = other.UnconditionalInputs.ToList();
        SudachiDictionary = other.SudachiDictionary;
        SudachiResources = other.SudachiResources;
        Codec = (DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.Codec);
        ConditionerDropout = other.ConditionerDropout;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        TrainingSteps = other.TrainingSteps;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>The backbone (the transformer).</summary>
    public ZonosBackbone Backbone { get; set; } = ZonosBackbone.Transformer;

    /// <summary>Layers (26; the hybrid has 46).</summary>
    public int Layers { get; set; } = 26;

    /// <summary>The layers that attend; null: all of them (the transformer). The hybrid's are 0, 4, …, 44.</summary>
    public int[]? AttentionLayers { get; set; }

    /// <summary>Query heads (16).</summary>
    public int Heads { get; set; } = 16;

    /// <summary>Key-value heads (4).</summary>
    public int KeyValueHeads { get; set; } = 4;

    /// <summary>The attention layers' gated feed-forward width (8192).</summary>
    public int AttentionIntermediate { get; set; } = 8192;

    /// <summary>The Mamba2 layers' gated feed-forward width (0: none).</summary>
    public int MambaIntermediate { get; set; }

    /// <summary>Head dimensions the rotary embedding turns (128: all).</summary>
    public int RotaryDim { get; set; } = 128;

    /// <summary>Whether the rotary embedding turns adjacent pairs (the transformer) rather than halves (the hybrid).</summary>
    public bool RotaryInterleaved { get; set; } = true;

    /// <summary>LayerNorm's ε (1e-5).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    /// <summary>Mamba2's state size (128).</summary>
    public int StateDim { get; set; } = 128;

    /// <summary>Mamba2's convolution width (4).</summary>
    public int ConvWidth { get; set; } = 4;

    /// <summary>Mamba2's expansion (2).</summary>
    public int Expand { get; set; } = 2;

    /// <summary>Mamba2's head width (64).</summary>
    public int MambaHeadDim { get; set; } = 64;

    /// <summary>Mamba2's groups of B and C (1).</summary>
    public int Groups { get; set; } = 1;

    /// <summary>The SSD algorithm's chunk length (256); it changes only the order of the arithmetic.</summary>
    public int ChunkSize { get; set; } = 256;

    /// <summary>The prefix conditioners, in order (the transformer's seven).</summary>
    public List<ZonosConditioner> Conditioners { get; set; } = ZonosConditioner.Transformer.ToList();

    /// <summary>DAC codebooks (9).</summary>
    public int Codebooks { get; set; } = 9;

    /// <summary>The end-of-audio token (1024).</summary>
    public int EndToken { get; set; } = 1024;

    /// <summary>The mask token of the delay pattern (1025).</summary>
    public int MaskToken { get; set; } = 1025;

    /// <summary>The most frames generated (2580: 30 seconds).</summary>
    public int MaxNewTokens { get; set; } = 86 * 30;

    /// <summary>Classifier-free guidance's scale (2.0; the release requires it to differ from 1).</summary>
    public double CfgScale { get; set; } = 2.0;

    /// <summary>The temperature (1.0; 0 takes the most likely code).</summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>Nucleus sampling's mass (0: off).</summary>
    public double TopP { get; set; }

    /// <summary>Top-k (0: off).</summary>
    public int TopK { get; set; }

    /// <summary>Min-p (0.1).</summary>
    public double MinP { get; set; } = 0.1;

    /// <summary>NovelAI's unified sampler's linear term (0: off).</summary>
    public double UnifiedLinear { get; set; }

    /// <summary>The unified sampler's confidence term.</summary>
    public double UnifiedConfidence { get; set; }

    /// <summary>The unified sampler's quadratic term.</summary>
    public double UnifiedQuadratic { get; set; }

    /// <summary>The repetition penalty (3.0; 1 for none).</summary>
    public double RepetitionPenalty { get; set; } = 3.0;

    /// <summary>The frames the repetition penalty looks back over (2).</summary>
    public int RepetitionWindow { get; set; } = 2;

    /// <summary>The eSpeak language (en-us); one of <see cref="Zonos{T}.Languages"/>.</summary>
    public string Language { get; set; } = "en-us";

    /// <summary>The emotion mix: happiness, sadness, disgust, fear, surprise, anger, other, neutral (normalized to sum 1).</summary>
    public double[] Emotion { get; set; } = { 0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077 };

    /// <summary>The highest frequency in Hz (22050; use 22050 for voice cloning).</summary>
    public double Fmax { get; set; } = 22050;

    /// <summary>The pitch's standard deviation in Hz (20; 20–45 normal, 60–150 expressive).</summary>
    public double PitchStd { get; set; } = 20;

    /// <summary>Phonemes a second (15; 10 slow, 30 very fast).</summary>
    public double SpeakingRate { get; set; } = 15;

    /// <summary>The hybrid's target VQScore for each eighth of the audio (0.78).</summary>
    public double[] VqScore { get; set; } = Enumerable.Repeat(0.78, 8).ToArray();

    /// <summary>The hybrid's target CTC loss (0).</summary>
    public double CtcLoss { get; set; }

    /// <summary>The hybrid's target DNSMOS (4).</summary>
    public double DnsmosOverall { get; set; } = 4;

    /// <summary>The hybrid's speaker-noised flag (false).</summary>
    public bool SpeakerNoised { get; set; }

    /// <summary>Conditioners left unconditional (<c>make_cond_dict</c>'s vqscore_8 and dnsmos_ovrl).</summary>
    public List<string> UnconditionalInputs { get; set; } = new() { "vqscore_8", "dnsmos_ovrl" };

    /// <summary>For Japanese: sudachidict_full's <c>system.dic</c> (null: the SUDACHI_DICT environment variable). Read at
    /// runtime, not shipped.</summary>
    public string? SudachiDictionary { get; set; }

    /// <summary>For Japanese: SudachiPy's resources directory with <c>char.def</c>, <c>unk.def</c> and <c>rewrite.def</c>
    /// (null: the SUDACHI_RESOURCES environment variable).</summary>
    public string? SudachiResources { get; set; }

    /// <summary>The codec (DAC 44.1 kHz, 9 codebooks); load its released weights with
    /// <see cref="DAC{T}.LoadPretrainedWeights"/>.</summary>
    public DACOptions Codec { get; set; }

    /// <summary>In training, the chance each optional conditioner is replaced by its unconditional vector (0.1).</summary>
    public double ConditionerDropout { get; set; } = 0.1;

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

    /// <summary>The seed of sampling, of conditioner dropout, and of the native model's initialization when the
    /// architecture has no seed (0).</summary>
    public int SamplingSeed { get; set; }
}
