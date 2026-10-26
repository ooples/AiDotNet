namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>How <see cref="OrpheusTTS{T}"/> turns the generated codes into audio.</summary>
public enum OrpheusDecoding
{
    /// <summary>All frames decoded together (the release's Colab notebooks).</summary>
    Full,

    /// <summary>The <c>orpheus_tts</c> package's streaming decoder: once 28 codes (4 frames) are available, and after every
    /// further 7, the last 4 frames are decoded and samples 2048–4095 (the second frame) kept.</summary>
    Streaming,
}

/// <summary>Options for <see cref="OrpheusTTS{T}"/>.</summary>
/// <remarks>
/// <para>Orpheus has no paper: the defaults are canopyai/Orpheus-TTS (release 0.1, the <c>orpheus_tts</c> package, the
/// fine-tuning script and the data-preparation notebook it links) and canopylabs/orpheus-3b-0.1-ft's <c>config.json</c>:
/// Llama 3.2 3B (width 3072, 28 layers, 24 heads over 8 key-value heads of 128, SwiGLU 8192, rotary θ 500000 with Llama 3
/// scaling ×32, tied embeddings) over Llama 3's tokenizer plus 28,683 audio tokens (156,940 in all), speaking SNAC 24 kHz
/// codes (hubertsiuzdak/snac_24khz: three codebooks of 4096 at 12, 23 and 47 Hz, seven tokens a frame). Generation follows
/// the package: the prompt "<c>{voice}: {text}</c>" with the voice "tara", temperature 0.6, top-p 0.8, a repetition
/// penalty of 1.3 and at most 1200 new tokens. Fine-tuning follows <c>finetune/train.py</c>: Hugging Face's Trainer at
/// 5e-5 (AdamW, betas 0.9 and 0.999, eps 1e-8, no weight decay, linear decay without warm-up, gradient norm clipped to 1),
/// the loss on every token.</para>
/// <para><b>For Beginners:</b> These settings size Orpheus's language model and its audio codec, and control how it picks
/// each next sound token; the defaults reproduce the released 3B model.</para>
/// </remarks>
public class OrpheusTTSOptions : TtsModelOptions
{
    /// <summary>Creates the orpheus-3b-0.1-ft configuration.</summary>
    public OrpheusTTSOptions()
    {
        SampleRate = 24000;
        HopSize = 512;
        HiddenDim = 3072;
        VocabSize = 156940;
        MaxTextLength = 8192;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
        DropoutRate = 0.0;
    }

    /// <summary>Copies another instance.</summary>
    public OrpheusTTSOptions(OrpheusTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Layers = other.Layers;
        Heads = other.Heads;
        KeyValueHeads = other.KeyValueHeads;
        HeadDim = other.HeadDim;
        Intermediate = other.Intermediate;
        RmsEpsilon = other.RmsEpsilon;
        RopeTheta = other.RopeTheta;
        RopeScalingFactor = other.RopeScalingFactor;
        RopeLowFrequencyFactor = other.RopeLowFrequencyFactor;
        RopeHighFrequencyFactor = other.RopeHighFrequencyFactor;
        RopeOriginalMaxPositions = other.RopeOriginalMaxPositions;
        Voice = other.Voice;
        Temperature = other.Temperature;
        TopP = other.TopP;
        RepetitionPenalty = other.RepetitionPenalty;
        MaxNewTokens = other.MaxNewTokens;
        Decoding = other.Decoding;
        ConstrainToAudioTokens = other.ConstrainToAudioTokens;
        CodecEncoderDim = other.CodecEncoderDim;
        CodecEncoderRates = (int[])other.CodecEncoderRates.Clone();
        CodecDecoderDim = other.CodecDecoderDim;
        CodecDecoderRates = (int[])other.CodecDecoderRates.Clone();
        CodecCodebookDim = other.CodecCodebookDim;
        CodecNoise = other.CodecNoise;
        CodecDepthwise = other.CodecDepthwise;
        RemoveDuplicateFrames = other.RemoveDuplicateFrames;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        TrainingSteps = other.TrainingSteps;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Decoder layers (28).</summary>
    public int Layers { get; set; } = 28;

    /// <summary>Attention heads (24).</summary>
    public int Heads { get; set; } = 24;

    /// <summary>Key-value heads shared by groups of query heads (8).</summary>
    public int KeyValueHeads { get; set; } = 8;

    /// <summary>Width of each head (128).</summary>
    public int HeadDim { get; set; } = 128;

    /// <summary>SwiGLU hidden width (8192).</summary>
    public int Intermediate { get; set; } = 8192;

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

    /// <summary>The voice named before the text ("tara"; the fine-tuned release also knows leah, jess, leo, dan, mia, zac and
    /// zoe), or null for the text alone (the pretrained model, or a single-speaker fine-tune).</summary>
    public string? Voice { get; set; } = "tara";

    /// <summary>Sampling temperature (0.6); 0 picks the most likely token.</summary>
    public double Temperature { get; set; } = 0.6;

    /// <summary>Nucleus sampling's probability mass (0.8).</summary>
    public double TopP { get; set; } = 0.8;

    /// <summary>The repetition penalty over every token of the prompt and the output (1.3; the release requires at least
    /// 1.1).</summary>
    public double RepetitionPenalty { get; set; } = 1.3;

    /// <summary>The most tokens generated after the prompt (1200: about 14 seconds).</summary>
    public int MaxNewTokens { get; set; } = 1200;

    /// <summary>How the codes become audio (<see cref="OrpheusDecoding.Full"/>).</summary>
    public OrpheusDecoding Decoding { get; set; } = OrpheusDecoding.Full;

    /// <summary>Whether each sampled token is restricted to the codes of the slot it fills, and the end of speech to frame
    /// boundaries (false). Not part of the release, which samples from the whole vocabulary and drops malformed frames;
    /// it guarantees well-formed frames from a model that has not learned them yet.</summary>
    public bool ConstrainToAudioTokens { get; set; }

    /// <summary>SNAC's first encoder width (48).</summary>
    public int CodecEncoderDim { get; set; } = 48;

    /// <summary>SNAC's encoder strides (2, 4, 8, 8).</summary>
    public int[] CodecEncoderRates { get; set; } = { 2, 4, 8, 8 };

    /// <summary>SNAC's first decoder width (1024).</summary>
    public int CodecDecoderDim { get; set; } = 1024;

    /// <summary>SNAC's decoder rates (8, 8, 4, 2).</summary>
    public int[] CodecDecoderRates { get; set; } = { 8, 8, 4, 2 };

    /// <summary>SNAC's factorized code width (8).</summary>
    public int CodecCodebookDim { get; set; } = 8;

    /// <summary>Whether SNAC's decoder blocks add noise (true).</summary>
    public bool CodecNoise { get; set; } = true;

    /// <summary>Whether SNAC's residual convolutions are depthwise (true).</summary>
    public bool CodecDepthwise { get; set; } = true;

    /// <summary>Whether training drops each frame whose coarse code repeats the previous kept frame's, as the release's data
    /// preparation does (true).</summary>
    public bool RemoveDuplicateFrames { get; set; } = true;

    /// <summary>AdamW β1 (0.9).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW β2 (0.999).</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>AdamW ε (1e-8).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The gradient norm clip (1.0; 0 for none).</summary>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>Training steps over which the learning rate decays linearly to 0 (0: a constant rate).</summary>
    public int TrainingSteps { get; set; }

    /// <summary>The seed of generation's sampling and the codec's decoder noise (0).</summary>
    public int SamplingSeed { get; set; }
}
