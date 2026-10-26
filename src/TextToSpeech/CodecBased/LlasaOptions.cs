namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for <see cref="Llasa{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are Llasa-1B (Ye et al. 2025, "Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based
/// Speech Synthesis"; <c>HKUSTAudio/Llasa-1B</c>): Llama 3.2 1B (width 2048, 16 layers, 32 heads over 8 key-value heads of
/// 64, SwiGLU 8192, rotary θ 500000 with Llama 3 scaling ×32, tied embeddings) over Llama 3's tokenizer plus 8 task markers
/// and 65,536 speech tokens (193,800 in all), speaking X-codec2 (<c>HKUSTAudio/xcodec2</c>: one 65,536-entry FSQ codebook
/// at 50 Hz, 16 kHz audio). The 3B and 8B models are Llama 3.2 3B and Llama 3.1 8B with the same tokens.</para>
/// <para>Generation follows the model card: Llama 3.2's chat template with "Convert the text to speech:" and the text
/// between the text-understanding markers, the assistant turn opened with the speech-generation marker, then sampling at
/// temperature 0.8 with top-p 1 up to 2048 tokens in all. Training follows the paper (§3.2.1) where it is explicit — peak
/// learning rate 5e-5, a cosine schedule with warm-up over 3% of an epoch and a final rate of 10% of the peak, 3 epochs,
/// sequences cropped to 2048 tokens, the loss on the speech tokens — and the official recipe (<c>train_tts.py</c>,
/// <c>config.json</c>) otherwise: AdamW with β₂ = 0.95 and weight decay 0.01, gradients clipped to norm 1.</para>
/// <para><b>For Beginners:</b> These settings size Llasa's language model and its speech codec, and control how it picks
/// each next speech token; the defaults reproduce the released 1B model.</para>
/// </remarks>
public class LlasaOptions : TtsModelOptions
{
    /// <summary>Creates the Llasa-1B configuration.</summary>
    public LlasaOptions()
    {
        SampleRate = 16000;
        HopSize = 320;
        HiddenDim = 2048;
        VocabSize = 193800;
        MaxTextLength = 2048;
        LearningRate = 5e-5;
        WeightDecay = 0.01;
        DropoutRate = 0.0;
    }

    /// <summary>Copies another instance.</summary>
    public LlasaOptions(LlasaOptions other)
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
        ChatDate = other.ChatDate;
        Temperature = other.Temperature;
        TopP = other.TopP;
        MaxLength = other.MaxLength;
        IncludePromptAudio = other.IncludePromptAudio;
        ConstrainToSpeechTokens = other.ConstrainToSpeechTokens;
        CodecEncoderDim = other.CodecEncoderDim;
        CodecEncoderRates = (int[])other.CodecEncoderRates.Clone();
        CodecAcousticDim = other.CodecAcousticDim;
        CodecSemanticDim = other.CodecSemanticDim;
        CodecDecoderDim = other.CodecDecoderDim;
        CodecDecoderDepth = other.CodecDecoderDepth;
        CodecDecoderHeads = other.CodecDecoderHeads;
        SemanticHiddenSize = other.SemanticHiddenSize;
        SemanticHeads = other.SemanticHeads;
        SemanticIntermediateSize = other.SemanticIntermediateSize;
        SemanticLayer = other.SemanticLayer;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        MaxGradNorm = other.MaxGradNorm;
        TrainingSteps = other.TrainingSteps;
        Epochs = other.Epochs;
        WarmupEpochFraction = other.WarmupEpochFraction;
        FinalLearningRateFraction = other.FinalLearningRateFraction;
        SamplingSeed = other.SamplingSeed;
    }

    /// <summary>Decoder layers (16).</summary>
    public int Layers { get; set; } = 16;

    /// <summary>Attention heads (32).</summary>
    public int Heads { get; set; } = 32;

    /// <summary>Key-value heads shared by groups of query heads (8).</summary>
    public int KeyValueHeads { get; set; } = 8;

    /// <summary>Width of each head (64).</summary>
    public int HeadDim { get; set; } = 64;

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

    /// <summary>The date the chat template writes into its system header ("Today Date: …", formatted like "26 Jul 2024"),
    /// or null for the current date, as the template does when it runs (null).</summary>
    public string? ChatDate { get; set; }

    /// <summary>Sampling temperature (0.8); 0 picks the most likely token.</summary>
    public double Temperature { get; set; } = 0.8;

    /// <summary>Nucleus sampling's probability mass (1: the whole distribution).</summary>
    public double TopP { get; set; } = 1.0;

    /// <summary>The most tokens in the prompt and the output together (2048, the training length).</summary>
    public int MaxLength { get; set; } = 2048;

    /// <summary>Whether synthesis from a speech prompt returns the prompt's own audio in front of the new speech (true, as
    /// the model card decodes the prompt's codes with the generated ones).</summary>
    public bool IncludePromptAudio { get; set; } = true;

    /// <summary>Whether each sampled token is restricted to the speech tokens and the end of speech (false). Not part of
    /// the release, which samples from the whole vocabulary and skips other tokens when reading the codes; it guarantees
    /// codes from a model that has not learned to write them yet.</summary>
    public bool ConstrainToSpeechTokens { get; set; }

    /// <summary>X-codec2's first acoustic-encoder width (48).</summary>
    public int CodecEncoderDim { get; set; } = 48;

    /// <summary>X-codec2's acoustic-encoder strides (2, 2, 4, 4, 5): 320 samples a code.</summary>
    public int[] CodecEncoderRates { get; set; } = { 2, 2, 4, 4, 5 };

    /// <summary>X-codec2's acoustic feature width (1024).</summary>
    public int CodecAcousticDim { get; set; } = 1024;

    /// <summary>X-codec2's semantic feature width (1024).</summary>
    public int CodecSemanticDim { get; set; } = 1024;

    /// <summary>X-codec2's decoder width (1024).</summary>
    public int CodecDecoderDim { get; set; } = 1024;

    /// <summary>X-codec2's decoder Transformer blocks (12).</summary>
    public int CodecDecoderDepth { get; set; } = 12;

    /// <summary>X-codec2's decoder attention heads (16).</summary>
    public int CodecDecoderHeads { get; set; } = 16;

    /// <summary>The w2v-BERT 2.0 encoder's width (1024).</summary>
    public int SemanticHiddenSize { get; set; } = 1024;

    /// <summary>The w2v-BERT 2.0 encoder's attention heads (16).</summary>
    public int SemanticHeads { get; set; } = 16;

    /// <summary>The w2v-BERT 2.0 encoder's feed-forward width (4096).</summary>
    public int SemanticIntermediateSize { get; set; } = 4096;

    /// <summary>The w2v-BERT 2.0 hidden state X-codec2 reads (16); only that many of its 24 layers are built.</summary>
    public int SemanticLayer { get; set; } = 16;

    /// <summary>AdamW's β₁ (0.9, Hugging Face's default).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>AdamW's β₂ (0.95, the official config).</summary>
    public double Beta2 { get; set; } = 0.95;

    /// <summary>AdamW's ε (1e-8, Hugging Face's default).</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>The gradient-norm clip (1.0, Hugging Face's default; 0 for none).</summary>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>The total optimizer steps the schedule spans (0: a constant learning rate).</summary>
    public int TrainingSteps { get; set; }

    /// <summary>The epochs <see cref="TrainingSteps"/> covers (3), which sizes the warm-up.</summary>
    public int Epochs { get; set; } = 3;

    /// <summary>The warm-up's length as a fraction of one epoch (0.03).</summary>
    public double WarmupEpochFraction { get; set; } = 0.03;

    /// <summary>The final learning rate as a fraction of the peak (0.1).</summary>
    public double FinalLearningRateFraction { get; set; } = 0.1;

    /// <summary>The seed of generation's sampling (and of the native model's initialization when the architecture has no
    /// seed) (0).</summary>
    public int SamplingSeed { get; set; }
}
