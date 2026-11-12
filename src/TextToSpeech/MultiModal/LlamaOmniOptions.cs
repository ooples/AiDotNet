namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Which of LLaMA-Omni's two training stages (§2.5) a step follows.</summary>
public enum LlamaOmniTrainingStage
{
    /// <summary>Stage 1: the speech adaptor and the language model learn the text response (eq. 3); the speech encoder
    /// is frozen and the speech decoder is not involved. The paper's peak rate is 2e-5.</summary>
    TextResponse,

    /// <summary>Stage 2: only the speech decoder learns the units by CTC (eq. 5); everything else is frozen. The paper's
    /// peak rate is 2e-4.</summary>
    SpeechDecoder,

    /// <summary>The release's joint option (<c>tune_speech_generator_only = False</c>): the text loss plus the CTC loss
    /// weighted by <see cref="LlamaOmniOptions.CtcLossWeight"/>, all of adaptor, language model and decoder training.</summary>
    Joint,
}

/// <summary>Which hidden state stands for a response token when the speech decoder trains.</summary>
public enum LlamaOmniAlignment
{
    /// <summary>The paper's c_i = M(P(S), Y_&lt;i): the state that predicts y_i, the one inference reads.</summary>
    Paper,

    /// <summary>The release's training code: the state at y_i's own position (the one that has read it), one position
    /// later than inference reads.</summary>
    Release,
}

/// <summary>Options for <see cref="LlamaOmni{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the release, ICTNLP/Llama-3.1-8B-Omni:
/// <list type="bullet">
/// <item>whisper-large-v3's encoder;</item>
/// <item>a 5× concatenating adaptor (2048 hidden);</item>
/// <item>Llama-3.1-8B-Instruct: 32 layers, d 4096, 32 heads with 8 key-value heads, SwiGLU 14336, rope θ 500000 with
/// Llama 3's ×8 scaling;</item>
/// <item>a 2-layer, 4096-wide CTC speech decoder (32 heads, 11008) over 1000 mHuBERT units, upsampled 25×;</item>
/// <item>fairseq's unit HiFi-GAN at 16 kHz.</item>
/// </list></para>
/// <para><b>For Beginners:</b> LLaMA-Omni lets a text language model listen and talk. A speech encoder turns your
/// voice into vectors the language model reads; while it writes its text answer, a small decoder reads the language
/// model's internal states and writes "speech units", which a vocoder turns into sound.</para>
/// </remarks>
public class LlamaOmniOptions : TtsModelOptions
{
    /// <summary>Creates the release's configuration.</summary>
    public LlamaOmniOptions()
    {
        SampleRate = 16000;
        HiddenDim = 4096;
        NumHeads = 32;
        VocabSize = 128256;
        MaxTextLength = 2048;
        DropoutRate = 0.0;
        LearningRate = 2e-5;
        WeightDecay = 0.0;
        Seed = 42;
    }

    // ---------------------------------------------------------------- language model

    /// <summary>Llama layers (32), key-value heads (8) and SwiGLU width (14336).</summary>
    public int Layers { get; set; } = 32;

    /// <inheritdoc cref="Layers"/>
    public int KeyValueHeads { get; set; } = 8;

    /// <inheritdoc cref="Layers"/>
    public int Intermediate { get; set; } = 14336;

    /// <summary>RMSNorm ε (1e-5).</summary>
    public double RmsEpsilon { get; set; } = 1e-5;

    /// <summary>The rotary base (500000) and Llama 3's scaling: factor 8, low and high frequency factors 1 and 4, over
    /// 8192 original positions.</summary>
    public double RopeTheta { get; set; } = 500000;

    /// <inheritdoc cref="RopeTheta"/>
    public double RopeScalingFactor { get; set; } = 8;

    /// <inheritdoc cref="RopeTheta"/>
    public double RopeLowFrequencyFactor { get; set; } = 1;

    /// <inheritdoc cref="RopeTheta"/>
    public double RopeHighFrequencyFactor { get; set; } = 4;

    /// <inheritdoc cref="RopeTheta"/>
    public int RopeOriginalMaxPositions { get; set; } = 8192;

    /// <summary>A Hugging Face <c>tokenizer.json</c> for Llama 3; text needs it.</summary>
    public string? TokenizerPath { get; set; }

    // ---------------------------------------------------------------- speech encoder and adaptor

    /// <summary>The Whisper encoder's mel bands (128), frames after its stride (1500, 30 s), width (1280), heads (20)
    /// and layers (32): whisper-large-v3.</summary>
    public int EncoderMels { get; set; } = 128;

    /// <inheritdoc cref="EncoderMels"/>
    public int EncoderContext { get; set; } = 1500;

    /// <inheritdoc cref="EncoderMels"/>
    public int EncoderWidth { get; set; } = 1280;

    /// <inheritdoc cref="EncoderMels"/>
    public int EncoderHeads { get; set; } = 20;

    /// <inheritdoc cref="EncoderMels"/>
    public int EncoderLayers { get; set; } = 32;

    /// <summary>Frames the adaptor concatenates (k = 5) and its hidden width (2048).</summary>
    public int AdaptorDownsample { get; set; } = 5;

    /// <inheritdoc cref="AdaptorDownsample"/>
    public int AdaptorHidden { get; set; } = 2048;

    // ---------------------------------------------------------------- speech decoder and vocoder

    /// <summary>The CTC decoder's layers (2), width (4096), heads (32) and feed-forward width (11008).</summary>
    public int SpeechDecoderLayers { get; set; } = 2;

    /// <inheritdoc cref="SpeechDecoderLayers"/>
    public int SpeechDecoderWidth { get; set; } = 4096;

    /// <inheritdoc cref="SpeechDecoderLayers"/>
    public int SpeechDecoderHeads { get; set; } = 32;

    /// <inheritdoc cref="SpeechDecoderLayers"/>
    public int SpeechDecoderIntermediate { get; set; } = 11008;

    /// <summary>Frames per text token, λ (25).</summary>
    public int UpsampleFactor { get; set; } = 25;

    /// <summary>Discrete units, K (1000 mHuBERT clusters).</summary>
    public int Units { get; set; } = 1000;

    /// <summary>The unit vocoder's embedding width (128), first width (512) and upsampling (5·4·4·2·2 = 320 samples per
    /// unit frame, 50 Hz at 16 kHz).</summary>
    public int VocoderEmbeddingDim { get; set; } = 128;

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int VocoderChannels { get; set; } = 512;

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int[] VocoderUpsampleRates { get; set; } = { 5, 4, 4, 2, 2 };

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int[] VocoderUpsampleKernels { get; set; } = { 11, 8, 8, 4, 4 };

    /// <summary>The minimum unit chunk Ω the vocoder voices at a time while text is generated; 0 waits for all the units
    /// (the paper's offline Ω = +∞).</summary>
    public int UnitChunkSize { get; set; }

    // ---------------------------------------------------------------- unit extractor (training targets)

    /// <summary>mHuBERT's width (768), layers built (11, the layer read), heads (12), feed-forward width (3072) and
    /// feature-encoder channels (512), whose layer-11 units are the decoder's targets.</summary>
    public int HubertWidth { get; set; } = 768;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertLayer { get; set; } = 11;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertHeads { get; set; } = 12;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertIntermediate { get; set; } = 3072;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertConvChannels { get; set; } = 512;

    // ---------------------------------------------------------------- generation

    /// <summary>The system prompt (the release's llama_3 template).</summary>
    public string SystemPrompt { get; set; } =
        "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language.";

    /// <summary>The instruction after the speech in the user's turn (the release's evaluation questions).</summary>
    public string SpeechInstruction { get; set; } = "Please directly answer the questions in the user's speech.";

    /// <summary>Sampling temperature: 0 is the paper's greedy search (§4.2).</summary>
    public double Temperature { get; set; }

    /// <summary>The nucleus when sampling (1: none).</summary>
    public double TopP { get; set; } = 1.0;

    /// <summary>The most tokens a response may have (the release's 256).</summary>
    public int MaxNewTokens { get; set; } = 256;

    // ---------------------------------------------------------------- training

    /// <summary>The stage a training step follows; the speech decoder's by default, which TTS training is.</summary>
    public LlamaOmniTrainingStage Stage { get; set; } = LlamaOmniTrainingStage.SpeechDecoder;

    /// <summary>Which hidden state stands for a response token in decoder training: the paper's.</summary>
    public LlamaOmniAlignment Alignment { get; set; } = LlamaOmniAlignment.Paper;

    /// <summary>The speech decoder's peak rate (§4.1: 2e-4); the text stage's is
    /// <see cref="TtsModelOptions.LearningRate"/> (2e-5).</summary>
    public double SpeechDecoderLearningRate { get; set; } = 2e-4;

    /// <summary>Steps of the cosine schedule (the paper's 3 epochs of 200K samples at batch 32: 18750), of which the
    /// first 3 % warm up.</summary>
    public int TotalSteps { get; set; } = 18750;

    /// <summary>The fraction of <see cref="TotalSteps"/> that warms up (§4.1: 3 %). As transformers' schedule, the first
    /// step's rate is then 0.</summary>
    public double WarmupFraction { get; set; } = 0.03;

    /// <summary>The CTC loss's weight in joint training (the release's <c>ctc_loss_weight</c>, 1).</summary>
    public double CtcLossWeight { get; set; } = 1.0;

    /// <summary>Gradient clipping (Hugging Face Trainer's default, 1.0).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
