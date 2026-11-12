namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The unit vocoder that voices SpeechGPT's units (Zhang et al. 2023 §4.1).</summary>
public enum SpeechGptVocoder
{
    /// <summary>The paper's multi-speaker unit HiFi-GAN (Polyak et al. 2021): a speaker's embedding concatenated to every
    /// upsampled frame, a duration predictor, trained here with HiFi-GAN's discriminators
    /// (<see cref="SpeechGPT{T}.TrainVocoder"/>). No weights were released.</summary>
    MultiSpeaker,

    /// <summary>The release's vocoder: fairseq's single-speaker LJSpeech unit HiFi-GAN
    /// (<c>mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj</c>), loadable with <see cref="SpeechGPT{T}.LoadVocoder"/>.</summary>
    ReleaseLJ,
}

/// <summary>SpeechGPT's three training stages (§4.2).</summary>
public enum SpeechGptTrainingStage
{
    /// <summary>Stage 1, modality-adaptation pre-training: next-token prediction on unlabelled speech's units (eq. 3).</summary>
    ModalityAdaptation,

    /// <summary>Stage 2, cross-modal instruction fine-tuning: ASR and TTS pairs, the loss on the response only (eq. 4).</summary>
    CrossModalInstruction,

    /// <summary>Stage 3, chain-of-modality instruction fine-tuning: LoRA on the attention's query and value projections,
    /// the same loss (eq. 4).</summary>
    ChainOfModality,
}

/// <summary>Options for <see cref="SpeechGPT{T}"/> and its speech recognizer.</summary>
/// <remarks>
/// <para>The defaults are the paper's: LLaMA-13B (40 layers, d 5120, 40 heads, SwiGLU 13824) with its vocabulary
/// expanded by 1000 mHuBERT units and six special tokens (§4.2, eq. 1–2); mHuBERT layer-11 units with repeats merged
/// (§3.1); the multi-speaker unit vocoder; sampling at temperature 0.8 with top-k 60 and top-p 0.8 over up to 2048
/// tokens (§5.1); a peak rate of 2e-4 in every stage (App. D). <see cref="Release7B"/> is the released
/// fnlp/SpeechGPT-7B-cm's LLaMA-7B.</para>
/// <para><b>For Beginners:</b> SpeechGPT teaches a text language model a new "language": speech units, little symbols a
/// speech model extracts from audio. It can then read units (listen), write units (speak) and translate between them
/// and text. A vocoder turns written units back into sound.</para>
/// </remarks>
public class SpeechGPTOptions : TtsModelOptions
{
    /// <summary>Creates the paper's configuration (LLaMA-13B).</summary>
    public SpeechGPTOptions()
    {
        SampleRate = 16000;
        HiddenDim = 5120;
        NumHeads = 40;
        VocabSize = 33006;
        MaxTextLength = 2048;
        DropoutRate = 0.0;
        LearningRate = 2e-4;
        WeightDecay = 0.0;
        Seed = 42;
    }

    /// <summary>The release's LLaMA-7B (fnlp/SpeechGPT-7B-cm): 32 layers, d 4096, 32 heads, SwiGLU 11008.</summary>
    public static SpeechGPTOptions Release7B() => new() { HiddenDim = 4096, NumHeads = 32, Layers = 32, Intermediate = 11008 };

    // ---------------------------------------------------------------- language model

    /// <summary>LLaMA layers (40) and SwiGLU width (13824).</summary>
    public int Layers { get; set; } = 40;

    /// <inheritdoc cref="Layers"/>
    public int Intermediate { get; set; } = 13824;

    /// <summary>RMSNorm ε (the release's 1e-6) and the rotary base (10000).</summary>
    public double RmsEpsilon { get; set; } = 1e-6;

    /// <inheritdoc cref="RmsEpsilon"/>
    public double RopeTheta { get; set; } = 10000;

    /// <summary>A directory with LLaMA's <c>tokenizer.model</c> and the release's <c>added_tokens.json</c>; text needs it.</summary>
    public string? TokenizerDirectory { get; set; }

    // ---------------------------------------------------------------- units

    /// <summary>Discrete units, K (1000 mHuBERT clusters).</summary>
    public int Units { get; set; } = 1000;

    /// <summary>mHuBERT's width (768), the layer read (11), heads (12), feed-forward width (3072) and feature-encoder
    /// channels (512).</summary>
    public int HubertWidth { get; set; } = 768;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertLayer { get; set; } = 11;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertHeads { get; set; } = 12;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertIntermediate { get; set; } = 3072;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertConvChannels { get; set; } = 512;

    // ---------------------------------------------------------------- vocoder

    /// <summary>The unit vocoder: the paper's multi-speaker one by default.</summary>
    public SpeechGptVocoder Vocoder { get; set; } = SpeechGptVocoder.MultiSpeaker;

    /// <summary>The multi-speaker vocoder's speaker table (speech-resynthesis' 200) and the speaker synthesis uses.</summary>
    public int VocoderSpeakers { get; set; } = 200;

    /// <inheritdoc cref="VocoderSpeakers"/>
    public int Speaker { get; set; }

    /// <summary>The vocoder's embedding width (128), first width (512), upsampling (5·4·4·2·2 = 320 samples per unit,
    /// 50 Hz) and kernels.</summary>
    public int VocoderEmbeddingDim { get; set; } = 128;

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int VocoderChannels { get; set; } = 512;

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int[] VocoderUpsampleRates { get; set; } = { 5, 4, 4, 2, 2 };

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int[] VocoderUpsampleKernels { get; set; } = { 11, 8, 8, 4, 4 };

    /// <summary>Whether to build HiFi-GAN's multi-period and multi-scale discriminators, which training the vocoder needs.</summary>
    public bool IncludeVocoderDiscriminators { get; set; }

    /// <summary>The vocoder's training segment in samples (speech-resynthesis' 8960, 28 units).</summary>
    public int VocoderSegmentSize { get; set; } = 8960;

    /// <summary>The vocoder's rate (2e-4, AdamW β 0.8 and 0.99), decayed by 0.999 per epoch of
    /// <see cref="VocoderUpdatesPerEpoch"/> updates; its losses weigh the mel L1 by 45 and the duration loss by 1.</summary>
    public double VocoderLearningRate { get; set; } = 2e-4;

    /// <inheritdoc cref="VocoderLearningRate"/>
    public double VocoderLearningRateDecay { get; set; } = 0.999;

    /// <inheritdoc cref="VocoderLearningRate"/>
    public int VocoderUpdatesPerEpoch { get; set; } = 1000;

    // ---------------------------------------------------------------- prompts and generation

    /// <summary>The release's meta instruction, before every prompt.</summary>
    public string MetaInstruction { get; set; } =
        "You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n";

    /// <summary>The task descriptions of TTS and ASR (App. B's first examples).</summary>
    public string TtsInstruction { get; set; } = "Can you please read this sentence out loud?";

    /// <inheritdoc cref="TtsInstruction"/>
    public string AsrInstruction { get; set; } = "Can you transcribe the speech into a written format?";

    /// <summary>Sampling (§5.1): temperature 0.8, top-k 60, top-p 0.8; 0 temperature decodes greedily.</summary>
    public double Temperature { get; set; } = 0.8;

    /// <inheritdoc cref="Temperature"/>
    public int TopK { get; set; } = 60;

    /// <inheritdoc cref="Temperature"/>
    public double TopP { get; set; } = 0.8;

    /// <summary>The most tokens a response may have, and the fewest before it may end (the release's 10).</summary>
    public int MaxNewTokens { get; set; } = 2048;

    /// <inheritdoc cref="MaxNewTokens"/>
    public int MinNewTokens { get; set; } = 10;

    // ---------------------------------------------------------------- training

    /// <summary>The stage a language-model step follows; cross-modal instruction fine-tuning by default, which TTS and
    /// ASR pairs are.</summary>
    public SpeechGptTrainingStage Stage { get; set; } = SpeechGptTrainingStage.CrossModalInstruction;

    /// <summary>Steps of the cosine schedule (App. D: 900, 4000 and 4200 for the three stages; the stage's default when
    /// 0), of which <see cref="WarmupFraction"/> warm up (the release's scripts: 3 %).</summary>
    public int TotalSteps { get; set; }

    /// <inheritdoc cref="TotalSteps"/>
    public double WarmupFraction { get; set; } = 0.03;

    /// <summary>Stage 3's LoRA on the query and value projections: rank 8, alpha 16, dropout 0.05 (App. D and the
    /// release's script).</summary>
    public int LoraRank { get; set; } = 8;

    /// <inheritdoc cref="LoraRank"/>
    public double LoraAlpha { get; set; } = 16;

    /// <inheritdoc cref="LoraRank"/>
    public double LoraDropout { get; set; } = 0.05;

    /// <summary>Gradient clipping (Hugging Face Trainer's default, 1.0).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
