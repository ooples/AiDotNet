using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Which voice encoder MinMo reads speech with.</summary>
public enum MinMoVoiceEncoder
{
    /// <summary>The paper's: SenseVoice-Large's encoder (636M; a Whisper-large-style encoder over 128 log-mel bands, 50
    /// frames a second). Its weights were not released; Whisper-large-v3's encoder loads into it.</summary>
    SenseVoiceLarge,

    /// <summary>SenseVoice-Small's SAN-M encoder (released): 80-band filterbanks stacked to 16.7 frames a second, 70 SAN-M
    /// blocks of 512.</summary>
    SenseVoiceSmall,
}

/// <summary>MinMo's training stages (§3.4).</summary>
public enum MinMoTrainingStage
{
    /// <summary>Speech-to-text pre-alignment: the input projector alone, on a tenth of the data.</summary>
    PreAlign,

    /// <summary>Speech-to-text full alignment: the input projector and the voice encoder.</summary>
    FullAlign,

    /// <summary>Speech-to-text instruction fine-tuning: the LLM through LoRA.</summary>
    InstructionTuning,

    /// <summary>Text-to-speech alignment, first part: the output projector alone.</summary>
    TextToSpeechProjector,

    /// <summary>Text-to-speech alignment, second part: the output projector and the voice token LM.</summary>
    TextToSpeech,

    /// <summary>Speech-to-speech alignment: the output projector and the voice token LM on spoken dialogues.</summary>
    SpeechToSpeech,

    /// <summary>Duplex interaction alignment: the full duplex predictor alone.</summary>
    Duplex,
}

/// <summary>What the full duplex predictor decides at a position of the conversation.</summary>
public enum MinMoDuplexAction
{
    /// <summary>Nothing changes: keep listening, or keep speaking through a back-channel.</summary>
    Continue,

    /// <summary>The user's turn has ended: respond.</summary>
    Respond,

    /// <summary>The user is taking the turn: stop speaking and listen.</summary>
    Stop,
}

/// <summary>Options for <see cref="MinMo{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's MinMo (Chen et al. 2025, Table 1): SenseVoice-Large's encoder, an input projector of
/// one 2×-downsampling convolution and two Transformer layers, Qwen2.5-7B-Instruct with LoRA, a linear output projector
/// into CosyVoice 2's voice token LM and its token2wav (<see cref="VoiceDecoder"/>), and a one-layer full duplex predictor.
/// Where the paper is silent the same team's released code decides: FunASR's LLM-ASR (<c>Transformer</c> adaptor, its
/// ChatML prompt, Qwen's generation settings) and CosyVoice 2.</para>
/// <para><b>For Beginners:</b> MinMo is a voice assistant built on a text chatbot: an ear turns speech into something the
/// chatbot reads, the chatbot writes its answer, and a voice reads the answer aloud as it is written.</para>
/// </remarks>
public class MinMoOptions : TtsModelOptions
{
    /// <summary>Creates the paper's MinMo.</summary>
    public MinMoOptions()
    {
        SampleRate = 24000;
        HiddenDim = 3584;
        NumHeads = 28;
        VocabSize = 152064;
        MaxTextLength = 512;
        DropoutRate = 0.0;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
        Seed = 42;
    }

    // ---------------------------------------------------------------- the voice encoder

    /// <summary>The voice encoder.</summary>
    public MinMoVoiceEncoder Encoder { get; set; } = MinMoVoiceEncoder.SenseVoiceLarge;

    /// <summary>SenseVoice-Large's encoder: log-mel bands (128), width (1280), heads (20) and layers (32).</summary>
    public int LargeEncoderMels { get; set; } = 128;

    /// <inheritdoc cref="LargeEncoderMels"/>
    public int LargeEncoderWidth { get; set; } = 1280;

    /// <inheritdoc cref="LargeEncoderMels"/>
    public int LargeEncoderHeads { get; set; } = 20;

    /// <inheritdoc cref="LargeEncoderMels"/>
    public int LargeEncoderLayers { get; set; } = 32;

    /// <summary>SenseVoice-Small's SAN-M encoder: filterbank bands (80), width (512), heads (4), feed-forward (2048),
    /// blocks (50) and blocks after the norm (20).</summary>
    public int SmallEncoderMels { get; set; } = 80;

    /// <inheritdoc cref="SmallEncoderMels"/>
    public int SmallEncoderWidth { get; set; } = 512;

    /// <inheritdoc cref="SmallEncoderMels"/>
    public int SmallEncoderHeads { get; set; } = 4;

    /// <inheritdoc cref="SmallEncoderMels"/>
    public int SmallEncoderFeedForward { get; set; } = 2048;

    /// <inheritdoc cref="SmallEncoderMels"/>
    public int SmallEncoderBlocks { get; set; } = 50;

    /// <inheritdoc cref="SmallEncoderMels"/>
    public int SmallEncoderTpBlocks { get; set; } = 20;

    // ---------------------------------------------------------------- the input projector

    /// <summary>The input projector (FunASR's <c>Transformer</c> adaptor): frames stacked by 2 (the 2× downsampling
    /// convolution), a 2048-wide ReLU bottleneck into the LLM's width, then 2 Transformer layers of 8 heads whose
    /// feed-forward is a quarter of the width.</summary>
    public int ProjectorDownsample { get; set; } = 2;

    /// <inheritdoc cref="ProjectorDownsample"/>
    public int ProjectorFeedForward { get; set; } = 2048;

    /// <inheritdoc cref="ProjectorDownsample"/>
    public int ProjectorLayers { get; set; } = 2;

    /// <inheritdoc cref="ProjectorDownsample"/>
    public int ProjectorHeads { get; set; } = 8;

    // ---------------------------------------------------------------- the LLM

    /// <summary>Qwen2.5-7B-Instruct: layers (28), key-value heads (4), feed-forward (18944), RoPE base (1e6) and RMSNorm
    /// epsilon (1e-6). <c>HiddenDim</c>, <c>NumHeads</c> and <c>VocabSize</c> are its width (3584), heads (28) and token
    /// table (152,064).</summary>
    public int Layers { get; set; } = 28;

    /// <inheritdoc cref="Layers"/>
    public int KeyValueHeads { get; set; } = 4;

    /// <inheritdoc cref="Layers"/>
    public int Intermediate { get; set; } = 18944;

    /// <inheritdoc cref="Layers"/>
    public double RopeTheta { get; set; } = 1_000_000;

    /// <inheritdoc cref="Layers"/>
    public double RmsEpsilon { get; set; } = 1e-6;

    /// <summary>The LoRA on the LLM's query and value projections for instruction tuning (not stated: rank 8, alpha 16,
    /// dropout 0.05).</summary>
    public int LoraRank { get; set; } = 8;

    /// <inheritdoc cref="LoraRank"/>
    public double LoraAlpha { get; set; } = 16;

    /// <inheritdoc cref="LoraRank"/>
    public double LoraDropout { get; set; } = 0.05;

    /// <summary>The system prompt (FunASR's LLM-ASR default).</summary>
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";

    /// <summary>The instruction Synthesize and Transcribe send (§3.3: "Speech Transcription" for recognition; Table 3's
    /// form for synthesis).</summary>
    public string TranscriptionInstruction { get; set; } = "Speech Transcription";

    /// <inheritdoc cref="TranscriptionInstruction"/>
    public string SynthesisInstruction { get; set; } = "Please read the following text aloud";

    /// <summary>Text generation: Qwen2.5-Instruct's sampling (temperature 0.7, top-p 0.8, top-k 20, repetition penalty
    /// 1.05); a temperature of 0 decodes greedily. At most <see cref="MaxNewTokens"/> tokens.</summary>
    public double Temperature { get; set; } = 0.7;

    /// <inheritdoc cref="Temperature"/>
    public double TopP { get; set; } = 0.8;

    /// <inheritdoc cref="Temperature"/>
    public int TopK { get; set; } = 20;

    /// <inheritdoc cref="Temperature"/>
    public double RepetitionPenalty { get; set; } = 1.05;

    /// <inheritdoc cref="Temperature"/>
    public int MaxNewTokens { get; set; } = 512;

    // ---------------------------------------------------------------- the voice decoder

    /// <summary>CosyVoice 2 (0.5B), whose language model is the voice token LM and whose flow and vocoder are the token2wav
    /// synthesizer. MinMo interleaves 5 semantic vectors with 15 speech tokens (its <c>TextChunk</c> and
    /// <c>SpeechChunk</c>).</summary>
    public CosyVoice2Options VoiceDecoder { get; set; } = new();

    // ---------------------------------------------------------------- the full duplex predictor

    /// <summary>The full duplex predictor (18M): the LLM's hidden states projected to 1024, one causal Transformer layer
    /// (8 heads, feed-forward 4096) and a linear softmax over <see cref="MinMoDuplexAction"/>. The paper fixes one layer
    /// and the parameter count; the width is chosen to match it.</summary>
    public int DuplexWidth { get; set; } = 1024;

    /// <inheritdoc cref="DuplexWidth"/>
    public int DuplexHeads { get; set; } = 8;

    /// <inheritdoc cref="DuplexWidth"/>
    public int DuplexFeedForward { get; set; } = 4096;

    // ---------------------------------------------------------------- training

    /// <summary>The stage a training step follows.</summary>
    public MinMoTrainingStage Stage { get; set; } = MinMoTrainingStage.PreAlign;

    /// <summary>Gradient clipping (not stated: 1.0).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
