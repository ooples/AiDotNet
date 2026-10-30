using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>How GLM-4-Voice lays out its spoken reply (Zeng et al. 2024b §3.3).</summary>
public enum GlmVoiceGeneration
{
    /// <summary>The paper's "streaming thoughts": after <c>&lt;|assistant|&gt;streaming_transcription</c> the reply
    /// alternates 13 text tokens and 26 speech tokens, so speech starts before the text is complete.</summary>
    StreamingThoughts,

    /// <summary>The decoupled two-step generation (§3.3, with the published template of Zeng et al. 2024a App. F): the
    /// whole text reply after <c>&lt;|assistant|&gt;transcript</c>, then the speech after
    /// <c>&lt;|assistant|&gt;&lt;|begin_of_audio|&gt;</c>.</summary>
    TextFirst,
}

/// <summary>How <see cref="GLM4Voice{T}.Synthesize"/> reads a text aloud.</summary>
public enum GlmVoiceTtsFormat
{
    /// <summary>The paper's speech-and-text-to-speech subtask (§3.3) on the chat model: the text as the user's turn
    /// and as the assistant's text, which is fed rather than generated; the model writes the speech tokens around it in
    /// the <see cref="GLM4VoiceOptions.Generation"/> layout.</summary>
    ForcedTextThoughts,

    /// <summary>A base model's text-to-speech in the published interleaved pre-training format (Zeng et al. 2024a
    /// App. D.3): <c>{text}&lt;|begin_of_audio|&gt;</c>, then speech tokens until <c>&lt;|end_of_audio|&gt;</c>. The
    /// release has no base checkpoint; the paper's own TTS prompt is not published.</summary>
    InterleavedBase,
}

/// <summary>Which of GLM-4-Voice's two language-model stages a training step follows (§4).</summary>
public enum GlmVoiceTrainingStage
{
    /// <summary>Stage 1, joint speech-text pre-training (§4.1.1): AdamW (0.9, 0.95), the rate decaying linearly from
    /// 6e-5 to 6e-6, no dropout.</summary>
    PreTraining,

    /// <summary>Stage 2, supervised fine-tuning (§4.2.2): the rate decaying from 1e-5 to 1e-6, weight decay 0.1,
    /// dropout 0.5 on the hidden states, gradients clipped at 1.0.</summary>
    FineTuning,
}

/// <summary>Options for <see cref="GLM4Voice{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the release (THUDM/glm-4-voice-9b, glm-4-voice-tokenizer, glm-4-voice-decoder):
/// <list type="bullet">
/// <item>the ChatGLM language model: 40 layers, d 4096, 32 heads of 128 in 2 key-value groups, SwiGLU 13696 and a
/// 168960-token vocabulary of text and speech tokens;</item>
/// <item>the Whisper-VQ speech tokenizer: whisper-large-v3's encoder up to layer 16, pooled 4× and quantized to 16384
/// codes at 12.5 Hz;</item>
/// <item>the CosyVoice flow decoder: a 6-block Conformer with 10-token attention blocks and a conditional flow-matching
/// U-Net, then HiFT at 22.05 kHz.</item>
/// </list></para>
/// <para><b>For Beginners:</b> GLM-4-Voice is a voice chatbot. It turns speech into "speech words" (tokens), a large
/// language model writes a reply that mixes text words and speech words, and a decoder turns the speech words back into
/// sound. These settings size each of those parts and control how replies are generated and how each part is
/// trained.</para>
/// </remarks>
public class GLM4VoiceOptions : TtsModelOptions
{
    /// <summary>Creates the release's configuration.</summary>
    public GLM4VoiceOptions()
    {
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        HiddenDim = 4096;
        NumHeads = 32;
        VocabSize = 168960;
        MaxTextLength = 8192;
        DropoutRate = 0.0;
        LearningRate = 1e-5;
        WeightDecay = 0.1;
        // Sampling, the decoder's per-call noise (the release seeds torch with 42 there) and training's draws.
        Seed = 42;
    }

    // ---------------------------------------------------------------- language model

    /// <summary>Transformer layers of the language model (40).</summary>
    public int Layers { get; set; } = 40;

    /// <summary>Key-value groups of the multi-query attention (2).</summary>
    public int KeyValueGroups { get; set; } = 2;

    /// <summary>Width of each attention head (<c>kv_channels</c>, 128).</summary>
    public int HeadDim { get; set; } = 128;

    /// <summary>The SwiGLU inner width (13696).</summary>
    public int FeedForwardDim { get; set; } = 13696;

    /// <summary>The RMSNorm ε (3.90625e-8).</summary>
    public double NormEpsilon { get; set; } = 3.90625e-08;

    /// <summary>The rotary base multiplier (<c>rope_ratio</c>, 1).</summary>
    public double RopeRatio { get; set; } = 1.0;

    /// <summary>A model directory with GLM-4's <c>tokenizer.model</c> and <c>tokenizer_config.json</c>; text needs it.</summary>
    public string? TokenizerDirectory { get; set; }

    // ---------------------------------------------------------------- speech tokenizer

    /// <summary>The speech tokenizer's width (whisper-large-v3: 1280).</summary>
    public int SpeechTokenizerDim { get; set; } = 1280;

    /// <summary>The speech tokenizer's attention heads (20).</summary>
    public int SpeechTokenizerHeads { get; set; } = 20;

    /// <summary>The speech tokenizer's feed-forward width (5120).</summary>
    public int SpeechTokenizerFeedForwardDim { get; set; } = 5120;

    /// <summary>Whisper encoder layers in all (32); tokenizing runs only the first <see cref="QuantizePosition"/>.</summary>
    public int SpeechTokenizerLayers { get; set; } = 32;

    /// <summary>The encoder's position table (1500 frames, 30 s).</summary>
    public int SpeechTokenizerMaxPositions { get; set; } = 1500;

    /// <summary>The average pooling's kernel and stride (4).</summary>
    public int PoolingKernel { get; set; } = 4;

    /// <summary>The layer after which the pooling runs (16).</summary>
    public int PoolingPosition { get; set; } = 16;

    /// <summary>The layer after which the vector quantizer runs (16).</summary>
    public int QuantizePosition { get; set; } = 16;

    /// <summary>Speech tokens (VQ codes): 16384.</summary>
    public int SpeechCodebookSize { get; set; } = 16384;

    /// <summary>Encoder frames per block-causal attention block before pooling (200: 4 s).</summary>
    public int CausalBlockSize { get; set; } = 200;

    /// <summary>The codebook's EMA decay (0.99).</summary>
    public double EmaDecay { get; set; } = 0.99;

    /// <summary>The commitment coefficient (0.25) and the quantize loss's scale (10): the release's
    /// <c>quantize_commit_coefficient</c> and <c>quantize_loss_scale</c>.</summary>
    public double CommitCoefficient { get; set; } = 0.25;

    /// <inheritdoc cref="CommitCoefficient"/>
    public double QuantizeLossScale { get; set; } = 10.0;

    /// <summary>Codebook updates between dead-code restarts (100).</summary>
    public int RestartInterval { get; set; } = 100;

    /// <summary>
    /// Whether to build the speech tokenizer's training stack: whisper-large-v3's remaining encoder layers and its
    /// decoder, which carry the ASR loss that trains the tokenizer (§3.1). Off by default: tokenizing and the other
    /// trainings do not need it, and at the release's size it is about 1.4 billion parameters.
    /// </summary>
    public bool IncludeSpeechTokenizerTraining { get; set; }

    /// <summary>Whisper decoder layers (32), vocabulary (51866) and positions (448), for the ASR loss.</summary>
    public int AsrDecoderLayers { get; set; } = 32;

    /// <inheritdoc cref="AsrDecoderLayers"/>
    public int AsrVocabulary { get; set; } = 51866;

    /// <inheritdoc cref="AsrDecoderLayers"/>
    public int AsrMaxTargetPositions { get; set; } = 448;

    /// <summary>Whisper's <c>decoder_start_token_id</c> (<c>&lt;|startoftranscript|&gt;</c>, 50258), which the ASR
    /// decoder's input starts with.</summary>
    public int AsrDecoderStartId { get; set; } = 50258;

    /// <summary>The speech tokenizer's fine-tuning rate (Zeng et al. 2024a App. B.1: 1e-5, from whisper-large-v3).</summary>
    public double SpeechTokenizerLearningRate { get; set; } = 1e-5;

    // ---------------------------------------------------------------- speech decoder

    /// <summary>The flow's token embedding and Conformer width (512).</summary>
    public int FlowDim { get; set; } = 512;

    /// <summary>The Conformer's heads (8), feed-forward width (2048) and blocks (6).</summary>
    public int FlowHeads { get; set; } = 8;

    /// <inheritdoc cref="FlowHeads"/>
    public int FlowFeedForwardDim { get; set; } = 2048;

    /// <inheritdoc cref="FlowHeads"/>
    public int FlowLayers { get; set; } = 6;

    /// <summary>Speech tokens per decoding block and Conformer attention block (10: b = 0.8 s, §3.2).</summary>
    public int FlowBlockTokens { get; set; } = 10;

    /// <summary>The U-Net estimator's channels ([256, 256]), heads (8 of 64), transformer blocks per level (4) and
    /// middle blocks (12).</summary>
    public int[] DecoderChannels { get; set; } = { 256, 256 };

    /// <inheritdoc cref="DecoderChannels"/>
    public int DecoderHeads { get; set; } = 8;

    /// <inheritdoc cref="DecoderChannels"/>
    public int DecoderHeadDim { get; set; } = 64;

    /// <inheritdoc cref="DecoderChannels"/>
    public int DecoderBlocks { get; set; } = 4;

    /// <inheritdoc cref="DecoderChannels"/>
    public int DecoderMidBlocks { get; set; } = 12;

    /// <summary>Euler steps of the flow (10).</summary>
    public int FlowTimesteps { get; set; } = 10;

    /// <summary>The classifier-free guidance weight at inference (0.7) and the condition-drop rate in training (0.2).</summary>
    public double InferenceCfgRate { get; set; } = 0.7;

    /// <inheritdoc cref="InferenceCfgRate"/>
    public double TrainingCfgRate { get; set; } = 0.2;

    /// <summary>The flow-matching loss: CosyVoice's paper's absolute error (the release's config also names
    /// <c>l1</c>), or its code's squared error.</summary>
    public CosyVoiceFlowLoss FlowLoss { get; set; } = CosyVoiceFlowLoss.AbsoluteError;

    /// <summary>HiFT's base channels (512) and F0 predictor channels (512).</summary>
    public int HiftBaseChannels { get; set; } = 512;

    /// <inheritdoc cref="HiftBaseChannels"/>
    public int HiftF0Channels { get; set; } = 512;

    /// <summary>How speech tokens are decoded to audio: the paper's 0.8 s blocks by default.</summary>
    public GlmVoiceDecoding Decoding { get; set; } = GlmVoiceDecoding.PaperBlocks;

    /// <summary>Whether decoder training uses the paper's truncated streaming samples (§3.2: the first n·b seconds of
    /// each utterance), as its fine-tuning stage does.</summary>
    public bool StreamingTruncation { get; set; } = true;

    /// <summary>The decoder's training rate (Zeng et al. 2024a App. B.2: 1e-3).</summary>
    public double DecoderLearningRate { get; set; } = 1e-3;

    // ---------------------------------------------------------------- generation

    /// <summary>The reply's layout: streaming thoughts by default.</summary>
    public GlmVoiceGeneration Generation { get; set; } = GlmVoiceGeneration.StreamingThoughts;

    /// <summary>What <see cref="GLM4Voice{T}.Synthesize"/> does with a text: forced-text thoughts by default.</summary>
    public GlmVoiceTtsFormat TtsFormat { get; set; } = GlmVoiceTtsFormat.ForcedTextThoughts;

    /// <summary>Text tokens, then speech tokens, per streaming-thoughts chunk (13 and 26).</summary>
    public int TextTokensPerChunk { get; set; } = 13;

    /// <inheritdoc cref="TextTokensPerChunk"/>
    public int SpeechTokensPerChunk { get; set; } = 26;

    /// <summary>The system prompt; null uses the release's (the web demo's for a speech or a text input under
    /// streaming thoughts, Zeng et al. 2024a App. F's under text-first generation).</summary>
    public string? SystemPrompt { get; set; }

    /// <summary>Sampling temperature (the web demo's 0.2) and nucleus (0.8); the release always samples. Each call
    /// samples from a generator seeded with <see cref="Models.Options.ModelOptions.Seed"/>, so a call is
    /// reproducible.</summary>
    public double Temperature { get; set; } = 0.2;

    /// <inheritdoc cref="Temperature"/>
    public double TopP { get; set; } = 0.8;

    /// <summary>The most tokens a reply may have (the web demo's 2000).</summary>
    public int MaxNewTokens { get; set; } = 2000;

    // ---------------------------------------------------------------- language-model training

    /// <summary>Which stage the language model's training steps follow: fine-tuning by default.</summary>
    public GlmVoiceTrainingStage Stage { get; set; } = GlmVoiceTrainingStage.FineTuning;

    /// <summary>Pre-training's peak rate (§4.1.1: 6e-5); fine-tuning's is <see cref="TtsModelOptions.LearningRate"/>
    /// (§4.2.2: 1e-5). Either decays linearly to a tenth.</summary>
    public double PreTrainingLearningRate { get; set; } = 6e-5;

    /// <summary>Steps over which the rate decays linearly to a tenth (§4.1.1, §4.2.2 give the endpoints, not the
    /// length).</summary>
    public int DecaySteps { get; set; } = 100000;

    /// <summary>Fine-tuning epochs of the speech part and of the text part of each split sample (§4.2.2: 20 and 4).</summary>
    public int SpeechEpochs { get; set; } = 20;

    /// <inheritdoc cref="SpeechEpochs"/>
    public int TextEpochs { get; set; } = 4;

    /// <summary>Dropout on the hidden states during fine-tuning (§4.2.2: 0.5).</summary>
    public double FineTuningDropout { get; set; } = 0.5;

    /// <summary>Gradient clipping (§4.2.2: 1.0).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
