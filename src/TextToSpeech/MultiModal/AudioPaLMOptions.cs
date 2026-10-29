using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Generation;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>How AudioPaLM turns speech into audio tokens (Rubenstein et al. 2023 §3.1).</summary>
public enum AudioPalmSpeechTokenizer
{
    /// <summary>The paper's w2v-BERT tokens: a w2v-BERT model's embeddings, not normalized, clustered by k-means into
    /// 1024 tokens at 25 Hz (<see cref="W2vBert{T}"/>).</summary>
    W2vBert,

    /// <summary>Meta's public w2v-BERT 2.0 (<c>facebook/w2v-bert-2.0</c>): a hidden layer's embeddings clustered by
    /// k-means (50 Hz), a stand-in for the unreleased USM encoders.</summary>
    W2vBert2,
}

/// <summary>How AudioPaLM turns audio tokens into speech (§3.3).</summary>
public enum AudioPalmDetokenizer
{
    /// <summary>SoundStorm's non-autoregressive acoustic model (<see cref="SoundStorm{T}"/>).</summary>
    SoundStorm,

    /// <summary>AudioLM's autoregressive stages 2 and 3 (<see cref="AudioLM{T}"/>).</summary>
    AudioLM,
}

/// <summary>Options for <see cref="AudioPaLM{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are PaLM 8B (Chowdhery et al. 2022 Table 1: 32 layers, d 4096, 16 heads of 256, multi-query,
/// d_ff 4·d, a 256k SentencePiece vocabulary) extended by 1024 audio tokens (§3.2), w2v-BERT tokens and SoundStorm,
/// fine-tuned as PaLM §6.1.2 fine-tunes (Adafactor at a constant 5e-5, dropout 0.1, loss masked on the inputs).
/// <see cref="Gemma2B"/> configures the public Gemma-2B decoder instead.</para>
/// <para><b>For Beginners:</b> AudioPaLM is a text language model that has also learned "audio words", so the same
/// model can read text aloud, transcribe speech or translate speech into another language's speech. These settings size
/// the language model, choose how speech becomes audio words and back, and control training and generation.</para>
/// </remarks>
public class AudioPaLMOptions : TtsModelOptions
{
    /// <summary>Creates PaLM 8B's configuration.</summary>
    public AudioPaLMOptions()
    {
        SampleRate = 16000;
        HiddenDim = 4096;
        NumHeads = 16;
        VocabSize = 256000;
        MaxTextLength = 2048;
        DropoutRate = 0.1;
        LearningRate = 5e-5;
        WeightDecay = 0.0;
    }

    /// <summary>§5.4.8's optimization for the PaLM-2 experiments: dropout 0.2 and a linear ramp to 1e-4 followed by an
    /// exponential decay to 1e-5 over the given steps (the paper does not state them).</summary>
    public static AudioPaLMOptions PaLM2Ablation(int warmupSteps, int decaySteps) => new()
    {
        DropoutRate = 0.2,
        RampWarmupSteps = warmupSteps,
        RampDecaySteps = decaySteps,
    };

    /// <summary>The public Gemma-2B decoder (Gemma Team 2024: 18 layers, d 2048, 8 query heads and one key/value head of
    /// 256, GeGLU d_ff 16384, its own 256k vocabulary with &lt;bos&gt; 2 and &lt;eos&gt; 1), which
    /// <c>LoadGemmaWeights</c> reads.</summary>
    public static AudioPaLMOptions Gemma2B() => new()
    {
        Decoder = PalmDecoderKind.Gemma,
        HiddenDim = 2048,
        Layers = 18,
        NumHeads = 8,
        KeyValueHeads = 1,
        HeadDim = 256,
        IntermediateSize = 16384,
        NormEpsilon = 1e-6,
        BeginId = 2,
        EndId = 1,
    };

    // ---------------------------------------------------------------- the decoder

    /// <summary>The decoder's architecture (PaLM).</summary>
    public PalmDecoderKind Decoder { get; set; } = PalmDecoderKind.Palm;

    /// <summary>Transformer layers (32).</summary>
    public int Layers { get; set; } = 32;

    /// <summary>Key/value heads (1: multi-query).</summary>
    public int KeyValueHeads { get; set; } = 1;

    /// <summary>Width of each head (256).</summary>
    public int HeadDim { get; set; } = 256;

    /// <summary>The MLP's inner width (16384 = 4·d).</summary>
    public int IntermediateSize { get; set; } = 16384;

    /// <summary>The norms' ε (1e-5, PaLM-pytorch's LayerNorm; Gemma's RMSNorm uses 1e-6).</summary>
    public double NormEpsilon { get; set; } = 1e-5;

    /// <summary>The rotary base (10000).</summary>
    public double RopeTheta { get; set; } = 10000;

    // ---------------------------------------------------------------- tokens

    /// <summary>Audio tokens appended after the text vocabulary (1024, §3.1).</summary>
    public int AudioTokens { get; set; } = 1024;

    /// <summary>The text SentencePiece model (PaLM's is not public; Gemma's <c>tokenizer.model</c> with
    /// <see cref="Gemma2B"/>).</summary>
    public string? SentencePieceModelPath { get; set; }

    /// <summary>The token every sequence starts with (0, T5X's decoder start; Gemma's &lt;bos&gt; 2).</summary>
    public int BeginId { get; set; }

    /// <summary>The end token (1).</summary>
    public int EndId { get; set; } = 1;

    /// <summary>The language the TTS task names (English).</summary>
    public string Language { get; set; } = "English";

    // ---------------------------------------------------------------- audio tokens and audio

    /// <summary>How speech becomes audio tokens (w2v-BERT).</summary>
    public AudioPalmSpeechTokenizer SpeechTokenizer { get; set; } = AudioPalmSpeechTokenizer.W2vBert;

    /// <summary>The w2v-BERT tokenizer's options (w2v-BERT XL, its 7th masked block, 1024 clusters, not normalized).</summary>
    public W2vBertOptions W2vBert { get; set; } = new();

    /// <summary>The w2v-BERT 2.0 hidden state clustered (12 of 24, the middle layer, as §3.1 takes USM's).</summary>
    public int W2vBert2Layer { get; set; } = 12;

    /// <summary>How audio tokens become speech (SoundStorm).</summary>
    public AudioPalmDetokenizer Detokenizer { get; set; } = AudioPalmDetokenizer.SoundStorm;

    /// <summary>SoundStorm's options (its acoustic model only; AudioPaLM supplies the semantic tokens).</summary>
    public SoundStormOptions SoundStorm { get; set; } = new() { IncludeTextToSemantic = false };

    /// <summary>AudioLM's options (its acoustic stages read AudioPaLM's tokens).</summary>
    public AudioLMOptions AudioLM { get; set; } = new();

    /// <summary>Seconds of voice prompt the detokenizer continues (3, repeated when the recording is shorter; §3.3).</summary>
    public double VoicePromptSeconds { get; set; } = 3.0;

    // ---------------------------------------------------------------- training and generation

    /// <summary>Adafactor's first-moment decay (0.9, PaLM §5).</summary>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Global gradient-norm clip (1.0, PaLM §5).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Warm-up steps of §5.4.8's schedule (a linear ramp to <see cref="RampPeakLearningRate"/>); 0 with
    /// <see cref="RampDecaySteps"/> 0 keeps §3.6's constant rate.</summary>
    public int RampWarmupSteps { get; set; }

    /// <summary>Steps over which §5.4.8's schedule decays exponentially to <see cref="RampFinalLearningRate"/>.</summary>
    public int RampDecaySteps { get; set; }

    /// <summary>§5.4.8's peak rate (1e-4).</summary>
    public double RampPeakLearningRate { get; set; } = 1e-4;

    /// <summary>§5.4.8's final rate (1e-5).</summary>
    public double RampFinalLearningRate { get; set; } = 1e-5;

    /// <summary>Weight of PaLM's auxiliary z-loss, 1e-4 · log²Z (§5).</summary>
    public double ZLossWeight { get; set; } = 1e-4;

    /// <summary>Seed of the training draws (dropout).</summary>
    public int TrainingSeed { get; set; } = 1;

    /// <summary>Most tokens a generation adds (1500, a minute of 25 Hz audio tokens).</summary>
    public int MaxNewTokens { get; set; } = 1500;

    /// <summary>Sampling temperature; 0 decodes greedily (0).</summary>
    public double Temperature { get; set; }

    /// <summary>Seed of sampling and of the detokenizer's draws.</summary>
    public int SamplingSeed { get; set; } = 1;
}
