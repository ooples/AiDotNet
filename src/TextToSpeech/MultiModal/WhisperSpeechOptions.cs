using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>What a WhisperSpeech training step trains.</summary>
public enum WhisperSpeechTrainingStage
{
    /// <summary>The text-to-semantic model (T2S): next-token cross-entropy over the recording's semantic tokens, plus 0.1 ×
    /// the causal text encoder's next-byte cross-entropy.</summary>
    TextToSemantic,

    /// <summary>The semantic-to-acoustic model (S2A): each EnCodec quantizer's next-token cross-entropy under the delay
    /// pattern, conditioned on the recording's speaker embedding.</summary>
    SemanticToAcoustic,

    /// <summary>The semantic tokenizer (the Whisper VQ bottleneck): the frozen Whisper decoder's cross-entropy on the
    /// transcript from the quantized states, plus KL divergence to Whisper's own logits, plus the commitment loss.</summary>
    SemanticTokenizer,
}

/// <summary>Options for <see cref="WhisperSpeech{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are what the release's <c>Pipeline()</c> loads (collabora/WhisperSpeech): t2s-small-en+pl and
/// s2a-q4-small-en+pl, whose semantic tokens are whisper-vq-stoks-medium-en+pl's (Whisper medium's encoder, quantized at 25
/// Hz to 512 codes), Vocos's EnCodec 24 kHz decoder over four EnCodec codebooks (3 kbps), and speechbrain's ECAPA speaker
/// embedding. <see cref="ForCheckpoints"/> configures the other released checkpoints. Training follows the release's
/// trainer (AdamW β (0.9, 0.95) with per-module rates and weight decay, the one-cycle cosine schedule with its momentum
/// cycle, gradient-norm clipping) at each model's tunables.</para>
/// <para><b>For Beginners:</b> WhisperSpeech turns text into "semantic tokens" (what Whisper hears), those into EnCodec
/// audio tokens in a chosen voice, and the audio tokens into sound.</para>
/// </remarks>
public class WhisperSpeechOptions : TtsModelOptions
{
    /// <summary>Creates the release's default pipeline.</summary>
    public WhisperSpeechOptions()
    {
        SampleRate = 24000;
        HopSize = 320;
        HiddenDim = 768;
        NumHeads = 12;
        VocabSize = 256;
        MaxTextLength = 550;
        DropoutRate = 0.0;
        LearningRate = 1.5e-3;
        WeightDecay = 0.1;
        Seed = 42;
        Codec = CreateReleaseCodec();
    }

    /// <summary>EnCodec's 24 kHz model at 3 kbps (four codebooks), the acoustic tokens the release's S2A models.</summary>
    public static EnCodecOptions CreateReleaseCodec()
    {
        var codec = EnCodecOptions.OfficialCheckpoint24kHz();
        codec.TargetBandwidthKbps = 3.0;
        return codec;
    }

    /// <summary>
    /// The configuration of released checkpoints (names as in the WhisperSpeech/WhisperSpeech repository, with or without
    /// <c>.model</c>): a T2S model, an S2A model and the semantic tokenizer their tokens come from (the en+pl models read
    /// whisper-vq-stoks-medium-en+pl, the 7- and 8-language ones whisper-vq-stoks-v3-7lang).
    /// </summary>
    public static WhisperSpeechOptions ForCheckpoints(string textToSemantic = "t2s-small-en+pl", string semanticToAcoustic = "s2a-q4-small-en+pl",
        string semanticTokenizer = "whisper-vq-stoks-medium-en+pl")
    {
        static string Name(string s) => (s ?? throw new ArgumentNullException(nameof(s))).EndsWith(".model", StringComparison.Ordinal) ? s.Substring(0, s.Length - 6) : s;
        var o = new WhisperSpeechOptions();
        // (depth, heads, encoder depth ratio, characters-per-second input, padding token offset, warmup)
        (int, int, double, bool, int, int) t2s = Name(textToSemantic) switch
        {
            "t2s-tiny-en+pl" => (4, 6, 0.25, true, -1, 4000),
            "t2s-base-en+pl" => (6, 8, 0.25, true, -1, 4000),
            "t2s-small-en+pl" or "t2s-small-yt" => (12, 12, 0.25, true, -1, 4000),
            "t2s-hq-fast-en+pl" or "t2s-fast-small-en+pl" or "t2s-fast-small-en+pl+yt" => (12, 12, 0.75, true, -1, 300),
            "t2s-fast-small-nocps-en+pl+yt" => (12, 12, 0.75, false, -1, 300),
            "t2s-fast-medium-en+pl+yt" => (24, 16, 0.75, true, -1, 300),
            "t2s-v1.1-small-en+pl" => (12, 12, 0.25, true, 0, 500),
            "t2s-v1.9-de+pl" => (4, 6, 0.25, true, -1, 300),
            "t2s-v1.9-medium-7lang" => (24, 16, 0.25, true, -1, 500),
            "t2s-v1.95-medium-7lang" => (24, 16, 0.25, true, 0, 500),
            "t2s-v1.95-small-8lang" => (12, 12, 0.25, true, 0, 500),
            _ => throw new ArgumentException($"Unknown T2S checkpoint '{textToSemantic}'.", nameof(textToSemantic)),
        };
        (o.T2SDepth, o.T2SHeads, o.T2SEncoderDepthRatio, o.T2SCharactersPerSecondInput, o.T2SPaddingTokenOffset, o.T2SWarmupSteps) = t2s;
        // (depth, heads, encoder depth ratio, hidden-to-embedding projection, warmup)
        (int, int, double, bool, int) s2a = Name(semanticToAcoustic) switch
        {
            "s2a-q4-tiny-en+pl" => (4, 6, 0.25, true, 2000),
            "s2a-q4-base-en+pl" => (6, 8, 0.25, true, 2000),
            "s2a-q4-small-en+pl" => (12, 12, 0.25, true, 2000),
            "s2a-q4-hq-fast-en+pl" => (12, 12, 0.75, false, 300),
            "s2a-v1.1-small-en+pl" or "s2a-v1.1-small-en+pl-noyt" => (12, 12, 0.25, false, 500),
            "s2a-v1.9-base-7lang" => (6, 8, 0.25, false, 300),
            "s2a-v1.9-medium-7lang" or "s2a-v1.95-medium-7lang" => (24, 16, 0.25, false, 500),
            "s2a-v1.95-small-fast-en" => (12, 12, 0.75, false, 500),
            _ => throw new ArgumentException($"Unknown S2A checkpoint '{semanticToAcoustic}'.", nameof(semanticToAcoustic)),
        };
        (o.S2ADepth, o.S2AHeads, o.S2AEncoderDepthRatio, o.S2AHiddenToEmbedding, o.S2AWarmupSteps) = s2a;
        o.VqWarmupSteps = Name(semanticTokenizer) switch
        {
            "whisper-vq-stoks-medium-en+pl" => 850,
            "whisper-vq-stoks-v3-7lang" => 300,
            _ => throw new ArgumentException($"Unknown semantic tokenizer checkpoint '{semanticTokenizer}'.", nameof(semanticTokenizer)),
        };
        return o;
    }

    // ---------------------------------------------------------------- inference

    /// <summary>The language spoken (a Whisper language code: "en", "pl", "de", …).</summary>
    public string Language { get; set; } = "en";

    /// <summary>The speaking rate in characters a second (15) that the T2S model is conditioned on.</summary>
    public double CharactersPerSecond { get; set; } = 15;

    /// <summary>The sampling temperature of both models (0.7).</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Sampling over the most likely k tokens only (null: all, the release's default).</summary>
    public int? TopK { get; set; }

    /// <summary>The Vocos bandwidth class decoding uses (null: the one of the codebook count, as the release picks it).</summary>
    public int? VocosBandwidthId { get; set; }

    // ---------------------------------------------------------------- semantic tokens

    /// <summary>The semantic codes (512); the T2S and S2A tables add the padding code.</summary>
    public int SemanticCodes { get; set; } = 512;

    /// <summary>The semantic codebook's width (64), which the T2S and S2A models read frozen.</summary>
    public int SemanticWidth { get; set; } = 64;

    // ---------------------------------------------------------------- text to semantic (t2s-small-en+pl)

    /// <summary>The T2S model's depth (12: 6 encoder and 18 decoder blocks at the 0.25 ratio).</summary>
    public int T2SDepth { get; set; } = 12;

    /// <summary>The T2S model's heads (12); with <see cref="T2SHeadWidth"/> its width.</summary>
    public int T2SHeads { get; set; } = 12;

    /// <summary>The T2S model's head width (64).</summary>
    public int T2SHeadWidth { get; set; } = 64;

    /// <summary>The T2S feed-forward multiple (4).</summary>
    public int T2SFfnMult { get; set; } = 4;

    /// <summary>The share of the T2S blocks in the text encoder (0.25).</summary>
    public double T2SEncoderDepthRatio { get; set; } = 0.25;

    /// <summary>The T2S text encoder is causal and adds its 0.1-weighted next-byte loss (true).</summary>
    public bool T2SCausalEncoder { get; set; } = true;

    /// <summary>The T2S decoder is conditioned on the characters-per-second bin (true).</summary>
    public bool T2SCharactersPerSecondInput { get; set; } = true;

    /// <summary>The T2S padding and start token relative to the semantic table (−1: the last code; 0: past it).</summary>
    public int T2SPaddingTokenOffset { get; set; } = -1;

    /// <summary>The T2S initialization scale (1) and embedding deviation (0.01).</summary>
    public double T2SInitStd { get; set; } = 1;

    /// <inheritdoc cref="T2SInitStd"/>
    public double T2SEmbeddingsStd { get; set; } = 0.01;

    /// <summary>The T2S logit multiplier (0.35) and query multiplier (1).</summary>
    public double T2SOutputMult { get; set; } = 0.35;

    /// <inheritdoc cref="T2SOutputMult"/>
    public double T2SQueryMult { get; set; } = 1;

    /// <summary>The T2S embeddings' (5) and embedding projectors' (2.5) learning-rate multiples.</summary>
    public double T2SEmbeddingsLrScale { get; set; } = 5;

    /// <inheritdoc cref="T2SEmbeddingsLrScale"/>
    public double T2SProjectorLrScale { get; set; } = 2.5;

    /// <summary>The T2S peak learning rate (1.5e-3), gradient-norm clip (0.2), weight decay (0.1) and warmup (4000).</summary>
    public double T2SLearningRate { get; set; } = 1.5e-3;

    /// <inheritdoc cref="T2SLearningRate"/>
    public double T2SMaxGradientNorm { get; set; } = 0.2;

    /// <inheritdoc cref="T2SLearningRate"/>
    public double T2SWeightDecay { get; set; } = 0.1;

    /// <inheritdoc cref="T2SLearningRate"/>
    public int T2SWarmupSteps { get; set; } = 4000;

    // ---------------------------------------------------------------- semantic to acoustic (s2a-q4-small-en+pl)

    /// <summary>The S2A model's depth (12), heads (12), head width (64) and feed-forward multiple (4).</summary>
    public int S2ADepth { get; set; } = 12;

    /// <inheritdoc cref="S2ADepth"/>
    public int S2AHeads { get; set; } = 12;

    /// <inheritdoc cref="S2ADepth"/>
    public int S2AHeadWidth { get; set; } = 64;

    /// <inheritdoc cref="S2ADepth"/>
    public int S2AFfnMult { get; set; } = 4;

    /// <summary>The share of the S2A blocks in the semantic encoder (0.25).</summary>
    public double S2AEncoderDepthRatio { get; set; } = 0.25;

    /// <summary>The S2A semantic encoder is causal and adds its next-token loss (false).</summary>
    public bool S2ACausalEncoder { get; set; }

    /// <summary>The S2A model keeps the hidden-to-embedding projection (<c>force_hidden_to_emb</c>; true for the q4
    /// checkpoints, false from v1.1 on).</summary>
    public bool S2AHiddenToEmbedding { get; set; } = true;

    /// <summary>The conditioned S2A variant (language, SNR and C50 embeddings with the speaker's).</summary>
    public bool S2AConditioned { get; set; }

    /// <summary>The S2A initialization scale (9) and embedding deviation (0.2).</summary>
    public double S2AInitStd { get; set; } = 9;

    /// <inheritdoc cref="S2AInitStd"/>
    public double S2AEmbeddingsStd { get; set; } = 0.2;

    /// <summary>The S2A logit multiplier (5.6) and query multiplier (0.3).</summary>
    public double S2AOutputMult { get; set; } = 5.6;

    /// <inheritdoc cref="S2AOutputMult"/>
    public double S2AQueryMult { get; set; } = 0.3;

    /// <summary>The S2A embeddings' learning-rate multiple (10).</summary>
    public double S2AEmbeddingsLrScale { get; set; } = 10;

    /// <summary>The S2A peak learning rate (3e-3), gradient-norm clip (2), weight decay (1e-3) and warmup (2000).</summary>
    public double S2ALearningRate { get; set; } = 3e-3;

    /// <inheritdoc cref="S2ALearningRate"/>
    public double S2AMaxGradientNorm { get; set; } = 2;

    /// <inheritdoc cref="S2ALearningRate"/>
    public double S2AWeightDecay { get; set; } = 1e-3;

    /// <inheritdoc cref="S2ALearningRate"/>
    public int S2AWarmupSteps { get; set; } = 2000;

    // ---------------------------------------------------------------- the semantic tokenizer (whisper-vq-stoks-medium-en+pl)

    /// <summary>The Whisper model the tokenizer quantizes (medium: 1024 wide, 16 heads, 24 encoder and 24 decoder layers,
    /// 1500 audio and 448 text positions, 80 mels, the 51,865-token multilingual vocabulary).</summary>
    public int WhisperWidth { get; set; } = 1024;

    /// <inheritdoc cref="WhisperWidth"/>
    public int WhisperHeads { get; set; } = 16;

    /// <inheritdoc cref="WhisperWidth"/>
    public int WhisperLayers { get; set; } = 24;

    /// <summary>Whisper's audio positions (1500, a 30-second window); the semantic tokens of a window are this over
    /// <see cref="VqDownsample"/>, and the S2A context three times that.</summary>
    public int WhisperAudioContext { get; set; } = 1500;

    /// <inheritdoc cref="WhisperWidth"/>
    public int WhisperTextContext { get; set; } = 448;

    /// <inheritdoc cref="WhisperWidth"/>
    public int WhisperMels { get; set; } = 80;

    /// <inheritdoc cref="WhisperWidth"/>
    public int WhisperVocabulary { get; set; } = 51865;

    /// <summary>The tokenizer's output blocks (1) and heads (16; their width is <see cref="WhisperWidth"/> over them).</summary>
    public int VqDepth { get; set; } = 1;

    /// <inheritdoc cref="VqDepth"/>
    public int VqHeads { get; set; } = 16;

    /// <summary>Whisper frames per semantic token (2, averaged).</summary>
    public int VqDownsample { get; set; } = 2;

    /// <summary>The tokenizer's initialization scale (1.5), position deviation (0.045) and query multiplier (2).</summary>
    public double VqInitStd { get; set; } = 1.5;

    /// <inheritdoc cref="VqInitStd"/>
    public double VqEmbeddingsStd { get; set; } = 0.045;

    /// <inheritdoc cref="VqInitStd"/>
    public double VqQueryMult { get; set; } = 2;

    /// <summary>The codebook's moving-average decay (0.9) and dead-code threshold (0: the release trains without
    /// replacement).</summary>
    public double VqCodebookDecay { get; set; } = 0.9;

    /// <inheritdoc cref="VqCodebookDecay"/>
    public double VqDeadCodeThreshold { get; set; }

    /// <summary>The tokenizer's peak learning rate (9e-4), gradient-norm clip (2), weight decay (1e-3) and warmup (850).</summary>
    public double VqLearningRate { get; set; } = 9e-4;

    /// <inheritdoc cref="VqLearningRate"/>
    public double VqMaxGradientNorm { get; set; } = 2;

    /// <inheritdoc cref="VqLearningRate"/>
    public double VqWeightDecay { get; set; } = 1e-3;

    /// <inheritdoc cref="VqLearningRate"/>
    public int VqWarmupSteps { get; set; } = 850;

    // ---------------------------------------------------------------- speaker, codec and vocoder

    /// <summary>The ECAPA-TDNN speaker extractor's stage widths (1024, 1024, 1024, 1024, 3072), attention and
    /// squeeze-excitation widths (128) and embedding size (192, the S2A speaker width).</summary>
    public int[] SpeakerChannels { get; set; } = { 1024, 1024, 1024, 1024, 3072 };

    /// <inheritdoc cref="SpeakerChannels"/>
    public int SpeakerAttentionChannels { get; set; } = 128;

    /// <inheritdoc cref="SpeakerChannels"/>
    public int SpeakerSeChannels { get; set; } = 128;

    /// <inheritdoc cref="SpeakerChannels"/>
    public int SpeakerEmbeddingSize { get; set; } = 192;

    /// <summary>The codec whose codes the S2A model writes (EnCodec 24 kHz at 3 kbps: four codebooks), frozen.</summary>
    public EnCodecOptions Codec { get; set; }

    /// <summary>Vocos's backbone width (384), ConvNeXt width (1152) and blocks (8).</summary>
    public int VocosDim { get; set; } = 384;

    /// <inheritdoc cref="VocosDim"/>
    public int VocosIntermediateDim { get; set; } = 1152;

    /// <inheritdoc cref="VocosDim"/>
    public int VocosLayers { get; set; } = 8;

    // ---------------------------------------------------------------- training

    /// <summary>What training steps train (the T2S model by default).</summary>
    public WhisperSpeechTrainingStage Stage { get; set; } = WhisperSpeechTrainingStage.TextToSemantic;

    /// <summary>The steps of the one-cycle schedule (8000, the release trainer's default iterations).</summary>
    public int TotalSteps { get; set; } = 8000;
}
