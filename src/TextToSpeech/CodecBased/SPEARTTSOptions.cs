using System;
using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for SPEAR-TTS (Kharitonov et al. 2023, "Speak, Read and Prompt: High-Fidelity Text-to-Speech with
/// Minimal Supervision").</summary>
/// <remarks>
/// <para>
/// <b>Tokens</b> (§6.1, §7.1): semantic tokens are the 7th-layer w2v-BERT embeddings quantized to 512 clusters at
/// 25 Hz, with consecutive repetitions removed; acoustic tokens are the first 3 RVQ levels (1024 codes each) of a
/// 16 kHz SoundStream at 50 frames per second, interleaved per frame (150 tokens a second).
/// </para>
/// <para>
/// <b>Reading</b>, S1 (§4, §7.2): text (phonemes) to semantic tokens with T5-Large (24 encoder and 24 decoder layers,
/// width 1024, feed-forward 4096, 16 heads of 64). It is pretrained on semantic tokens to reconstruct each sequence
/// from a copy with every token deleted with probability 0.6, at dropout 0.5 (1M updates, batch 256), then finetuned.
/// Finetuning on parallel data updates the encoder's embeddings and its lower <see cref="ReadingTrainableEncoderLayers"/>
/// with the rest frozen, at a dropout searched over {0.1, 0.3, 0.5} and label smoothing 0.1. Backtranslation
/// finetunes the same pretrained model the other way, semantic tokens to text, updating the top and bottom
/// <see cref="BacktranslationTrainableDecoderLayers"/> decoder layers. Every stage uses Adafactor with inverse
/// square-root decay. Synthesis reads with beam search (beam 10).
/// </para>
/// <para>
/// <b>Speaking</b>, S2 (§5, §7.3): semantic to acoustic tokens with a decoder-only Transformer (12 layers, 12 heads of
/// 64, width 768, feed-forward 2048), trained on two non-overlapping windows of each utterance: the prompt's semantic
/// tokens, the target's semantic tokens and the prompt's acoustic tokens are the prefix, and the model learns the
/// target's acoustic tokens. Synthesis samples at temperature 0.75, prompted by 3 seconds of the speaker.
/// </para>
/// <para>
/// No official code exists. What the paper leaves open follows lucidrains/spear-tts-pytorch for S1's procedure and
/// AudioLM's sequence layout (<see cref="AudioLmCore{T}"/>) for S2; see <c>docs/text-to-speech/paper-fidelity-decisions.md</c>.
/// </para>
/// <para><b>For Beginners:</b> SPEAR-TTS learns to talk from hours of untranscribed audio and to read from only
/// minutes of transcribed speech, then speaks in any voice it hears for three seconds.</para>
/// </remarks>
public class SPEARTTSOptions : TtsModelOptions
{
    /// <summary>Creates the paper's options.</summary>
    public SPEARTTSOptions()
    {
        SampleRate = 16000;
        HopSize = 320;
        // S1: T5-Large (Raffel et al. 2020).
        HiddenDim = 1024;
        NumHeads = 16;
        NumEncoderLayers = 24;
        NumDecoderLayers = 24;
        DropoutRate = 0.1;
        // Adafactor's relative step size is its own learning rate.
        LearningRate = 0.0;
        WeightDecay = 0.0;
        MaxTextLength = 512;
        SoundStream = new SoundStreamOptions { SampleRate = 16000, NumQuantizers = 12, CodebookSize = 1024, TargetBandwidthKbps = 1.5 };
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public SPEARTTSOptions(SPEARTTSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        SemanticCodes = other.SemanticCodes;
        SemanticFrameRate = other.SemanticFrameRate;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        ReadingFeedForwardDim = other.ReadingFeedForwardDim;
        ReadingKeyValueDim = other.ReadingKeyValueDim;
        PretrainingDeletionProbability = other.PretrainingDeletionProbability;
        PretrainingDropoutRate = other.PretrainingDropoutRate;
        LabelSmoothing = other.LabelSmoothing;
        ReadingTrainableEncoderLayers = other.ReadingTrainableEncoderLayers;
        BacktranslationTrainableDecoderLayers = other.BacktranslationTrainableDecoderLayers;
        BeamSize = other.BeamSize;
        MaxSemanticTokens = other.MaxSemanticTokens;
        SpeakingModelDim = other.SpeakingModelDim;
        SpeakingLayers = other.SpeakingLayers;
        SpeakingHeads = other.SpeakingHeads;
        SpeakingKeyValueDim = other.SpeakingKeyValueDim;
        SpeakingFeedForwardDim = other.SpeakingFeedForwardDim;
        SpeakingDropoutRate = other.SpeakingDropoutRate;
        Temperature = other.Temperature;
        PromptSeconds = other.PromptSeconds;
        MaxAcousticFrames = other.MaxAcousticFrames;
        SoundStream = (SoundStreamOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(other.SoundStream);
        SamplingSeed = other.SamplingSeed;
    }

    // ---------------------------------------------------------------- tokens (§6.1, §7.1)

    /// <summary>Semantic token vocabulary: k-means clusters of w2v-BERT's 7th layer (512).</summary>
    public int SemanticCodes { get; set; } = 512;

    /// <summary>Semantic tokens per second before repetitions are removed (25).</summary>
    public int SemanticFrameRate { get; set; } = 25;

    /// <summary>SoundStream RVQ levels the acoustic tokens keep (3).</summary>
    public int NumCodebooks { get; set; } = 3;

    /// <summary>Codes per RVQ level (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    // ---------------------------------------------------------------- reading, S1 (§4, §7.2)

    /// <summary>S1's feed-forward width (T5-Large: 4096). Its width, heads and depths are <see cref="TtsModelOptions.HiddenDim"/>,
    /// <see cref="TtsModelOptions.NumHeads"/>, <see cref="TtsModelOptions.NumEncoderLayers"/> and <see cref="TtsModelOptions.NumDecoderLayers"/>.</summary>
    public int ReadingFeedForwardDim { get; set; } = 4096;

    /// <summary>S1's per-head width (T5-Large: 64).</summary>
    public int ReadingKeyValueDim { get; set; } = 64;

    /// <summary>The probability of deleting each semantic token in the pretraining task (0.6).</summary>
    public double PretrainingDeletionProbability { get; set; } = 0.6;

    /// <summary>S1's dropout during pretraining (0.5). Finetuning uses <see cref="TtsModelOptions.DropoutRate"/>, which the
    /// paper searches over {0.1, 0.3, 0.5}.</summary>
    public double PretrainingDropoutRate { get; set; } = 0.5;

    /// <summary>Label smoothing of S1's finetuning losses (0.1); pretraining uses none.</summary>
    public double LabelSmoothing { get; set; } = 0.1;

    /// <summary>The lower encoder layers finetuning on parallel data updates, with the encoder's embeddings (the paper
    /// searches {4, 6, 8} on ground-truth data and finetunes the lower 4 last; 4).</summary>
    public int ReadingTrainableEncoderLayers { get; set; } = 4;

    /// <summary>N, the decoder's top and bottom layers backtranslation finetuning updates (the paper searches
    /// {2, 3, 4, 12}; 2).</summary>
    public int BacktranslationTrainableDecoderLayers { get; set; } = 2;

    /// <summary>S1's beam size when reading (10).</summary>
    public int BeamSize { get; set; } = 10;

    /// <summary>The most semantic tokens S1 reads out for one text.</summary>
    public int MaxSemanticTokens { get; set; } = 500;

    // ---------------------------------------------------------------- speaking, S2 (§5, §7.3)

    /// <summary>S2's width (768).</summary>
    public int SpeakingModelDim { get; set; } = 768;

    /// <summary>S2's layers (12).</summary>
    public int SpeakingLayers { get; set; } = 12;

    /// <summary>S2's attention heads (12).</summary>
    public int SpeakingHeads { get; set; } = 12;

    /// <summary>S2's per-head width (64).</summary>
    public int SpeakingKeyValueDim { get; set; } = 64;

    /// <summary>S2's feed-forward width (2048).</summary>
    public int SpeakingFeedForwardDim { get; set; } = 2048;

    /// <summary>S2's dropout. The paper states none; AudioLM's Transformers (which S2 follows) use 0.1.</summary>
    public double SpeakingDropoutRate { get; set; } = 0.1;

    /// <summary>S2's sampling temperature (0.75, chosen on LJSpeech's validation CER).</summary>
    public double Temperature { get; set; } = 0.75;

    /// <summary>The length of a voice prompt, and of the prompt window in S2's training, in seconds (3).</summary>
    public double PromptSeconds { get; set; } = 3.0;

    /// <summary>The most codec frames S2 generates for one utterance.</summary>
    public int MaxAcousticFrames { get; set; } = 1500;

    // ---------------------------------------------------------------- components

    /// <summary>The SoundStream codec. Its bandwidth must give <see cref="NumCodebooks"/> levels (3 at 50 Hz is
    /// 1.5 kbps).</summary>
    public SoundStreamOptions SoundStream { get; set; }

    /// <summary>The seed of synthesis and of the training corruption.</summary>
    public int SamplingSeed { get; set; } = 1234;
}
