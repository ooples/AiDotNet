namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>SpiritLM's two versions (Nguyen et al. 2024 §3).</summary>
public enum SpiritLmVariant
{
    /// <summary>SpiritLM Base (§3.1): speech as HuBERT units only.</summary>
    Base,

    /// <summary>SpiritLM Expressive (§3.2): HuBERT units interleaved with pitch and style tokens.</summary>
    Expressive,
}

/// <summary>How HuBERT frames become units.</summary>
public enum SpiritLmUnitQuantizer
{
    /// <summary>The release's CTC-trained linear quantizer (<c>L11_quantizer_500.pt</c>: 500 units and a blank, blank
    /// frames taking the unit on their left); the released language model's units.</summary>
    LinearCtc,

    /// <summary>A k-means over the layer's features, as TWIST's tokenizer (Hassid et al. 2023), which the paper names;
    /// load or fit the centroids with <see cref="SpiritLM{T}.LoadUnitKMeans"/>.</summary>
    KMeans,
}

/// <summary>The F0 extractor of SpiritLM Expressive's pitch tokens (§3.2).</summary>
public enum SpiritLmPitchExtractor
{
    /// <summary>FCPE (torchfcpe's bundled model), which the paper uses for the language model's tokens and the release by
    /// default.</summary>
    Fcpe,

    /// <summary>pYAAPT, with which the paper trains the pitch VQ-VAE.</summary>
    Yaapt,
}

/// <summary>Where SpiritLM Expressive's style tokens come from.</summary>
public enum SpiritLmStyleTokenizer
{
    /// <summary>The release's wav2vec 2.0 style classifier (<c>style_encoder_w2v2</c>), one class a second; the released
    /// language model's style tokens.</summary>
    ReleaseClassifier,

    /// <summary>The paper's (§3.2): SONAR Expressive's SpeechProp features of the 25 Hz EnCodec units, pooled per second,
    /// fine-tuned to predict Expresso's styles and clustered by a 100-centroid k-means. SpeechProp was never released; it
    /// is built and trained here from Duquenne et al. 2023 (<see cref="SpiritLM{T}.TrainSonarExpressive(Tensor{T}, Tensor{T}, bool)"/>,
    /// <see cref="SpiritLM{T}.TrainSpeechPropStyle"/>, <see cref="SpiritLM{T}.FitStyleCentroids"/>).</summary>
    SpeechProp,
}

/// <summary>The output modality of a SpiritLM generation (the release's <c>OutputModality</c>).</summary>
public enum SpiritLmModality
{
    /// <summary>Text: speech tokens and the modality markers are forbidden.</summary>
    Text,

    /// <summary>Speech: text tokens and the modality markers are forbidden.</summary>
    Speech,

    /// <summary>Either, switching as the model chooses.</summary>
    Arbitrary,
}

/// <summary>Options for <see cref="SpiritLM{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's SpiritLM Base 7B: LLaMA 2 7B (32 layers, d 4096, 32 heads, SwiGLU 11008, RMSNorm
/// ε 1e-5) with RoPE's base raised to 100,000 (App. A) and its vocabulary extended by <c>[Text]</c>, <c>[Speech]</c> and
/// 501 HuBERT tokens (32,512 rows in the release); the 25 Hz mHuBERT's layer 11 for units; the unit HiFi-GAN trained on
/// Expresso with 4 speakers (and 34 styles); generation at temperature 0.8 and top-p 0.95 with at most 50 tokens of text,
/// 200 of speech from text and 300 of speech from speech (App. D); continued pre-training at a constant rate of 3e-5 over
/// 4,096-token sequences (App. A).</para>
/// <para><see cref="Expressive"/> is SpiritLM Expressive: 64 pitch and 100 style tokens added (32,768 rows), FCPE F0 and a
/// pitch VQ-VAE, the released wav2vec 2.0 style classifier, and a vocoder that reads both.</para>
/// <para><b>For Beginners:</b> SpiritLM is a text language model that also learned a "language" of speech tokens. Text is
/// marked <c>[Text]</c>, speech <c>[Speech]</c>, and it can switch between them mid-sentence, so it can continue speech
/// as speech or text, read text aloud, or transcribe.</para>
/// </remarks>
public class SpiritLMOptions : TtsModelOptions
{
    /// <summary>Creates SpiritLM Base 7B.</summary>
    public SpiritLMOptions()
    {
        SampleRate = 16000;
        HiddenDim = 4096;
        NumHeads = 32;
        VocabSize = 32512;
        MaxTextLength = 4096;
        DropoutRate = 0.0;
        LearningRate = 3e-5;
        WeightDecay = 0.1;
        Seed = 42;
    }

    /// <summary>SpiritLM Expressive 7B: pitch and style tokens, a 32,768-row vocabulary, the expressive vocoder.</summary>
    public static SpiritLMOptions Expressive() => new()
    {
        Variant = SpiritLmVariant.Expressive, VocabSize = 32768, VocoderStyles = 0, Speaker = 2, VocoderSegmentSize = 17920,
    };

    /// <summary>Base or Expressive.</summary>
    public SpiritLmVariant Variant { get; set; } = SpiritLmVariant.Base;

    // ---------------------------------------------------------------- language model

    /// <summary>LLaMA 2 layers (32) and SwiGLU width (11008).</summary>
    public int Layers { get; set; } = 32;

    /// <inheritdoc cref="Layers"/>
    public int Intermediate { get; set; } = 11008;

    /// <summary>RMSNorm ε (1e-5) and the rotary base (100,000, App. A).</summary>
    public double RmsEpsilon { get; set; } = 1e-5;

    /// <inheritdoc cref="RmsEpsilon"/>
    public double RopeTheta { get; set; } = 100000;

    /// <summary>The release's SentencePiece <c>tokenizer.model</c>, whose user-defined pieces are the modality markers and
    /// speech tokens; text needs it.</summary>
    public string? TokenizerPath { get; set; }

    /// <summary>LLaMA's text vocabulary (32,000); the markers and speech tokens follow it.</summary>
    public int TextVocabulary { get; set; } = 32000;

    // ---------------------------------------------------------------- speech tokens

    /// <summary>HuBERT units (500) and their quantizer; the token vocabulary holds one more (the CTC blank).</summary>
    public int Units { get; set; } = 500;

    /// <inheritdoc cref="Units"/>
    public SpiritLmUnitQuantizer UnitQuantizer { get; set; } = SpiritLmUnitQuantizer.LinearCtc;

    /// <summary>The 25 Hz mHuBERT's width (768), layer read (11), heads (12), feed-forward width (3072) and feature-encoder
    /// channels (512); its convolutions stride 5·2⁷ = 640 samples.</summary>
    public int HubertWidth { get; set; } = 768;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertLayer { get; set; } = 11;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertHeads { get; set; } = 12;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertIntermediate { get; set; } = 3072;

    /// <inheritdoc cref="HubertWidth"/>
    public int HubertConvChannels { get; set; } = 512;

    /// <summary>Pitch tokens (64) and the F0 extractor feeding the pitch VQ-VAE.</summary>
    public int PitchTokens { get; set; } = 64;

    /// <inheritdoc cref="PitchTokens"/>
    public SpiritLmPitchExtractor PitchExtractor { get; set; } = SpiritLmPitchExtractor.Fcpe;

    /// <summary>The pitch VQ-VAE's width (32) and latent width (128).</summary>
    public int PitchQuantizerWidth { get; set; } = 32;

    /// <inheritdoc cref="PitchQuantizerWidth"/>
    public int PitchQuantizerLatent { get; set; } = 128;

    /// <summary>FCPE's mel bands (128), width (512), layers (6) and salience bins (360).</summary>
    public int FcpeMels { get; set; } = 128;

    /// <inheritdoc cref="FcpeMels"/>
    public int FcpeWidth { get; set; } = 512;

    /// <inheritdoc cref="FcpeMels"/>
    public int FcpeLayers { get; set; } = 6;

    /// <inheritdoc cref="FcpeMels"/>
    public int FcpeBins { get; set; } = 360;

    /// <summary>Style tokens (100) and the style classifier's wav2vec 2.0 width (768), layers (12), heads (12), feed-forward
    /// width (3072), convolution channels (512) and projection (256).</summary>
    public int StyleTokens { get; set; } = 100;

    /// <inheritdoc cref="StyleTokens"/>
    public int StyleEncoderWidth { get; set; } = 768;

    /// <inheritdoc cref="StyleTokens"/>
    public int StyleEncoderLayers { get; set; } = 12;

    /// <inheritdoc cref="StyleTokens"/>
    public int StyleEncoderHeads { get; set; } = 12;

    /// <inheritdoc cref="StyleTokens"/>
    public int StyleEncoderIntermediate { get; set; } = 3072;

    /// <inheritdoc cref="StyleTokens"/>
    public int StyleEncoderConvChannels { get; set; } = 512;

    /// <inheritdoc cref="StyleTokens"/>
    public int StyleEncoderProjection { get; set; } = 256;

    /// <summary>The style tokenizer: the release's classifier by default, or the paper's SpeechProp features.</summary>
    public SpiritLmStyleTokenizer StyleTokenizer { get; set; } = SpiritLmStyleTokenizer.ReleaseClassifier;

    // ---------------------------------------------------------------- SpeechProp (SONAR Expressive)

    /// <summary>SONAR Expressive's EnCodec (§3.1): 16 kHz, SEANet ratios 8·5·4·4 (25 frames a second), 128-dimensional
    /// latents, 8 codebooks of 1024 (2 kbps).</summary>
    public AiDotNet.Audio.Generation.EnCodecOptions SpeechPropCodec { get; set; } = new()
    {
        SampleRate = 16000, Ratios = new[] { 8, 5, 4, 4 }, Dimension = 128, NumQuantizers = 8, CodebookSize = 1024,
        TargetBandwidths = new[] { 2.0 }, TargetBandwidthKbps = 2.0, SegmentSize = 16000, IncludeDiscriminators = false,
    };

    /// <summary>The SpeechProp encoder's and the decoders' width (1024), heads (16) and feed-forward width (4096); the
    /// paper states only the depths (12 and 16).</summary>
    public int SpeechPropWidth { get; set; } = 1024;

    /// <inheritdoc cref="SpeechPropWidth"/>
    public int SpeechPropHeads { get; set; } = 16;

    /// <inheritdoc cref="SpeechPropWidth"/>
    public int SpeechPropFeedForward { get; set; } = 4096;

    /// <summary>The SpeechProp encoder's layers (12) and each decoder's (16).</summary>
    public int SpeechPropEncoderLayers { get; set; } = 12;

    /// <inheritdoc cref="SpeechPropEncoderLayers"/>
    public int SpeechPropDecoderLayers { get; set; } = 16;

    /// <summary>The SpeechProp embedding's size (1024, as SONAR's).</summary>
    public int SpeechPropDim { get; set; } = 1024;

    /// <summary>Expresso's style classes the SpeechProp features are fine-tuned to predict (the release's 34-style list).</summary>
    public int ExpressoStyles { get; set; } = 34;

    /// <summary>The rate of SONAR Expressive's and the style fine-tuning's AdamW (1e-4; neither paper states one).</summary>
    public double SpeechPropLearningRate { get; set; } = 1e-4;

    /// <summary>Whether to build SONAR's text encoder (NLLB-200's 1.3B encoder) for the decoders' semantic conditioning;
    /// without it, training takes SONAR embeddings computed elsewhere.</summary>
    public bool IncludeSonarTextEncoder { get; set; }

    /// <summary>SONAR's text encoder: vocabulary (256,206), width (1024), layers (24), heads (16), feed-forward (8192).</summary>
    public int SonarVocabSize { get; set; } = 256206;

    /// <inheritdoc cref="SonarVocabSize"/>
    public int SonarWidth { get; set; } = 1024;

    /// <inheritdoc cref="SonarVocabSize"/>
    public int SonarLayers { get; set; } = 24;

    /// <inheritdoc cref="SonarVocabSize"/>
    public int SonarHeads { get; set; } = 16;

    /// <inheritdoc cref="SonarVocabSize"/>
    public int SonarFeedForward { get; set; } = 8192;

    /// <summary>A directory with NLLB's <c>sentencepiece.bpe.model</c> and <c>special_tokens_map.json</c>.</summary>
    public string? SonarTokenizerDirectory { get; set; }

    // ---------------------------------------------------------------- vocoder

    /// <summary>The vocoder's speakers (Expresso's 4) and the speaker synthesis uses (the release's default, 2).</summary>
    public int VocoderSpeakers { get; set; } = 4;

    /// <inheritdoc cref="VocoderSpeakers"/>
    public int Speaker { get; set; } = 2;

    /// <summary>The Base vocoder's utterance styles (34, its <c>styles.txt</c>; 0 for none) and the style synthesis uses
    /// (the release's default, 8, "conv-default").</summary>
    public int VocoderStyles { get; set; } = 34;

    /// <inheritdoc cref="VocoderStyles"/>
    public int VocoderStyle { get; set; } = 8;

    /// <summary>The vocoder's embedding width (128), first width (512), upsampling (5·4·4·4·2 = 640 samples per unit, 25 Hz)
    /// and kernels.</summary>
    public int VocoderEmbeddingDim { get; set; } = 128;

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int VocoderChannels { get; set; } = 512;

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int[] VocoderUpsampleRates { get; set; } = { 5, 4, 4, 4, 2 };

    /// <inheritdoc cref="VocoderEmbeddingDim"/>
    public int[] VocoderUpsampleKernels { get; set; } = { 11, 8, 8, 8, 4 };

    /// <summary>Whether to build HiFi-GAN's discriminators, which training the vocoder needs.</summary>
    public bool IncludeVocoderDiscriminators { get; set; }

    /// <summary>The vocoder's training segment in samples (the Base config's 8960, 14 units; the Expressive config's
    /// 17920).</summary>
    public int VocoderSegmentSize { get; set; } = 8960;

    /// <summary>The vocoder's and the pitch VQ-VAE's rate (2e-4, AdamW β 0.8 and 0.99), decayed by 0.999 per epoch of
    /// <see cref="VocoderUpdatesPerEpoch"/> updates.</summary>
    public double VocoderLearningRate { get; set; } = 2e-4;

    /// <inheritdoc cref="VocoderLearningRate"/>
    public double VocoderLearningRateDecay { get; set; } = 0.999;

    /// <inheritdoc cref="VocoderLearningRate"/>
    public int VocoderUpdatesPerEpoch { get; set; } = 1000;

    // ---------------------------------------------------------------- generation

    /// <summary>Sampling (App. D): temperature 0.8 and top-p 0.95; 0 temperature decodes greedily.</summary>
    public double Temperature { get; set; } = 0.8;

    /// <inheritdoc cref="Temperature"/>
    public double TopP { get; set; } = 0.95;

    /// <summary>Top-k before top-p: the release builds a fresh <c>GenerationConfig</c>, so transformers' default of 50 applies
    /// (the paper names only top-p); 0 samples from the nucleus alone.</summary>
    public int TopK { get; set; } = 50;

    /// <summary>The most tokens a generation adds (App. D): 50 of text, 200 of speech after text, 300 of speech after
    /// speech; an arbitrary-modality generation takes the speech limit of its prompt's last modality.</summary>
    public int MaxNewTextTokens { get; set; } = 50;

    /// <inheritdoc cref="MaxNewTextTokens"/>
    public int MaxNewSpeechTokensFromText { get; set; } = 200;

    /// <inheritdoc cref="MaxNewTextTokens"/>
    public int MaxNewSpeechTokensFromSpeech { get; set; } = 300;

    // ---------------------------------------------------------------- training

    /// <summary>The word spans of interleaved training (App. A): 10 to 30 words of text and 5 to 15 of speech.</summary>
    public int MinTextSpanWords { get; set; } = 10;

    /// <inheritdoc cref="MinTextSpanWords"/>
    public int MaxTextSpanWords { get; set; } = 30;

    /// <inheritdoc cref="MinTextSpanWords"/>
    public int MinSpeechSpanWords { get; set; } = 5;

    /// <inheritdoc cref="MinTextSpanWords"/>
    public int MaxSpeechSpanWords { get; set; } = 15;

    /// <summary>The language model's optimizer: LLaMA 2's AdamW (β 0.9 and 0.95, ε 1e-5, weight decay 0.1, clipping 1.0)
    /// at the constant <c>LearningRate</c>.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
