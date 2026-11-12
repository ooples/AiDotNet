using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for SeACo-Paraformer: Paraformer-large with a hotword bias encoder, bias decoder and bias output
/// layer.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the released SeACo-Paraformer (FunASR's seaco config and the paper's section
/// 4.2): the Paraformer-large backbone, a two-layer LSTM bias encoder of width 512, a four-block bias decoder with 1,024
/// hidden units, hotwords of 2 to 8 tokens sampled with ratios 0.75, and 8,404 characters.</para>
/// </remarks>
public class SeACoOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SeACoOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SeACoOptions(SeACoOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        WarmupSteps = other.WarmupSteps;
        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        NumMels = other.NumMels;
        LfrM = other.LfrM;
        LfrN = other.LfrN;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        FsmnKernelSize = other.FsmnKernelSize;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        LabelSmoothing = other.LabelSmoothing;
        SamplingRatio = other.SamplingRatio;
        PredictorWeight = other.PredictorWeight;
        TailThreshold = other.TailThreshold;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
        TrainingStage = other.TrainingStage;
        NumBiasEncoderLayers = other.NumBiasEncoderLayers;
        NumBiasDecoderLayers = other.NumBiasDecoderLayers;
        BiasFeedForwardDim = other.BiasFeedForwardDim;
        BiasFsmnKernelSize = other.BiasFsmnKernelSize;
        BiasLabelSmoothing = other.BiasLabelSmoothing;
        HotwordMaskTokenId = other.HotwordMaskTokenId;
        HotwordBatchRatio = other.HotwordBatchRatio;
        HotwordUtteranceRatio = other.HotwordUtteranceRatio;
        HotwordMinLength = other.HotwordMinLength;
        HotwordMaxLength = other.HotwordMaxLength;
        BiasMergeLambda = other.BiasMergeLambda;
        AttentionFilterTopK = other.AttentionFilterTopK;
    }

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Kaldi filterbank bands (80).</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Filterbank frames stacked by the low frame rate front end (7).</summary>
    public int LfrM { get; set; } = 7;
    /// <summary>The stride between stacks (6).</summary>
    public int LfrN { get; set; } = 6;
    /// <summary>The model width (512), shared by the encoder, the predictor, both decoders and the bias encoder.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>SAN-M encoder blocks (50).</summary>
    public int NumEncoderLayers { get; set; } = 50;
    /// <summary>SAN-M ASR decoder blocks with cross-attention (16), before the final feed-forward block.</summary>
    public int NumDecoderLayers { get; set; } = 16;
    /// <summary>Gets or sets the number of attention heads. Defaults to <c>4</c>.</summary>
    public int NumAttentionHeads { get; set; } = 4;
    /// <summary>The encoder's and the ASR decoder's feed-forward width (2,048).</summary>
    public int FeedForwardDim { get; set; } = 2048;
    /// <summary>The encoder's and the ASR decoder's FSMN kernel (11).</summary>
    public int FsmnKernelSize { get; set; } = 11;
    /// <summary>Output vocabulary, blank (0), start (1) and end (2) included (8,404). The bias output layer adds the
    /// no-bias class.</summary>
    public int VocabSize { get; set; } = 8404;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the language code used for speech processing. Defaults to <c>"zh"</c>.</summary>
    public string Language { get; set; } = "zh";
    /// <summary>The ASR decoder loss's label smoothing (0.1).</summary>
    public double LabelSmoothing { get; set; } = 0.1;
    /// <summary>The glancing sampler's λ (0.75).</summary>
    public double SamplingRatio { get; set; } = 0.75;
    /// <summary>The MAE token-count loss's weight (1).</summary>
    public double PredictorWeight { get; set; } = 1.0;
    /// <summary>The CIF tail weight at inference (0.45).</summary>
    public double TailThreshold { get; set; } = 0.45;
    /// <summary>The learning rate's warmup steps (30,000, FunASR's warmuplr).</summary>
    public int WarmupSteps { get; set; } = 30000;
    /// <summary>Optional vocabulary mapping token IDs to text tokens (the release's tokens.json); hotwords are tokenized
    /// with it.</summary>
    public string[]? Vocabulary { get; set; }

    /// <summary>
    /// Which parameter group <c>Train</c> updates. Default: <see cref="SeACoTrainingStage.Backbone"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Train the backbone first. Then select <see cref="SeACoTrainingStage.Bias"/> to teach
    /// hotword customization without changing the recognizer.</para>
    /// <para>SeACo-Paraformer (Shi et al., arXiv 2308.03266, section 3.1) trains the bias modules "with a well-trained
    /// Paraformer model freezing", separately from the ASR training: <see cref="SeACoTrainingStage.Backbone"/> then
    /// <see cref="SeACoTrainingStage.Bias"/>.</para>
    /// </remarks>
    public SeACoTrainingStage TrainingStage { get; set; } = SeACoTrainingStage.Backbone;

    /// <summary>LSTM layers in the bias encoder (2; section 4.2).</summary>
    public int NumBiasEncoderLayers { get; set; } = 2;
    /// <summary>SAN-M blocks with cross-attention in the bias decoder (4; section 4.2), before its final feed-forward
    /// block.</summary>
    public int NumBiasDecoderLayers { get; set; } = 4;
    /// <summary>The bias decoder's feed-forward width (1,024; section 4.2).</summary>
    public int BiasFeedForwardDim { get; set; } = 1024;
    /// <summary>The bias decoder's FSMN kernel (21; the release's seaco_decoder_conf).</summary>
    public int BiasFsmnKernelSize { get; set; } = 21;
    /// <summary>The bias loss's label smoothing (0.1; the release's seaco_lsm_weight).</summary>
    public double BiasLabelSmoothing { get; set; } = 0.1;

    /// <summary>
    /// The no-bias class ('#'). Defaults to <c>null</c>, which resolves to <see cref="VocabSize"/>: the class the paper
    /// appends to the ASR vocabulary.
    /// </summary>
    /// <remarks>
    /// SeACo section 3.1: "an additional token (counted as #, means no-bias) is appended to the ASR output vocabulary to
    /// mark non-hotword position outputs", which is why the bias output layer emits <see cref="VocabSize"/> + 1 logits.
    /// Set it only to point at a different reserved class (the release reuses id 8377).
    /// </remarks>
    public int? HotwordMaskTokenId { get; set; }

    /// <summary>Resolves the no-bias class, defaulting to the appended class at <see cref="VocabSize"/>.</summary>
    internal int ResolveHotwordMaskTokenId()
    {
        int tokenId = HotwordMaskTokenId ?? VocabSize;
        if (tokenId < 0 || tokenId > VocabSize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(HotwordMaskTokenId),
                tokenId,
                $"HotwordMaskTokenId must identify one of the {VocabSize + 1} bias-output classes.");
        }

        return tokenId;
    }

    /// <summary>r_b: the share of training batches that sample hotwords (0.75); the others train with the default hotword
    /// alone.</summary>
    public double HotwordBatchRatio { get; set; } = 0.75;
    /// <summary>r_u: the share of utterances in a sampling batch that contribute a hotword (0.75).</summary>
    public double HotwordUtteranceRatio { get; set; } = 0.75;
    /// <summary>l_min: the shortest sampled hotword in tokens (2).</summary>
    public int HotwordMinLength { get; set; } = 2;
    /// <summary>l_max: the longest sampled hotword in tokens (8).</summary>
    public int HotwordMaxLength { get; set; } = 8;
    /// <summary>λ in Equation 5: the weight on the bias probabilities where the bias decoder detects a hotword (1).</summary>
    public double BiasMergeLambda { get; set; } = 1.0;
    /// <summary>k for attention score filtering (50; section 4.3): with more hotwords, only the k the bias decoder's last
    /// layer attends to most are kept. Zero disables the filter.</summary>
    public int AttentionFilterTopK { get; set; } = 50;
}
