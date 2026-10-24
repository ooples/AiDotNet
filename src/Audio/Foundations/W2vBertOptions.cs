using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Foundations;

/// <summary>Options for w2v-BERT (Chung et al. 2021, "w2v-BERT: Combining Contrastive Learning and Masked Language
/// Modeling for Self-Supervised Speech Pre-Training").</summary>
/// <remarks>
/// <para>
/// The defaults are w2v-BERT XL (Table 1, 0.6B parameters): 80-bin log-mel features (25 ms windows every 10 ms, as the
/// Conformer setup it follows uses), a feature encoder of two stride-2 2-D convolutions (4× fewer frames, so 25 frames
/// a second), a contrastive module of a linear projection and 12 Conformer blocks, and a masked-prediction module of
/// 12 more, all 1024 wide with 8 heads, convolution kernel 5 and no relative attention. The quantizer has one codebook
/// of 1024 entries 1024 wide (Table 1) and otherwise follows wav2vec 2.0's (§3.2: K = 100 distractors, diversity weight
/// α = 0.1, κ = 0.1, the Gumbel temperature annealed from 2 to 0.5). Masking starts a 10-step span at each step with
/// probability 0.065, and a masked step is replaced by a random vector (§4.2). The loss is L_c + L_m (β = γ = 1).
/// Pre-training uses Adam with the Transformer schedule (Vaswani et al. 2017, §5.3: β = 0.9 / 0.98, ε = 1e-9) at a peak
/// of 2e-3 after 25k warm-up steps (§4.2).
/// </para>
/// <para>
/// Not stated by the paper or the works it follows, and decided here: the convolutions' 3×3 kernels and their channel
/// count (the model width), as Conformer's convolutional subsampling is reproduced by ESPnet's
/// <c>Conv2dSubsampling</c>; the random vectors are drawn from N(0, 1); the masked-prediction loss is summed over the
/// masked steps, as wav2vec 2.0's contrastive loss is; and the log-mel floor (<see cref="LogMelFloor"/>).
/// </para>
/// <para>
/// Google released no w2v-BERT code or weights. Meta's w2v-BERT 2.0 (<c>facebook/w2v-bert-2.0</c>) is a different
/// network (stacked-frame features, relative-key attention, one 24-layer stack), so its checkpoint does not load here.
/// </para>
/// <para><b>For Beginners:</b> w2v-BERT learns what speech sounds like from recordings alone, with no transcripts. Its
/// middle layers make good "what is being said" tokens for speech generators such as AudioLM and SoundStorm.</para>
/// </remarks>
public class W2vBertOptions : ModelOptions
{
    /// <summary>Creates w2v-BERT XL's options.</summary>
    public W2vBertOptions()
    {
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public W2vBertOptions(W2vBertOptions other)
    {
        if (other is null) throw new System.ArgumentNullException(nameof(other));
        SampleRate = other.SampleRate;
        MelBins = other.MelBins;
        WindowLength = other.WindowLength;
        HopLength = other.HopLength;
        LogMelFloor = other.LogMelFloor;
        ModelDim = other.ModelDim;
        NumHeads = other.NumHeads;
        FeedForwardMultiplier = other.FeedForwardMultiplier;
        ConvKernel = other.ConvKernel;
        ContrastiveLayers = other.ContrastiveLayers;
        MaskedLayers = other.MaskedLayers;
        CodebookSize = other.CodebookSize;
        CodeDim = other.CodeDim;
        Negatives = other.Negatives;
        DiversityWeight = other.DiversityWeight;
        MaskProbability = other.MaskProbability;
        MaskLength = other.MaskLength;
        LearningRate = other.LearningRate;
        WarmupSteps = other.WarmupSteps;
        SemanticLayer = other.SemanticLayer;
        SemanticClusters = other.SemanticClusters;
        FixedDraw = other.FixedDraw;
        Seed = other.Seed;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
    }

    /// <summary>w2v-BERT XXL (Table 1, 1.0B parameters): 30 masked-prediction blocks. The paper trains it with
    /// Adafactor (β = 0.9 / 0.98); this options class keeps Adam, so pass that optimizer to the model yourself.</summary>
    public static W2vBertOptions Xxl() => new() { MaskedLayers = 30 };

    /// <summary>Audio sample rate (16 kHz).</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Log-mel filterbank bins (80).</summary>
    public int MelBins { get; set; } = 80;

    /// <summary>Analysis window in samples (400: 25 ms).</summary>
    public int WindowLength { get; set; } = 400;

    /// <summary>Frame step in samples (160: 10 ms).</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>The floor added before the log, log(mel + floor). Not stated; 1e-6 keeps silence finite.</summary>
    public double LogMelFloor { get; set; } = 1e-6;

    /// <summary>Model width (1024).</summary>
    public int ModelDim { get; set; } = 1024;

    /// <summary>Attention heads (8).</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Feed-forward width as a multiple of the model width (4, Conformer's).</summary>
    public int FeedForwardMultiplier { get; set; } = 4;

    /// <summary>Conformer convolution kernel (5).</summary>
    public int ConvKernel { get; set; } = 5;

    /// <summary>Conformer blocks in the contrastive module (12).</summary>
    public int ContrastiveLayers { get; set; } = 12;

    /// <summary>Conformer blocks in the masked-prediction module (12 for XL, 30 for XXL).</summary>
    public int MaskedLayers { get; set; } = 12;

    /// <summary>Quantizer codebook entries, which are the masked-prediction targets (1024).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Quantized vector width (1024).</summary>
    public int CodeDim { get; set; } = 1024;

    /// <summary>Distractors per masked step (100, wav2vec 2.0's).</summary>
    public int Negatives { get; set; } = 100;

    /// <summary>Codebook diversity loss weight α (0.1).</summary>
    public double DiversityWeight { get; set; } = 0.1;

    /// <summary>Probability that a step starts a masked span (0.065).</summary>
    public double MaskProbability { get; set; } = 0.065;

    /// <summary>Steps in a masked span (10).</summary>
    public int MaskLength { get; set; } = 10;

    /// <summary>Peak learning rate (2e-3).</summary>
    public double LearningRate { get; set; } = 2e-3;

    /// <summary>Transformer-schedule warm-up steps (25 000).</summary>
    public int WarmupSteps { get; set; } = 25_000;

    /// <summary>The masked-prediction block, counted from 1, whose output AudioLM clusters into semantic tokens (7).</summary>
    public int SemanticLayer { get; set; } = 7;

    /// <summary>k-means clusters of the semantic tokens (1024, AudioLM's).</summary>
    public int SemanticClusters { get; set; } = 1024;

    /// <summary>
    /// A seed that fixes the pre-training draw (mask, Gumbel noise, distractors, dropout) for every step and every
    /// evaluated objective, and keeps each input's codebook entries from its first evaluation; null, the paper's
    /// setting, draws afresh each step and lets the quantizer choose.
    /// </summary>
    /// <remarks>
    /// The pre-training objective is an expectation over those draws, and on a small model or a short clip one draw's
    /// loss moves by more than training lowers the expectation (a contrastive term of exactly 0 when every distractor
    /// shares its positive's code). Fixing the draw makes the objective a deterministic function of the weights, and
    /// keeping the entries makes it a continuous one (the masked-prediction targets are otherwise the quantizer's argmax,
    /// so the loss jumps when training moves a frame to another entry). That is what an overfitting or gradient check
    /// needs; it is not a pre-training setting.
    /// </remarks>
    public int? FixedDraw { get; set; }

    /// <summary>Path of an ONNX export, for inference.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
