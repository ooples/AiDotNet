using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The positional encoding of NaturalSpeech 2's Transformers.</summary>
public enum NaturalSpeech2Positions
{
    /// <summary>The sinusoidal table indexed by position (the Transformer's).</summary>
    Sinusoidal,

    /// <summary>Amphion's: the table indexed by the batch position of a batch-first input, so at a batch of one every
    /// frame gets row 0. Its checkpoint was trained so.</summary>
    AmphionBatchIndexed,
}

/// <summary>What NaturalSpeech 2's latent diffusion learns.</summary>
public enum NaturalSpeech2Diffusion
{
    /// <summary>The paper's variance-preserving SDE with a data-predicting denoiser (Eqs. 2-6).</summary>
    ScoreSde,

    /// <summary>Amphion's alternative: rectified flow, z_t = (1 − t) z₀ + t ε, the denoiser predicting ε − z₀.</summary>
    FlowMatching,
}

/// <summary>The form of Eq. 6's score term.</summary>
public enum NaturalSpeech2ScoreLoss
{
    /// <summary>‖ε̂ − ε‖², the noise the predicted ẑ₀ implies against the drawn noise (Eq. 6 without its 1/Σ_t weight).</summary>
    NoiseSpace,

    /// <summary>Eq. 6 as written, ‖Σ_t⁻¹(ρ(ẑ₀, t) − z_t) − ∇log p_t(z_t)‖², which is the noise error weighted by 1/Σ_t.</summary>
    ScoreSpace,
}

/// <summary>The residual each quantizer's cross-entropy term (L_ce-rvq) is measured on.</summary>
public enum NaturalSpeech2CeResidual
{
    /// <summary>The paper's: ẑ₀ minus the ground-truth embeddings of the earlier quantizers.</summary>
    GroundTruth,

    /// <summary>Amphion's: ẑ₀ minus the embeddings ẑ₀ itself quantizes to at the earlier quantizers.</summary>
    Predicted,
}

/// <summary>Where the training prompt is cut from the utterance.</summary>
public enum NaturalSpeech2PromptPosition
{
    /// <summary>Anywhere ("randomly cut off a segment", §3.3; Amphion's "mid").</summary>
    Anywhere,

    /// <summary>At the start (Amphion's configured "start").</summary>
    Start,
}

/// <summary>The ODE solver of reverse diffusion.</summary>
public enum NaturalSpeech2Solver
{
    /// <summary>Euler steps (the paper's).</summary>
    Euler,

    /// <summary>Amphion's midpoint steps.</summary>
    Midpoint,
}

/// <summary>Options for <see cref="NaturalSpeech2{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are the paper's (Shen et al. 2023, §4.1, §5.1, Table 11, App. B). Where it is silent the values come
/// from Amphion's implementation (open-mmlab/Amphion, <c>models/tts/naturalspeech2</c>), and where Amphion departs from
/// the paper its choice is an option; <see cref="Amphion"/> selects all of them, which its released checkpoint needs.</para>
/// <para><b>For Beginners:</b> NaturalSpeech 2 speaks in the voice of a few seconds of reference audio. A neural codec turns
/// speech into continuous vectors; a diffusion model, told the phonemes, how long each lasts and the pitch, generates
/// those vectors for new text; the codec turns them back into audio. Train the codec first (see <see cref="Codec"/>),
/// then the diffusion model.</para>
/// </remarks>
public class NaturalSpeech2Options : TtsModelOptions
{
    /// <summary>Creates the paper's configuration.</summary>
    public NaturalSpeech2Options()
    {
        SampleRate = 16000;
        HopSize = 200;
        HiddenDim = 512;
        NumHeads = 8;
        NumEncoderLayers = 6;
        VocabSize = 100;
        DropoutRate = 0.2;
        LearningRate = 5e-4;
        // AdamW's weight decay is unstated; torch's default.
        WeightDecay = 0.01;
        Codec = PaperCodec();
    }

    /// <summary>
    /// The paper's codec (§3.1, Table 11): SoundStream's model at 16 kHz with a total stride of 200 (12.5 ms frames;
    /// strides 2, 4, 5, 5), 16 residual quantizers of 1024 codes of dimension 256, without the denoising FiLM.
    /// </summary>
    public static SoundStreamOptions PaperCodec() => new()
    {
        SampleRate = 16000,
        SegmentSize = 16000,
        Ratios = [5, 5, 4, 2],
        Dimension = 256,
        NumQuantizers = 16,
        CodebookSize = 1024,
        TargetBandwidthKbps = 12.8,
        FilmPosition = SoundStreamFilmPosition.None,
    };

    /// <summary>Amphion's codec: EnCodec's released 24 kHz model at 12 kbps (16 codebooks of dimension 128).</summary>
    public static EnCodecOptions AmphionCodec()
    {
        var codec = EnCodecOptions.OfficialCheckpoint24kHz();
        codec.TargetBandwidthKbps = 12.0;
        return codec;
    }

    /// <summary>
    /// Amphion's configuration (<c>config/ns2.json</c>, <c>egs/tts/NaturalSpeech2/exp_config.json</c>) and every departure
    /// of its code from the paper, as its LibriTTS checkpoint (<c>amphion/naturalspeech2_libritts</c>) was trained.
    /// </summary>
    public static NaturalSpeech2Options Amphion() => new()
    {
        SampleRate = 24000,
        HopSize = 320,
        Codec = AmphionCodec(),
        StyleAdaptivePhonemeEncoder = true,
        Positions = NaturalSpeech2Positions.AmphionBatchIndexed,
        DenoiserDilationCycle = 2,
        DenoiserAttentionOffset = 0,
        DataLossL1 = true,
        NoiseLossL1 = true,
        CeLossWeight = 0.5,
        CeResidual = NaturalSpeech2CeResidual.Predicted,
        PromptPosition = NaturalSpeech2PromptPosition.Start,
        Temperature = 1.44,
        SamplingSteps = 200,
        LearningRate = 1e-4,
        WarmupSteps = 5000,
        CosineSchedule = true,
    };

    // ---------------------------------------------------------------- codec

    /// <summary>
    /// The neural audio codec whose quantized latents the diffusion model generates (<see cref="PaperCodec"/>, or
    /// <see cref="AmphionCodec"/>). <see cref="TtsModelOptions.SampleRate"/> and <see cref="TtsModelOptions.HopSize"/> must
    /// be its. Train it before the diffusion model, through <see cref="NaturalSpeech2{T}.Codec"/>.
    /// </summary>
    public NeuralAudioCodecOptions Codec { get; set; }

    // ---------------------------------------------------------------- text

    /// <summary>
    /// A pronunciation lexicon (word, then ARPAbet phones, per line; Amphion's <c>librispeech-lexicon.txt</c>) consulted
    /// before g2p_en, as Amphion's front end does; null uses g2p_en alone.
    /// </summary>
    public string? LexiconPath { get; set; }

    // ---------------------------------------------------------------- networks (Table 11)

    /// <summary>The phoneme encoder's and the prompt encoder's feed-forward filter (2048).</summary>
    public int FilterChannels { get; set; } = 2048;

    /// <summary>Their feed-forward convolution's kernel (9).</summary>
    public int EncoderKernel { get; set; } = 9;

    /// <summary>The speech prompt encoder's Transformer layers (6).</summary>
    public int NumPromptLayers { get; set; } = 6;

    /// <summary>Whether the phoneme encoder's LayerNorms are style-adaptive on the prompt (false; Amphion's true).</summary>
    public bool StyleAdaptivePhonemeEncoder { get; set; }

    /// <summary>The Transformers' positional encoding (Sinusoidal; Amphion's batch-indexed table).</summary>
    public NaturalSpeech2Positions Positions { get; set; } = NaturalSpeech2Positions.Sinusoidal;

    /// <summary>The duration predictor's convolution layers (30).</summary>
    public int DurationLayers { get; set; } = 30;

    /// <summary>The duration predictor's kernel (3).</summary>
    public int DurationKernel { get; set; } = 3;

    /// <summary>The pitch predictor's convolution layers (30).</summary>
    public int PitchLayers { get; set; } = 30;

    /// <summary>The pitch predictor's kernel (5).</summary>
    public int PitchKernel { get; set; } = 5;

    /// <summary>The predictors attend to the prompt before every this-many convolutions (3: ten attention layers of 30).</summary>
    public int PredictorAttentionEvery { get; set; } = 3;

    /// <summary>The predictors' dropout (0.5).</summary>
    public double PredictorDropout { get; set; } = 0.5;

    /// <summary>The lowest pitch bucket boundary in Hz (50; Amphion's).</summary>
    public double PitchMin { get; set; } = 50;

    /// <summary>The highest pitch bucket boundary in Hz (1100; Amphion's).</summary>
    public double PitchMax { get; set; } = 1100;

    /// <summary>The pitch embedding's buckets (512; Amphion's), log-spaced between the boundaries.</summary>
    public int PitchBins { get; set; } = 512;

    /// <summary>The prompt's query tokens m (32).</summary>
    public int QueryTokens { get; set; } = 32;

    /// <summary>The denoiser's WaveNet layers (40).</summary>
    public int DenoiserLayers { get; set; } = 40;

    /// <summary>The denoiser's dilation (2, App. B), when <see cref="DenoiserDilationCycle"/> is null.</summary>
    public int DenoiserDilation { get; set; } = 2;

    /// <summary>A dilation cycle c giving layer i the dilation 2^(i mod c) (null; Amphion's 2: 1, 2, 1, 2 ...).</summary>
    public int? DenoiserDilationCycle { get; set; }

    /// <summary>The denoiser attends to the query tokens and applies FiLM every this-many layers (3).</summary>
    public int DenoiserAttentionEvery { get; set; } = 3;

    /// <summary>
    /// The first such layer, 0-based (2: layers 3, 6 ... 39 counted from 1, Table 11's 13 attention layers of 40;
    /// Amphion's 0 gives 14).
    /// </summary>
    public int DenoiserAttentionOffset { get; set; } = 2;

    /// <summary>
    /// Dropout on the denoiser's attention weights (0.1, Amphion's). The paper's WaveNet dropout of 0.2 has no stated
    /// place; Amphion builds a 0.2 dropout in each block and never applies it.
    /// </summary>
    public double DenoiserAttentionDropout { get; set; } = 0.1;

    // ---------------------------------------------------------------- diffusion

    /// <summary>The diffusion formulation (the paper's score SDE; Amphion's flow matching is an option).</summary>
    public NaturalSpeech2Diffusion Diffusion { get; set; } = NaturalSpeech2Diffusion.ScoreSde;

    /// <summary>β at t = 0 of the linear noise schedule (0.05; Amphion's, the paper's schedule is unstated).</summary>
    public double BetaMin { get; set; } = 0.05;

    /// <summary>β at t = 1 (20; Amphion's).</summary>
    public double BetaMax { get; set; } = 20;

    /// <summary>Training steps t are drawn from U(0, 1) clamped to [ε, 1 − ε] (1e-5; Amphion's).</summary>
    public double TimeEpsilon { get; set; } = 1e-5;

    /// <summary>Eq. 6's score term (noise space; your choice, the literal form is an option).</summary>
    public NaturalSpeech2ScoreLoss ScoreLoss { get; set; } = NaturalSpeech2ScoreLoss.NoiseSpace;

    /// <summary>Whether the data term is an L1 distance (false: Eq. 6's squared L2; Amphion's L1).</summary>
    public bool DataLossL1 { get; set; }

    /// <summary>Whether the score term is an L1 distance (false; Amphion's L1).</summary>
    public bool NoiseLossL1 { get; set; }

    /// <summary>The score term's weight (1).</summary>
    public double NoiseLossWeight { get; set; } = 1.0;

    /// <summary>λ_ce-rvq (0.1, §3.2; Amphion's 0.5).</summary>
    public double CeLossWeight { get; set; } = 0.1;

    /// <summary>The residual the RVQ cross-entropy reads (the paper's ground truth; Amphion's prediction).</summary>
    public NaturalSpeech2CeResidual CeResidual { get; set; } = NaturalSpeech2CeResidual.GroundTruth;

    // ---------------------------------------------------------------- prompt (§3.3)

    /// <summary>The training prompt's smallest share of the phonemes (0.1; Amphion's).</summary>
    public double PromptMinFraction { get; set; } = 0.1;

    /// <summary>Its largest share (0.5; Amphion's).</summary>
    public double PromptMaxFraction { get; set; } = 0.5;

    /// <summary>Where the training prompt is cut (anywhere; Amphion's configured start).</summary>
    public NaturalSpeech2PromptPosition PromptPosition { get; set; } = NaturalSpeech2PromptPosition.Anywhere;

    // ---------------------------------------------------------------- inference (§5.1)

    /// <summary>The sampling temperature τ: z₁ ~ N(0, τ⁻¹ I) (1.22; Amphion divides by 1.2, τ = 1.44).</summary>
    public double Temperature { get; set; } = 1.22;

    /// <summary>The reverse ODE's steps (150; Amphion's inference script 200).</summary>
    public int SamplingSteps { get; set; } = 150;

    /// <summary>The ODE solver (Euler).</summary>
    public NaturalSpeech2Solver Solver { get; set; } = NaturalSpeech2Solver.Euler;

    // ---------------------------------------------------------------- training (§5.1)

    /// <summary>The learning-rate warm-up steps (32000; Amphion's 5000).</summary>
    public int WarmupSteps { get; set; } = 32000;

    /// <summary>
    /// Whether the learning rate follows a cosine decay to zero at <see cref="TrainingSteps"/> after the warm-up
    /// (false: the inverse square root; Amphion's cosine).
    /// </summary>
    public bool CosineSchedule { get; set; }

    /// <summary>The cosine schedule's length (800000; Amphion's).</summary>
    public int TrainingSteps { get; set; } = 800000;

    /// <summary>The gradient-norm clip (0.5; Amphion's, the paper's is unstated).</summary>
    public double GradientClipNorm { get; set; } = 0.5;
}
