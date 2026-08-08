using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for SeACo-Paraformer: hot-word customizable ASR.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SeACo model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
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

        // INHERITED FROM ModelOptions, AND THEREFORE EASY TO MISS. Every declared property is copied
        // below; Seed is not declared here, so it was silently dropped and a copied configuration
        // produced a DIFFERENT model from the one it was copied from -- the failure mode that costs
        // the most to diagnose, because the two configurations compare equal on everything visible.
        Seed = other.Seed;

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;

        // SeACo's own properties, as opposed to the shared ASR ones above. Omitting any of them
        // is silent data loss: the clone keeps the default while the original keeps the
        // configured value. LearningRate is the one that bites hardest - a clone would train at
        // a different rate than the model it was copied from.
        NumDecoderLayers = other.NumDecoderLayers;
        NumBiasEncoderLayers = other.NumBiasEncoderLayers;
        FeedForwardDim = other.FeedForwardDim;
        LearningRate = other.LearningRate;
        TrainingStage = other.TrainingStage;
        CeWeight = other.CeWeight;
        MaeWeight = other.MaeWeight;
        BiasMergeLambda = other.BiasMergeLambda;
        SamplerLambda = other.SamplerLambda;
        HotwordMinLength = other.HotwordMinLength;
        HotwordMaxLength = other.HotwordMaxLength;
        HotwordMaskTokenId = other.HotwordMaskTokenId;
        HotwordBatchRatio = other.HotwordBatchRatio;
        HotwordUtteranceRatio = other.HotwordUtteranceRatio;
    }

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 12;
    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 8404;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";

    /// <summary>
    /// Gets or sets the Adam learning rate. The default, 5e-4, is the peak rate Paraformer /
    /// SeACo-Paraformer train with (Gao et al., arXiv 2206.08317; Shi et al., arXiv 2308.03266).
    /// </summary>
    public double LearningRate { get; set; } = 5e-4;

    /// <summary>
    /// Which parameter group <c>Train</c> updates. Default: <see cref="SeACoTrainingStage.Joint"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SeACo-Paraformer (Shi et al., arXiv 2308.03266, §3) trains the ASR backbone first, then FREEZES
    /// it and trains only the bias parameters, "separate from the ASR training". Select
    /// <see cref="SeACoTrainingStage.Backbone"/> then <see cref="SeACoTrainingStage.Bias"/> to
    /// reproduce that exactly.
    /// </para>
    /// <para>
    /// The default is <see cref="SeACoTrainingStage.Joint"/> so a single <c>Train</c> call updates every
    /// parameter, which is what callers expect from one entry point. It is deliberately NOT the paper's
    /// recipe: joint training lets hot-word supervision reshape the recognizer, which the paper's
    /// freeze exists to prevent.
    /// </para>
    /// </remarks>
    public SeACoTrainingStage TrainingStage { get; set; } = SeACoTrainingStage.Joint;

    /// <summary>
    /// Weight gamma on the cross-entropy term of Paraformer's objective,
    /// L_total = gamma * L_CE + L_MAE + L_MWER (Gao et al., arXiv 2206.08317, Eq 6). Default: 1.0.
    /// </summary>
    /// <remarks>
    /// The paper introduces gamma in Eq 6 without fixing a value in the text, so the neutral 1.0 is
    /// used rather than inventing one. Raise it to weight transcription accuracy over the predictor's
    /// token-count objective.
    /// </remarks>
    public double CeWeight { get; set; } = 1.0;

    /// <summary>
    /// Weight on the CIF predictor's mean-absolute-error term in Paraformer Eq 6. Default: 1.0, the
    /// unweighted form the equation states.
    /// </summary>
    /// <remarks>
    /// This term supervises the predicted token COUNT. Paraformer §2.2/2.4 describe it as guiding the
    /// predictor to convergence; with the weight at zero the predictor head trains unsupervised.
    /// </remarks>
    public double MaeWeight { get; set; } = 1.0;

    /// <summary>
    /// Token id substituted at non-hotword positions when computing SeACo's bias loss. Defaults to
    /// <c>null</c>, which resolves to <see cref="VocabSize"/> -- the appended no-bias slot.
    /// </summary>
    /// <remarks>
    /// SeACo §3.1 states that "an additional token (counted as #, means no-bias) is appended to the ASR
    /// output vocabulary to mark non-hotword position outputs", and §3 defines L_bias with a
    /// "hotword-position-aware criterion in which labels in non-hotword positions are replaced by #".
    /// APPENDED means the id is <see cref="VocabSize"/> itself — the new last slot — which is why the
    /// bias output layer emits VocabSize + 1 logits. A null value resolves to that appended index;
    /// set it explicitly only to point at a different reserved slot.
    /// </remarks>
    public int? HotwordMaskTokenId { get; set; }

    /// <summary>
    /// Resolves the '#' no-bias token id, defaulting to the appended slot at <see cref="VocabSize"/>.
    /// </summary>
    internal int ResolveHotwordMaskTokenId() => HotwordMaskTokenId ?? VocabSize;

    /// <summary>
    /// Depth of SeACo's bias (hotword) encoder. Default: 1.
    /// </summary>
    /// <remarks>
    /// SeACo §3 describes the bias encoder as a light module over the sampled hotword token sequences;
    /// the paper does not publish a depth, so the minimal single block is used rather than inventing a
    /// larger figure.
    /// </remarks>
    public int NumBiasEncoderLayers { get; set; } = 1;

    /// <summary>
    /// SeACo's r_b: ratio of BATCHES on which hotword sampling is performed. Default: 0.5.
    /// </summary>
    /// <remarks>
    /// SeACo §3: "r_b for the ratio of batches to conduct sampling, the forward of the other batches
    /// will be conducted with a default hotword &lt;blank&gt;". The paper names the hyper-parameter but
    /// does not publish its value, so an even split is used rather than inventing a tuned figure.
    /// </remarks>
    public double HotwordBatchRatio { get; set; } = 0.5;

    /// <summary>
    /// SeACo's r_u: same idea as <see cref="HotwordBatchRatio"/> but at UTTERANCE level inside an
    /// active batch. Default: 0.5.
    /// </summary>
    /// <remarks>
    /// SeACo §3: "the average number of hotwords sampled for active batch is r_u x bs + 1 (one for the
    /// default hotword)". Value not published; even split used.
    /// </remarks>
    public double HotwordUtteranceRatio { get; set; } = 0.5;

    /// <summary>
    /// SeACo's l_min: minimum length, in characters, of a sampled hotword. Default: 2.
    /// </summary>
    /// <remarks>
    /// SeACo §3 introduces "l_min and l_max for the minimum and maximum lengths of sampled hotwords"
    /// without publishing values. 2 excludes single-character fragments that carry no entity signal.
    /// </remarks>
    public int HotwordMinLength { get; set; } = 2;

    /// <summary>
    /// SeACo's l_max: maximum length, in characters, of a sampled hotword. Default: 10.
    /// </summary>
    /// <remarks>
    /// Companion to <see cref="HotwordMinLength"/>; the paper publishes no value. 10 spans the named
    /// entities its Aishell-1-NE test sets are built from.
    /// </remarks>
    public int HotwordMaxLength { get; set; } = 10;

    /// <summary>
    /// SeACo's lambda in Eq 5, the trust placed in the bias decoder. Default: 1.0.
    /// </summary>
    /// <remarks>
    /// arXiv 2308.03266 Eq 5: <c>P_m = P_ASR</c> when <c>argmax P_bi = &lt;no-bias&gt;</c>, otherwise
    /// <c>lambda * P_bi + (1 - lambda) * P_ASR</c>. §4.3 states "we use lambda = 1.0 for Equation (5) in
    /// inference", so that is the default; §3.2 calls it "a tunable parameter to adjust the degree of
    /// trust bias decoder output".
    /// </remarks>
    public double BiasMergeLambda { get; set; } = 1.0;

    /// <summary>
    /// Paraformer's sampling factor lambda, controlling how many target-embedding positions the GLM
    /// sampler substitutes into the acoustic embedding. Default: 1.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Gao et al. 2022 (arXiv 2206.08317) §2.3 Eq 4:
    /// <c>GLM(Y, Yhat) = Sampler(Es | Ea, Ec, ceil(lambda * d(Y, Yhat)))</c>, where
    /// <c>d(Y, Yhat)</c> is the Hamming distance between the target and the Pass-1 prediction — a count
    /// that is "larger when the model is poorly trained, and should decrease along with the training
    /// process", so the substitution rate anneals on its own without a schedule.
    /// </para>
    /// <para>
    /// §3 reports Paraformer "is robust to lambda in a range from 0.5 to 1.0" and Table 2 sweeps
    /// 0.2 / 0.5 / 0.75 / 1.0 / 1.5. 1.0 is the top of the robust band, hence the default; it is the
    /// paper's own value rather than an invented one.
    /// </para>
    /// </remarks>
    public double SamplerLambda { get; set; } = 1.0;

    /// <summary>
    /// Number of parallel-decoder blocks. Default: 6.
    /// </summary>
    /// <remarks>
    /// Gao et al. 2022 §4.2 describes the Paraformer base model as a "50-layers SAN-M encoder and
    /// 16-layer NAR Transformer decoder"; SeACo §4.2 reuses that backbone. 6 is retained as the default
    /// because it is what <c>LayerHelper.CreateDefaultParaformerLayers</c> already used, so exposing the
    /// knob does not silently re-shape existing models — set 16 to match the published base config.
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 6;

    /// <summary>
    /// Feed-forward inner width inside encoder and decoder blocks. Default: 2048.
    /// </summary>
    /// <remarks>
    /// Gao et al. 2022 §4.2: the Paraformer base model runs "with 2048 hidden units".
    /// </remarks>
    public int FeedForwardDim { get; set; } = 2048;
}
