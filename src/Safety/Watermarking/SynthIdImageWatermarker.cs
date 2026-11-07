using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>The outcome of a calibrated watermark decision.</summary>
public enum WatermarkVerdict
{
    /// <summary>The image is not watermarked: its score is implausible for a watermarked image and typical of a clean one.</summary>
    NotWatermarked,

    /// <summary>The image is watermarked: its score is implausible for a clean image and typical of a watermarked one.</summary>
    Watermarked,

    /// <summary>Neither hypothesis can be rejected alone at the configured level, so no claim is made.</summary>
    Abstain,
}

/// <summary>A calibrated three-state decision on one image, with the evidence behind it.</summary>
public sealed class WatermarkDecision
{
    /// <summary>The decision.</summary>
    public WatermarkVerdict Verdict { get; init; }

    /// <summary>The decoder's detection logit for the image.</summary>
    public double DetectionLogit { get; init; }

    /// <summary>
    /// The conformal p-value against the clean calibration set: (1 + clean scores at least this high) / (n + 1).
    /// Small values mean a clean image rarely scores this high.
    /// </summary>
    public double CleanPValue { get; init; }

    /// <summary>
    /// The conformal p-value against the watermarked calibration set: (1 + watermarked scores at most this high) /
    /// (n + 1). Small values mean a watermarked image rarely scores this low.
    /// </summary>
    public double WatermarkedPValue { get; init; }

    /// <summary>The level alpha both p-values were compared against.</summary>
    public double Alpha { get; init; }
}

/// <summary>
/// A SynthID-Image-style post-hoc watermark. A trained encoder adds a content-dependent residual carrying a payload.
/// A trained decoder outputs a detection logit and one logit per payload bit. Detection is a three-state decision
/// calibrated on held-out clean and watermarked images.
/// </summary>
/// <remarks>
/// <para>
/// Gowal et al. (2025) describe SynthID-Image's objective and decision rule but not its networks, losses or training
/// schedule, and the weights are not public. This is an open implementation of what the paper states:
/// <list type="bullet">
/// <item>A post-hoc encoder and decoder ⟨f, g⟩, where g returns a detection logit and per-bit payload logits.</item>
/// <item>Detection trained by cross-entropy, on transformed watermarked images (label 1) and the same transformations
/// of clean images (label 0), under a perceptual constraint.</item>
/// <item>A separate payload loss.</item>
/// <item>A mark that depends on the content, so that it cannot be lifted onto another image.</item>
/// <item>Conformal p-values against held-out clean and watermarked calibration sets. Thresholding both at alpha
/// gives a watermarked, not-watermarked or abstain verdict.</item>
/// </list>
/// </para>
/// <para>
/// What the paper does not give comes from <see cref="StegaStampWatermarker{T}"/>, which this class extends. The
/// encoder is its U-Net and the decoder its spatial-transformer network, with one extra logit. The perceptual
/// constraint is its border-weighted YUV loss plus the WGAN critic. The transformations are its differentiable
/// warp, blur, noise, colour and JPEG distortions. Content dependence is trained with a specificity loss: one image's
/// residual is added to another image and labelled unwatermarked. The paper describes no such loss. Treat this as an
/// open design of the same family, not a reproduction of Google's system.
/// </para>
/// <para>
/// <see cref="Decide"/> needs <see cref="Calibrate"/> first. The guarantee is the conformal one: if a clean test image
/// is exchangeable with the clean calibration images, P(verdict = watermarked) &lt;= alpha. Under the same condition, a
/// watermarked image is called not-watermarked with probability at most alpha. A level of alpha can only be reached
/// with at least 1/alpha - 1 calibration images per set. With fewer, the decision abstains.
/// </para>
/// <para>
/// The keyed <see cref="Detect"/> answers a narrower question: does the image carry this key's payload? It scores
/// the payload logits against the key codeword's signs. That z is exactly standard normal, given the logits, for
/// images that do not carry the payload. A detection is reported only when that z is significant and the detection
/// logit also agrees: the calibrated verdict is watermarked, or, before calibration, the logit is positive.
/// </para>
/// <para>
/// <b>For Beginners:</b> one network hides a code in an image by changing it almost invisibly. A second network
/// answers two questions: "is there a mark at all?" and "what does the code say?". Before you trust its answer, show
/// it many clean and many marked images. It then learns how its scores are distributed and can say "marked", "not
/// marked" or "not sure", with a known error rate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SynthID-Image: Image watermarking at internet scale",
    "https://arxiv.org/abs/2510.09263",
    Year = 2025,
    Authors = "Sven Gowal, Rudy Bunel, Florian Stimberg, David Stutz, Guillermo Ortiz-Jimenez, et al.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class SynthIdImageWatermarker<T> : StegaStampWatermarker<T>
{
    // Sorted detection logits of the calibration sets. Their length varies with the calibration, so they stay out of
    // the parameter vector and persist through RegisterState instead.
    [Scratch]
    private double[] _cleanScores = Array.Empty<double>();

    [Scratch]
    private double[] _watermarkedScores = Array.Empty<double>();

    /// <summary>Creates a watermarker with SynthID-O's stated sizes (512 x 512 images, 136 payload bits).</summary>
    public SynthIdImageWatermarker() : this(new SynthIdImageOptions()) { }

    /// <summary>Creates a watermarker; <paramref name="key"/> derives the payload <see cref="StegaStampWatermarker{T}.EmbedWatermark"/> hides.</summary>
    public SynthIdImageWatermarker(SynthIdImageOptions options, string? key = null)
        : base(Checked(options), key)
    {
    }

    private static SynthIdImageOptions Checked(SynthIdImageOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        options.ValidateDetection();
        return options;
    }

    /// <inheritdoc />
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        state.DeclareDoubleVector("SynthIdImageWatermarker.CleanScores",
            () => new Vector<double>(_cleanScores), restored => _cleanScores = restored?.ToArray() ?? Array.Empty<double>());
        state.DeclareDoubleVector("SynthIdImageWatermarker.WatermarkedScores",
            () => new Vector<double>(_watermarkedScores), restored => _watermarkedScores = restored?.ToArray() ?? Array.Empty<double>());
    }

    private SynthIdImageOptions Detection => (SynthIdImageOptions)Settings;

    /// <inheritdoc />
    public override string ModuleName => "SynthIdImageWatermarker";

    private protected override string WatermarkName => "SynthID-style image watermark";

    // One logit per payload bit, then the detection logit.
    private protected override int DecoderOutputs => Settings.SecretBits + 1;

    /// <summary>Whether <see cref="Calibrate"/> has supplied both calibration sets.</summary>
    public bool IsCalibrated => _cleanScores.Length > 0 && _watermarkedScores.Length > 0;

    /// <summary>
    /// This trained watermarker under another key: the same networks and calibration, a different payload.
    /// </summary>
    public new SynthIdImageWatermarker<T> WithKey(string? key)
    {
        var keyed = new SynthIdImageWatermarker<T>(Detection, key);
        keyed.CopyTrainedStateFrom(this);
        keyed._cleanScores = (double[])_cleanScores.Clone();
        keyed._watermarkedScores = (double[])_watermarkedScores.Clone();
        return keyed;
    }

    /// <summary>The detection logit of each image in <paramref name="images"/>, [B, 3, H, W] or [3, H, W].</summary>
    public double[] DetectionLogits(Tensor<T> images)
    {
        var logits = DecodeLogits(Resized(AsBatch(images)));
        int b = logits.Shape[0], column = Settings.SecretBits;
        var scores = new double[b];
        for (int n = 0; n < b; n++) scores[n] = NumOps.ToDouble(logits[n, column]);
        return scores;
    }

    /// <summary>
    /// Records the detection-logit distributions the decision is calibrated against. Pass held-out images, never the
    /// training images. Include the transformations you want covered (crops, JPEG and so on) in both sets.
    /// </summary>
    /// <param name="cleanImages">Images without the watermark, [N, 3, H, W].</param>
    /// <param name="watermarkedImages">Watermarked images, [M, 3, H, W].</param>
    /// <remarks>
    /// Each image is scored on its own, through the same single-image path <see cref="Decide"/> uses. The conformal
    /// guarantee needs calibration and test scores to be exchangeable, and a batched forward can round differently
    /// from a single-image one. When the scores are close together, that difference alone would put every test image
    /// above the whole calibration set.
    /// </remarks>
    public void Calibrate(Tensor<T> cleanImages, Tensor<T> watermarkedImages)
    {
        var clean = SingleImageLogits(cleanImages);
        var marked = SingleImageLogits(watermarkedImages);
        Array.Sort(clean);
        Array.Sort(marked);
        _cleanScores = clean;
        _watermarkedScores = marked;
    }

    private double[] SingleImageLogits(Tensor<T> images)
    {
        var batch = AsBatch(images);
        var scores = new double[batch.Shape[0]];
        for (int n = 0; n < scores.Length; n++) scores[n] = DetectionLogits(batch.Slice(n))[0];
        return scores;
    }

    /// <summary>
    /// The calibrated decision for <paramref name="image"/> at <see cref="SynthIdImageOptions.FalsePositiveRate"/>.
    /// Watermarked when only the clean hypothesis is rejected, not watermarked when only the watermarked hypothesis is
    /// rejected, and abstain otherwise.
    /// </summary>
    public WatermarkDecision Decide(Tensor<T> image)
    {
        if (!IsCalibrated)
            throw new InvalidOperationException("Call Calibrate with held-out clean and watermarked images before Decide.");
        double score = DetectionLogits(image)[0];
        double alpha = Detection.FalsePositiveRate;
        double cleanP = (1.0 + CountAtLeast(_cleanScores, score)) / (_cleanScores.Length + 1.0);
        double markedP = (1.0 + CountAtMost(_watermarkedScores, score)) / (_watermarkedScores.Length + 1.0);
        bool notClean = cleanP <= alpha, notMarked = markedP <= alpha;
        var verdict = notClean && !notMarked ? WatermarkVerdict.Watermarked
            : notMarked && !notClean ? WatermarkVerdict.NotWatermarked
            : WatermarkVerdict.Abstain;
        return new WatermarkDecision
        {
            Verdict = verdict, DetectionLogit = score, CleanPValue = cleanP, WatermarkedPValue = markedP, Alpha = alpha,
        };
    }

    // Scores >= value in a sorted array.
    private static int CountAtLeast(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (sorted[mid] < value) lo = mid + 1; else hi = mid;
        }

        return sorted.Length - lo;
    }

    // Scores <= value in a sorted array.
    private static int CountAtMost(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (sorted[mid] <= value) lo = mid + 1; else hi = mid;
        }

        return lo;
    }

    /// <summary>
    /// Tests <paramref name="image"/> for this key's payload. The reported z is the payload score of
    /// <see cref="StegaStampWatermarker{T}.Detect"/>. A detection is reported only when the detection logit also
    /// agrees: the calibrated verdict is watermarked, or, before calibration, the logit is positive.
    /// </summary>
    public override WatermarkResult Detect(Tensor<T> image)
    {
        var payload = base.Detect(image);
        bool marked = IsCalibrated
            ? Decide(image).Verdict == WatermarkVerdict.Watermarked
            : DetectionLogits(image)[0] > 0;
        return new WatermarkResult
        {
            WatermarkDetected = payload.WatermarkDetected && marked,
            Confidence = payload.Confidence,
            ZScore = payload.ZScore,
            PValue = payload.PValue,
            ScoredUnits = payload.ScoredUnits,
            Modality = payload.Modality,
            WatermarkType = payload.WatermarkType,
        };
    }

    /// <summary>
    /// StegaStamp's secret, image and critic losses, plus the detection loss. The detection loss is cross-entropy on
    /// the detection logit: label 1 for the distorted watermarked images, label 0 for the same distortions of the
    /// clean images. It also includes the specificity term: each image plus its neighbour's residual, labelled 0. Both ramp
    /// in with the image loss, after the secret-only phase.
    /// </summary>
    private protected override Tensor<T> EncoderDecoderLoss(IReadOnlyList<Tensor<T>> data)
    {
        var images = data[0];
        var secrets = data[1];
        int b = images.Shape[0], s = Settings.ImageSize, bits = Settings.SecretBits;
        Tensor<T> Distorted(Tensor<T> x) => Distort(x, data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
        Tensor<T> DetectionColumn(Tensor<T> logits) => Engine.TensorSlice(logits, new[] { 0, bits }, new[] { b, 1 });
        Tensor<T> MeanSoftplus(Tensor<T> x) => Engine.Reshape(Engine.ReduceMean(Engine.Softplus(x), new[] { 0, 1 }, keepDims: false), new[] { 1 });
        var minusOne = NumOps.FromDouble(-1.0);

        var residual = EncodeResidual(images, secrets);
        var encoded = Engine.TensorAdd(images, residual);
        var markedLogits = DecodeLogits(Distorted(encoded));
        var cleanLogits = DecodeLogits(Distorted(images));

        // Binary cross-entropy with logits: softplus(-x) for label 1, softplus(x) for label 0.
        var detection = Engine.TensorAdd(
            MeanSoftplus(Engine.TensorMultiplyScalar(DetectionColumn(markedLogits), minusOne)),
            MeanSoftplus(DetectionColumn(cleanLogits)));
        if (b >= 2 && Detection.SpecificityLossScale > 0)
        {
            var neighbour = Engine.TensorConcatenate(new[]
            {
                Engine.TensorSlice(residual, new[] { 1, 0, 0, 0 }, new[] { b - 1, 3, s, s }),
                Engine.TensorSlice(residual, new[] { 0, 0, 0, 0 }, new[] { 1, 3, s, s }),
            }, axis: 0);
            var liftedLogits = DecodeLogits(Distorted(Engine.TensorAdd(images, neighbour)));
            detection = Engine.TensorAdd(detection,
                Engine.TensorMultiplyScalar(MeanSoftplus(DetectionColumn(liftedLogits)), NumOps.FromDouble(Detection.SpecificityLossScale)));
        }

        // The detection terms ramp in with the image loss: w[1] / ImageLossScale, which is 0 during the secret-only
        // phase. With detection on from step 0, the encoder learned a payload-free mark that the detection logit reads
        // easily, and the payload stayed at chance. That separates the two objectives, as the paper trains them.
        var ramp = Engine.TensorMultiplyScalar(Engine.TensorSlice(data[9], new[] { 1 }, new[] { 1 }),
            NumOps.FromDouble(Detection.DetectionLossScale / Settings.ImageLossScale));
        detection = Engine.TensorMultiply(detection, ramp);

        var secretLoss = SecretLoss(markedLogits, secrets);
        var imageLoss = ResidualLoss(residual, data[9]);
        var criticLoss = Engine.ReduceMean(CriticScore(encoded), new[] { 0 }, keepDims: false);
        var w = data[9];
        Tensor<T> Weighted(Tensor<T> loss, int index) =>
            Engine.TensorMultiply(Engine.Reshape(loss, new[] { 1 }), Engine.TensorSlice(w, new[] { index }, new[] { 1 }));
        var total = Engine.TensorAdd(Engine.TensorAdd(Weighted(secretLoss, 0), Weighted(imageLoss, 1)), Weighted(criticLoss, 2));
        return Engine.TensorAdd(total, detection);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SynthID-style image watermark" }, { "ImageSize", Settings.ImageSize },
            { "SecretBits", Settings.SecretBits }, { "TrainingStep", TrainingStep },
            { "CleanCalibrationImages", _cleanScores.Length }, { "WatermarkedCalibrationImages", _watermarkedScores.Length },
        },
        ModelDataProvider = () => this.Serialize()
    };
}
