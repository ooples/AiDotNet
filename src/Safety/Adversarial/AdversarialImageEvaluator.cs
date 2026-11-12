using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Safety;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Adversarial;

/// <summary>
/// Detects adversarial perturbations in images via the learnable feature-squeezing
/// ensemble described by Xu et al. 2018.
/// </summary>
/// <remarks>
/// <para>
/// Adversarial attacks add imperceptible perturbations that cause classifiers to
/// misclassify content. This module detects such perturbations by extracting three
/// statistical features from the image and combining them with a learnable linear
/// ensemble:
/// </para>
/// <list type="bullet">
///   <item>High-frequency energy ratio — adversarial perturbations elevate HF content.</item>
///   <item>Histogram smoothness — adversarial images produce non-natural histograms with gaps.</item>
///   <item>Feature-squeezing residual — bit-depth reduction removes the perturbation, leaving a measurable L2.</item>
/// </list>
/// <para>
/// Per Xu et al. 2018 §4 the final score is a learnable weighted combination of these
/// detectors (the paper describes it as a logistic ensemble). This implementation models
/// that as a single <see cref="DenseLayer{T}"/> mapping <c>[B, 3]</c> features → <c>[B, 1]</c>
/// score, with a sigmoid activation for the [0, 1] range that the
/// <see cref="EvaluateImage(Tensor{T})"/> threshold expects.
/// </para>
/// <para>
/// <b>For Beginners:</b> An adversarial image looks normal to humans but tricks AI
/// classifiers. This module looks for three telltale patterns of injected noise and
/// learns the right weight on each (some attacks reveal themselves more in HF energy,
/// others in pixel-histogram gaps).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks",
    "https://arxiv.org/abs/1704.01155",
    Year = 2018,
    Authors = "Weilin Xu, David Evans, Yanjun Qi")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class AdversarialImageEvaluator<T> : NeuralNetworkBase<T>, IImageSafetyModule<T>
{
    private const int FeatureCount = 3;  // HF energy, histogram, feature-squeezing
    private static readonly INumericOperations<T> StaticNumOps = MathHelper.GetNumericOperations<T>();
    private double _threshold;

    /// <inheritdoc />
    public string ModuleName => "AdversarialImageEvaluator";

    /// <inheritdoc />
    public bool IsReady => true;

    /// <summary>
    /// Initializes a new adversarial image evaluator.
    /// </summary>
    public AdversarialImageEvaluator(AdversarialImageEvaluatorOptions? options = null)
        : base(new NeuralNetworkArchitecture<T>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.BinaryClassification,
                inputHeight: 32, inputWidth: 32, inputDepth: 3,
                outputSize: 1),
              new BinaryCrossEntropyLoss<T>())
    {
        options ??= new AdversarialImageEvaluatorOptions();
        if (options.Threshold < 0 || options.Threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options.Threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = options.Threshold;

        // Build the layer chain now rather than on first Predict. Until this ran, Layers was
        // EMPTY, so both parameter surfaces described a model with no layers at all while the
        // model plainly has a Dense(3 -> 1). ParameterCount papered over that by returning
        // FeatureCount + 1 whenever the base returned less -- four parameters GetParameters
        // could not produce, because the tensors did not exist yet. With the layer built here
        // and sized explicitly in InitializeLayers, both surfaces report the true 4 from
        // construction, and neither changes value across the first Predict.
        EnsureArchitectureInitialized();
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // ONLY THE SINGLE HEAD IS ACCEPTED, BECAUSE ONLY THE SINGLE HEAD IS USED. PredictCore
            // always runs Layers[0].Forward(features) and nothing else, while ForwardForTraining
            // delegates to the base path, which walks EVERY layer in Layers. A multi-layer
            // architecture was therefore accepted, trained end to end, and then predicted from its
            // first layer alone -- training updated layers the prediction path never reaches, so the
            // model appeared to train while its predictions ignored most of what was learned.
            //
            // Rejecting the mismatch is the honest option: this evaluator is three fixed heuristic
            // features and one scoring head (Xu et al. 2018 SS4), so there is no meaningful
            // multi-layer variant for PredictCore to consume.
            if (Architecture.Layers.Count != 1)
            {
                throw new ArgumentException(
                    $"{nameof(AdversarialImageEvaluator<T>)} scores images from {FeatureCount} fixed "
                    + "heuristic features through a single learnable head, so a supplied architecture "
                    + $"must contain exactly one layer; got {Architecture.Layers.Count}. Training would "
                    + "update every layer while prediction reads only the first.",
                    nameof(Architecture));
            }

            Layers.AddRange(Architecture.Layers);
            return;
        }

        // Single learnable Dense (3 → 1) with Sigmoid for the [0, 1] score range,
        // per Xu et al. 2018 §4 (HF energy, histogram, feature squeezing).
        //
        // Sized EXPLICITLY at FeatureCount inputs rather than deferred. The input width is not
        // actually unknown here — PredictCore always feeds exactly FeatureCount features, which
        // it computes itself — so deferring it only hid that fact from the parameter surfaces.
        // A deferred layer gets its width from the architecture's declared input during lazy
        // resolution, and this model's architecture describes the IMAGE ([3, 32, 32]), not the
        // feature vector the layer consumes. Resolution therefore built a 32-input Dense worth
        // 33 parameters, which the first forward pass discarded and rebuilt at the real width of
        // 3. Stating the width the layer is always given makes ParameterCount and GetParameters
        // both report the true 4 from construction onward, and they no longer change value
        // across the first Predict.
        Layers.Add(new FullyConnectedLayer<T>(FeatureCount, 1, new SigmoidActivation<T>()));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Extracts the three heuristic features from the input image, then runs them
    /// through the learnable Dense layer to produce a per-image adversarial score
    /// in [0, 1]. Input shape: <c>[C, H, W]</c> (single sample) or <c>[B, C, H, W]</c>
    /// (batched). Output shape: <c>[1]</c> or <c>[B, 1]</c> respectively.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Defensive lazy init: the base's EnsureArchitectureInitialized
        // only fires from train / first-Predict in NeuralNetworkBase, but
        // model-family invariant tests can hit Predict on a freshly-
        // constructed model where Layers is still empty —
        // `Layers[0].Forward(features)` then throws IndexOutOfRange and
        // cascade-fails every test in AdversarialImageEvaluatorTests.
        if (Layers.Count == 0) InitializeLayers();

        var features = ExtractFeatures(input, out bool wasUnbatched);

        // Run through the learnable Dense layer → [batch, 1]
        var score = Layers[0].Forward(features);

        if (wasUnbatched && score.Rank > 1 && score.Shape[0] == 1)
        {
            var squeezed = new int[score.Rank - 1];
            for (int i = 0; i < squeezed.Length; i++) squeezed[i] = score.Shape[i + 1];
            score = score.Reshape(squeezed);
        }
        return score;
    }

    /// <summary>
    /// Builds the <c>[B, FeatureCount]</c> heuristic feature tensor that the learnable layer
    /// consumes: HF energy, histogram anomaly, and feature-squeezing distance (Xu et al. 2018 §4).
    /// </summary>
    /// <remarks>
    /// Shared by <see cref="PredictCore"/> and <see cref="ForwardForTraining"/> so both paths feed
    /// the layer the same representation. They previously did not, and the consequences were
    /// silent: training pushed the RAW image through the layer chain, sizing the deferred Dense to
    /// the image width, and the first Predict then re-resolved that same layer to the 3-feature
    /// width -- discarding every weight training had learned. The model trained and predicted on
    /// different inputs, and nothing failed, because lazy re-resolution quietly rebuilt the layer
    /// instead of reporting the conflict.
    /// </remarks>
    private Tensor<T> ExtractFeatures(Tensor<T> input, out bool wasUnbatched)
    {
        // Promote rank-3 single-sample to rank-4 batched, mirror the
        // MobileNetV3 / Mask2Former pattern.
        wasUnbatched = input.Rank == 3;
        var batched = wasUnbatched
            ? input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] })
            : input;
        if (batched.Rank != 4)
            throw new ArgumentException(
                $"AdversarialImageEvaluator expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {input.Rank}.",
                nameof(input));

        int batch = batched.Shape[0];
        int channels = batched.Shape[1];
        int height = batched.Shape[2];
        int width = batched.Shape[3];

        // The heuristics read each sample as its flat C*H*W buffer: rows of `width` values with
        // the channels stacked, i.e. a [batch, channels * height, width] view. They are fixed
        // feature extractors (not trained), so nothing here is recorded on the tape.
        using var noGrad = new NoGradScope<T>();
        var engine = Engine;
        var samples = engine.Reshape(batched.Contiguous(), new[] { batch, channels * height, width });

        // Build per-batch [batch, 3] feature tensor.
        return engine.TensorStack(new[]
        {
            ComputeHighFrequencyAnomalyScore(samples),
            ComputeHistogramAnomalyScore(samples),
            ComputeFeatureSqueezingScore(samples)
        }, 1);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Routes training through the SAME feature extraction <see cref="PredictCore"/> uses. Without
    /// this the base walked the layer chain over the raw image, so the weights training adjusted
    /// belonged to a differently-shaped layer than the one inference used. Mirrors the
    /// <c>PANNs</c> pattern of normalising the model input before delegating.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (Layers.Count == 0) InitializeLayers();

        // Already-extracted features pass straight through: callers that have done the extraction
        // themselves (and the layer-only path) must not be made to round-trip an image.
        if (input.Rank == 2 && input.Shape[1] == FeatureCount)
            return base.ForwardForTraining(input);

        return base.ForwardForTraining(ExtractFeatures(input, out _));
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        if (image is null || image.Length < 64) return findings;

        var score = Predict(image);
        double combinedScore = NumOps.ToDouble(score.Length > 0 ? score[0] : NumOps.Zero);

        if (combinedScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Manipulated,
                Severity = combinedScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, combinedScore),
                Description = $"Adversarial perturbation detected (score: {combinedScore:F3}). " +
                              "Image may have been adversarially modified to evade classifiers.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null) return new List<SafetyFinding>();
        // Reshape the flat content vector into a square grayscale image so the
        // feature heuristics still have spatial structure to work with.
        int side = (int)Math.Sqrt(content.Length);
        if (side * side > content.Length) side -= 1;
        if (side < 4) return new List<SafetyFinding>();
        var tensor = new Tensor<T>(new[] { 1, side, side }, content.Subvector(0, side * side));
        return EvaluateImage(tensor);
    }

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    /// <summary>
    /// AIE's <see cref="Predict"/> doesn't feed the input image directly into
    /// <c>Layers[0]</c> — it first extracts a 3-element feature vector
    /// (<see cref="ComputeHighFrequencyAnomalyScore"/>,
    /// <see cref="ComputeHistogramAnomalyScore"/>,
    /// <see cref="ComputeFeatureSqueezingScore"/>) and only then runs the
    /// Dense(3 → 1) head. The base
    /// <see cref="NeuralNetworkBase{T}.GetNamedLayerActivations"/> iterates
    /// <c>Layers</c> and calls <c>Forward</c> with the raw image, which both
    /// (a) bypasses the feature-extraction stage that defines AIE's design
    /// and (b) would attempt <c>Dense.Forward([B, C, H, W])</c> — a shape
    /// the lazy layer hasn't been sized for. Worse, on a freshly-constructed
    /// network where the layers haven't been materialised yet (Layers.Count
    /// is 0 pre-first-Predict) the base loop produces an empty dictionary,
    /// trivially failing
    /// <c>NeuralNetworkModelTestBase.NamedLayerActivations_ShouldBeNonEmpty</c>.
    /// Override to run the actual Predict pipeline and expose the Dense
    /// sigmoid output as the single named activation.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (Layers.Count == 0) InitializeLayers();
        var output = Predict(input);
        return new Dictionary<string, Tensor<T>>
        {
            // Use the layer's runtime type name so the key shape matches the
            // base's "Layer_0_DenseLayer"-style convention.
            [$"Layer_0_{Layers[0].GetType().Name}"] = output.Clone()
        };
    }


    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = "AdversarialImageEvaluator",
        Description = "Feature-squeezing ensemble adversarial-image detector (Xu et al. 2018).",
        FeatureCount = FeatureCount,
        Complexity = 1,
        // Populate AdditionalInfo (the canonical InputShape/OutputShape/hyperparameter report) and
        // ModelData — an empty shell fails the Metadata_ShouldExist invariant every model is held to.
        AdditionalInfo = new System.Collections.Generic.Dictionary<string, object>
        {
            { "FeatureCount", FeatureCount },
            { "DetectionThreshold", _threshold },
            { "InputShape", Architecture.GetInputShape() },
            { "OutputShape", Architecture.GetOutputShape() },
        },
        ModelDataProvider = () => SerializeForMetadata(),
    };

    /// <inheritdoc />


    /// <inheritdoc />


    private static Tensor<T> ComputeHighFrequencyAnomalyScore(Tensor<T> samples)
    {
        // Compute energy in high-frequency components using Laplacian approximation.
        // Natural images concentrate energy in low frequencies; adversarial perturbations
        // raise the HF ratio. `samples` is [batch, rows, width] (each sample's flat buffer).
        var engine = AiDotNetEngine.Current;
        int batch = samples.Shape[0];
        int height = samples.Shape[1];
        int width = samples.Shape[2];

        if (width < 4 || height < 4) return Tensor<T>.CreateZeros(batch);

        // Interior points y in [1, maxRows), x in [1, maxCols); every 4-neighbour is in range.
        int rows = Math.Min(height - 1, 128) - 1;
        int cols = Math.Min(width - 1, 128) - 1;
        var center = engine.TensorSlice(samples, new[] { 0, 1, 1 }, new[] { batch, rows, cols });
        var left = engine.TensorSlice(samples, new[] { 0, 1, 0 }, new[] { batch, rows, cols });
        var right = engine.TensorSlice(samples, new[] { 0, 1, 2 }, new[] { batch, rows, cols });
        var up = engine.TensorSlice(samples, new[] { 0, 0, 1 }, new[] { batch, rows, cols });
        var down = engine.TensorSlice(samples, new[] { 0, 2, 1 }, new[] { batch, rows, cols });

        var laplacian = engine.TensorSubtract(
            engine.TensorSubtract(
                engine.TensorSubtract(
                    engine.TensorSubtract(engine.TensorMultiplyScalar(center, StaticNumOps.FromDouble(4)), left),
                    right),
                up),
            down);
        var axes = new[] { 1, 2 };
        var hfEnergy = engine.ReduceSum(engine.TensorSquare(laplacian), axes, false);
        var lfEnergy = engine.ReduceSum(engine.TensorSquare(center), axes, false);

        // lfEnergy < 1e-10 -> 0; hfRatio < 0.15 -> 0; else min(1, (hfRatio - 0.15) / 0.5).
        var tooDark = SafetyTensorOps<T>.Lt(lfEnergy, 1e-10);
        var total = engine.TensorAdd(lfEnergy, hfEnergy);
        var hfRatio = engine.TensorDivide(hfEnergy, SafetyTensorOps<T>.Where(tooDark, 1.0, total));
        var score = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Lt(hfRatio, 0.15), 0.0,
            SafetyTensorOps<T>.MinOne(SafetyTensorOps<T>.Affine(hfRatio, 1.0 / 0.5, -0.15 / 0.5)));
        return SafetyTensorOps<T>.Where(tooDark, 0.0, score);
    }

    /// <summary>
    /// Scales pixels above 1 from the 0-255 range down to 0-1 (<c>val &gt; 1 ? val / 255 : val</c>).
    /// </summary>
    private static Tensor<T> NormalizePixelRange(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Gt(values, 1.0),
            engine.TensorDivideScalar(values, StaticNumOps.FromDouble(255.0)), values);
    }

    private static Tensor<T> ComputeHistogramAnomalyScore(Tensor<T> samples)
    {
        var engine = AiDotNetEngine.Current;
        int bins = 64;
        int batch = samples.Shape[0];
        int sampleSize = samples.Shape[1] * samples.Shape[2];
        int totalPixels = Math.Min(sampleSize, 65536);

        if (totalPixels < 100) return Tensor<T>.CreateZeros(batch);

        // Bin each value of the first 65536 pixels into [0, 1): pixels above 1 are rescaled from
        // 0-255, everything is clamped to [0, 1 - 1e-10]; a NaN pixel lands in bin 0.
        var flat = engine.TensorNarrow(engine.Reshape(samples, new[] { batch, sampleSize }), 1, 0, totalPixels);
        var values = NormalizePixelRange(engine.TensorWhere(engine.TensorIsNan(flat),
            SafetyTensorOps<T>.FullLike(flat, 0.0), flat));
        values = SafetyTensorOps<T>.MaxZero(values);
        values = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Gt(values, 1.0 - 1e-10), 1.0 - 1e-10, values);
        var bin = engine.TensorFloor(engine.TensorMultiplyScalar(values, StaticNumOps.FromDouble(bins)));
        bin = SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Gt(bin, bins - 1), bins - 1.0, bin);

        // histogram[b, i] = number of pixels of sample b in bin i (bin ids are exact integers,
        // so a [0, bins) histogram with `bins` buckets counts each id in its own bucket)
        var perSample = new Tensor<T>[batch];
        T lower = StaticNumOps.Zero;
        T upper = StaticNumOps.FromDouble(bins);
        for (int b = 0; b < batch; b++)
            perSample[b] = engine.TensorHistc(engine.TensorSliceAxis(bin, 0, b), bins, lower, upper);
        var histogram = engine.TensorStack(perSample, 0);

        var axis1 = new[] { 1 };
        var emptyBins = engine.ReduceSum(SafetyTensorOps<T>.Lt(histogram, 0.5), axis1, false);

        int inner = bins - 2;
        var laplacian = engine.TensorSubtract(
            engine.TensorAdd(engine.TensorNarrow(histogram, 1, 0, inner), engine.TensorNarrow(histogram, 1, 2, inner)),
            engine.TensorMultiplyScalar(engine.TensorNarrow(histogram, 1, 1, inner), StaticNumOps.FromDouble(2.0)));
        var smoothnessViolation = engine.ReduceSum(engine.TensorAbs(laplacian), axis1, false);
        var avgSmoothness = engine.TensorDivideScalar(
            engine.TensorDivideScalar(smoothnessViolation, StaticNumOps.FromDouble(inner)),
            StaticNumOps.FromDouble(Math.Max(totalPixels / bins, 1)));
        var smoothnessScore = SafetyTensorOps<T>.ClampUnit(SafetyTensorOps<T>.Affine(avgSmoothness, 1.0, -0.5));
        var emptyBinRatio = engine.TensorDivideScalar(emptyBins, StaticNumOps.FromDouble(bins));
        var gapScore = SafetyTensorOps<T>.ClampUnit(SafetyTensorOps<T>.Affine(emptyBinRatio, 1.0 / 0.4, -0.1 / 0.4));

        return engine.TensorAdd(
            engine.TensorMultiplyScalar(smoothnessScore, StaticNumOps.FromDouble(0.6)),
            engine.TensorMultiplyScalar(gapScore, StaticNumOps.FromDouble(0.4)));
    }

    private static Tensor<T> ComputeFeatureSqueezingScore(Tensor<T> samples)
    {
        var engine = AiDotNetEngine.Current;
        int batch = samples.Shape[0];
        int sampleSize = samples.Shape[1] * samples.Shape[2];
        int numPixels = Math.Min(sampleSize, 16384);

        var values = NormalizePixelRange(
            engine.TensorNarrow(engine.Reshape(samples, new[] { batch, sampleSize }), 1, 0, numPixels));

        // Squeeze to 4-bit depth (round half to even, like Math.Round) and measure the RMS change.
        var fifteen = StaticNumOps.FromDouble(15.0);
        var squeezed = engine.TensorDivideScalar(engine.TensorRound(engine.TensorMultiplyScalar(values, fifteen)), fifteen);
        var diff = engine.TensorSubtract(values, squeezed);
        var l2Sum = engine.ReduceSum(engine.TensorSquare(diff), new[] { 1 }, false);

        var rms = engine.TensorSqrt(engine.TensorDivideScalar(l2Sum, StaticNumOps.FromDouble(numPixels)));
        return SafetyTensorOps<T>.Where(SafetyTensorOps<T>.Lt(rms, 0.01), 0.0,
            SafetyTensorOps<T>.MinOne(SafetyTensorOps<T>.Affine(rms, 1.0 / 0.03, -0.01 / 0.03)));
    }
}
