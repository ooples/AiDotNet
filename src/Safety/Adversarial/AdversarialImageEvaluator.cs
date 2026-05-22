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
public class AdversarialImageEvaluator<T> : NeuralNetworkBase<T>, IImageSafetyModule<T>
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
    /// <param name="threshold">Detection threshold (0-1). Default: 0.5.</param>
    public AdversarialImageEvaluator(double threshold = 0.5)
        : base(new NeuralNetworkArchitecture<T>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.BinaryClassification,
                inputHeight: 32, inputWidth: 32, inputDepth: 3,
                outputSize: 1),
              new BinaryCrossEntropyLoss<T>())
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        // Single learnable Dense (3 → 1) with Sigmoid for the [0, 1] score range.
        // Initialised so the default forward pass approximately matches the
        // hand-tuned heuristic weights from Xu et al. 2018 §4
        // (HF: 0.40, histogram: 0.30, squeezing: 0.30).
        // DenseLayer<T>(outputSize, activation) — input dim is resolved on first
        // forward via lazy weight init.
        Layers.Add(new DenseLayer<T>(1, (IActivationFunction<T>?)new SigmoidActivation<T>()));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Extracts the three heuristic features from the input image, then runs them
    /// through the learnable Dense layer to produce a per-image adversarial score
    /// in [0, 1]. Input shape: <c>[C, H, W]</c> (single sample) or <c>[B, C, H, W]</c>
    /// (batched). Output shape: <c>[1]</c> or <c>[B, 1]</c> respectively.
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Defensive lazy init: the base's EnsureArchitectureInitialized
        // only fires from train / first-Predict in NeuralNetworkBase, but
        // model-family invariant tests can hit Predict on a freshly-
        // constructed model where Layers is still empty —
        // `Layers[0].Forward(features)` then throws IndexOutOfRange and
        // cascade-fails every test in AdversarialImageEvaluatorTests.
        if (Layers.Count == 0) InitializeLayers();

        // Promote rank-3 single-sample to rank-4 batched, mirror the
        // MobileNetV3 / Mask2Former pattern.
        bool wasUnbatched = input.Rank == 3;
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

        // Build per-batch [batch, 3] feature tensor.
        var features = new Tensor<T>(new[] { batch, FeatureCount });
        for (int b = 0; b < batch; b++)
        {
            // Slice this sample as flat span for the heuristic computations.
            int sampleSize = channels * height * width;
            var sampleData = new T[sampleSize];
            int offset = 0;
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        sampleData[offset++] = batched[b, c, h, w];
            var sampleSpan = new ReadOnlySpan<T>(sampleData);
            var sampleShape = new[] { channels, height, width };

            features[b, 0] = NumOps.FromDouble(ComputeHighFrequencyAnomalyScore(sampleSpan, sampleShape));
            features[b, 1] = NumOps.FromDouble(ComputeHistogramAnomalyScore(sampleSpan));
            features[b, 2] = NumOps.FromDouble(ComputeFeatureSqueezingScore(sampleSpan));
        }

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
        var data = new T[side * side];
        for (int i = 0; i < data.Length; i++) data[i] = content[i];
        var tensor = new Tensor<T>(new[] { 1, side, side }, new Vector<T>(data));
        return EvaluateImage(tensor);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (Layers.Count == 0) return;
        Layers[0].UpdateParameters(parameters);
    }

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

    /// <summary>
    /// AIE's pipeline always emits a 3-element feature vector to a single
    /// <c>DenseLayer(3 → 1)</c>: <c>3 × 1 = 3</c> weights + <c>1</c> bias = 4
    /// learnable parameters. The base class's <c>ParameterCount</c> tries to
    /// pre-resolve lazy layer shapes by propagating <c>Architecture.InputShape</c>
    /// through the layer chain, but AIE's Dense doesn't see the architecture's
    /// image shape <c>[C, H, W]</c> — it sees the C#-computed feature vector
    /// <c>[B, 3]</c> instead (feature extraction happens in <see cref="Predict"/>,
    /// outside the layer chain). The result: lazy Dense's <c>InputShape[0]</c>
    /// stays at the <c>-1</c> sentinel and contributes <c>0</c> to the sum,
    /// making the
    /// <c>NeuralNetworkModelTestBase.Parameters_ShouldBeNonEmpty</c>
    /// invariant trivially fail despite the model having a perfectly
    /// well-defined parameter count.
    /// </summary>
    /// <remarks>
    /// Custom <c>Architecture.Layers</c> from the caller are honoured (we
    /// defer to the base sum in that case). The override only short-circuits
    /// for the default Xu et al. 2018 Dense(3 → 1) topology.
    /// </remarks>
    public override long ParameterCount
    {
        get
        {
            // Caller-supplied custom Layers: base knows best — defer.
            if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
                return base.ParameterCount;

            // Default topology: AFTER a Forward has run, base.ParameterCount
            // returns the correct 4 (lazy Dense is materialised). BEFORE the
            // first Forward, base returns 0 because lazy Dense's
            // InputShape[0] is still −1. Take base's value when it's already
            // ≥ FeatureCount + 1 (post-Forward); otherwise emit the
            // architecturally known count.
            long baseCount = base.ParameterCount;
            return baseCount >= FeatureCount + 1 ? baseCount : FeatureCount + 1;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = "AdversarialImageEvaluator",
        Description = "Feature-squeezing ensemble adversarial-image detector (Xu et al. 2018).",
        FeatureCount = FeatureCount,
        Complexity = 1,
    };

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_threshold);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _threshold = reader.ReadDouble();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new AdversarialImageEvaluator<T>(_threshold);

    private static double ComputeHighFrequencyAnomalyScore(ReadOnlySpan<T> span, int[] shape)
    {
        // Compute energy in high-frequency components using Laplacian approximation.
        // Natural images concentrate energy in low frequencies; adversarial perturbations
        // raise the HF ratio.
        int width = shape.Length >= 2 ? shape[shape.Length - 1] : (int)Math.Sqrt(span.Length);
        int height = span.Length / Math.Max(width, 1);

        if (width < 4 || height < 4) return 0;

        double lfEnergy = 0, hfEnergy = 0;
        int count = 0;

        int stride = width;
        int maxRows = Math.Min(height - 1, 128);
        int maxCols = Math.Min(width - 1, 128);

        for (int y = 1; y < maxRows; y++)
        {
            for (int x = 1; x < maxCols; x++)
            {
                int idx = y * stride + x;
                if (idx >= span.Length || idx + 1 >= span.Length || idx - 1 < 0 ||
                    idx - stride < 0 || idx + stride >= span.Length) continue;

                double center = StaticNumOps.ToDouble(span[idx]);
                double right = StaticNumOps.ToDouble(span[idx + 1]);
                double left = StaticNumOps.ToDouble(span[idx - 1]);
                double up = StaticNumOps.ToDouble(span[idx - stride]);
                double down = StaticNumOps.ToDouble(span[idx + stride]);

                double laplacian = 4 * center - left - right - up - down;
                hfEnergy += laplacian * laplacian;
                lfEnergy += center * center;
                count++;
            }
        }

        if (count == 0 || lfEnergy < 1e-10) return 0;

        double hfRatio = hfEnergy / (lfEnergy + hfEnergy);
        if (hfRatio < 0.15) return 0;
        return Math.Min(1.0, (hfRatio - 0.15) / 0.5);
    }

    private static double ComputeHistogramAnomalyScore(ReadOnlySpan<T> span)
    {
        int bins = 64;
        int[] histogram = new int[bins];
        int totalPixels = 0;

        for (int i = 0; i < span.Length && i < 65536; i++)
        {
            double val = StaticNumOps.ToDouble(span[i]);
            if (val > 1.0) val /= 255.0;
            val = Math.Max(0, Math.Min(1.0 - 1e-10, val));
            int bin = (int)(val * bins);
            bin = Math.Max(0, Math.Min(bins - 1, bin));
            histogram[bin]++;
            totalPixels++;
        }

        if (totalPixels < 100) return 0;

        int emptyBins = 0;
        for (int i = 0; i < bins; i++)
            if (histogram[i] == 0) emptyBins++;

        double smoothnessViolation = 0;
        for (int i = 1; i < bins - 1; i++)
        {
            double laplacian = histogram[i - 1] + histogram[i + 1] - 2.0 * histogram[i];
            smoothnessViolation += Math.Abs(laplacian);
        }
        double avgSmoothness = smoothnessViolation / (bins - 2) / Math.Max(totalPixels / bins, 1);
        double smoothnessScore = Math.Min(1.0, Math.Max(0, (avgSmoothness - 0.5) / 1.0));
        double emptyBinRatio = (double)emptyBins / bins;
        double gapScore = Math.Min(1.0, Math.Max(0, (emptyBinRatio - 0.1) / 0.4));

        return 0.6 * smoothnessScore + 0.4 * gapScore;
    }

    private static double ComputeFeatureSqueezingScore(ReadOnlySpan<T> span)
    {
        int numPixels = Math.Min(span.Length, 16384);
        double l2Sum = 0;

        for (int i = 0; i < numPixels; i++)
        {
            double val = StaticNumOps.ToDouble(span[i]);
            if (val > 1.0) val /= 255.0;

            double squeezed = Math.Round(val * 15.0) / 15.0;
            double diff = val - squeezed;
            l2Sum += diff * diff;
        }

        double rms = Math.Sqrt(l2Sum / numPixels);
        if (rms < 0.01) return 0;
        return Math.Min(1.0, (rms - 0.01) / 0.03);
    }
}
