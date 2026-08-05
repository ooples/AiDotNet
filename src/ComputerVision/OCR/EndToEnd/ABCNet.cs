using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.OCR.EndToEnd;

/// <summary>
/// ABCNet: single-shot scene text spotting that regresses a cubic Bezier boundary per text instance and
/// rectifies each instance with BezierAlign before recognizing it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Liu, Chen, Shen, He, Jin and Wang, "ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve
/// Network" (CVPR 2020 oral, pp. 9809-9818, arXiv:2002.10200).
/// </para>
/// <para>
/// The two contributions live in <see cref="CubicBezierCurve{T}"/> and <see cref="BezierAlign"/>, each
/// independently testable; this class is what composes them into a spotter. What makes it ABCNet rather
/// than a generic two-stage pipeline is that the SAME feature map feeds both branches and the curve
/// parameters are what connect them: the detection head regresses eight control points, and BezierAlign
/// reads the recognition branch's input through those very control points. A pipeline that detects a box,
/// crops the image, and hands the crop to a separate recognizer — which is what
/// <see cref="SceneTextReader{T}"/> does, honestly and usefully — shares no gradient between the two and
/// cannot represent a curved instance at all.
/// </para>
/// <para>
/// SINGLE SHOT, and that is the source of the paper's real-time claim: text/not-text scores and the 16
/// curve coordinates come from two 1x1 convolutions over one shared feature map, with no proposal stage
/// and no per-anchor refinement.
/// </para>
/// <para>
/// The recognition branch is deliberately lightweight — convolutions plus a per-column classifier over
/// the rectified strip, trained with CTC. The paper's argument is that BezierAlign has already done the
/// hard part: once the instance is straightened, an attention-based recognizer earns very little over a
/// cheap one, and its cost would remove the speed advantage.
/// </para>
/// <para><b>For Beginners:</b> Text in real photos is often curved — around a logo, along a sign. ABCNet
/// describes each piece of text with a smooth curve along its top and bottom edges, uses those curves to
/// straighten the text out, and then reads the straightened version. It finds and reads text in one pass
/// rather than two separate stages.</para>
/// </remarks>
/// <example>
/// <code>
/// var model = new ABCNet&lt;double&gt;(new ABCNetOptions&lt;double&gt; { InputHeight = 256, InputWidth = 256 });
/// foreach (var instance in model.Spot(image))
///     Console.WriteLine($"score {instance.Score}, {instance.CharacterIndices.Count} characters");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network",
    "https://arxiv.org/abs/2002.10200",
    Year = 2020,
    Authors = "Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin, Liangwei Wang")]
public class ABCNet<T> : NeuralNetworkBase<T>
{
    /// <summary>Coordinates the Bezier head regresses: 8 control points, (x, y) each.</summary>
    public const int BezierCoordinateCount = 16;

    private const int BackboneLayerCount = 3;
    private const int DetectionHeadLayerCount = 2;
    private const int RecognitionLayerCount = 3;

    /// <summary>Total layers this model routes, in order: backbone, then heads, then recognition.</summary>
    public const int ExpectedLayerCount =
        BackboneLayerCount + DetectionHeadLayerCount + RecognitionLayerCount;

    private readonly ABCNetOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    // The tensor engine comes from NeuralNetworkBase's own Engine member rather than a local one, so
    // this model records on the same engine as every other layer it drives. Declaring a second one here
    // shadowed the base's and would have let the two diverge after an engine switch.

    /// <summary>A spotted text instance: where it is, how confident, and what it says.</summary>
    /// <param name="Score">Text confidence at the detecting feature-map position.</param>
    /// <param name="ControlPoints">
    /// The 8 control points in IMAGE pixel coordinates — four along the top edge then four along the
    /// bottom. Image rather than feature-map coordinates because that is what a caller drawing the
    /// boundary needs; the conversion is by <see cref="ABCNetOptions{T}.FeatureStride"/>.
    /// </param>
    /// <param name="CharacterIndices">
    /// CTC-decoded character class indices, blanks and repeats already collapsed.
    /// </param>
    public readonly record struct TextInstance(
        double Score,
        IReadOnlyList<(double X, double Y)> ControlPoints,
        IReadOnlyList<int> CharacterIndices);

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates the model with the paper's defaults.</summary>
    public ABCNet()
        : this(new ABCNetOptions<T>())
    {
    }

    /// <summary>Creates the model.</summary>
    /// <param name="options">Configuration; defaults are the paper's where it states them.</param>
    /// <param name="architecture">
    /// Optional custom architecture. If it supplies layers there must be exactly
    /// <see cref="ExpectedLayerCount"/> of them, in this class's documented order — see
    /// <see cref="InitializeLayers"/>.
    /// </param>
    /// <param name="optimizer">Optional optimizer; Adam by default.</param>
    /// <param name="lossFunction">
    /// Optional loss for the inherited training surface. The paper's objective is a sum of a detection
    /// loss over the score map, an L1 loss on the Bezier coordinates, and a CTC loss on the recognition
    /// branch; a pointwise loss over the detection tensor is what the base training surface can express.
    /// </param>
    public ABCNet(
        ABCNetOptions<T> options,
        NeuralNetworkArchitecture<T>? architecture = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(ResolveArchitecture(options, architecture), lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    private static NeuralNetworkArchitecture<T> ResolveArchitecture(
        ABCNetOptions<T> options, NeuralNetworkArchitecture<T>? architecture)
    {
        Guard.NotNull(options);
        options.Validate();

        return architecture ?? new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: options.InputHeight,
            inputWidth: options.InputWidth,
            inputDepth: options.InputChannels,
            outputSize: 1 + BezierCoordinateCount);
    }

    /// <summary>Gets the detection feature-map height.</summary>
    public int FeatureHeight => _options.InputHeight / _options.FeatureStride;

    /// <summary>Gets the detection feature-map width.</summary>
    public int FeatureWidth => _options.InputWidth / _options.FeatureStride;

    /// <summary>
    /// Builds the layers, in the ONE order this class routes them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The order is part of the contract because the forward pass is BRANCHED and therefore cannot be a
    /// plain sequential walk: layers 0-2 are the shared backbone, 3 is the text-score head, 4 is the
    /// Bezier coordinate head, and 5-7 are the recognition branch. A custom layer list is accepted only
    /// at exactly this length, rather than being padded or truncated — silently misrouting a
    /// caller's layers would produce a model that trains and predicts while computing something else
    /// entirely.
    /// </para>
    /// <para>
    /// The Bezier head is LINEAR while the score head is sigmoid. That asymmetry matters: the head
    /// regresses signed OFFSETS from each feature position to each control point, so squashing it to
    /// [0, 1] would make every control point fall below and to the right of its own position and no
    /// curve could ever bend upward.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            if (Architecture.Layers.Count != ExpectedLayerCount)
            {
                throw new ArgumentException(
                    $"ABCNet routes a branched forward pass and needs exactly {ExpectedLayerCount} layers "
                    + $"in its documented order (backbone x{BackboneLayerCount}, score head, Bezier head, "
                    + $"recognition x{RecognitionLayerCount}); got {Architecture.Layers.Count}.");
            }

            Layers.AddRange(Architecture.Layers);
            return;
        }

        int c = _options.FeatureChannels;

        // Shared backbone, total stride 4.
        Layers.Add(new ConvolutionalLayer<T>(c / 4, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>));
        Layers.Add(new ConvolutionalLayer<T>(c / 2, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>));
        Layers.Add(new ConvolutionalLayer<T>(c, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>));

        // Detection heads, both 1x1 over the shared map.
        Layers.Add(new ConvolutionalLayer<T>(1, 1, 1, 0, new SigmoidActivation<T>() as IActivationFunction<T>));
        Layers.Add(new ConvolutionalLayer<T>(BezierCoordinateCount, 1, 1, 0));

        // Recognition branch, over the BezierAlign-rectified strip.
        Layers.Add(new ConvolutionalLayer<T>(c / 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>));
        Layers.Add(new ConvolutionalLayer<T>(c / 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>));
        Layers.Add(new DenseLayer<T>(_options.NumCharacterClasses));
    }

    /// <summary>Runs the shared backbone.</summary>
    private Tensor<T> Backbone(Tensor<T> image)
    {
        var f = image;
        for (int i = 0; i < BackboneLayerCount; i++) f = Layers[i].Forward(f);
        return f;
    }

    /// <summary>
    /// The detection output: the text score map stacked with the Bezier coordinate map,
    /// <c>[1 + 16, FeatureHeight, FeatureWidth]</c>.
    /// </summary>
    /// <remarks>
    /// Both heads are returned together because they are jointly what the detection loss is computed
    /// against, and because keeping them in one tensor is what lets the inherited training surface reach
    /// the whole detection branch in a single step.
    /// </remarks>
    protected Tensor<T> ComputeDetection(Tensor<T> input)
    {
        Guard.NotNull(input);

        var features = Backbone(input);
        var scores = Layers[BackboneLayerCount].Forward(features);
        var coords = Layers[BackboneLayerCount + 1].Forward(features);

        int channelAxis = scores.Rank == 4 ? 1 : 0;
        return Engine.Concat(new[] { scores, coords }, channelAxis);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input) => ComputeDetection(input);

    /// <summary>
    /// Detects text instances, decoding each feature position's regressed offsets into control points.
    /// </summary>
    /// <param name="image">Input image, <c>[C, H, W]</c> or <c>[1, C, H, W]</c>.</param>
    /// <returns>
    /// Instances above <see cref="ABCNetOptions{T}.ConfidenceThreshold"/>, highest scoring first, capped
    /// at <see cref="ABCNetOptions{T}.MaxInstances"/>. Character indices are empty — use
    /// <see cref="Spot"/> to recognize as well.
    /// </returns>
    /// <remarks>
    /// Offsets are decoded RELATIVE to the detecting position, which is what makes a single-shot head
    /// able to place a curve anywhere in the image: an absolute-coordinate head would have to learn the
    /// position of every instance from the feature values alone.
    /// </remarks>
    public IReadOnlyList<TextInstance> DetectInstances(Tensor<T> image)
    {
        Guard.NotNull(image);

        var detection = ComputeDetection(image);
        return DecodeDetections(detection);
    }

    private IReadOnlyList<TextInstance> DecodeDetections(Tensor<T> detection)
    {
        // Batch is 1 in both the [C,H,W] and [1,C,H,W] cases, so the channel planes start at element 0
        // either way and the rank does not change the indexing.
        int h = detection.Shape[detection.Rank - 2];
        int w = detection.Shape[detection.Rank - 1];
        int plane = h * w;

        var found = new List<TextInstance>();
        double stride = _options.FeatureStride;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double score = Convert.ToDouble(detection[(y * w) + x]);
                if (score < _options.ConfidenceThreshold) continue;

                var control = new (double X, double Y)[BezierAlign.ControlPointCount];
                for (int k = 0; k < BezierAlign.ControlPointCount; k++)
                {
                    // Channel 0 is the score, so coordinate channels start at 1.
                    double dx = Convert.ToDouble(detection[((1 + (2 * k)) * plane) + (y * w) + x]);
                    double dy = Convert.ToDouble(detection[((1 + (2 * k) + 1) * plane) + (y * w) + x]);
                    control[k] = ((x + dx) * stride, (y + dy) * stride);
                }

                found.Add(new TextInstance(score, control, Array.Empty<int>()));
            }
        }

        found.Sort(static (a, b) => b.Score.CompareTo(a.Score));
        if (found.Count > _options.MaxInstances) found.RemoveRange(_options.MaxInstances, found.Count - _options.MaxInstances);
        return found;
    }

    /// <summary>
    /// Detects and recognizes every text instance in one pass over the shared features.
    /// </summary>
    /// <param name="image">Input image, <c>[C, H, W]</c> or <c>[1, C, H, W]</c>.</param>
    /// <remarks>
    /// The backbone runs ONCE and both branches read its output, which is the whole point of the
    /// single-shot design. Re-running a backbone per instance is what the two-stage pipelines do and is
    /// where their cost goes.
    /// </remarks>
    public IReadOnlyList<TextInstance> Spot(Tensor<T> image)
    {
        Guard.NotNull(image);

        var features = Backbone(image);
        var scores = Layers[BackboneLayerCount].Forward(features);
        var coords = Layers[BackboneLayerCount + 1].Forward(features);
        int channelAxis = scores.Rank == 4 ? 1 : 0;
        var detected = DecodeDetections(Engine.Concat(new[] { scores, coords }, channelAxis));

        var spotted = new List<TextInstance>(detected.Count);
        foreach (var instance in detected)
        {
            // Back to FEATURE-MAP coordinates: BezierAlign samples the feature map, while the reported
            // control points are in image pixels.
            var featureSpace = new (double X, double Y)[BezierAlign.ControlPointCount];
            for (int k = 0; k < BezierAlign.ControlPointCount; k++)
            {
                featureSpace[k] = (instance.ControlPoints[k].X / _options.FeatureStride,
                                   instance.ControlPoints[k].Y / _options.FeatureStride);
            }

            var logits = RecognizeRectified(features, ControlPointTensor(featureSpace));
            spotted.Add(instance with { CharacterIndices = CtcGreedyDecode(logits) });
        }

        return spotted;
    }

    private static Tensor<T> ControlPointTensor(IReadOnlyList<(double X, double Y)> points)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(new[] { BezierAlign.ControlPointCount, 2 });
        for (int k = 0; k < BezierAlign.ControlPointCount; k++)
        {
            t[(k * 2) + 0] = numOps.FromDouble(points[k].X);
            t[(k * 2) + 1] = numOps.FromDouble(points[k].Y);
        }
        return t;
    }

    /// <summary>
    /// Rectifies one instance with BezierAlign and returns per-column character logits.
    /// </summary>
    /// <param name="features">The shared feature map.</param>
    /// <param name="controlPoints">
    /// <c>[8, 2]</c> control points in FEATURE-MAP coordinates, top curve first.
    /// </param>
    /// <returns>Logits shaped <c>[BezierSampleWidth, NumCharacterClasses]</c>.</returns>
    /// <remarks>
    /// One classifier per rectified COLUMN, which is what makes CTC applicable: the columns are the time
    /// steps. Collapsing the strip to a single vector instead would discard the left-to-right ordering
    /// that tells the characters apart.
    /// </remarks>
    public Tensor<T> RecognizeRectified(Tensor<T> features, Tensor<T> controlPoints)
    {
        Guard.NotNull(features);
        Guard.NotNull(controlPoints);

        var strip = BezierAlign.Sample(
            Engine, features, controlPoints, _options.BezierSampleHeight, _options.BezierSampleWidth);

        var f = Layers[BackboneLayerCount + DetectionHeadLayerCount].Forward(strip);
        f = Layers[BackboneLayerCount + DetectionHeadLayerCount + 1].Forward(f);

        // [C, h, w] -> [w, C, h] -> [w, C*h], so the dense head classifies each column independently
        // with the leading axis acting as the batch.
        int channels = f.Shape[f.Rank - 3];
        int height = f.Shape[f.Rank - 2];
        int width = f.Shape[f.Rank - 1];

        var columnsFirst = Engine.TensorPermute(f, new[] { 2, 0, 1 });
        var flattened = Engine.Reshape(columnsFirst, new[] { width, channels * height });

        return Layers[ExpectedLayerCount - 1].Forward(flattened);
    }

    /// <summary>
    /// Greedy CTC decode: argmax per column, then collapse repeats and drop blanks.
    /// </summary>
    /// <param name="logits">Per-column logits, <c>[columns, classes]</c>.</param>
    /// <remarks>
    /// <para>
    /// REPEATS ARE COLLAPSED BEFORE BLANKS ARE DROPPED, and the order is not interchangeable. CTC's
    /// blank is what separates a genuine double letter from one character spanning two columns: dropping
    /// blanks first would turn the sequence <c>l - blank - l</c> into <c>l l</c> and then into a single
    /// <c>l</c>, silently deleting a letter from every word containing a double.
    /// </para>
    /// <para>
    /// Class 0 is the blank, following the convention that the blank precedes the alphabet.
    /// </para>
    /// </remarks>
    public static IReadOnlyList<int> CtcGreedyDecode(Tensor<T> logits)
    {
        Guard.NotNull(logits);
        if (logits.Rank != 2)
            throw new ArgumentException($"Logits must be [columns, classes]; got rank {logits.Rank}.", nameof(logits));

        const int blank = 0;
        int columns = logits.Shape[0];
        int classes = logits.Shape[1];

        var decoded = new List<int>(columns);
        int previous = -1;

        for (int c = 0; c < columns; c++)
        {
            int best = 0;
            double bestValue = double.NegativeInfinity;
            for (int k = 0; k < classes; k++)
            {
                double v = Convert.ToDouble(logits[(c * classes) + k]);
                if (v > bestValue) { bestValue = v; best = k; }
            }

            if (best != previous && best != blank) decoded.Add(best);
            previous = best;
        }

        return decoded;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        Guard.NotNull(parameters);

        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            if (p.Length > 0 && offset + p.Length <= parameters.Length)
            {
                var slice = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += p.Length;
            }
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "ABCNet" },
            { "FeatureChannels", _options.FeatureChannels },
            { "FeatureStride", _options.FeatureStride },
            { "BezierSampleHeight", _options.BezierSampleHeight },
            { "BezierSampleWidth", _options.BezierSampleWidth },
            { "NumCharacterClasses", _options.NumCharacterClasses },
        },
        ModelData = this.Serialize(),
    };

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        Guard.NotNull(writer);
        writer.Write(_options.InputHeight);
        writer.Write(_options.InputWidth);
        writer.Write(_options.InputChannels);
        writer.Write(_options.FeatureChannels);
        writer.Write(_options.FeatureStride);
        writer.Write(_options.BezierSampleHeight);
        writer.Write(_options.BezierSampleWidth);
        writer.Write(_options.NumCharacterClasses);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        Guard.NotNull(reader);
        for (int i = 0; i < 8; i++) _ = reader.ReadInt32();
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new ABCNet<T>(_options, Architecture, _optimizer, _lossFunction);
}
