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
using AiDotNet.NeuralNetworks.Graph;
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
/// var image = Tensor&lt;double&gt;.CreateRandom(1, 3, 32, 32);
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
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class ABCNet<T> : NeuralNetworkBase<T>, ICompositeLoss<T>
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

    // The declared wiring. Two graphs because the model has two entry points; see BuildGraph.
    private LayerGraph<T>? _detectionGraph;
    private LayerGraph<T>? _recognitionGraph;

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
    /// <param name="optimizer">
    /// Optional optimizer. The default is the paper's SGD with momentum, at
    /// <see cref="ABCNetOptions{T}.LearningRate"/> and <see cref="ABCNetOptions{T}.Momentum"/>.
    /// </param>
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
        // THE OPTIMIZER NOW MATCHES THE RATE. ABCNetOptions.LearningRate defaults to the paper's
        // 0.01, which is an SGD-WITH-MOMENTUM rate; the default optimizer here was Adam, whose step
        // is roughly the learning rate itself, so 0.01 was about two orders of magnitude too large
        // for it. The note previously left on this line said exactly that -- reproducing the paper
        // means injecting SGD, and staying on Adam means dropping to about 1e-4 -- but the code kept
        // the one combination it warned against, and the oversized steps drove the shared backbone's
        // ReLUs negative until every input produced the same constant detection map.
        //
        // Take the paper's side of that choice rather than retuning its published rate: SGD with
        // momentum, which this optimizer applies with InitialMomentum defaulting to the paper's 0.9.
        // Injecting an optimizer still overrides all of it.
        _optimizer = optimizer ?? new StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                InitialMomentum = _options.Momentum,
            });

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

        BuildGraph();
        ValidateLayerContracts();
    }

    /// <summary>
    /// Checks the declared layer layouts and throws when the SEQUENTIAL parts of the chain disagree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ADJACENT IN <c>Layers</c> IS NOT THE SAME AS ADJACENT IN DATAFLOW, and this model breaks that
    /// assumption in both available ways — which is precisely why it was worth piloting the contract here.
    /// </para>
    /// <para>
    /// (1) BRANCHING: layers 3 and 4 are the two detection heads. Both read layer 2's output and neither
    /// feeds the other, so treating 2-&gt;3-&gt;4-&gt;5 as a pipeline would report mismatches that are not real.
    /// </para>
    /// <para>
    /// (2) EXPLICIT RESHAPES BETWEEN LAYERS: the recognition branch permutes and flattens
    /// <c>[C, h, w]</c> into <c>[w, C*h]</c> between the last convolution and the dense head — see
    /// <see cref="RecognizeRectified"/>. That transformation is real dataflow but it is NOT a layer, so a
    /// naive adjacency check reports conv-&gt;dense as incompatible. It genuinely is incompatible AS A DIRECT
    /// HAND-OFF; the reshape is what makes it correct.
    /// </para>
    /// <para>
    /// So only the runs that are actually contiguous get validated: the backbone (0-2) and the two
    /// recognition convolutions (5-6). The dense head is entered through a reshape and therefore has no
    /// layout-adjacent predecessor to check against. Validating a false pair would be worse than
    /// validating nothing — it would train readers to ignore the failure.
    /// </para>
    /// </remarks>
    private void ValidateLayerContracts()
    {
        if (_detectionGraph is null || _recognitionGraph is null) return;

        // The runs come from the GRAPH now, not from hand-written index ranges. That is the whole point of
        // migrating: ContiguousRuns breaks a run at a fan-out and at an edge transform, which is exactly
        // where two layers stop meeting directly — so the false positive the pilot hit (conv -> dense
        // across a permute) cannot be constructed any more, and nobody has to remember which indices are
        // safe to compare.
        Validate(_detectionGraph, "detection");
        Validate(_recognitionGraph, "recognition");

        void Validate(LayerGraph<T> graph, string what)
        {
            int run = 0;
            foreach (var contiguous in graph.ContiguousRuns())
                LayerContractValidator.ValidateOrThrow(contiguous, $"{nameof(ABCNet<T>)} {what} run {run++}");
        }
    }

    /// <summary>
    /// Declares the model's real wiring: a shared backbone feeding two detection heads, and a separate
    /// per-instance recognition chain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// TWO GRAPHS, because ABCNet genuinely has two entry points. The detection graph runs once over the
    /// image. The recognition graph runs once PER DETECTED INSTANCE over a BezierAlign-rectified strip, so
    /// it is not reachable from the image input and cannot be a sub-path of the first.
    /// </para>
    /// <para>
    /// The detection graph is where the FAN-OUT lives: both heads read the backbone's output and neither
    /// feeds the other. That is precisely the structure the inherited sequential training path cannot
    /// represent — it would run head A into head B — and precisely the structure whose backward needs the
    /// gradient contributions of both heads SUMMED at the backbone.
    /// </para>
    /// <para>
    /// The recognition graph carries the permute-and-flatten as an EDGE rather than a layer, so the flat
    /// <c>Layers</c> projection still contains exactly the eight parameter-owning layers that parameter
    /// counting, cloning and serialization expect.
    /// </para>
    /// </remarks>
    /// <inheritdoc />
    /// <remarks>
    /// Both graphs, because this model has two entry points. Declaring them is what stops the base's
    /// linear reading of <c>Layers</c> from pairing the score head with the Bezier head, and the last
    /// recognition convolution with the dense head it only reaches through a permute+reshape - neither
    /// of which is a hand-off that exists.
    /// </remarks>
    protected override IEnumerable<LayerGraph<T>>? DeclaredLayerGraphs
    {
        get
        {
            if (_detectionGraph is null || _recognitionGraph is null) return null;
            return new[] { _detectionGraph, _recognitionGraph };
        }
    }
    private void BuildGraph()
    {
        if (Layers.Count != ExpectedLayerCount) return;   // a custom list; its author owns the wiring

        var detection = new LayerGraphBuilder<T>();
        int stem = detection.Add(Layers[0]);
        int mid = detection.Add(Layers[1], stem);
        int trunk = detection.Add(Layers[2], mid);
        int scores = detection.Add(Layers[BackboneLayerCount], trunk);
        int coords = detection.Add(Layers[BackboneLayerCount + 1], trunk);
        int stacked = detection.AddJoin(
            layer: null,
            inputs: new[] { scores, coords },
            combine: parts =>
            {
                int channelAxis = parts[0].Rank == 4 ? 1 : 0;
                return Engine.Concat(new[] { parts[0], parts[1] }, channelAxis);
            },
            description: "concat score map with the 16 Bezier coordinate channels");
        _detectionGraph = detection.Output(stacked).Build();

        int recogStart = BackboneLayerCount + DetectionHeadLayerCount;
        var recognition = new LayerGraphBuilder<T>();
        int r0 = recognition.Add(Layers[recogStart]);
        int r1 = recognition.Add(Layers[recogStart + 1], r0);
        int r2 = recognition.AddVia(
            Layers[ExpectedLayerCount - 1], r1,
            transform: f => ColumnsAsBatch(f),
            description: "permute [C,h,w] -> [w,C,h] then flatten to [w, C*h] so each column is classified");
        _recognitionGraph = recognition.Output(r2).Build();
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

        // Executed through the declared graph, so the branch structure lives in exactly one place. Written
        // out by hand here as well, it could drift from what BuildGraph says and the layout validation
        // would then be checking a shape the forward pass never produces.
        if (_detectionGraph is not null) return _detectionGraph.Forward(input);

        var features = Backbone(input);
        var scores = Layers[BackboneLayerCount].Forward(features);
        var coords = Layers[BackboneLayerCount + 1].Forward(features);

        int channelAxis = scores.Rank == 4 ? 1 : 0;
        return Engine.Concat(new[] { scores, coords }, channelAxis);
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// The inherited resolver walks <c>Layers</c> in list order and hands each layer its predecessor's
    /// output shape. For this model that is the wrong dataflow twice over: the Bezier head would be sized
    /// against the SCORE head's single output channel instead of the trunk's <c>FeatureChannels</c>, and
    /// the recognition branch would be sized against the Bezier head instead of the rectified strip. The
    /// first is not a subtle mis-size — it fails outright, because the head is a convolution whose input
    /// depth is then pinned to 1 while every real forward feeds it the full trunk.
    /// </para>
    /// <para>
    /// Resolving through the declared graphs uses the same wiring the forward pass uses, so the two cannot
    /// drift. The recognition graph starts from the RECTIFIED STRIP rather than the image: BezierAlign
    /// samples a fixed <c>BezierSampleHeight</c>×<c>BezierSampleWidth</c> grid off the trunk, so its input
    /// shape is known without running detection first.
    /// </para>
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved) return;

        if (_detectionGraph is null || _recognitionGraph is null)
        {
            // A caller-supplied layer list, whose author owns the wiring: no graph was built, so there is
            // nothing better on offer than the inherited sequential walk.
            base.ResolveLazyLayerShapes();
            return;
        }

        var inputShape = TryGetArchitectureInputShape();
        if (inputShape is null)
        {
            base.ResolveLazyLayerShapes();
            return;
        }

        _detectionGraph.ResolveShapes(inputShape);

        // The strip BezierAlign produces off the trunk: trunk channels over the configured sample grid.
        // Batch-free by construction — Sample rectifies one instance at a time.
        _recognitionGraph.ResolveShapes(
            new[] { _options.FeatureChannels, _options.BezierSampleHeight, _options.BezierSampleWidth });

        MarkLayerShapesResolved();
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// GRAPH-DRIVEN, and it has to be. The inherited training forward walks <c>Layers</c> in order, which
    /// for this model would feed the score head into the coordinate head and the coordinate head into the
    /// recognition convolutions — a graph this model does not have. That mistake raises nothing: it trains,
    /// the loss falls, and every generic invariant passes while the computation is wrong. Routing the
    /// training forward through the same graph the prediction path uses is what makes the two agree by
    /// construction rather than by review.
    /// </para>
    /// <para>
    /// The tape sees a genuine fan-out here — the trunk feeds both heads — so the backward must SUM the two
    /// heads' contributions into the shared trunk rather than take either one. That accumulation is the
    /// tape's, and it is covered by dedicated finite-difference checks
    /// (<c>FanOutGradientAccumulationTests</c> in the Tensors package) precisely because its failure mode is
    /// silent: a backward that overwrote instead of accumulating would still produce finite, plausible
    /// gradients.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // Required of every override that does not call base: stochastic layers derive their masks from
        // RandomSeed, and without this wiring they would train on an unseeded stream.
        EnsureLayerRandomSeedsWired();
        return ComputeDetection(input);
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// THE PAPER'S OBJECTIVE IS A SUM, and declaring it is what finally supervises the recognizer.
    /// Liu et al. train detection and recognition JOINTLY - a score-map term, an L1 term on the Bezier
    /// coordinates, and a CTC term over the recognition head. Returning only the detection tensor from
    /// the training forward meant layers 5-7 received no gradient whatsoever: the model trained its
    /// detector and left its recognizer at initialisation, while the loss fell and every generic
    /// invariant passed.
    /// </para>
    /// <para>
    /// Two terms rather than three because the detection tensor already carries the score map AND the
    /// sixteen Bezier channels concatenated - one loss over it covers both of the paper's detection
    /// terms. Splitting them would need separate weights the paper does not give.
    /// </para>
    /// <para>
    /// Equal weights, deliberately: the paper states its objective as an unweighted sum. A weight is
    /// visible here precisely so it can be checked against the paper rather than buried in a loss.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputSpec<T>> DeclaredOutputs => new[]
    {
        new OutputSpec<T>("detection", _lossFunction, 1.0),
        new OutputSpec<T>("recognition", _lossFunction, 1.0),
    };

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// RECOGNITION IS COMPUTED THROUGH THE PREDICTED CURVE, which is the coupling that makes this one
    /// model rather than a detector bolted to a recognizer. BezierAlign samples the shared trunk
    /// features THROUGH the control points the Bezier head just regressed, so gradient from the
    /// recognition term flows back into the coordinate head. Sampling through detached coordinates
    /// would reproduce the arithmetic and lose the paper's actual argument.
    /// </para>
    /// <para>
    /// The control points come from the feature-space decode of the CURRENT prediction, so the two
    /// heads are genuinely wired together during training and not merely summed.
    /// </para>
    /// </remarks>
    public IReadOnlyList<Tensor<T>> ComputeOutputs(Tensor<T> input)
    {
        Guard.NotNull(input);
        EnsureLayerRandomSeedsWired();

        var detection = ComputeDetection(input);
        var features = Backbone(input);
        var controlPoints = TrainingControlPoints(detection);
        var recognition = RecognizeRectified(features, controlPoints);

        return new[] { detection, recognition };
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

    /// <summary>
    /// Control points for the TRAINING path, sliced from the detection tensor with engine ops so the
    /// tape survives.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SEPARATE FROM THE INFERENCE DECODE, and it has to be. <see cref="DecodeDetections"/> reads
    /// values out into CLR doubles to rank and threshold instances - correct for inference, fatal here:
    /// the moment a value leaves the tensor the tape is severed, and the recognition term would train
    /// nothing in the coordinate head. Everything below stays in engine ops for exactly that reason.
    /// </para>
    /// <para>
    /// A FIXED feature position, not the highest-scoring one. Selecting by argmax is not
    /// differentiable, and a straight-through approximation would put a fiction in the gradient path.
    /// A fixed position still routes the recognition loss through coordinates the Bezier head actually
    /// regressed, which is the coupling the paper relies on; which position it is does not change that.
    /// </para>
    /// <para>
    /// The head regresses OFFSETS from the sampling position, so the position is added back to reach
    /// feature-map coordinates - the same decode the inference path performs, done tensor-side.
    /// </para>
    /// </remarks>
    private Tensor<T> TrainingControlPoints(Tensor<T> detection)
    {
        int rank = detection.Rank;
        int channelAxis = rank == 4 ? 1 : 0;
        int heightAxis = rank - 2;
        int widthAxis = rank - 1;

        // Channels 1..16 are the Bezier coordinates; channel 0 is the score map.
        var coords = Engine.TensorNarrow(detection, channelAxis, 1, BezierCoordinateCount);

        // One spatial position, kept near the middle so the decoded curve lands inside the feature map
        // rather than hard against a corner.
        int py = detection.Shape[heightAxis] / 2;
        int px = detection.Shape[widthAxis] / 2;
        coords = Engine.TensorNarrow(coords, heightAxis, py, 1);
        coords = Engine.TensorNarrow(coords, widthAxis, px, 1);

        var offsets = Engine.Reshape(coords, new[] { BezierAlign.ControlPointCount, 2 });

        // Offsets are relative to the sampling position, so add it back. A constant addend, so it
        // contributes no gradient of its own while leaving the offsets fully differentiable.
        var origin = new Tensor<T>(new[] { BezierAlign.ControlPointCount, 2 });
        for (int k = 0; k < BezierAlign.ControlPointCount; k++)
        {
            origin[(k * 2) + 0] = NumOps.FromDouble(px);
            origin[(k * 2) + 1] = NumOps.FromDouble(py);
        }

        return Engine.TensorAdd(offsets, origin);
    }

    /// <summary>
    /// Turns a rectified strip into one row per COLUMN, so the dense head classifies each column.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RANK-AWARE, because the two callers do not agree on rank. Inference rectifies one instance at a
    /// time and hands over <c>[C, h, w]</c>; training runs through the base, which auto-promotes an
    /// unbatched input to <c>[1, C, H, W]</c>, so the strip arrives as <c>[B, C, h, w]</c>. The previous
    /// version read its axes rank-agnostically (<c>Rank - 3</c>) but then permuted with a hardcoded
    /// <c>[2, 0, 1]</c> - correct for rank 3 and a throw for rank 4, which is exactly what the joint
    /// objective hit the moment it ran the recognition branch during training.
    /// </para>
    /// <para>
    /// Columns become the leading axis either way, since CTC treats them as the time steps: collapsing
    /// the strip to one vector would discard the left-to-right ordering that distinguishes characters.
    /// </para>
    /// </remarks>
    private Tensor<T> ColumnsAsBatch(Tensor<T> strip)
    {
        int channels = strip.Shape[strip.Rank - 3];
        int height = strip.Shape[strip.Rank - 2];
        int width = strip.Shape[strip.Rank - 1];

        if (strip.Rank == 3)
        {
            // [C, h, w] -> [w, C, h] -> [w, C*h]
            var columnsFirst = Engine.TensorPermute(strip, new[] { 2, 0, 1 });
            return Engine.Reshape(columnsFirst, new[] { width, channels * height });
        }

        if (strip.Rank == 4)
        {
            // [B, C, h, w] -> [B, w, C, h] -> [B*w, C*h]. Batch and column fold into one leading axis
            // because the classifier is per-column and indifferent to which image a column came from.
            int batch = strip.Shape[0];
            var columnsFirst = Engine.TensorPermute(strip, new[] { 0, 3, 1, 2 });
            return Engine.Reshape(columnsFirst, new[] { batch * width, channels * height });
        }

        throw new ArgumentException(
            $"The rectified strip must be [C, h, w] or [B, C, h, w]; got rank {strip.Rank}.",
            nameof(strip));
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

        // Through the declared graph, whose edge transform IS the permute-and-flatten below. Keeping the
        // sequence in one place is what stops the executed path and the validated path from disagreeing.
        if (_recognitionGraph is not null) return _recognitionGraph.Forward(strip);

        var f = Layers[BackboneLayerCount + DetectionHeadLayerCount].Forward(strip);
        f = Layers[BackboneLayerCount + DetectionHeadLayerCount + 1].Forward(f);

        var flattened = ColumnsAsBatch(f);

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

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
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


    /// <inheritdoc />

}
