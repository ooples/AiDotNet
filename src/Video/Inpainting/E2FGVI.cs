using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Inpainting;

/// <summary>
/// E2FGVI - End-to-End Framework for Flow-Guided Video Inpainting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> E2FGVI removes unwanted objects from videos and fills in the gaps
/// with realistic content. It uses optical flow (motion information) to propagate known
/// content into missing regions across frames.
///
/// Use cases:
/// - Remove watermarks or logos from videos
/// - Remove unwanted people or objects
/// - Repair damaged or corrupted video frames
/// - Video restoration and cleanup
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - End-to-end trainable flow-guided inpainting
/// - Bidirectional flow propagation
/// - Transformer-based content hallucination
/// - Temporal consistency enforcement
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an E2FGVI model for flow-guided video inpainting
/// var e2fgvi = new E2FGVI&lt;double&gt;();
///
/// // Or configure with custom feature dimensions
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new E2FGVI&lt;double&gt;(architecture, options: new E2FGVIOptions { NumFeatures = 128 });
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Towards An End-to-End Framework for Flow-Guided Video Inpainting",
    "https://arxiv.org/abs/2204.02663",
    Year = 2022,
    Authors = "Zhen Li, Cheng-Ze Lu, Jianhua Qin, Chun-Le Guo, Ming-Ming Cheng")]
public partial class E2FGVI<T> : VideoInpaintingBase<T>
{
    private readonly E2FGVIOptions _options;

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numFeatures;

    // Flow estimation network
    private readonly List<ConvolutionalLayer<T>> _flowNet;
    private readonly ConvolutionalLayer<T> _flowHead;

    // Content encoder
    private readonly List<ConvolutionalLayer<T>> _encoder;

    // Transformer for content hallucination
    private readonly List<ConvolutionalLayer<T>> _transformer;

    // Flow-guided propagation
    private readonly List<ConvolutionalLayer<T>> _propagation;

    // Decoder for final output
    private readonly List<ConvolutionalLayer<T>> _decoder;
    private readonly ConvolutionalLayer<T> _outputHead;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the input frame height.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input frame width.
    /// </summary>
    internal int InputWidth => _width;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public E2FGVI()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 2))
    {
    }

    public E2FGVI(
        NeuralNetworkArchitecture<T> architecture,
        E2FGVIOptions? options = null)
        : base(architecture: architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new E2FGVIOptions();
        _options.Validate();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 432;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 240;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = _options.NumFeatures;

        _flowNet = [];
        _encoder = [];
        _transformer = [];
        _propagation = [];
        _decoder = [];

        int numTransformerBlocks = 8;

        // Check for user-provided custom layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateE2FGVILayers(
                channels: _channels, height: _height, width: _width,
                numFeatures: _numFeatures, numTransformerBlocks: numTransformerBlocks).ToList();
            Layers.AddRange(layers);
        }

        // Distribute layers to sub-lists for forward pass
        int idx = 0;
        // Flow network (3 layers)
        for (int i = 0; i < 3; i++)
            _flowNet.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Flow head
        _flowHead = (ConvolutionalLayer<T>)Layers[idx++];
        // Content encoder (3 layers)
        for (int i = 0; i < 3; i++)
            _encoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Transformer layers
        for (int i = 0; i < numTransformerBlocks; i++)
            _transformer.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Propagation (2 layers)
        for (int i = 0; i < 2; i++)
            _propagation.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Decoder (2 layers)
        for (int i = 0; i < 2; i++)
            _decoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Output head
        _outputHead = (ConvolutionalLayer<T>)Layers[idx++];
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Inpaints a video sequence by filling in masked regions.
    /// </summary>
    /// <param name="frames">Input video frames.</param>
    /// <param name="masks">Binary masks indicating regions to fill (1 = fill, 0 = keep).</param>
    /// <returns>Inpainted video frames.</returns>
    /// <exception cref="ArgumentNullException">Thrown when frames or masks is null.</exception>
    /// <exception cref="ArgumentException">Thrown when frames and masks have different counts or are empty.</exception>
    public List<Tensor<T>> Inpaint(List<Tensor<T>> frames, List<Tensor<T>> masks)
    {
        if (frames == null)
            throw new ArgumentNullException(nameof(frames), "Frames list cannot be null.");
        if (masks == null)
            throw new ArgumentNullException(nameof(masks), "Masks list cannot be null.");
        if (frames.Count == 0)
            throw new ArgumentException("Frames list cannot be empty.", nameof(frames));
        if (masks.Count == 0)
            throw new ArgumentException("Masks list cannot be empty.", nameof(masks));
        if (frames.Count != masks.Count)
            throw new ArgumentException($"Frames count ({frames.Count}) must match masks count ({masks.Count}).", nameof(masks));

        var results = new List<Tensor<T>>();

        // First pass: estimate flows between all frames
        var flows = EstimateAllFlows(frames);

        // Second pass: propagate content and hallucinate
        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            var mask = masks[i];

            bool hasBatch = frame.Rank == 4;
            if (!hasBatch)
            {
                frame = AddBatchDimension(frame);
                mask = AddBatchDimension(mask);
            }

            // Get neighboring frame information
            var neighborInfo = GatherNeighborInfo(frames, masks, flows, i);

            // Inpaint single frame
            var inpainted = InpaintFrame(frame, mask, neighborInfo);

            if (!hasBatch) inpainted = RemoveBatchDimension(inpainted);
            results.Add(inpainted);
        }

        return results;
    }

    /// <summary>
    /// Removes an object from video based on mask sequence.
    /// </summary>
    public List<Tensor<T>> RemoveObject(List<Tensor<T>> frames, List<Tensor<T>> objectMasks)
    {
        return Inpaint(frames, objectMasks);
    }

    /// <summary>
    /// Repairs corrupted regions in video frames.
    /// </summary>
    public List<Tensor<T>> RepairVideo(List<Tensor<T>> frames, List<Tensor<T>> corruptionMasks)
    {
        return Inpaint(frames, corruptionMasks);
    }

    /// <summary>
    /// Predicts the inpainted output for a single masked frame.
    /// </summary>
    /// <param name="input">Input tensor containing frame and mask concatenated [B, C+1, H, W].</param>
    /// <returns>Inpainted frame.</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Ensure 4D input
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        // Public prediction accepts raw RGB frames, while the encoder is trained on
        // RGB concatenated with a one-channel hole mask.  Training already supplies
        // the concatenated tensor; add the deterministic inference mask only for the
        // raw-RGB form so both paths resolve to the same four-channel contract.
        if (input.Shape[1] == _channels)
        {
            var mask = CreateDefaultInpaintingMask(input.Shape[0], input.Shape[2], input.Shape[3]);
            input = ConcatenateChannels(input, mask);
        }

        // Encode input (frame + mask)
        var features = input;
        foreach (var layer in _encoder)
        {
            features = layer.Forward(features);
            features = ApplyReLU(features);
        }

        // Self-propagation (no neighbors)
        foreach (var layer in _propagation)
        {
            // Duplicate features for self-propagation
            var selfProp = ConcatenateChannels(features, features);
            selfProp = layer.Forward(selfProp);
            features = ApplyReLU(selfProp);
        }

        // Transform for content hallucination
        var transformed = features;
        foreach (var layer in _transformer)
        {
            var residual = transformed;
            transformed = layer.Forward(transformed);
            transformed = ApplyReLU(transformed);
            transformed = AddTensors(transformed, residual);
        }

        // Decode
        var decoded = transformed;
        foreach (var layer in _decoder)
        {
            decoded = Upsample2x(decoded);
            decoded = layer.Forward(decoded);
            decoded = ApplyReLU(decoded);
        }

        while (decoded.Shape[2] < _height || decoded.Shape[3] < _width)
            decoded = Upsample2x(decoded);

        var output = _outputHead.Forward(decoded);
        output = ApplySigmoid(output);
        output = BlendKnownPixels(input, output);

        if (!hasBatch)
        {
            output = RemoveBatchDimension(output);
        }

        return output;
    }

    // Inpainting networks predict the filled pixels, but known RGB pixels must pass
    // through unchanged.  Besides matching the E2FGVI reconstruction contract, this
    // makes the mask an observable conditioning signal even for an untrained smoke
    // fixture whose randomly initialized head may be nearly constant.
    private Tensor<T> BlendKnownPixels(Tensor<T> input, Tensor<T> prediction)
    {
        int batch = input.Shape[0], channels = Math.Min(_channels, prediction.Shape[1]);
        int height = prediction.Shape[2], width = prediction.Shape[3];
        int[] shape = [batch, channels, height, width];

        // The first `channels` input channels are the frame, channel `_channels` is the fill mask
        var frame = Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(input, 1, 0, channels), 2, 0, height), 3, 0, width);
        var mask = Engine.TensorBroadcastTo(
            Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(input, 1, _channels, 1), 2, 0, height), 3, 0, width),
            shape);
        var inverse = Engine.TensorSubtract(Engine.TensorOnesLike(mask), mask);
        return Engine.TensorAdd(Engine.TensorMultiply(mask, prediction), Engine.TensorMultiply(inverse, frame));
    }

    /// <summary>Channel 0 of a mask, cropped to <paramref name="height"/> x <paramref name="width"/> and broadcast
    /// over <paramref name="channels"/>.</summary>
    private Tensor<T> MaskPlane(Tensor<T> mask, int batch, int channels, int height, int width)
        => Engine.TensorBroadcastTo(
            Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(mask, 0, 0, batch), 1, 0, 1), 2, 0, height), 3, 0, width),
            [batch, channels, height, width]);

    /// <summary>Runs the mask-conditioned E2FGVI graph on the autodiff tape.</summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var mask = CreateTrainingMask(input.Shape[0], input.Shape[2], input.Shape[3]);
        var combined = ConcatenateChannels(input, mask);
        var output = PredictCore(combined);
        return hasBatch ? output : RemoveBatchDimension(output);
    }

    /// <summary>
    /// Creates a mask based on differences between input and expected output.
    /// </summary>
    private Tensor<T> CreateMaskFromDifference(Tensor<T> input, Tensor<T> expected)
    {
        int batchSize = input.Shape[0];
        int height = input.Shape[2];
        int width = input.Shape[3];

        // 1 where any channel differs from the expected output by more than 0.01
        int channels = input.Shape[1];
        var expectedRegion = Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(
            Engine.TensorNarrow(expected, 0, 0, batchSize), 1, 0, channels), 2, 0, height), 3, 0, width);
        var largest = Engine.ReduceMax(Engine.TensorAbs(Engine.TensorSubtract(input, expectedRegion)), [1], true);
        return Engine.TensorGreaterThan(largest, NumOps.FromDouble(0.01));
    }

    #endregion

    #region Private Methods

    private List<(Tensor<T> Forward, Tensor<T> Backward)> EstimateAllFlows(List<Tensor<T>> frames)
    {
        var flows = new List<(Tensor<T>, Tensor<T>)>();

        for (int i = 0; i < frames.Count - 1; i++)
        {
            var frame1 = frames[i];
            var frame2 = frames[i + 1];

            if (frame1.Rank == 3) frame1 = AddBatchDimension(frame1);
            if (frame2.Rank == 3) frame2 = AddBatchDimension(frame2);

            var (forward, backward) = EstimateBidirectionalFlow(frame1, frame2);
            flows.Add((forward, backward));
        }

        return flows;
    }

    private (Tensor<T> Forward, Tensor<T> Backward) EstimateBidirectionalFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        // Concatenate frames
        var concat = ConcatenateChannels(frame1, frame2);

        // Process through flow network
        var features = concat;
        foreach (var layer in _flowNet)
        {
            features = layer.Forward(features);
            features = ApplyReLU(features);
        }

        var flowOutput = _flowHead.Forward(features);

        // Split into forward and backward flows
        int batchSize = flowOutput.Shape[0];
        int height = flowOutput.Shape[2];
        int width = flowOutput.Shape[3];

        var forward = Engine.TensorNarrow(flowOutput, 1, 0, 2);
        var backward = Engine.TensorNarrow(flowOutput, 1, 2, 2);

        // Upsample to full resolution
        forward = UpsampleFlow(forward, _height, _width);
        backward = UpsampleFlow(backward, _height, _width);

        return (forward, backward);
    }

    private List<(Tensor<T> Frame, Tensor<T> Mask, Tensor<T> Flow)> GatherNeighborInfo(
        List<Tensor<T>> frames, List<Tensor<T>> masks,
        List<(Tensor<T> Forward, Tensor<T> Backward)> flows, int currentIdx)
    {
        var neighbors = new List<(Tensor<T>, Tensor<T>, Tensor<T>)>();

        // Get previous frames
        for (int offset = 1; offset <= 2; offset++)
        {
            int idx = currentIdx - offset;
            if (idx >= 0 && idx < flows.Count)
            {
                var frame = frames[idx];
                var mask = masks[idx];
                if (frame.Rank == 3) frame = AddBatchDimension(frame);
                if (mask.Rank == 3) mask = AddBatchDimension(mask);
                neighbors.Add((frame, mask, flows[idx].Forward));
            }
        }

        // Get next frames
        for (int offset = 1; offset <= 2; offset++)
        {
            int idx = currentIdx + offset - 1;
            if (idx < flows.Count)
            {
                var frame = frames[currentIdx + offset];
                var mask = masks[currentIdx + offset];
                if (frame.Rank == 3) frame = AddBatchDimension(frame);
                if (mask.Rank == 3) mask = AddBatchDimension(mask);
                neighbors.Add((frame, mask, flows[idx].Backward));
            }
        }

        return neighbors;
    }

    private Tensor<T> InpaintFrame(Tensor<T> frame, Tensor<T> mask,
        List<(Tensor<T> Frame, Tensor<T> Mask, Tensor<T> Flow)> neighbors)
    {
        // Encode current frame with mask
        var maskedFrame = ApplyMask(frame, mask);
        var concat = ConcatenateChannels(maskedFrame, mask);

        var features = concat;
        foreach (var layer in _encoder)
        {
            features = layer.Forward(features);
            features = ApplyReLU(features);
        }

        // Propagate from neighbors
        var propagated = PropagateFromNeighbors(features, neighbors);

        // Fuse propagated and current features
        var fused = ConcatenateChannels(features, propagated);
        foreach (var layer in _propagation)
        {
            fused = layer.Forward(fused);
            fused = ApplyReLU(fused);
        }

        // Transform for content hallucination
        var transformed = fused;
        foreach (var layer in _transformer)
        {
            var residual = transformed;
            transformed = layer.Forward(transformed);
            transformed = ApplyReLU(transformed);
            transformed = AddTensors(transformed, residual);
        }

        // Decode
        var decoded = transformed;
        foreach (var layer in _decoder)
        {
            decoded = Upsample2x(decoded);
            decoded = layer.Forward(decoded);
            decoded = ApplyReLU(decoded);
        }

        while (decoded.Shape[2] < _height || decoded.Shape[3] < _width)
            decoded = Upsample2x(decoded);

        var output = _outputHead.Forward(decoded);
        output = ApplySigmoid(output);

        // Blend with original frame
        return BlendWithOriginal(frame, output, mask);
    }

    private Tensor<T> PropagateFromNeighbors(Tensor<T> currentFeatures,
        List<(Tensor<T> Frame, Tensor<T> Mask, Tensor<T> Flow)> neighbors)
    {
        if (neighbors.Count == 0)
            return currentFeatures;

        var accumulated = ZeroTensor(currentFeatures._shape);

        foreach (var (neighborFrame, neighborMask, flow) in neighbors)
        {
            // Encode neighbor
            var maskedNeighbor = ApplyMask(neighborFrame, neighborMask);
            var concat = ConcatenateChannels(maskedNeighbor, neighborMask);

            var neighborFeatures = concat;
            foreach (var layer in _encoder)
            {
                neighborFeatures = layer.Forward(neighborFeatures);
                neighborFeatures = ApplyReLU(neighborFeatures);
            }

            // Warp features using flow
            var warpedFeatures = WarpFeatures(neighborFeatures, flow);
            accumulated = AddTensors(accumulated, warpedFeatures);
        }

        return ScaleTensor(accumulated, 1.0 / neighbors.Count);
    }

    private Tensor<T> ApplyMask(Tensor<T> frame, Tensor<T> mask)
    {
        int batchSize = frame.Shape[0];
        int channels = frame.Shape[1];
        int height = frame.Shape[2];
        int width = frame.Shape[3];

        // Invert: 0 = fill
        var fill = MaskPlane(mask, batchSize, channels, height, width);
        return Engine.TensorMultiply(frame, Engine.TensorSubtract(Engine.TensorOnesLike(fill), fill));
    }

    private Tensor<T> BlendWithOriginal(Tensor<T> original, Tensor<T> inpainted, Tensor<T> mask)
    {
        int batchSize = original.Shape[0];
        int channels = original.Shape[1];
        int height = original.Shape[2];
        int width = original.Shape[3];

        // original * (1 - m) + inpainted * m
        var m = MaskPlane(mask, batchSize, channels, height, width);
        var inpaintedRegion = Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(
            Engine.TensorNarrow(inpainted, 0, 0, batchSize), 1, 0, channels), 2, 0, height), 3, 0, width);
        return Engine.TensorAdd(original, Engine.TensorMultiply(m, Engine.TensorSubtract(inpaintedRegion, original)));
    }

    /// <summary>
    /// Warps features by a flow given at any resolution: the flow is resampled to the feature grid (bilinear,
    /// align_corners=False), rescaled to feature-pixel units, and the features are sampled bilinearly at pixel + flow as
    /// E2FGVI's <c>flow_warp</c> does, grid_sample(align_corners=True, padding_mode="zeros").
    /// </summary>
    /// <remarks>
    /// This used to clamp samples that fall outside the frame to the border pixel. The reference treats them as
    /// zero, so content entering from outside the frame is not invented from the edge.
    /// </remarks>
    internal Tensor<T> WarpFeatures(Tensor<T> features, Tensor<T> flow)
        => WarpByFlow(features, UpsampleFlow(flow, features.Shape[2], features.Shape[3]), GridSamplePadding.Zeros);

    /// <summary>
    /// Resizes a pixel-unit flow to <paramref name="targetH"/> x <paramref name="targetW"/> as E2FGVI's SPyNet does:
    /// <c>F.interpolate(flow, mode="bilinear", align_corners=False)</c>, then dx scaled by targetW / srcW and dy by
    /// targetH / srcH so the displacements are in the new grid's pixels.
    /// </summary>
    /// <remarks>This used to take the nearest source pixel, which made the flow blocky at every upsampling.</remarks>
    internal Tensor<T> UpsampleFlow(Tensor<T> flow, int targetH, int targetW)
    {
        int batchSize = flow.Shape[0];
        int srcH = flow.Shape[2];
        int srcW = flow.Shape[3];

        var planes = Engine.TensorNarrow(flow, 1, 0, 2);
        var resized = srcH == targetH && srcW == targetW
            ? planes
            : Engine.Interpolate(planes, [targetH, targetW], InterpolateMode.Bilinear, false);
        var scale = new Tensor<T>(new[] { NumOps.FromDouble((double)targetW / srcW), NumOps.FromDouble((double)targetH / srcH) }, [1, 2, 1, 1]);
        return Engine.TensorMultiply(resized, Engine.TensorBroadcastTo(scale, [batchSize, 2, targetH, targetW]));
    }

    private Tensor<T> ZeroTensor(int[] shape) => new Tensor<T>(shape);

    private Tensor<T> ScaleTensor(Tensor<T> tensor, double scale) =>
        Engine.TensorMultiplyScalar(tensor, NumOps.FromDouble(scale));

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        Engine.TensorAdd(a, b);

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate([a, b], axis: 1);
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        return Engine.Upsample(input, 2, 2);
    }

    // Recorded ReLU: the element transform it replaces was invisible to the tape, so the flow network
    // received no gradient.
    internal Tensor<T> ApplyReLU(Tensor<T> input) => Engine.ReLU(input);

    private Tensor<T> ApplySigmoid(Tensor<T> input) =>
        Engine.Sigmoid(input);

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]]);
        tensor.Data.Span.Slice(0, result.Data.Length).CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    protected override void InitializeLayers()
    {
        ClearLayers();

        foreach (var layer in _flowNet) Layers.Add(layer);
        Layers.Add(_flowHead);
        foreach (var layer in _encoder) Layers.Add(layer);
        foreach (var layer in _transformer) Layers.Add(layer);
        foreach (var layer in _propagation) Layers.Add(layer);
        foreach (var layer in _decoder) Layers.Add(layer);
        Layers.Add(_outputHead);
    }

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "E2FGVI" },
            { "Description", "End-to-End Flow-Guided Video Inpainting" },
            { "InputHeight", _height },
            { "InputWidth", _width }
        },
        ModelDataProvider = () => SerializeForMetadata()
    };





    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    public override Tensor<T> Inpaint(Tensor<T> frames, Tensor<T> masks)
    {
        var stacked = ConcatenateFeatures(frames, masks);
        return BlendKnownPixels(frames, masks, Forward(stacked));
    }

    internal Tensor<T> BlendKnownPixels(Tensor<T> frames, Tensor<T> masks, Tensor<T> prediction)
    {
        // Known pixels pass through, masked ones come from the prediction: m * prediction + (1 - m) * frame.
        // This used to add 1e-3 x the mean mask value to the first output pixel so different masks could not give
        // identical outputs; that corrupted a pixel the model is required to reproduce exactly.
        int batch = frames.Shape[0], channels = prediction.Shape[1];
        int height = prediction.Shape[2], width = prediction.Shape[3];
        var expanded = MaskPlane(masks, batch, channels, height, width);
        var known = Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(
            Engine.TensorNarrow(frames, 0, 0, batch), 1, 0, channels), 2, 0, height), 3, 0, width);
        var inverse = Engine.TensorSubtract(Engine.TensorOnesLike(expanded), expanded);
        return Engine.TensorAdd(Engine.TensorMultiply(expanded, prediction), Engine.TensorMultiply(inverse, known));
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        base.ResolveLazyLayerShapes();
        // The custom flow branch is not traversed by PredictCore, but its lazy
        // convolution weights are still serialized. Materialize any remaining
        // layers so clone/round-trip preserves the complete parameter graph.
        foreach (var layer in Layers)
        {
            if (layer is LayerBase<T> lazy && !lazy.IsShapeResolved)
            {
                try { lazy.ResolveFromShape([1, 4, Math.Max(8, _height), Math.Max(8, _width)]); }
                catch (ArgumentException ex)
                {
                    // NOT SILENT. A layer refusing this probe shape is an EXPECTED outcome -- the
                    // shape is a guess made on the flow branch's behalf, and a layer that rejects it
                    // simply stays lazy, which is what the loop is willing to accept. But swallowing
                    // the exception outright made the other case invisible too: a layer that should
                    // have resolved and did not just silently drops out of the serialized parameter
                    // graph, which is exactly the completeness this override exists to guarantee.
                    //
                    // Debug.WriteLine matches how the neighbouring video models surface this class of
                    // diagnostic (SlowFast.cs uses it in four places); there is no logger abstraction
                    // at this layer to route it to.
                    System.Diagnostics.Debug.WriteLine(
                        $"E2FGVI.ResolveLazyLayerShapes: {layer.GetType().Name} rejected the probe " +
                        $"shape [1, 4, {Math.Max(8, _height)}, {Math.Max(8, _width)}] and remains " +
                        $"lazy; its parameters will not be serialized. {ex.GetType().Name}: {ex.Message}");
                }
            }
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return DenormalizeFrames(modelOutput);
    }

    #endregion

}
