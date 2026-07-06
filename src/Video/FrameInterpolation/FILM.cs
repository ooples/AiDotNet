using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// FILM (Frame Interpolation for Large Motion) model for high-quality frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FILM generates smooth intermediate frames between two input frames,
/// even when there's significant motion between them. It's particularly good at:
/// - Large motion scenes (fast camera movements, rapid object motion)
/// - Creating slow-motion effects from regular video
/// - Increasing video frame rate (24fps to 60fps)
/// - Smooth transitions between keyframes
///
/// Unlike older methods that struggle with large motions, FILM uses a multi-scale
/// feature extraction approach that handles both small and large movements gracefully.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Multi-scale feature pyramid for handling large motions
/// - Bi-directional flow estimation with occlusion handling
/// - Feature-based frame synthesis (not just flow warping)
/// - Scale-agnostic architecture for arbitrary resolution
/// </para>
/// <para>
/// <b>Reference:</b> Reda et al., "FILM: Frame Interpolation for Large Motion"
/// ECCV 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a FILM model for frame interpolation with large motion handling
/// var film = new FILM&lt;double&gt;();
///
/// // Or configure with custom multi-scale feature pyramid settings
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 6, outputSize: 3);
/// var model = new FILM&lt;double&gt;(architecture, numScales: 7, numFeatures: 64);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FILM: Frame Interpolation for Large Motion",
    "https://arxiv.org/abs/2202.04901",
    Year = 2022,
    Authors = "Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, Brian Curless")]
public class FILM<T> : FrameInterpolationBase<T>
{
    private readonly FILMOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numScales;
    private int _numFeatures;

    // Multi-scale feature extractor. NOTE: these per-role references are re-derived from Layers each
    // forward (SyncLayerReferences) — NOT readonly — because the base's deserialize/clone replaces the
    // Layers list with freshly-loaded layers, and stale ctor references would make a clone run untrained
    // convs (Clone_AfterTraining).
    private readonly List<ConvolutionalLayer<T>> _featureExtractor;
    private readonly List<ConvolutionalLayer<T>> _pyramidLayers;

    // Bi-directional flow estimator
    private readonly List<ConvolutionalLayer<T>> _flowEstimator;
    private ConvolutionalLayer<T> _flowRefinement;

    // Feature fusion and synthesis
    private readonly List<ConvolutionalLayer<T>> _fusionLayers;
    private ConvolutionalLayer<T> _synthesisHead;

    // Occlusion estimator
    private ConvolutionalLayer<T> _occlusionEstimator;

    // Number of pyramid layers (computed from dims in the ctor); needed to re-index Layers.
    private int _pyramidCount;

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

    /// <summary>
    /// Gets the number of pyramid scales.
    /// </summary>
    internal int NumScales => _numScales;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FILM class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numScales">Number of pyramid scales for multi-scale processing.</param>
    /// <param name="numFeatures">Base number of feature channels.</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public FILM()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 6,
            outputSize: 3))
    {
    }

    public FILM(
        NeuralNetworkArchitecture<T> architecture,
        int numScales = 7,
        int numFeatures = 64,
        FILMOptions? options = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _options = options ?? new FILMOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numScales = numScales;
        _numFeatures = numFeatures;

        _featureExtractor = [];
        _pyramidLayers = [];
        _flowEstimator = [];
        _fusionLayers = [];

        // Check for user-provided custom layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateFILMLayers(
                channels: _channels, height: _height, width: _width,
                numScales: _numScales, numFeatures: _numFeatures).ToList();
            Layers.AddRange(layers);
        }

        // Compute pyramid count to distribute layers correctly
        int pyramidCount = 0;
        {
            int cH = _height / 2, cW = _width / 2, cC = _numFeatures * 2;
            for (int s = 0; s < _numScales - 1; s++)
            {
                if (cH < 4 || cW < 4) break;
                pyramidCount++;
                cH /= 2; cW /= 2; cC = Math.Min(cC * 2, 512);
            }
        }

        _pyramidCount = pyramidCount;

        // Distribute layers to sub-lists for forward pass
        int idx = 0;
        // Feature extractor (3 layers)
        for (int i = 0; i < 3; i++)
            _featureExtractor.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Pyramid layers (variable)
        for (int i = 0; i < pyramidCount; i++)
            _pyramidLayers.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Flow estimator (3 layers)
        for (int i = 0; i < 3; i++)
            _flowEstimator.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Flow refinement
        _flowRefinement = (ConvolutionalLayer<T>)Layers[idx++];
        // Occlusion estimator
        _occlusionEstimator = (ConvolutionalLayer<T>)Layers[idx++];
        // Fusion layers (2 layers)
        for (int i = 0; i < 2; i++)
            _fusionLayers.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Synthesis head
        _synthesisHead = (ConvolutionalLayer<T>)Layers[idx++];
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Interpolates a frame between two input frames.
    /// </summary>
    /// <param name="frame1">First input frame [C, H, W] or [B, C, H, W].</param>
    /// <param name="frame2">Second input frame [C, H, W] or [B, C, H, W].</param>
    /// <param name="timestep">Interpolation position (0.0 = frame1, 1.0 = frame2, 0.5 = middle).</param>
    /// <returns>Interpolated frame at the specified timestep.</returns>
    public override Tensor<T> Interpolate(Tensor<T> frame1, Tensor<T> frame2, double timestep = 0.5)
    {
        // Re-sync per-role layer refs to the current Layers (a clone/deserialize replaces Layers with the
        // loaded trained layers after construction; without this the forward would run the ctor's
        // untrained layers).
        SyncLayerReferences();

        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        // Fully tape-connected forward (autodiff handles the backward pass — no manual backprop / no
        // cached activations). Every op below is a trainable layer or a tape-aware Engine op, so
        // TrainWithTape's ComputeGradients reaches every parameter.
        var features1 = ExtractFeatures(frame1);
        var features2 = ExtractFeatures(frame2);

        var (flow1to2, flow2to1) = EstimateFlow(features1, features2);

        // Scale flows by timestep.
        var flowToT1 = ScaleFlow(flow2to1, timestep);
        var flowToT2 = ScaleFlow(flow1to2, 1.0 - timestep);

        var (occ1, occ2) = EstimateOcclusion(features1, features2, flow1to2, flow2to1);

        // Backward-warp features by the timestep-scaled flows.
        var warped1 = WarpFeatures(features1, flowToT1);
        var warped2 = WarpFeatures(features2, flowToT2);

        // Occlusion-aware fusion + synthesis.
        var fused = FuseFeatures(warped1, warped2, occ1, occ2, flowToT1, flowToT2, timestep);
        var output = SynthesizeFrame(fused);

        if (!hasBatch)
        {
            output = RemoveBatchDimension(output);
        }

        return output;
    }

    /// <summary>
    /// Generates multiple intermediate frames between two input frames.
    /// </summary>
    /// <param name="frame1">First input frame.</param>
    /// <param name="frame2">Second input frame.</param>
    /// <param name="numIntermediateFrames">Number of frames to generate.</param>
    /// <returns>List of interpolated frames (excluding input frames).</returns>
    public List<Tensor<T>> InterpolateMultiple(Tensor<T> frame1, Tensor<T> frame2, int numIntermediateFrames)
    {
        var results = new List<Tensor<T>>();

        for (int i = 1; i <= numIntermediateFrames; i++)
        {
            double t = (double)i / (numIntermediateFrames + 1);
            var interpolated = Interpolate(frame1, frame2, t);
            results.Add(interpolated);
        }

        return results;
    }

    /// <summary>
    /// Increases video frame rate by a given factor.
    /// </summary>
    /// <param name="frames">Input video frames.</param>
    /// <param name="factor">Frame rate multiplication factor (2, 4, or 8).</param>
    /// <returns>Frame rate enhanced video.</returns>
    public List<Tensor<T>> IncreaseFrameRate(List<Tensor<T>> frames, int factor = 2)
    {
        var result = new List<Tensor<T>>();
        int intermediateFrames = factor - 1;

        for (int i = 0; i < frames.Count - 1; i++)
        {
            result.Add(frames[i]);
            var interpolated = InterpolateMultiple(frames[i], frames[i + 1], intermediateFrames);
            result.AddRange(interpolated);
        }

        result.Add(frames[frames.Count - 1]);
        return result;
    }

    /// <summary>
    /// Creates slow-motion effect from video frames.
    /// </summary>
    /// <param name="frames">Input video frames.</param>
    /// <param name="slowdownFactor">Slowdown factor (2 = half speed, 4 = quarter speed).</param>
    /// <returns>Slow-motion video frames.</returns>
    public List<Tensor<T>> CreateSlowMotion(List<Tensor<T>> frames, int slowdownFactor = 4)
    {
        return IncreaseFrameRate(frames, slowdownFactor);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => InterpolatePair(input);

    /// <inheritdoc/>
    // Training forward must run the SAME custom forward as inference (split the concatenated pair, then
    // Interpolate). Without this override the base tape-training runs the flat Layers list on the raw
    // 2-frame input, feeding the 3-channel feature conv a 6-channel tensor ("Expected input depth 3, got
    // 6"). The forward is fully tape-connected, so ComputeGradients reaches every parameter.
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => InterpolatePair(input);

    /// <summary>
    /// Splits a concatenated frame pair (rank-4 [B, C*2, H, W] or, as the ModelFamily harness feeds,
    /// rank-3 [C*2, H, W]) along the channel axis with the tape-aware Engine slice, then interpolates at
    /// t=0.5. Interpolate is rank-flexible (hasBatch = frame1.Rank == 4).
    /// </summary>
    private Tensor<T> InterpolatePair(Tensor<T> input)
    {
        int rank = input.Rank;
        int channelAxis = rank == 4 ? 1 : 0;
        int channels = input.Shape[channelAxis] / 2;

        var begin1 = new int[rank];
        var begin2 = new int[rank];
        var size = new int[rank];
        for (int i = 0; i < rank; i++) size[i] = input.Shape[i];
        size[channelAxis] = channels;
        begin2[channelAxis] = channels;

        var frame1 = Engine.TensorSlice(input, begin1, size);
        var frame2 = Engine.TensorSlice(input, begin2, size);
        return Interpolate(frame1, frame2, 0.5);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> ExtractFeatures(Tensor<T> frame)
    {
        var features = frame;
        foreach (var layer in _featureExtractor)
        {
            features = layer.Forward(features);
            features = ApplyLeakyReLU(features, 0.2);
        }
        return features;
    }

    private (Tensor<T> flow1to2, Tensor<T> flow2to1) EstimateFlow(Tensor<T> features1, Tensor<T> features2)
    {
        // Concatenate features
        var concat = ConcatenateChannels(features1, features2);

        // Estimate flow
        var flowFeatures = concat;
        for (int i = 0; i < _flowEstimator.Count; i++)
        {
            flowFeatures = _flowEstimator[i].Forward(flowFeatures);
            // Only apply activation to intermediate layers, not final flow output
            // Flow represents pixel displacements (positive or negative, unbounded)
            if (i < _flowEstimator.Count - 1)
            {
                flowFeatures = ApplyLeakyReLU(flowFeatures, 0.2);
            }
        }

        // Split the 4-channel head output into two 2-channel flows via the tape-aware Engine slice
        // (was a scalar per-pixel copy that detached the tape).
        int b = flowFeatures.Shape[0];
        int h = flowFeatures.Shape[2];
        int w = flowFeatures.Shape[3];
        var flow1to2 = Engine.TensorSlice(flowFeatures, [0, 0, 0, 0], [b, 2, h, w]);
        var flow2to1 = Engine.TensorSlice(flowFeatures, [0, 2, 0, 0], [b, 2, h, w]);
        return (flow1to2, flow2to1);
    }

    private Tensor<T> ScaleFlow(Tensor<T> flow, double scale)
    {
        return Engine.TensorMultiplyScalar(flow, NumOps.FromDouble(scale));
    }

    private (Tensor<T> occ1, Tensor<T> occ2) EstimateOcclusion(
        Tensor<T> features1, Tensor<T> features2,
        Tensor<T> flow1to2, Tensor<T> flow2to1)
    {
        // Concatenate all inputs
        var concat = ConcatenateChannels(features1, features2);
        concat = ConcatenateChannels(concat, flow1to2);
        concat = ConcatenateChannels(concat, flow2to1);

        // Estimate occlusion
        var occFeatures = _occlusionEstimator.Forward(concat);
        occFeatures = ApplySigmoid(occFeatures);

        // Split into two masks
        // Split the 2-channel sigmoid output into two 1-channel occlusion masks via the tape-aware
        // Engine slice (was a scalar per-pixel copy that detached the tape).
        int b = occFeatures.Shape[0];
        int h = occFeatures.Shape[2];
        int w = occFeatures.Shape[3];
        var occ1 = Engine.TensorSlice(occFeatures, [0, 0, 0, 0], [b, 1, h, w]);
        var occ2 = Engine.TensorSlice(occFeatures, [0, 1, 0, 0], [b, 1, h, w]);
        return (occ1, occ2);
    }

    // Backward-warp features by optical flow via the tape-aware Engine.GridSample (identity affine grid
    // + normalized flow offset) so gradients flow back into BOTH features and flow — the standard
    // differentiable warp (cf. RIFE.WarpImage). Was a scalar per-pixel bilinear loop that detached the
    // tape. flow: [B, 2, H, W] in pixel units (channel 0 = dx, 1 = dy).
    private Tensor<T> WarpFeatures(Tensor<T> features, Tensor<T> flow)
    {
        int b = features.Shape[0];
        int h = features.Shape[2];
        int w = features.Shape[3];

        var identityTheta = new Tensor<T>([b, 2, 3]);
        for (int bi = 0; bi < b; bi++)
        {
            identityTheta[bi, 0, 0] = NumOps.One; // x scale
            identityTheta[bi, 1, 1] = NumOps.One; // y scale
        }
        var baseGrid = Engine.AffineGrid(identityTheta, h, w);

        var flowNHWC = Engine.TensorPermute(flow, [0, 2, 3, 1]); // [B, h, w, 2]
        double sx = w > 1 ? 2.0 / (w - 1) : 0.0;
        double sy = h > 1 ? 2.0 / (h - 1) : 0.0;
        var scale = new Tensor<T>([b, h, w, 2]);
        var scaleSpan = scale.Data.Span;
        for (int idx = 0; idx + 1 < scaleSpan.Length; idx += 2)
        {
            scaleSpan[idx] = NumOps.FromDouble(sx);
            scaleSpan[idx + 1] = NumOps.FromDouble(sy);
        }
        var grid = Engine.TensorAdd(baseGrid, Engine.TensorMultiply(flowNHWC, scale));

        var featNHWC = Engine.TensorPermute(features, [0, 2, 3, 1]);
        var warpedNHWC = Engine.GridSample(featNHWC, grid);
        return Engine.TensorPermute(warpedNHWC, [0, 3, 1, 2]);
    }

    private Tensor<T> FuseFeatures(
        Tensor<T> warped1, Tensor<T> warped2,
        Tensor<T> occ1, Tensor<T> occ2,
        Tensor<T> flow1, Tensor<T> flow2,
        double timestep)
    {
        // Concatenate all inputs
        var concat = ConcatenateChannels(warped1, warped2);
        concat = ConcatenateChannels(concat, flow1);
        concat = ConcatenateChannels(concat, flow2);
        concat = ConcatenateChannels(concat, occ1);
        concat = ConcatenateChannels(concat, occ2);

        // Process through fusion layers
        var features = concat;
        foreach (var layer in _fusionLayers)
        {
            features = layer.Forward(features);
            features = ApplyLeakyReLU(features, 0.2);
        }

        return features;
    }

    private Tensor<T> SynthesizeFrame(Tensor<T> features)
    {
        // Upsample to full resolution
        var upsampled = Upsample2x(features);

        // Final synthesis
        var output = _synthesisHead.Forward(upsampled);
        output = ApplySigmoid(output);

        return output;
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate([a, b], axis: 1);
    }

    // Nearest-neighbour 2x upsample via the tape-aware Engine op (was a scalar per-pixel loop that
    // detached the autodiff tape).
    private Tensor<T> Upsample2x(Tensor<T> input)
        => Engine.Upsample(input, 2, 2);

    // Tape-aware LeakyReLU (was a .Transform closure that detaches the tape).
    private Tensor<T> ApplyLeakyReLU(Tensor<T> input, double negativeSlope)
        => Engine.LeakyReLU(input, NumOps.FromDouble(negativeSlope));

    private Tensor<T> ApplySigmoid(Tensor<T> input)
        => Engine.Sigmoid(input);

    // Batch-dim add/remove via Engine.Reshape (tape-aware; a raw span-copy detached the tape at the
    // output boundary, so the loss gradient never reached the network on the unbatched path).
    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, [1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        var newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++) newShape[i] = tensor.Shape[i + 1];
        return Engine.Reshape(tensor, newShape);
    }


    #endregion

    #region Abstract Implementation

    protected override void InitializeLayers()
    {
        ClearLayers();

        // MUST match the constructor's layer-distribution order (the order CreateFILMLayers emits them:
        // featureExtractor, pyramid, flowEstimator, flowRefinement, occlusionEstimator, fusion,
        // synthesisHead). InitializeLayers rebuilds Layers on deserialize; if this order differs from the
        // order the parameters were serialized in, UpdateParameters loads each layer's weights into the
        // WRONG layer, so a clone/round-trip predicts with mismatched weights (Clone_AfterTraining).
        foreach (var layer in _featureExtractor) Layers.Add(layer);
        foreach (var layer in _pyramidLayers) Layers.Add(layer);
        foreach (var layer in _flowEstimator) Layers.Add(layer);
        Layers.Add(_flowRefinement);
        Layers.Add(_occlusionEstimator);
        foreach (var layer in _fusionLayers) Layers.Add(layer);
        Layers.Add(_synthesisHead);
    }

    /// <summary>
    /// Re-derives the per-role layer references (_featureExtractor / _pyramidLayers / _flowEstimator /
    /// _flowRefinement / _occlusionEstimator / _fusionLayers / _synthesisHead) from the CURRENT Layers
    /// list, by the constructor's distribution layout. Run before every forward: the base's
    /// deserialize/clone repopulates Layers with freshly-loaded (trained) layers, so without re-syncing
    /// the forward keeps running the constructor's untrained layer objects and a clone of a trained model
    /// reproduces the untrained output (Clone_AfterTraining). Mirrors RIFE/UPRNet.ExtractLayerReferences.
    /// </summary>
    private void SyncLayerReferences()
    {
        int total = 11 + _pyramidCount; // 3 FE + pyramid + 3 flow + flowRefine + occlusion + 2 fusion + synth
        if (Layers.Count < total) return; // not fully built yet

        _featureExtractor.Clear();
        _pyramidLayers.Clear();
        _flowEstimator.Clear();
        _fusionLayers.Clear();

        int idx = 0;
        for (int i = 0; i < 3; i++) _featureExtractor.Add((ConvolutionalLayer<T>)Layers[idx++]);
        for (int i = 0; i < _pyramidCount; i++) _pyramidLayers.Add((ConvolutionalLayer<T>)Layers[idx++]);
        for (int i = 0; i < 3; i++) _flowEstimator.Add((ConvolutionalLayer<T>)Layers[idx++]);
        _flowRefinement = (ConvolutionalLayer<T>)Layers[idx++];
        _occlusionEstimator = (ConvolutionalLayer<T>)Layers[idx++];
        for (int i = 0; i < 2; i++) _fusionLayers.Add((ConvolutionalLayer<T>)Layers[idx++]);
        _synthesisHead = (ConvolutionalLayer<T>)Layers[idx++];
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // Use layer.ParameterCount + layer.UpdateParameters and ALWAYS advance the offset (the proven
        // AMT pattern). The previous version keyed off GetParameters().Length and called SetParameters,
        // then SKIPPED a layer whose count read as 0 (an unresolved/lazy layer post-deserialize) WITHOUT
        // advancing the offset — so every subsequent layer read its weights from the wrong slice and a
        // clone/round-trip predicted with mismatched weights (Clone_AfterTraining, issue #1221 class).
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = checked((int)layer.ParameterCount);
            if (count == 0) continue;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FILM" },
                { "Description", "Frame Interpolation for Large Motion" },
                { "InputHeight", _height },
                { "InputWidth", _width },
                { "NumScales", _numScales }
            },
            ModelData = SerializeForMetadata()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numScales);
        writer.Write(_numFeatures);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numScales = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new FILM<T>(Architecture, _numScales, _numFeatures);

    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return DenormalizeFrames(modelOutput);
    }

    #endregion

}
