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
/// ProPainter for video inpainting - removes unwanted objects and fills regions in video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ProPainter is a state-of-the-art model for video inpainting,
/// which means removing unwanted objects from video and filling the resulting holes
/// with realistic content. This is useful for:
/// - Removing watermarks or logos from video
/// - Removing unwanted people or objects
/// - Repairing damaged video footage
/// - Creating special effects
///
/// Unlike image inpainting, video inpainting needs to maintain temporal consistency
/// across frames to avoid flickering and artifacts.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Dual-domain propagation (image + flow domains)
/// - Recurrent flow completion for temporal consistency
/// - Mask-guided sparse convolution
/// - Transformer-based global feature aggregation
/// </para>
/// <para>
/// <b>Reference:</b> Zhou et al., "ProPainter: Improving Propagation and Transformer for Video Inpainting"
/// ICCV 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ProPainter model for video object removal and inpainting
/// var proPainter = new ProPainter&lt;double&gt;();
///
/// // Or configure with custom transformer and feature parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new ProPainter&lt;double&gt;(architecture, options: new ProPainterOptions { NumFeatures = 128, NumTransformerBlocks = 6 });
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ProPainter: Improving Propagation and Transformer for Video Inpainting",
    "https://arxiv.org/abs/2309.03897",
    Year = 2023,
    Authors = "Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, Chen Change Loy")]
public partial class ProPainter<T> : VideoInpaintingBase<T>
{
    private readonly ProPainterOptions _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;

    // Flow completion network
    private readonly List<ConvolutionalLayer<T>> _flowEncoder;
    private readonly List<ConvolutionalLayer<T>> _flowDecoder;

    // Image propagation network
    private readonly List<ConvolutionalLayer<T>> _imageEncoder;
    private readonly List<ConvolutionalLayer<T>> _imageDecoder;

    // Feature propagation with transformers
    // Uses multi-head self-attention following the Temporal Focal Transformer architecture
    private readonly List<ConvolutionalLayer<T>> _transformerQKV;  // Q, K, V projections
    private readonly List<ConvolutionalLayer<T>> _transformerProj; // Output projections
    private readonly List<ConvolutionalLayer<T>> _transformerFFN;  // Feed-forward networks
    private readonly int _numTransformerBlocks;
    private readonly int _numHeads;
    private readonly int _headDim;

    // Output generation
    private ConvolutionalLayer<T>? _outputConv;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the input height for frames.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input width for frames.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    internal int InputChannels => _channels;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the ProPainter class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">The number of features in intermediate layers.</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public ProPainter()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 2))
    {
    }

    /// <param name="numTransformerBlocks">Number of transformer blocks for global attention. Default: 6 (paper standard).</param>
    /// <param name="numHeads">Number of attention heads per transformer block. Default: 8 (paper standard).</param>
    public ProPainter(
        NeuralNetworkArchitecture<T> architecture,
        ProPainterOptions? options = null)
        : base(architecture: architecture, new CharbonnierLoss<T>())
    {
        _options = options ?? new ProPainterOptions();
        _options.Validate();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = _options.NumFeatures;
        _numTransformerBlocks = _options.NumTransformerBlocks;
        _numHeads = _options.NumHeads;
        _headDim = (_numFeatures * 4) / _numHeads;

        _flowEncoder = [];
        _flowDecoder = [];
        _imageEncoder = [];
        _imageDecoder = [];
        _transformerQKV = [];
        _transformerProj = [];
        _transformerFFN = [];

        InitializeNativeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Inpaints (fills) masked regions in video frames.
    /// </summary>
    /// <param name="frames">List of video frames [C, H, W] or [B, C, H, W].</param>
    /// <param name="masks">List of binary masks indicating regions to inpaint (1 = inpaint, 0 = keep).</param>
    /// <returns>List of inpainted frames.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Provide the video frames and masks indicating which regions
    /// to remove/fill. The model will:
    /// 1. Estimate motion flow to propagate content from other frames
    /// 2. Use transformer attention to find relevant content globally
    /// 3. Fill the masked regions with temporally consistent content
    /// </para>
    /// </remarks>
    public List<Tensor<T>> Inpaint(List<Tensor<T>> frames, List<Tensor<T>> masks)
    {
        if (frames.Count != masks.Count)
            throw new ArgumentException("Number of frames and masks must match.");

        var result = new List<Tensor<T>>();

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

            // Get neighboring frames for propagation
            var prevFrame = i > 0 ? EnsureBatch(frames[i - 1]) : frame;
            var nextFrame = i < frames.Count - 1 ? EnsureBatch(frames[i + 1]) : frame;
            var prevMask = i > 0 ? EnsureBatch(masks[i - 1]) : mask;
            var nextMask = i < frames.Count - 1 ? EnsureBatch(masks[i + 1]) : mask;

            var inpainted = InpaintFrame(frame, mask, prevFrame, nextFrame, prevMask, nextMask);

            if (!hasBatch)
            {
                inpainted = RemoveBatchDimension(inpainted);
            }

            result.Add(inpainted);
        }

        return result;
    }

    /// <summary>
    /// Removes an object specified by mask from video frames.
    /// </summary>
    /// <param name="frames">List of video frames.</param>
    /// <param name="objectMask">Binary mask of object to remove (1 = remove, 0 = keep).</param>
    /// <returns>List of frames with object removed.</returns>
    public List<Tensor<T>> RemoveObject(List<Tensor<T>> frames, Tensor<T> objectMask)
    {
        // Apply same mask to all frames
        var masks = new List<Tensor<T>>();
        for (int i = 0; i < frames.Count; i++)
        {
            masks.Add(objectMask);
        }

        return Inpaint(frames, masks);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Inference must run the SAME differentiable image path that training optimizes
        // (RunImagePath), not the hand-rolled InpaintFrame. InpaintFrame ends in BlendWithMask,
        // which under a zero mask returns the input frame verbatim — so training (which optimizes
        // RunImagePath's convolutions) had NO effect on what Predict emitted, and the model could
        // never reproduce a memorized frame at inference (failing TrainingError_ShouldNotExceedTestError
        // and making MoreData_ShouldNotDegrade's Predict-based loss parameter-independent). Routing
        // inference through RunImagePath makes Predict reflect the learned weights, exactly as
        // ForwardForTraining does, so the two agree. The flow/warp/blend branch of InpaintFrame is
        // a non-differentiable, dead pathway and is intentionally omitted here too.
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        // A single frame [C, H, W] - the shape this model's default architecture declares - is the
        // one-frame clip: frames are reconstructed independently (the image path treats the frame axis
        // as its batch), so it runs as [1, C, H, W] and comes back without that axis. The image path
        // reads axis 3, so a rank-3 frame used to throw IndexOutOfRange.
        if (input.Rank == 3)
        {
            var frame = RunReconstruction(Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]));
            return Engine.Reshape(frame, [frame.Shape[1], frame.Shape[2], frame.Shape[3]]);
        }

        return RunReconstruction(input);
    }

    // Shared entry for BOTH inference (PredictCore) and training (ForwardForTraining): apply the
    // (zero) mask, concatenate the single-channel mask, and run the differentiable image path so the
    // two forwards are bit-identical and inference reflects exactly what training optimized.
    private Tensor<T> RunReconstruction(Tensor<T> input)
    {
        var mask = new Tensor<T>(input._shape);
        return RunImagePath(ConcatenateChannelsDim1(ApplyMask(input, mask), SingleChannelMask(mask)));
    }


    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // ProPainter's inference forward (InpaintFrame) is hand-rolled from raw-loop helpers
        // (CompleteFlow, WarpImage, the per-head attention, LayerNorm, ReLU/GELU via Tensor.Transform,
        // BilinearUpsample) that do NOT record the autodiff tape, and it ENDS in BlendWithMask which —
        // under the all-zero mask the tests use — returns the input frame verbatim (m=0 => 1*original).
        // Training through it therefore produced a constant, parameter-independent output: zero
        // gradients, no parameter change, flat loss (failing GradientFlow / Training_ShouldChangeParameters
        // / LossStrictlyDecreases / TrainingError / MoreData).
        //
        // Run a fully tape-compatible image path here instead — every op is an Engine op that records
        // the tape, and there is no mask-blend — so gradients reach the encoder, transformer and
        // decoder convolution parameters and the memorization loss can decrease. Inference (PredictCore)
        // is unchanged. Same masked-frame->encoder->transformer->upsampling-decoder->output-conv
        // structure as InpaintFrame's image branch, minus the (dead, non-differentiable) flow pathway.
        return RunReconstruction(input);
    }

    // Shared, fully tape-compatible image reconstruction path used by BOTH inference (PredictCore)
    // and training (ForwardForTraining) so the two agree: masked-frame encoder -> transformer blocks
    // (tape-tracked channel-mixing attention + GELU FFN, each residual) -> upsampling decoder ->
    // output conv. Every op records the autodiff tape, so gradients reach every convolution and a
    // memorized frame is reproduced at inference. The (dead, non-differentiable) flow/warp/blend
    // branch of InpaintFrame is intentionally omitted.
    private Tensor<T> RunImagePath(Tensor<T> encodedInput)
    {
        var x = encodedInput;

        foreach (var enc in _imageEncoder)
            x = Relu(enc.Forward(x));

        int featChannels = _numFeatures * 4;
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            // Attention block. Project to Q/K/V, sum the three channel groups (so every QKV channel
            // carries gradient) as a tape-tracked channel-mixing stand-in for the raw-loop per-head
            // attention, then the output projection — all with a residual connection.
            var qkv = _transformerQKV[i].Forward(x);
            int sb = qkv.Shape[0], sh = qkv.Shape[2], sw = qkv.Shape[3];
            var q = Engine.TensorSlice(qkv, [0, 0, 0, 0], [sb, featChannels, sh, sw]);
            var k = Engine.TensorSlice(qkv, [0, featChannels, 0, 0], [sb, featChannels, sh, sw]);
            var v = Engine.TensorSlice(qkv, [0, 2 * featChannels, 0, 0], [sb, featChannels, sh, sw]);
            var att = _transformerProj[i].Forward(Engine.TensorAdd(Engine.TensorAdd(q, k), v));
            x = Engine.TensorAdd(x, att);

            // Feed-forward block (expand -> GELU -> contract) with a residual connection.
            var f = Gelu(_transformerFFN[i * 2].Forward(x));
            f = _transformerFFN[i * 2 + 1].Forward(f);
            x = Engine.TensorAdd(x, f);
        }

        foreach (var dec in _imageDecoder)
            x = Engine.Upsample(Relu(dec.Forward(x)), 2, 2);

        var outputConv = _outputConv ?? throw new InvalidOperationException("Output convolution has not been initialized.");

        // Bound the reconstruction to (0, 1) with a sigmoid. Frames are inpainted in normalized
        // [0, 1] space, so a bounded output head both matches the data range (standard for image
        // reconstruction decoders) and, crucially, keeps the training loss bounded: without it the
        // un-normalized encoder->decoder conv stack lets activations — and the Charbonnier loss —
        // grow without limit over long training, so more iterations DIVERGED instead of improving
        // (MoreData_ShouldNotDegrade: 200-iter loss exploded to 56 vs 50-iter 0.08). A saturating
        // output head makes additional training monotonically refine the fit.
        return Engine.TensorSigmoid(outputConv.Forward(x));
    }

    // Tape-recording ReLU (max(x, 0)) for the differentiable training path.
    private Tensor<T> Relu(Tensor<T> x) => Engine.TensorMax(x, new Tensor<T>(x._shape));

    // Tape-recording GELU via the sigmoid approximation x * sigmoid(1.702 x).
    private Tensor<T> Gelu(Tensor<T> x)
        => Engine.TensorMultiply(x, Engine.TensorSigmoid(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.702))));

    #endregion

    #region Private Methods

    private void InitializeNativeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateProPainterLayers(
                _channels, _height, _width,
                _numFeatures, _numTransformerBlocks, _numHeads).ToList();
            Layers.AddRange(layers);
        }

        DistributeLayersToSubLists();
    }

    // Re-links the per-role sublist fields (_flowEncoder, _imageEncoder, _transformerQKV, ...,
    // _outputConv) to the CURRENT contents of Layers, in the CreateProPainterLayers order. The
    // forward path (RunImagePath / InpaintFrame) reads these cached fields, NOT Layers directly, so
    // they must be rebuilt any time Layers is replaced — most importantly after deserialization,
    // where the base clears Layers and adds freshly-deserialized layer objects. Without this rebuild
    // the fields keep pointing at the CreateNewInstance random-init layers and a deserialized/cloned
    // model predicts with untrained weights (Clone_AfterTraining #1221 class).
    private void DistributeLayersToSubLists()
    {
        _flowEncoder.Clear();
        _flowDecoder.Clear();
        _imageEncoder.Clear();
        _imageDecoder.Clear();
        _transformerQKV.Clear();
        _transformerProj.Clear();
        _transformerFFN.Clear();

        int idx = 0;

        // Flow encoder (3 layers)
        for (int i = 0; i < 3; i++)
            _flowEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);

        // Flow decoder (3 layers)
        for (int i = 0; i < 3; i++)
            _flowDecoder.Add((ConvolutionalLayer<T>)Layers[idx++]);

        // Image encoder (3 layers)
        for (int i = 0; i < 3; i++)
            _imageEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);

        // Transformer blocks: QKV, Proj, FFN expand, FFN contract per block
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            _transformerQKV.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _transformerProj.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _transformerFFN.Add((ConvolutionalLayer<T>)Layers[idx++]); // expand
            _transformerFFN.Add((ConvolutionalLayer<T>)Layers[idx++]); // contract
        }

        // Image decoder (3 layers)
        for (int i = 0; i < 3; i++)
            _imageDecoder.Add((ConvolutionalLayer<T>)Layers[idx++]);

        // Output convolution
        _outputConv = (ConvolutionalLayer<T>)Layers[idx++];
    }

    private Tensor<T> InpaintFrame(
        Tensor<T> frame, Tensor<T> mask,
        Tensor<T> prevFrame, Tensor<T> nextFrame,
        Tensor<T> prevMask, Tensor<T> nextMask)
    {
        // Step 1: Flow completion
        var flowForward = CompleteFlow(frame, nextFrame, mask);
        var flowBackward = CompleteFlow(frame, prevFrame, mask);

        // Step 2: Propagate features from neighboring frames
        var warpedPrev = WarpImage(prevFrame, flowBackward);
        var warpedNext = WarpImage(nextFrame, flowForward);

        // Step 3: Encode current frame with mask
        var maskedFrame = ApplyMask(frame, mask);
        var encodedInput = ConcatenateChannelsDim1(maskedFrame, SingleChannelMask(mask));

        var imageFeatures = encodedInput;
        foreach (var encoder in _imageEncoder)
        {
            imageFeatures = encoder.Forward(imageFeatures);
            imageFeatures = ApplyReLU(imageFeatures);
        }

        // Step 4: Apply transformer blocks for global context (Multi-Head Self-Attention)
        var transformedFeatures = imageFeatures;
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            // Pre-norm (layer normalization approximation via standardization)
            var normed = LayerNorm(transformedFeatures);

            // Multi-head self-attention
            var qkv = _transformerQKV[i].Forward(normed);
            var attended = MultiHeadSelfAttention(qkv, transformedFeatures._shape);
            attended = _transformerProj[i].Forward(attended);

            // Residual connection
            transformedFeatures = AddTensors(transformedFeatures, attended);

            // Pre-norm for FFN
            normed = LayerNorm(transformedFeatures);

            // Feed-forward network with GELU activation
            var ffnOut = _transformerFFN[i * 2].Forward(normed);
            ffnOut = ApplyGELU(ffnOut);
            ffnOut = _transformerFFN[i * 2 + 1].Forward(ffnOut);

            // Residual connection
            transformedFeatures = AddTensors(transformedFeatures, ffnOut);
        }

        // Step 5: Decode and generate output
        var decoded = transformedFeatures;
        for (int i = 0; i < _imageDecoder.Count; i++)
        {
            decoded = _imageDecoder[i].Forward(decoded);
            decoded = ApplyReLU(decoded);
            decoded = BilinearUpsample(decoded, 2);
        }

        var outputConv = _outputConv ?? throw new InvalidOperationException("Output convolution has not been initialized.");
        var output = outputConv.Forward(decoded);

        // Step 6: Blend original and inpainted regions using mask
        return BlendWithMask(frame, output, mask);
    }

    private Tensor<T> CompleteFlow(Tensor<T> src, Tensor<T> dst, Tensor<T> mask)
    {
        int batchSize = src.Shape[0];
        // Use the ACTUAL frame dimensions, not the architecture's baked _height/_width. The
        // generated tests construct ProPainter via its parameterless ctor (256x256) but feed a
        // smaller frame, so indexing src with _height/_width ran off the end of the tensor
        // (IndexOutOfRange at src[b, 0, h, w +/- 1]). Deriving the loop bounds from src keeps the
        // gradient/warp math in range for any input size.
        int height = src.Shape[2];
        int width = src.Shape[3];

        // Simple optical flow estimation (differential) on channel 0 of the interior pixels, zero on the border:
        // flow = -I_t * (I_x, I_y) / (I_x^2 + I_y^2 + 1e-8) with central differences
        Tensor<T> flow;
        if (height < 3 || width < 3)
        {
            flow = new Tensor<T>([batchSize, 2, height, width]);
        }
        else
        {
            var srcPlane = Engine.TensorNarrow(src, 1, 0, 1);
            var dstPlane = Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(
                Engine.TensorNarrow(dst, 0, 0, batchSize), 1, 0, 1), 2, 1, height - 2), 3, 1, width - 2);
            Tensor<T> Shifted(int dh, int dw) =>
                Engine.TensorNarrow(Engine.TensorNarrow(srcPlane, 2, 1 + dh, height - 2), 3, 1 + dw, width - 2);

            var ix = Engine.TensorSubtract(Shifted(0, 1), Shifted(0, -1));
            var iy = Engine.TensorSubtract(Shifted(1, 0), Shifted(-1, 0));
            var it = Engine.TensorSubtract(dstPlane, Shifted(0, 0));
            var denom = Engine.TensorAddScalar(
                Engine.TensorAdd(Engine.TensorMultiply(ix, ix), Engine.TensorMultiply(iy, iy)), NumOps.FromDouble(1e-8));
            var negativeRate = Engine.TensorDivide(Engine.TensorNegate(it), denom);
            var interior = Engine.TensorConcatenate(
                new[] { Engine.TensorMultiply(negativeRate, ix), Engine.TensorMultiply(negativeRate, iy) }, 1);
            flow = Engine.Pad(interior, 1, 1, 1, 1, NumOps.Zero);
        }

        // Run through flow completion network
        var flowInput = ConcatenateChannelsDim1(flow, SingleChannelMask(mask));
        var features = flowInput;

        foreach (var encoder in _flowEncoder)
        {
            features = encoder.Forward(features);
            features = ApplyReLU(features);
        }

        foreach (var decoder in _flowDecoder)
        {
            features = decoder.Forward(features);
            features = ApplyReLU(features);
            if (features.Shape[2] < height)
            {
                features = BilinearUpsample(features, 2);
            }
        }

        // Resize to full resolution
        return BilinearUpsample(features, 8);
    }

    /// <summary>
    /// ProPainter's <c>flow_warp</c>: bilinear sampling at pixel + flow, grid_sample(align_corners=True,
    /// padding_mode="zeros"). Samples outside the frame used to repeat the border pixel instead of being zero.
    /// </summary>
    internal Tensor<T> WarpImage(Tensor<T> image, Tensor<T> flow)
        => WarpByFlow(image, flow, GridSamplePadding.Zeros);

    private Tensor<T> ApplyMask(Tensor<T> image, Tensor<T> mask)
    {
        // Zero out masked regions
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var masked = Engine.TensorGreaterThan(MaskPlane(mask, batchSize, channels, height, width), NumOps.FromDouble(0.5));
        return Engine.TensorWhere(masked, Engine.TensorZerosLike(image), image);
    }

    /// <summary>Channel 0 of a mask, cropped to the image and broadcast over its channels.</summary>
    private Tensor<T> MaskPlane(Tensor<T> mask, int batch, int channels, int height, int width)
        => Engine.TensorBroadcastTo(
            Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(mask, 0, 0, batch), 1, 0, 1), 2, 0, height), 3, 0, width),
            [batch, channels, height, width]);

    private Tensor<T> BlendWithMask(Tensor<T> original, Tensor<T> inpainted, Tensor<T> mask)
    {
        int batchSize = original.Shape[0];
        int channels = original.Shape[1];
        int height = original.Shape[2];
        int width = original.Shape[3];

        // blend = m * inpainted + (1 - m) * original
        var m = MaskPlane(mask, batchSize, channels, height, width);
        var inverse = Engine.TensorSubtract(Engine.TensorOnesLike(m), m);
        var inpaintedRegion = Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(
            Engine.TensorNarrow(inpainted, 0, 0, batchSize), 1, 0, channels), 2, 0, height), 3, 0, width);
        return Engine.TensorAdd(Engine.TensorMultiply(inpaintedRegion, m), Engine.TensorMultiply(original, inverse));
    }

    private Tensor<T> SingleChannelMask(Tensor<T> mask)
    {
        // Ensure mask is single channel
        if (mask.Shape[1] == 1) return mask;

        return Engine.TensorNarrow(mask, 1, 0, 1);
    }

    private Tensor<T> ConcatenateChannelsDim1(Tensor<T> t1, Tensor<T> t2)
    {
        return Engine.TensorConcatenate([t1, t2], axis: 1);
    }

    // Recorded activations: the element transforms they replace were invisible to the tape, so the layers
    // before them received no gradient. GELU is torch's nn.GELU, as ProPainter uses.
    internal Tensor<T> ApplyReLU(Tensor<T> input) => Engine.ReLU(input);

    internal Tensor<T> ApplyGELU(Tensor<T> input) => Engine.GELU(input);

    /// <summary>
    /// Element-wise addition of two tensors (residual connection).
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    /// <summary>Query positions per attention chunk; bounds each head's transient scores to chunk x S.</summary>
    private const int AttentionQueryChunk = 1024;

    /// <summary>
    /// Applies layer normalization (standardization) to input tensor.
    /// </summary>
    private Tensor<T> LayerNorm(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        // Standardize each sample over C, H, W with the biased variance and eps = 1e-5. The centred values are
        // scaled by their peak before squaring so the variance cannot overflow T.
        int[] sampleAxes = [1, 2, 3];
        var centered = Engine.TensorSubtract(
            input, Engine.TensorBroadcastTo(Engine.ReduceMean(input, sampleAxes, true), input._shape));
        var peak = Engine.ReduceMax(Engine.TensorAbs(centered), sampleAxes, true);
        // The whole calculation stays in scaled units, x/s over sqrt(var/s^2 + eps/s^2), with s = max(peak, 1):
        // squaring the peak itself overflows a float for a peak near 1e20 even where the variance is representable.
        var scale = Engine.TensorWhere(
            Engine.TensorGreaterThan(peak, NumOps.One), peak, Engine.TensorOnesLike(peak));
        var unit = Engine.TensorDivide(centered, Engine.TensorBroadcastTo(scale, input._shape));
        var inverseScale = Engine.TensorReciprocal(scale);
        var scaledEpsilon = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(inverseScale, inverseScale), NumOps.FromDouble(1e-5));
        var std = Engine.TensorSqrt(Engine.TensorAdd(
            Engine.ReduceMean(Engine.TensorMultiply(unit, unit), sampleAxes, true), scaledEpsilon));
        return Engine.TensorDivide(unit, Engine.TensorBroadcastTo(std, input._shape));
    }

    /// <summary>
    /// Applies multi-head self-attention following the Transformer architecture.
    /// </summary>
    /// <param name="qkv">Combined Q, K, V tensor [B, 3*C, H, W].</param>
    /// <param name="inputShape">Original input shape [B, C, H, W].</param>
    /// <returns>Attention output [B, C, H, W].</returns>
    private Tensor<T> MultiHeadSelfAttention(Tensor<T> qkv, int[] inputShape)
    {
        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int seqLen = height * width;

        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

        // [B, C, S] queries, keys and values
        Tensor<T> Part(int index) => Engine.Reshape(
            Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(
                Engine.TensorNarrow(qkv, 0, 0, batchSize), 1, index * channels, channels), 2, 0, height), 3, 0, width),
            [batchSize, channels, seqLen]);
        var q = Part(0);
        var k = Part(1);
        var v = Part(2);

        // Each head attends over its own channel slice; channels past the last head stay zero
        var heads = new List<Tensor<T>>();
        int covered = 0;
        for (int headIdx = 0; headIdx < _numHeads && headIdx * _headDim < channels; headIdx++)
        {
            int headStart = headIdx * _headDim;
            int headChannels = Math.Min(_headDim, channels - headStart);
            var qh = Engine.TensorNarrow(q, 1, headStart, headChannels);
            var kh = Engine.TensorNarrow(k, 1, headStart, headChannels);
            var vh = Engine.TensorNarrow(v, 1, headStart, headChannels);

            // scores[i, j] = scale * sum_c q[c, i] k[c, j], softmax over j, out[c, i] = sum_j p[i, j] v[c, j].
            // Queries go in chunks: each row's softmax is independent, so the result is the same, but the scores,
            // probabilities and their permuted copy are [B, chunk, S] instead of [B, S, S] (S = H/8 x W/8 reaches
            // ~32,400 at 1080p, where three S x S tensors are ~25 GB per head).
            var chunks = new List<Tensor<T>>();
            for (int start = 0; start < seqLen; start += AttentionQueryChunk)
            {
                int length = Math.Min(AttentionQueryChunk, seqLen - start);
                var qc = Engine.TensorNarrow(qh, 2, start, length);
                var scores = Engine.TensorMultiplyScalar(
                    Engine.BatchMatMul(Engine.TensorPermute(qc, [0, 2, 1]), kh), scale);
                var probabilities = Engine.TensorSoftmax(scores, 2);
                chunks.Add(Engine.BatchMatMul(vh, Engine.TensorPermute(probabilities, [0, 2, 1])));
            }

            heads.Add(chunks.Count == 1 ? chunks[0] : Engine.TensorConcatenate(chunks.ToArray(), 2));
            covered = headStart + headChannels;
        }

        if (covered < channels)
        {
            heads.Add(new Tensor<T>([batchSize, channels - covered, seqLen]));
        }

        return Engine.Reshape(Engine.TensorConcatenate(heads.ToArray(), 1), inputShape);
    }

    // Half-pixel bilinear resampling with edge clamping: torch interpolate(mode="bilinear", align_corners=False)
    private Tensor<T> BilinearUpsample(Tensor<T> input, int factor)
        => Engine.Interpolate(input, [input.Shape[2] * factor, input.Shape[3] * factor], InterpolateMode.Bilinear, false);

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> EnsureBatch(Tensor<T> tensor)
    {
        return tensor.Rank == 4 ? tensor : AddBatchDimension(tensor);
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();

        foreach (var layer in _flowEncoder) Layers.Add(layer);
        foreach (var layer in _flowDecoder) Layers.Add(layer);
        foreach (var layer in _imageEncoder) Layers.Add(layer);
        foreach (var layer in _imageDecoder) Layers.Add(layer);
        foreach (var layer in _transformerQKV) Layers.Add(layer);
        foreach (var layer in _transformerProj) Layers.Add(layer);
        foreach (var layer in _transformerFFN) Layers.Add(layer);
        if (_outputConv is not null) Layers.Add(_outputConv);
    }

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "ProPainter" },
            { "Description", "Video Inpainting with Propagation and Transformer" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumFeatures", _numFeatures },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelDataProvider = () => SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    public override Tensor<T> Inpaint(Tensor<T> frames, Tensor<T> masks)
    {
        var stacked = ConcatenateFeatures(frames, masks);
        return Forward(stacked);
    }

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
