using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Motion;

/// <summary>
/// GMFlow (Global Matching Flow) for accurate optical flow estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GMFlow estimates how pixels move between video frames using
/// a global matching approach. Unlike local methods that only look at small neighborhoods,
/// GMFlow considers the entire image when matching pixels, making it better at:
/// - Large displacements (fast motion)
/// - Textureless regions
/// - Occlusions and disocclusions
/// - Repetitive patterns
///
/// The output is a "flow field" where each pixel has (dx, dy) values indicating
/// where that pixel moved to in the next frame.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Transformer-based global matching architecture
/// - Cross-attention for finding correspondences
/// - Hierarchical refinement for sub-pixel accuracy
/// - Self-attention for context aggregation
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "GMFlow: Learning Optical Flow via Global Matching"
/// CVPR 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GMFlow model for global matching optical flow estimation
/// var gmFlow = new GMFlow&lt;double&gt;();
///
/// // Or configure with custom architecture
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new GMFlow&lt;double&gt;(architecture, numFeatures: 128);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GMFlow: Learning Optical Flow via Global Matching",
    "https://arxiv.org/abs/2111.13680",
    Year = 2022,
    Authors = "Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Dacheng Tao")]
public class GMFlow<T> : OpticalFlowBase<T>
{
    private readonly GMFlowOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _numTransformerLayers;
    private readonly int _numHeads;

    // Feature encoder
    private readonly List<ConvolutionalLayer<T>> _encoder;

    // Transformer layers for global matching
    private readonly List<ConvolutionalLayer<T>> _selfAttention;
    private readonly List<ConvolutionalLayer<T>> _crossAttention;

    // Flow decoder
    private readonly List<ConvolutionalLayer<T>> _flowDecoder;
    private ConvolutionalLayer<T>? _flowHead;

    // Refinement module
    private readonly List<ConvolutionalLayer<T>> _refinement;

    // Guards the one-time real-graph lazy-shape warmup (see ResolveLazyLayerShapes).
    private bool _lazyShapesWarmed;

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
    /// Gets the number of transformer layers.
    /// </summary>
    internal int NumTransformerLayers => _numTransformerLayers;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public GMFlow()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 2))
    {
    }

    public GMFlow(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 128,
        int numTransformerLayers = 6,
        int numHeads = 8,
        GMFlowOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new GMFlowOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _numTransformerLayers = numTransformerLayers;
        _numHeads = numHeads;

        _encoder = [];
        _selfAttention = [];
        _crossAttention = [];
        _flowDecoder = [];
        _refinement = [];

        // Check for user-provided custom layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateGMFlowLayers(
                channels: _channels, height: _height, width: _width,
                numFeatures: _numFeatures, numTransformerLayers: _numTransformerLayers).ToList();
            Layers.AddRange(layers);
        }

        // Distribute layers to sub-lists for the forward pass.
        ExtractLayerReferences();
    }

    /// <summary>
    /// (Re)builds the sub-list references (<see cref="_encoder"/>,
    /// <see cref="_selfAttention"/>, etc.) that the forward pass uses, from the
    /// canonical <see cref="NeuralNetworks.NeuralNetworkBase{T}.Layers"/> list.
    /// Must be called both after the layers are built in the constructor and after
    /// deserialization replaces <c>Layers</c> with the loaded weights — otherwise a
    /// clone would keep running the constructor's random-init layers while the loaded
    /// weights sit unused (Clone_ShouldProduceIdenticalOutput / Clone_AfterTraining).
    /// Idempotent.
    /// </summary>
    private void ExtractLayerReferences()
    {
        _encoder.Clear();
        _selfAttention.Clear();
        _crossAttention.Clear();
        _flowDecoder.Clear();
        _refinement.Clear();

        int idx = 0;
        // Encoder (6 layers)
        for (int i = 0; i < 6; i++)
            _encoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Transformer layers: 4 per layer (2 self-attn + 2 cross-attn)
        for (int i = 0; i < _numTransformerLayers; i++)
        {
            _selfAttention.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _selfAttention.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _crossAttention.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _crossAttention.Add((ConvolutionalLayer<T>)Layers[idx++]);
        }
        // Flow decoder (2 layers)
        _flowDecoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        _flowDecoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Flow head
        _flowHead = (ConvolutionalLayer<T>)Layers[idx++];
        // Refinement (3 layers)
        for (int i = 0; i < 3; i++)
            _refinement.Add((ConvolutionalLayer<T>)Layers[idx++]);
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Estimates optical flow between two frames.
    /// </summary>
    public override Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        // The ModelFamily harness feeds rank-3 [C, H, W] unbatched frames, but the
        // conv/attention pipeline needs rank-4 [B, C, H, W]. Promote via a tape-aware
        // reshape (NOT a CopyTo, which severs the autodiff tape) and squeeze the batch
        // dim back off the output below.
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = Engine.Reshape(frame1, [1, frame1.Shape[0], frame1.Shape[1], frame1.Shape[2]]);
            frame2 = Engine.Reshape(frame2, [1, frame2.Shape[0], frame2.Shape[1], frame2.Shape[2]]);
        }

        // Extract features
        var feat1 = EncodeFeatures(frame1);
        var feat2 = EncodeFeatures(frame2);

        // Global matching with transformers
        var (matchedFeat1, matchedFeat2) = GlobalMatching(feat1, feat2);

        // Decode flow
        var coarseFlow = DecodeFlow(matchedFeat1, matchedFeat2);

        // Upsample and refine
        var refinedFlow = RefineFlow(frame1, frame2, coarseFlow);

        if (!hasBatch)
        {
            refinedFlow = Engine.Reshape(refinedFlow,
                [refinedFlow.Shape[1], refinedFlow.Shape[2], refinedFlow.Shape[3]]);
        }

        return refinedFlow;
    }

    /// <summary>
    /// Computes forward and backward flow for consistency checking.
    /// </summary>
    public (Tensor<T> Forward, Tensor<T> Backward) EstimateBidirectionalFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        var forward = EstimateFlow(frame1, frame2);
        var backward = EstimateFlow(frame2, frame1);
        return (forward, backward);
    }

    /// <summary>
    /// Estimates flow with occlusion mask.
    /// </summary>
    public (Tensor<T> Flow, Tensor<T> Occlusion) EstimateFlowWithOcclusion(Tensor<T> frame1, Tensor<T> frame2)
    {
        var (forward, backward) = EstimateBidirectionalFlow(frame1, frame2);
        var occlusion = ComputeOcclusionMask(forward, backward);
        return (forward, occlusion);
    }

    /// <summary>
    /// Warps an image using the estimated flow.
    /// </summary>
    public Tensor<T> WarpImage(Tensor<T> image, Tensor<T> flow)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
            flow = AddBatchDimension(flow);
        }

        var warped = BilinearWarp(image, flow);

        if (!hasBatch)
        {
            warped = RemoveBatchDimension(warped);
        }

        return warped;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => ForwardPair(input);

    /// <inheritdoc/>
    /// <remarks>
    /// GMFlow's real computation graph is <see cref="EstimateFlow"/> (encode →
    /// global matching → decode → refine), not a sequential pass over the flat
    /// <c>Layers</c> list — the attention / warp / concat stages interleave non-layer
    /// tensor ops. The base <see cref="NeuralNetworks.NeuralNetworkBase{T}.ForwardForTraining"/>
    /// runs the layers in order, which produces channel-count mismatches. Route the
    /// training forward through the same tape-connected graph Predict uses so gradients
    /// actually flow (GradientFlow / Training_ShouldReduceLoss).
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardPair(input);

    /// <summary>
    /// Splits a concatenated frame pair and runs the flow-estimation graph.
    /// Expects <c>[B, 2C, H, W]</c> (batched) or <c>[2C, H, W]</c> (unbatched); the
    /// channel split uses a tape-aware <see cref="IEngine.TensorSlice{T}"/> so the
    /// autodiff tape is preserved end-to-end.
    /// </summary>
    private Tensor<T> ForwardPair(Tensor<T> input)
    {
        if (input.Rank == 3)
        {
            int c3 = input.Shape[0] / 2;
            var f1 = Engine.TensorSlice(input, [0, 0, 0], [c3, input.Shape[1], input.Shape[2]]);
            var f2 = Engine.TensorSlice(input, [c3, 0, 0], [c3, input.Shape[1], input.Shape[2]]);
            return EstimateFlow(f1, f2);
        }

        int batchSize = input.Shape[0];
        int channels = input.Shape[1] / 2;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var frame1 = Engine.TensorSlice(input, [0, 0, 0, 0], [batchSize, channels, height, width]);
        var frame2 = Engine.TensorSlice(input, [0, channels, 0, 0], [batchSize, channels, height, width]);

        return EstimateFlow(frame1, frame2);
    }

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

    private Tensor<T> EncodeFeatures(Tensor<T> input)
    {
        var features = input;
        foreach (var layer in _encoder)
        {
            features = layer.Forward(features);
            features = Engine.ReLU(features);
        }
        return features;
    }

    private (Tensor<T>, Tensor<T>) GlobalMatching(Tensor<T> feat1, Tensor<T> feat2)
    {
        var f1 = feat1;
        var f2 = feat2;

        for (int i = 0; i < _numTransformerLayers; i++)
        {
            // Self-attention on each feature (query/key projected from the SAME feature,
            // value = feature). Residual connection per the Transformer block, which also
            // keeps gradients flowing through the stack (GradientFlow).
            int selfIdx = i * 2;
            var q1 = _selfAttention[selfIdx].Forward(f1);
            var k1 = _selfAttention[selfIdx + 1].Forward(f1);
            f1 = AddTensors(f1, SpatialAttention(q1, k1, f1));

            var q2 = _selfAttention[selfIdx].Forward(f2);
            var k2 = _selfAttention[selfIdx + 1].Forward(f2);
            f2 = AddTensors(f2, SpatialAttention(q2, k2, f2));

            // Cross-attention between features: GMFlow's global matching. Each feature
            // queries the OTHER feature (query proj on self, key proj on the other, value =
            // the other feature). This is genuine cross-attention — NOT a conv over the
            // channel-concatenation of both features — so every projection conv sees a
            // consistent numFeatures-channel input.
            //
            // BOTH updates are computed from the PRE-update f1/f2 and applied
            // simultaneously. Updating f1 first and then feeding the new f1 into f2's
            // update would break the symmetry between the two branches: for identical
            // input frames (feat1 == feat2) the two features must stay equal through the
            // whole stack so the decoded flow is ~zero (IdenticalFrames_NearZeroFlow).
            int crossIdx = i * 2;
            var q1c = _crossAttention[crossIdx].Forward(f1);
            var k2c = _crossAttention[crossIdx + 1].Forward(f2);
            var q2c = _crossAttention[crossIdx].Forward(f2);
            var k1c = _crossAttention[crossIdx + 1].Forward(f1);

            var f1Next = AddTensors(f1, SpatialAttention(q1c, k2c, f2));
            var f2Next = AddTensors(f2, SpatialAttention(q2c, k1c, f1));
            f1 = f1Next;
            f2 = f2Next;
        }

        return (f1, f2);
    }

    /// <summary>
    /// Applies global scaled dot-product attention over all spatial positions.
    /// </summary>
    /// <param name="query">Query tensor [batch, channels, height, width].</param>
    /// <param name="key">Key tensor [batch, channels, height, width].</param>
    /// <param name="value">Value tensor [batch, channels, height, width].</param>
    /// <returns>Attention output tensor [batch, channels, height, width].</returns>
    /// <remarks>
    /// Implements Attention(Q, K, V) = softmax(Q · Kᵀ / sqrt(d_k)) · V entirely with
    /// tape-aware <see cref="IEngine"/> ops (reshape → permute → BatchMatMul → softmax
    /// → BatchMatMul → reshape) so gradients flow to the Q/K projection convolutions.
    /// GMFlow performs GLOBAL matching (attention across every spatial position, not a
    /// local window), which is exactly this dense HW×HW attention.
    /// <para>
    /// <b>Memory:</b> the score/attention tensors are dense <c>[B, HW, HW]</c>, so memory grows
    /// as O(B·(HW)²) in the feature-map resolution. GMFlow runs this on the 1/8-downsampled
    /// encoder features, so a 512×512 input becomes a 64×64 (HW=4096) map — already ~134 M score
    /// entries per batch item. To keep the OOM from surfacing as an opaque allocation failure deep
    /// inside <see cref="IEngine.BatchMatMul"/>, we guard the token count up front and throw an
    /// actionable message naming the resolution. Callers needing higher resolution should tile the
    /// image or switch to a windowed/local-attention variant.
    /// </para>
    /// </remarks>
    private Tensor<T> SpatialAttention(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        int b = query.Shape[0];
        int c = query.Shape[1];
        int h = query.Shape[2];
        int w = query.Shape[3];
        int hw = h * w;

        // Dense global attention materializes a [B, HW, HW] score matrix. Guard the token count so
        // an over-large feature map fails with a clear diagnostic instead of an opaque OOM inside
        // BatchMatMul/Softmax. 16384 tokens (e.g. 128×128) => ~2.1e8 float scores per batch item.
        const int MaxAttentionTokens = 16384;
        if (hw > MaxAttentionTokens)
        {
            throw new InvalidOperationException(
                $"GMFlow global attention feature map is {h}×{w} ({hw} tokens), which exceeds the " +
                $"{MaxAttentionTokens}-token limit for the dense O(HW²) score matrix (~{(long)hw * hw} " +
                "entries per batch item). Reduce the input resolution, tile the image, or use a " +
                "windowed/local-attention variant for higher resolutions.");
        }

        // [B, C, H, W] -> [B, C, HW] -> [B, HW, C] (spatial positions as tokens)
        var q = Engine.TensorPermute(Engine.Reshape(query, [b, c, hw]), [0, 2, 1]);
        var k = Engine.TensorPermute(Engine.Reshape(key, [b, c, hw]), [0, 2, 1]);
        var v = Engine.TensorPermute(Engine.Reshape(value, [b, c, hw]), [0, 2, 1]);

        // scores = Q · Kᵀ / sqrt(d_k)  -> [B, HW, HW]
        var kT = Engine.TensorPermute(k, [0, 2, 1]);            // [B, C, HW]
        var scores = Engine.BatchMatMul(q, kT);                 // [B, HW, HW]
        scores = Engine.TensorMultiplyScalar(scores, NumOps.FromDouble(1.0 / Math.Sqrt(c)));

        var attn = Engine.Softmax(scores, axis: -1);            // [B, HW, HW]
        var outTokens = Engine.BatchMatMul(attn, v);            // [B, HW, C]

        // back to [B, C, H, W]
        var outCHW = Engine.TensorPermute(outTokens, [0, 2, 1]); // [B, C, HW]
        return Engine.Reshape(outCHW, [b, c, h, w]);
    }

    private Tensor<T> DecodeFlow(Tensor<T> feat1, Tensor<T> feat2)
    {
        var diff = Engine.TensorSubtract(feat1, feat2);

        foreach (var layer in _flowDecoder)
        {
            diff = layer.Forward(diff);
            diff = Engine.ReLU(diff);
        }

        var flowHead = _flowHead ?? throw new InvalidOperationException("Flow head has not been initialized.");
        return flowHead.Forward(diff);
    }

    private Tensor<T> RefineFlow(Tensor<T> frame1, Tensor<T> frame2, Tensor<T> coarseFlow)
    {
        // Upsample coarse flow to the ACTUAL input resolution (frame1's H/W), NOT the
        // constructor's _height/_width. The encoder downsamples 8x, so coarseFlow is at
        // H/8 x W/8; the refinement stage concatenates it with the full-resolution frame
        // pair, which requires matching spatial dims. Using _height/_width upsampled the
        // flow to the fixed default (e.g. 256) while the harness feeds a smaller frame
        // (e.g. 64), producing a "concatenation axis mismatch 64 vs 256" crash.
        int targetH = frame1.Shape[2];
        int targetW = frame1.Shape[3];
        var upFlow = UpsampleFlow(coarseFlow, targetH, targetW);

        // Concatenate inputs
        var concat = ConcatenateChannels(frame1, frame2);
        concat = ConcatenateChannels(concat, upFlow);

        // Refine. Each conv already applies its own activation (ReLU on the two hidden
        // convs, linear on the final flow-residual conv), so no extra explicit ReLU is
        // added here — an explicit ReLU on the final output would re-clamp the signed
        // flow residual to >= 0 and collapse input sensitivity.
        var residual = concat;
        foreach (var layer in _refinement)
        {
            residual = layer.Forward(residual);
        }

        return AddTensors(upFlow, residual);
    }

    private Tensor<T> UpsampleFlow(Tensor<T> flow, int targetH, int targetW)
    {
        int srcH = flow.Shape[2];
        int srcW = flow.Shape[3];
        if (srcH == targetH && srcW == targetW)
            return flow;

        // Bilinearly resize the flow field (tape-aware), then scale the displacement
        // magnitudes by the resolution ratio: a flow vector measured in coarse-grid
        // pixels must be multiplied by target/src when re-expressed on the fine grid.
        // The encoder downsamples H and W by the same factor (three stride-2 convs), so
        // the vertical and horizontal ratios are equal and a single scalar is exact.
        var upsampled = Engine.Interpolate(flow, [targetH, targetW], InterpolateMode.Bilinear, alignCorners: false);
        double ratio = (double)targetH / srcH;
        return Engine.TensorMultiplyScalar(upsampled, NumOps.FromDouble(ratio));
    }

    private Tensor<T> ComputeOcclusionMask(Tensor<T> forward, Tensor<T> backward)
    {
        int batchSize = forward.Shape[0];
        int height = forward.Shape[2];
        int width = forward.Shape[3];

        var occlusion = new Tensor<T>([batchSize, 1, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double fx = Convert.ToDouble(forward[b, 0, h, w]);
                    double fy = Convert.ToDouble(forward[b, 1, h, w]);

                    int tx = Math.Max(0, Math.Min((int)(w + fx), width - 1));
                    int ty = Math.Max(0, Math.Min((int)(h + fy), height - 1));

                    double bx = Convert.ToDouble(backward[b, 0, ty, tx]);
                    double by = Convert.ToDouble(backward[b, 1, ty, tx]);

                    double consistency = Math.Sqrt((fx + bx) * (fx + bx) + (fy + by) * (fy + by));
                    double occ = consistency > 1.0 ? 1.0 : 0.0;

                    occlusion[b, 0, h, w] = NumOps.FromDouble(occ);
                }
            }
        }

        return occlusion;
    }

    private Tensor<T> BilinearWarp(Tensor<T> image, Tensor<T> flow)
    {
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var warped = new Tensor<T>(image._shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double srcX = w + Convert.ToDouble(flow[b, 0, h, w]);
                    double srcY = h + Convert.ToDouble(flow[b, 1, h, w]);

                    int x0 = (int)Math.Floor(srcX);
                    int y0 = (int)Math.Floor(srcY);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    x0 = Math.Max(0, Math.Min(x0, width - 1));
                    x1 = Math.Max(0, Math.Min(x1, width - 1));
                    y0 = Math.Max(0, Math.Min(y0, height - 1));
                    y1 = Math.Max(0, Math.Min(y1, height - 1));

                    double dx = srcX - Math.Floor(srcX);
                    double dy = srcY - Math.Floor(srcY);

                    for (int c = 0; c < channels; c++)
                    {
                        double v00 = Convert.ToDouble(image[b, c, y0, x0]);
                        double v01 = Convert.ToDouble(image[b, c, y0, x1]);
                        double v10 = Convert.ToDouble(image[b, c, y1, x0]);
                        double v11 = Convert.ToDouble(image[b, c, y1, x1]);

                        double val = v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) +
                                     v10 * (1 - dx) * dy + v11 * dx * dy;
                        warped[b, c, h, w] = NumOps.FromDouble(val);
                    }
                }
            }
        }

        return warped;
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate([a, b], axis: 1);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        Engine.TensorAdd(a, b);

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

        foreach (var layer in _encoder) Layers.Add(layer);
        foreach (var layer in _selfAttention) Layers.Add(layer);
        foreach (var layer in _crossAttention) Layers.Add(layer);
        foreach (var layer in _flowDecoder) Layers.Add(layer);
        if (_flowHead is not null) Layers.Add(_flowHead);
        foreach (var layer in _refinement) Layers.Add(layer);
    }

    private bool _shapesProbed;

    /// <inheritdoc/>
    /// <remarks>
    /// GMFlow's real graph is <see cref="EstimateFlow"/> (encode → global self/cross-attention
    /// matching → decode → refine), not a sequential pass over the flat Layers list — so the base
    /// linear walk mis-sizes the matching/decoder convs and the real forward throws
    /// "Expected input depth 3, but got 6". Resolve every lazy conv through a real forward on a small
    /// dummy frame-pair instead, so each conv sees exactly the input its production forward feeds it.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_shapesProbed || _encoder.Count == 0) return;
        _shapesProbed = true;
        int c = _channels > 0 ? _channels : 3;
        // Small 32×32 probe keeps the attention/matching resolution pass cheap enough to stay well
        // under the per-test construction budget while still exercising every conv.
        _ = PredictCore(new Tensor<T>([1, c * 2, 32, 32]));
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "GMFlow" },
            { "Description", "Global Matching Optical Flow" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumTransformerLayers", _numTransformerLayers }
        },
        ModelData = SerializeForMetadata()
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFeatures);
        writer.Write(_numTransformerLayers);
        writer.Write(_numHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        for (int i = 0; i < 6; i++) _ = reader.ReadInt32();

        // The base deserializer has replaced Layers with the loaded-weight layers;
        // re-point the forward-pass sub-lists at them so a clone/reload runs the
        // trained weights rather than the constructor's random-init layers
        // (Clone_ShouldProduceIdenticalOutput / Clone_AfterTraining).
        ExtractLayerReferences();
    }

    /// <summary>
    /// Resolves each convolution's lazy input depth by running the REAL computation
    /// graph once, instead of the base's sequential walk over <c>Layers</c>.
    /// </summary>
    /// <remarks>
    /// GMFlow's flat <c>Layers</c> list is not a valid sequential forward: the
    /// refinement stage consumes a channel-concatenation of the frame pair plus the
    /// upsampled flow (2C + 2 = 8 channels for the default 3-channel frames), but the
    /// base's sequential shape walk would feed the first refinement conv only the flow
    /// head's 2-channel output and lock its input depth to 2 — so the next real forward
    /// throws "Expected input depth 2, but got 8". Because a convolution's input depth
    /// depends only on channel count (not spatial size), a single forward over a small
    /// dummy frame pair resolves every layer correctly and cheaply; delegating to the
    /// base afterwards just marks the walk complete (every layer is already resolved, so
    /// its sequential pass is a no-op).
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesWarmed) return;
        _lazyShapesWarmed = true; // set first so any reentrancy is a no-op

        // Resolve every conv's input depth via the REAL graph. Deliberately does NOT
        // delegate to base.ResolveLazyLayerShapes(): the base's sequential walk over the
        // flat Layers list would re-resolve the first refinement conv from the flow
        // head's 2-channel output and clobber the correct 8-channel depth this forward
        // establishes.
        const int dummyHW = 64;   // comfortably above the encoder's 8x downsample floor
        var dummy = new Tensor<T>([1, _channels * 2, dummyHW, dummyHW]);
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try { ForwardPair(dummy); }
        catch { /* best-effort; a genuine forward failure surfaces on the real Train/Predict */ }
        finally { if (wasTraining) SetTrainingMode(true); }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new GMFlow<T>(Architecture, _numFeatures, _numTransformerLayers, _numHeads);

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
        return modelOutput;
    }

    #endregion

}
