using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
public class ProPainter<T> : VideoInpaintingBase<T>
{
    private readonly ProPainterOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// <param name="numTransformerBlocks">Number of transformer blocks for global attention. Default: 6 (paper standard).</param>
    /// <param name="numHeads">Number of attention heads per transformer block. Default: 8 (paper standard).</param>
    public ProPainter(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 128,
        int numTransformerBlocks = 6,
        int numHeads = 8,
        ProPainterOptions? options = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _options = options ?? new ProPainterOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _numTransformerBlocks = numTransformerBlocks;
        _numHeads = numHeads;
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For single-frame prediction, use zero mask (no inpainting)
        var mask = new Tensor<T>(input.Shape);
        return InpaintFrame(input, mask, input, input, mask, mask);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var predicted = Predict(input);
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));

        BackwardPass(lossGradient);

        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(lr);
        }
    }

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

        // Distribute layers to sub-lists for forward pass
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
        var encoderOutputs = new List<Tensor<T>>();
        foreach (var encoder in _imageEncoder)
        {
            imageFeatures = encoder.Forward(imageFeatures);
            imageFeatures = ApplyReLU(imageFeatures);
            encoderOutputs.Add(imageFeatures);
        }

        // Step 4: Apply transformer blocks for global context (Multi-Head Self-Attention)
        var transformedFeatures = imageFeatures;
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            // Pre-norm (layer normalization approximation via standardization)
            var normed = LayerNorm(transformedFeatures);

            // Multi-head self-attention
            var qkv = _transformerQKV[i].Forward(normed);
            var attended = MultiHeadSelfAttention(qkv, transformedFeatures.Shape);
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

        var output = _outputConv!.Forward(decoded);

        // Step 6: Blend original and inpainted regions using mask
        return BlendWithMask(frame, output, mask);
    }

    private Tensor<T> CompleteFlow(Tensor<T> src, Tensor<T> dst, Tensor<T> mask)
    {
        int batchSize = src.Shape[0];
        int height = _height;
        int width = _width;

        // Initialize coarse flow (zero)
        var flow = new Tensor<T>([batchSize, 2, height, width]);

        // Simple optical flow estimation (differential)
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    // Compute gradients
                    double ix = Convert.ToDouble(src[b, 0, h, w + 1]) - Convert.ToDouble(src[b, 0, h, w - 1]);
                    double iy = Convert.ToDouble(src[b, 0, h + 1, w]) - Convert.ToDouble(src[b, 0, h - 1, w]);
                    double it = Convert.ToDouble(dst[b, 0, h, w]) - Convert.ToDouble(src[b, 0, h, w]);

                    // Simple flow computation
                    double denom = ix * ix + iy * iy + 1e-8;
                    flow[b, 0, h, w] = NumOps.FromDouble(-it * ix / denom);
                    flow[b, 1, h, w] = NumOps.FromDouble(-it * iy / denom);
                }
            }
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

    private Tensor<T> WarpImage(Tensor<T> image, Tensor<T> flow)
    {
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var result = new Tensor<T>(image.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double dx = Convert.ToDouble(flow[b, 0, h, w]);
                    double dy = Convert.ToDouble(flow[b, 1, h, w]);

                    double srcH = h + dy;
                    double srcW = w + dx;

                    for (int c = 0; c < channels; c++)
                    {
                        result[b, c, h, w] = BilinearSample(image, b, c, srcH, srcW, height, width);
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyMask(Tensor<T> image, Tensor<T> mask)
    {
        // Zero out masked regions
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var result = new Tensor<T>(image.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double m = Convert.ToDouble(mask[b, 0, h, w]);
                    for (int c = 0; c < channels; c++)
                    {
                        result[b, c, h, w] = m > 0.5
                            ? NumOps.Zero
                            : image[b, c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> BlendWithMask(Tensor<T> original, Tensor<T> inpainted, Tensor<T> mask)
    {
        int batchSize = original.Shape[0];
        int channels = original.Shape[1];
        int height = original.Shape[2];
        int width = original.Shape[3];

        var result = new Tensor<T>(original.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double m = Convert.ToDouble(mask[b, 0, h, w]);
                    for (int c = 0; c < channels; c++)
                    {
                        T orig = original[b, c, h, w];
                        T inp = inpainted[b, c, h, w];
                        // blend = m * inpainted + (1-m) * original
                        result[b, c, h, w] = NumOps.Add(
                            NumOps.Multiply(inp, NumOps.FromDouble(m)),
                            NumOps.Multiply(orig, NumOps.FromDouble(1 - m)));
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> SingleChannelMask(Tensor<T> mask)
    {
        // Ensure mask is single channel
        if (mask.Shape[1] == 1) return mask;

        int batchSize = mask.Shape[0];
        int height = mask.Shape[2];
        int width = mask.Shape[3];

        var result = new Tensor<T>([batchSize, 1, height, width]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[b, 0, h, w] = mask[b, 0, h, w];
                }
            }
        }

        return result;
    }

    private Tensor<T> ConcatenateChannelsDim1(Tensor<T> t1, Tensor<T> t2)
    {
        int batchSize = t1.Shape[0];
        int c1 = t1.Shape[1];
        int c2 = t2.Shape[1];
        int height = t1.Shape[2];
        int width = t1.Shape[3];

        var result = new Tensor<T>([batchSize, c1 + c2, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < c1; c++)
                        result[b, c, h, w] = t1[b, c, h, w];
                    for (int c = 0; c < c2; c++)
                        result[b, c1 + c, h, w] = t2[b, c, h, w];
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            return NumOps.FromDouble(Math.Max(0, x));
        });
    }

    private Tensor<T> ApplyGELU(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double c = Math.Sqrt(2.0 / Math.PI);
            double gelu = 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
            return NumOps.FromDouble(gelu);
        });
    }

    /// <summary>
    /// Element-wise addition of two tensors (residual connection).
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));
    }

    /// <summary>
    /// Applies layer normalization (standardization) to input tensor.
    /// </summary>
    private Tensor<T> LayerNorm(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var result = new Tensor<T>(input.Shape);
        const double eps = 1e-5;

        for (int b = 0; b < batchSize; b++)
        {
            // Compute mean and variance across C, H, W
            double sum = 0;
            double sumSq = 0;
            int count = channels * height * width;

            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double val = Convert.ToDouble(input[b, c, h, w]);
                        sum += val;
                        sumSq += val * val;
                    }
                }
            }

            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double std = Math.Sqrt(variance + eps);

            // Normalize
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double val = Convert.ToDouble(input[b, c, h, w]);
                        result[b, c, h, w] = NumOps.FromDouble((val - mean) / std);
                    }
                }
            }
        }

        return result;
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

        var output = new Tensor<T>(inputShape);
        double scale = 1.0 / Math.Sqrt(_headDim);

        for (int b = 0; b < batchSize; b++)
        {
            // Split QKV into separate tensors
            var q = new double[channels, seqLen];
            var k = new double[channels, seqLen];
            var v = new double[channels, seqLen];

            for (int c = 0; c < channels; c++)
            {
                int pos = 0;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        q[c, pos] = Convert.ToDouble(qkv[b, c, h, w]);
                        k[c, pos] = Convert.ToDouble(qkv[b, channels + c, h, w]);
                        v[c, pos] = Convert.ToDouble(qkv[b, channels * 2 + c, h, w]);
                        pos++;
                    }
                }
            }

            // Multi-head attention
            var attOutput = new double[channels, seqLen];

            for (int headIdx = 0; headIdx < _numHeads; headIdx++)
            {
                int headStart = headIdx * _headDim;
                int headEnd = Math.Min(headStart + _headDim, channels);

                // Compute attention scores for this head
                var scores = new double[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    double maxScore = double.MinValue;
                    for (int j = 0; j < seqLen; j++)
                    {
                        double score = 0;
                        for (int c = headStart; c < headEnd; c++)
                        {
                            score += q[c, i] * k[c, j];
                        }
                        score *= scale;
                        scores[i, j] = score;
                        if (score > maxScore) maxScore = score;
                    }

                    // Softmax normalization
                    double sumExp = 0;
                    for (int j = 0; j < seqLen; j++)
                    {
                        scores[i, j] = Math.Exp(scores[i, j] - maxScore);
                        sumExp += scores[i, j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        scores[i, j] /= Math.Max(sumExp, 1e-12);
                    }
                }

                // Apply attention weights to values
                for (int c = headStart; c < headEnd; c++)
                {
                    for (int i = 0; i < seqLen; i++)
                    {
                        double weightedSum = 0;
                        for (int j = 0; j < seqLen; j++)
                        {
                            weightedSum += scores[i, j] * v[c, j];
                        }
                        attOutput[c, i] = weightedSum;
                    }
                }
            }

            // Copy to output tensor
            for (int c = 0; c < channels; c++)
            {
                int pos = 0;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        output[b, c, h, w] = NumOps.FromDouble(attOutput[c, pos]);
                        pos++;
                    }
                }
            }
        }

        return output;
    }

    private T BilinearSample(Tensor<T> tensor, int b, int c, double h, double w, int height, int width)
    {
        int h0 = (int)Math.Floor(h);
        int w0 = (int)Math.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        h0 = Math.Max(0, Math.Min(h0, height - 1));
        h1 = Math.Max(0, Math.Min(h1, height - 1));
        w0 = Math.Max(0, Math.Min(w0, width - 1));
        w1 = Math.Max(0, Math.Min(w1, width - 1));

        double hWeight = h - Math.Floor(h);
        double wWeight = w - Math.Floor(w);

        T v00 = tensor[b, c, h0, w0];
        T v01 = tensor[b, c, h0, w1];
        T v10 = tensor[b, c, h1, w0];
        T v11 = tensor[b, c, h1, w1];

        T top = NumOps.Add(
            NumOps.Multiply(v00, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v01, NumOps.FromDouble(wWeight)));
        T bottom = NumOps.Add(
            NumOps.Multiply(v10, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v11, NumOps.FromDouble(wWeight)));

        return NumOps.Add(
            NumOps.Multiply(top, NumOps.FromDouble(1 - hWeight)),
            NumOps.Multiply(bottom, NumOps.FromDouble(hWeight)));
    }

    private Tensor<T> BilinearUpsample(Tensor<T> input, int factor)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = inHeight * factor;
        int outWidth = inWidth * factor;

        var output = new Tensor<T>([batchSize, channels, outHeight, outWidth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        double srcH = (h + 0.5) / factor - 0.5;
                        double srcW = (w + 0.5) / factor - 0.5;
                        output[b, c, h, w] = BilinearSample(input, b, c, srcH, srcW, inHeight, inWidth);
                    }
                }
            }
        }

        return output;
    }

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

    private void BackwardPass(Tensor<T> gradient)
    {
        gradient = _outputConv!.Backward(gradient);

        for (int i = _imageDecoder.Count - 1; i >= 0; i--)
        {
            gradient = _imageDecoder[i].Backward(gradient);
        }

        // Backward through transformer blocks (in reverse order)
        for (int i = _numTransformerBlocks - 1; i >= 0; i--)
        {
            // Backward through FFN (contract then expand)
            gradient = _transformerFFN[i * 2 + 1].Backward(gradient);
            gradient = _transformerFFN[i * 2].Backward(gradient);

            // Backward through attention projection and QKV
            gradient = _transformerProj[i].Backward(gradient);
            gradient = _transformerQKV[i].Backward(gradient);
        }

        for (int i = _imageEncoder.Count - 1; i >= 0; i--)
        {
            gradient = _imageEncoder[i].Backward(gradient);
        }
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();
    }

    /// <inheritdoc/>
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
            ModelType = ModelType.VideoInpainting,
            AdditionalInfo = additionalInfo,
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFeatures);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ProPainter<T>(Architecture, _numFeatures, _numTransformerBlocks, _numHeads);
    }

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
