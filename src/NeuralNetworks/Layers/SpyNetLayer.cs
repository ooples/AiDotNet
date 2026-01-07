using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// SPyNet (Spatial Pyramid Network) layer for optical flow estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SPyNet uses a coarse-to-fine spatial pyramid approach to estimate optical flow
/// between two consecutive video frames. It's widely used in video super-resolution
/// and frame interpolation models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Optical flow tells us how pixels move between two frames.
/// SPyNet is a lightweight network that estimates this motion efficiently by
/// processing the images at multiple scales (pyramid levels).
///
/// The network works by:
/// 1. Building image pyramids at different resolutions
/// 2. Estimating flow at the coarsest level first
/// 3. Refining the flow at each finer level
/// 4. Combining all levels for the final flow
/// </para>
/// <para>
/// <b>Reference:</b> Ranjan and Black, "Optical Flow Estimation using a Spatial Pyramid Network",
/// CVPR 2017. https://arxiv.org/abs/1611.00850
/// </para>
/// </remarks>
public class SpyNetLayer<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    #region Fields

    private readonly IEngine _engine;
    private readonly int _numLevels;
    private readonly int _inputChannels;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly List<ConvolutionalLayer<T>> _basicModules;
    private Tensor<T>? _lastInput1;
    private Tensor<T>? _lastInput2;
    private Tensor<T>? _lastFlow;

    // Cached values for backward pass
    private readonly List<Tensor<T>> _cachedPyramid1 = [];
    private readonly List<Tensor<T>> _cachedPyramid2 = [];
    private readonly List<Tensor<T>> _cachedWarped = [];
    private readonly List<Tensor<T>> _cachedFlows = [];
    private readonly List<Tensor<T>> _cachedModuleInputs = [];
    private readonly List<Tensor<T>> _cachedGrids = [];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new SPyNet layer for optical flow estimation.
    /// </summary>
    /// <param name="inputHeight">Height of input frames.</param>
    /// <param name="inputWidth">Width of input frames.</param>
    /// <param name="inputChannels">Number of input channels (typically 3 for RGB).</param>
    /// <param name="numLevels">Number of pyramid levels (default: 5).</param>
    /// <param name="engine">Optional computation engine (CPU or GPU). If null, uses default CPU engine.</param>
    public SpyNetLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels = 3,
        int numLevels = 5,
        IEngine? engine = null)
        : base([inputChannels, inputHeight, inputWidth], [2, inputHeight, inputWidth])
    {
        _engine = engine ?? new CpuEngine();
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _numLevels = numLevels;
        _basicModules = [];

        // Initialize basic modules for each pyramid level
        // Each module processes concatenated (img1, img2, flow_estimate)
        // Input channels: 2 * inputChannels (two frames) + 2 (flow)
        int moduleInputChannels = 2 * inputChannels + 2;

        for (int i = 0; i < numLevels; i++)
        {
            // Simple 5-layer CNN for each pyramid level
            // ConvolutionalLayer(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
            var conv = new ConvolutionalLayer<T>(
                moduleInputChannels,
                inputHeight >> i,
                inputWidth >> i,
                32, // outputDepth (filters)
                7,  // kernelSize
                1,  // stride
                3); // padding
            _basicModules.Add(conv);
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Input should be concatenated [frame1, frame2] along channel dimension
        // Shape: [2*C, H, W] or [B, 2*C, H, W]
        bool hasBatch = input.Rank == 4;
        int channels = hasBatch ? input.Shape[1] : input.Shape[0];
        int height = hasBatch ? input.Shape[2] : input.Shape[1];
        int width = hasBatch ? input.Shape[3] : input.Shape[2];

        int singleFrameChannels = channels / 2;

        // Split into two frames
        var (frame1, frame2) = SplitFrames(input, singleFrameChannels, hasBatch);
        _lastInput1 = frame1;
        _lastInput2 = frame2;

        // Estimate flow using spatial pyramid
        var flow = EstimateFlow(frame1, frame2, hasBatch);
        _lastFlow = flow;

        return flow;
    }

    /// <summary>
    /// Estimates optical flow between two frames using separate tensors.
    /// </summary>
    /// <param name="frame1">First frame tensor.</param>
    /// <param name="frame2">Second frame tensor.</param>
    /// <returns>Optical flow tensor [2, H, W] representing (dx, dy) per pixel.</returns>
    public Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        _lastInput1 = frame1;
        _lastInput2 = frame2;

        bool hasBatch = frame1.Rank == 4;
        var flow = EstimateFlow(frame1, frame2, hasBatch);
        _lastFlow = flow;

        return flow;
    }

    private Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2, bool hasBatch)
    {
        int batch = hasBatch ? frame1.Shape[0] : 1;
        int height = hasBatch ? frame1.Shape[2] : frame1.Shape[1];
        int width = hasBatch ? frame1.Shape[3] : frame1.Shape[2];

        // Clear cached values for backward pass
        _cachedPyramid1.Clear();
        _cachedPyramid2.Clear();
        _cachedWarped.Clear();
        _cachedFlows.Clear();
        _cachedModuleInputs.Clear();
        _cachedGrids.Clear();

        // Build image pyramids
        var pyramid1 = BuildPyramid(frame1, hasBatch);
        var pyramid2 = BuildPyramid(frame2, hasBatch);

        // Cache pyramids for backward pass
        _cachedPyramid1.AddRange(pyramid1);
        _cachedPyramid2.AddRange(pyramid2);

        // Initialize flow at coarsest level (zeros)
        int coarseH = height >> (_numLevels - 1);
        int coarseW = width >> (_numLevels - 1);
        var flowShape = hasBatch ? new[] { batch, 2, coarseH, coarseW } : new[] { 2, coarseH, coarseW };
        var flow = new Tensor<T>(flowShape);

        // Coarse-to-fine refinement
        for (int level = _numLevels - 1; level >= 0; level--)
        {
            var img1 = pyramid1[level];
            var img2 = pyramid2[level];
            int levelH = hasBatch ? img1.Shape[2] : img1.Shape[1];
            int levelW = hasBatch ? img1.Shape[3] : img1.Shape[2];

            // Upsample flow if not at coarsest level
            if (level < _numLevels - 1)
            {
                flow = UpsampleFlow(flow, levelH, levelW, hasBatch);
            }

            // Cache flow before warping for backward pass
            _cachedFlows.Insert(0, flow);

            // Warp img2 using current flow estimate (uses IEngine.GridSample)
            var (warped2, grid) = WarpImageWithGrid(img2, flow, hasBatch);
            _cachedWarped.Insert(0, warped2);
            _cachedGrids.Insert(0, grid);

            // Concatenate inputs for basic module
            var moduleInput = ConcatenateForModule(img1, warped2, flow, hasBatch);
            _cachedModuleInputs.Insert(0, moduleInput);

            // Predict residual flow
            var residualFlow = _basicModules[level].Forward(moduleInput);

            // Extract flow channels and add residual
            flow = AddResidualFlow(flow, residualFlow, hasBatch);
        }

        return flow;
    }

    #endregion

    #region Backward Pass

    /// <inheritdoc/>
    /// <remarks>
    /// Full backward pass through the spatial pyramid with proper gradient flow:
    /// 1. Backprop through residual flow addition
    /// 2. Backprop through each pyramid level's CNN module
    /// 3. Backprop through concatenation to get gradients for warped image
    /// 4. Backprop through GridSample warping using IEngine
    /// 5. Accumulate gradients across pyramid levels
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (_lastInput1 == null || _lastInput2 == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        bool hasBatch = _lastInput1.Rank == 4;
        int batch = hasBatch ? _lastInput1.Shape[0] : 1;

        // Gradient w.r.t. final flow
        var gradFlow = gradOutput;

        // Initialize gradient accumulators for input frames
        var gradInput1 = new Tensor<T>(_lastInput1.Shape);
        var gradInput2 = new Tensor<T>(_lastInput2.Shape);

        // Fine-to-coarse gradient propagation (reverse of forward)
        for (int level = 0; level < _numLevels; level++)
        {
            // 1. Backprop through residual flow addition
            // gradFlow already contains gradient w.r.t. output flow
            // We need gradients w.r.t. input flow and residual

            // 2. Backprop through CNN module
            var moduleGrad = _basicModules[level].Backward(gradFlow);

            // 3. Backprop through concatenation
            // moduleInput was: [img1, warped2, flow]
            // Split gradient back to these components
            var (gradImg1Contrib, gradWarped2, gradFlowFromConcat) =
                SplitConcatenationGradient(moduleGrad, _cachedPyramid1[level].Shape, hasBatch);

            // 4. Backprop through GridSample warping using IEngine
            if (_cachedGrids.Count > level && _cachedPyramid2.Count > level)
            {
                // Get gradient w.r.t. input image (img2) through GridSample
                var gradImg2FromWarp = _engine.GridSampleBackwardInput(
                    gradWarped2,
                    _cachedGrids[level],
                    _cachedPyramid2[level].Shape);

                // Get gradient w.r.t. grid (which depends on flow)
                var gradGrid = _engine.GridSampleBackwardGrid(
                    gradWarped2,
                    _cachedPyramid2[level],
                    _cachedGrids[level]);

                // Convert grid gradient back to flow gradient
                var gradFlowFromWarp = ConvertGridGradientToFlowGradient(gradGrid, hasBatch);

                // Accumulate flow gradient
                gradFlow = AddTensors(gradFlowFromConcat, gradFlowFromWarp);

                // Accumulate image gradients (upsampled to full resolution later)
                AccumulatePyramidGradient(gradInput2, gradImg2FromWarp, level, hasBatch);
            }

            // Accumulate img1 gradients
            AccumulatePyramidGradient(gradInput1, gradImg1Contrib, level, hasBatch);

            // 5. Backprop through upsampling if not at coarsest level
            if (level < _numLevels - 1)
            {
                gradFlow = DownsampleFlowGradient(gradFlow, hasBatch);
            }
        }

        // Concatenate gradients for output (frame1, frame2)
        return ConcatenateFrameGradients(gradInput1, gradInput2, hasBatch);
    }

    private (Tensor<T> gradImg1, Tensor<T> gradWarped2, Tensor<T> gradFlow) SplitConcatenationGradient(
        Tensor<T> gradient, int[] imgShape, bool hasBatch)
    {
        int batch = hasBatch ? imgShape[0] : 1;
        int channels = hasBatch ? imgShape[1] : imgShape[0];
        int height = hasBatch ? gradient.Shape[^2] : gradient.Shape[^2];
        int width = hasBatch ? gradient.Shape[^1] : gradient.Shape[^1];

        var gradImg1Shape = hasBatch ? new[] { batch, channels, height, width } : new[] { channels, height, width };
        var gradFlowShape = hasBatch ? new[] { batch, 2, height, width } : new[] { 2, height, width };

        var gradImg1 = new Tensor<T>(gradImg1Shape);
        var gradWarped2 = new Tensor<T>(gradImg1Shape);
        var gradFlow = new Tensor<T>(gradFlowShape);

        int pixelsPerChannel = height * width;

        for (int b = 0; b < batch; b++)
        {
            // Copy gradient for img1
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch
                    ? b * (2 * channels + 2) * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;
                int dstOffset = hasBatch
                    ? b * channels * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    gradImg1.Data[dstOffset + i] = gradient.Data[srcOffset + i];
                }
            }

            // Copy gradient for warped2
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch
                    ? b * (2 * channels + 2) * pixelsPerChannel + (channels + c) * pixelsPerChannel
                    : (channels + c) * pixelsPerChannel;
                int dstOffset = hasBatch
                    ? b * channels * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    gradWarped2.Data[dstOffset + i] = gradient.Data[srcOffset + i];
                }
            }

            // Copy gradient for flow
            for (int c = 0; c < 2; c++)
            {
                int srcOffset = hasBatch
                    ? b * (2 * channels + 2) * pixelsPerChannel + (2 * channels + c) * pixelsPerChannel
                    : (2 * channels + c) * pixelsPerChannel;
                int dstOffset = hasBatch
                    ? b * 2 * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    gradFlow.Data[dstOffset + i] = gradient.Data[srcOffset + i];
                }
            }
        }

        return (gradImg1, gradWarped2, gradFlow);
    }

    private Tensor<T> ConvertGridGradientToFlowGradient(Tensor<T> gradGrid, bool hasBatch)
    {
        // Grid is in [-1, 1] normalized coordinates
        // Flow is in pixel coordinates
        // gradFlow = gradGrid * (dim - 1) / 2
        int batch = hasBatch ? gradGrid.Shape[0] : 1;
        int height = hasBatch ? gradGrid.Shape[1] : gradGrid.Shape[0];
        int width = hasBatch ? gradGrid.Shape[2] : gradGrid.Shape[1];

        var flowShape = hasBatch ? new[] { batch, 2, height, width } : new[] { 2, height, width };
        var gradFlow = new Tensor<T>(flowShape);

        T scaleW = NumOps.FromDouble((width - 1) / 2.0);
        T scaleH = NumOps.FromDouble((height - 1) / 2.0);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Grid is [batch, height, width, 2] where last dim is (x, y)
                    int gridIdxX = hasBatch
                        ? b * height * width * 2 + h * width * 2 + w * 2
                        : h * width * 2 + w * 2;
                    int gridIdxY = gridIdxX + 1;

                    int flowIdxX = hasBatch
                        ? b * 2 * height * width + 0 * height * width + h * width + w
                        : 0 * height * width + h * width + w;
                    int flowIdxY = hasBatch
                        ? b * 2 * height * width + 1 * height * width + h * width + w
                        : 1 * height * width + h * width + w;

                    gradFlow.Data[flowIdxX] = NumOps.Multiply(gradGrid.Data[gridIdxX], scaleW);
                    gradFlow.Data[flowIdxY] = NumOps.Multiply(gradGrid.Data[gridIdxY], scaleH);
                }
            }
        }

        return gradFlow;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result.Data[i] = NumOps.Add(a.Data[i], b.Data[i]);
        }
        return result;
    }

    private void AccumulatePyramidGradient(Tensor<T> target, Tensor<T> gradient, int level, bool hasBatch)
    {
        // Upsample gradient from pyramid level to full resolution and accumulate
        int targetH = hasBatch ? target.Shape[2] : target.Shape[1];
        int targetW = hasBatch ? target.Shape[3] : target.Shape[2];

        // Simple bilinear upsampling to target resolution
        var upsampled = UpsampleGradient(gradient, targetH, targetW, hasBatch);

        for (int i = 0; i < target.Length; i++)
        {
            target.Data[i] = NumOps.Add(target.Data[i], upsampled.Data[i]);
        }
    }

    private Tensor<T> UpsampleGradient(Tensor<T> gradient, int targetH, int targetW, bool hasBatch)
    {
        int batch = hasBatch ? gradient.Shape[0] : 1;
        int channels = hasBatch ? gradient.Shape[1] : gradient.Shape[0];
        int srcH = hasBatch ? gradient.Shape[2] : gradient.Shape[1];
        int srcW = hasBatch ? gradient.Shape[3] : gradient.Shape[2];

        if (srcH == targetH && srcW == targetW)
            return gradient;

        var outShape = hasBatch ? new[] { batch, channels, targetH, targetW } : new[] { channels, targetH, targetW };
        var output = new Tensor<T>(outShape);

        double scaleH = (double)srcH / targetH;
        double scaleW = (double)srcW / targetW;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        double srcHf = h * scaleH;
                        double srcWf = w * scaleW;

                        int h0 = (int)Math.Floor(srcHf);
                        int w0 = (int)Math.Floor(srcWf);
                        int h1 = Math.Min(h0 + 1, srcH - 1);
                        int w1 = Math.Min(w0 + 1, srcW - 1);

                        double hWeight = srcHf - h0;
                        double wWeight = srcWf - w0;

                        T v00 = GetGradValue(gradient, b, c, h0, w0, hasBatch, srcH, srcW, channels);
                        T v01 = GetGradValue(gradient, b, c, h0, w1, hasBatch, srcH, srcW, channels);
                        T v10 = GetGradValue(gradient, b, c, h1, w0, hasBatch, srcH, srcW, channels);
                        T v11 = GetGradValue(gradient, b, c, h1, w1, hasBatch, srcH, srcW, channels);

                        T top = NumOps.Add(
                            NumOps.Multiply(v00, NumOps.FromDouble(1 - wWeight)),
                            NumOps.Multiply(v01, NumOps.FromDouble(wWeight)));
                        T bottom = NumOps.Add(
                            NumOps.Multiply(v10, NumOps.FromDouble(1 - wWeight)),
                            NumOps.Multiply(v11, NumOps.FromDouble(wWeight)));
                        T value = NumOps.Add(
                            NumOps.Multiply(top, NumOps.FromDouble(1 - hWeight)),
                            NumOps.Multiply(bottom, NumOps.FromDouble(hWeight)));

                        int outIdx = hasBatch
                            ? b * channels * targetH * targetW + c * targetH * targetW + h * targetW + w
                            : c * targetH * targetW + h * targetW + w;
                        output.Data[outIdx] = value;
                    }
                }
            }
        }

        return output;
    }

    private T GetGradValue(Tensor<T> grad, int b, int c, int h, int w, bool hasBatch, int height, int width, int channels)
    {
        int idx = hasBatch
            ? b * channels * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return grad.Data[idx];
    }

    private Tensor<T> DownsampleFlowGradient(Tensor<T> gradFlow, bool hasBatch)
    {
        int batch = hasBatch ? gradFlow.Shape[0] : 1;
        int height = hasBatch ? gradFlow.Shape[2] : gradFlow.Shape[1];
        int width = hasBatch ? gradFlow.Shape[3] : gradFlow.Shape[2];

        int newH = height / 2;
        int newW = width / 2;

        var outShape = hasBatch ? new[] { batch, 2, newH, newW } : new[] { 2, newH, newW };
        var output = new Tensor<T>(outShape);

        // Average pooling 2x2 for gradient downsampling
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int h = 0; h < newH; h++)
                {
                    for (int w = 0; w < newW; w++)
                    {
                        T sum = NumOps.Zero;
                        for (int dh = 0; dh < 2; dh++)
                        {
                            for (int dw = 0; dw < 2; dw++)
                            {
                                int ih = h * 2 + dh;
                                int iw = w * 2 + dw;
                                int idx = hasBatch
                                    ? b * 2 * height * width + c * height * width + ih * width + iw
                                    : c * height * width + ih * width + iw;
                                sum = NumOps.Add(sum, gradFlow.Data[idx]);
                            }
                        }
                        int outIdx = hasBatch
                            ? b * 2 * newH * newW + c * newH * newW + h * newW + w
                            : c * newH * newW + h * newW + w;
                        // Scale flow gradient by 0.5 for each dimension (consistent with flow scaling)
                        output.Data[outIdx] = NumOps.Multiply(sum, NumOps.FromDouble(0.5));
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ConcatenateFrameGradients(Tensor<T> gradFrame1, Tensor<T> gradFrame2, bool hasBatch)
    {
        int batch = hasBatch ? gradFrame1.Shape[0] : 1;
        int channels = hasBatch ? gradFrame1.Shape[1] : gradFrame1.Shape[0];
        int height = hasBatch ? gradFrame1.Shape[2] : gradFrame1.Shape[1];
        int width = hasBatch ? gradFrame1.Shape[3] : gradFrame1.Shape[2];

        var outShape = hasBatch
            ? new[] { batch, 2 * channels, height, width }
            : new[] { 2 * channels, height, width };
        var output = new Tensor<T>(outShape);

        int pixelsPerChannel = height * width;

        for (int b = 0; b < batch; b++)
        {
            // Copy frame1 gradients
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch ? b * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = hasBatch ? b * 2 * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    output.Data[dstOffset + i] = gradFrame1.Data[srcOffset + i];
                }
            }
            // Copy frame2 gradients
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch ? b * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = hasBatch
                    ? b * 2 * channels * pixelsPerChannel + (channels + c) * pixelsPerChannel
                    : (channels + c) * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    output.Data[dstOffset + i] = gradFrame2.Data[srcOffset + i];
                }
            }
        }

        return output;
    }

    #endregion

    #region Helper Methods

    private (Tensor<T> frame1, Tensor<T> frame2) SplitFrames(Tensor<T> input, int channels, bool hasBatch)
    {
        int height = hasBatch ? input.Shape[2] : input.Shape[1];
        int width = hasBatch ? input.Shape[3] : input.Shape[2];
        int batch = hasBatch ? input.Shape[0] : 1;

        var shape = hasBatch ? new[] { batch, channels, height, width } : new[] { channels, height, width };
        var frame1 = new Tensor<T>(shape);
        var frame2 = new Tensor<T>(shape);

        int pixelsPerChannel = height * width;
        var inputData = input.Data.ToArray();

        var frame1Data = frame1.Data;
        var frame2Data = frame2.Data;
        for (int b = 0; b < batch; b++)
        {
            int batchOffset = b * 2 * channels * pixelsPerChannel;
            int outBatchOffset = b * channels * pixelsPerChannel;

            // Copy first frame channels
            for (int i = 0; i < channels * pixelsPerChannel; i++)
            {
                frame1Data[outBatchOffset + i] = inputData[batchOffset + i];
            }
            // Copy second frame channels
            for (int i = 0; i < channels * pixelsPerChannel; i++)
            {
                frame2Data[outBatchOffset + i] = inputData[batchOffset + channels * pixelsPerChannel + i];
            }
        }

        return (frame1, frame2);
    }

    private List<Tensor<T>> BuildPyramid(Tensor<T> image, bool hasBatch)
    {
        var pyramid = new List<Tensor<T>> { image };

        var current = image;
        for (int i = 1; i < _numLevels; i++)
        {
            current = Downsample(current, hasBatch);
            pyramid.Add(current);
        }

        return pyramid;
    }

    private Tensor<T> Downsample(Tensor<T> input, bool hasBatch)
    {
        int batch = hasBatch ? input.Shape[0] : 1;
        int channels = hasBatch ? input.Shape[1] : input.Shape[0];
        int height = hasBatch ? input.Shape[2] : input.Shape[1];
        int width = hasBatch ? input.Shape[3] : input.Shape[2];

        int newH = height / 2;
        int newW = width / 2;

        var outShape = hasBatch ? new[] { batch, channels, newH, newW } : new[] { channels, newH, newW };
        var output = new Tensor<T>(outShape);

        // Simple average pooling 2x2
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < newH; h++)
                {
                    for (int w = 0; w < newW; w++)
                    {
                        T sum = NumOps.Zero;
                        for (int dh = 0; dh < 2; dh++)
                        {
                            for (int dw = 0; dw < 2; dw++)
                            {
                                int ih = h * 2 + dh;
                                int iw = w * 2 + dw;
                                int idx = hasBatch
                                    ? b * channels * height * width + c * height * width + ih * width + iw
                                    : c * height * width + ih * width + iw;
                                sum = NumOps.Add(sum, input.Data[idx]);
                            }
                        }
                        int outIdx = hasBatch
                            ? b * channels * newH * newW + c * newH * newW + h * newW + w
                            : c * newH * newW + h * newW + w;
                        output.Data[outIdx] = NumOps.Divide(sum, NumOps.FromDouble(4.0));
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> UpsampleFlow(Tensor<T> flow, int targetH, int targetW, bool hasBatch)
    {
        int batch = hasBatch ? flow.Shape[0] : 1;
        int height = hasBatch ? flow.Shape[2] : flow.Shape[1];
        int width = hasBatch ? flow.Shape[3] : flow.Shape[2];

        var outShape = hasBatch ? new[] { batch, 2, targetH, targetW } : new[] { 2, targetH, targetW };
        var output = new Tensor<T>(outShape);

        // Bilinear upsampling with scale factor for flow values
        double scaleH = (double)targetH / height;
        double scaleW = (double)targetW / width;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        double srcH = h / scaleH;
                        double srcW = w / scaleW;

                        int h0 = (int)Math.Floor(srcH);
                        int w0 = (int)Math.Floor(srcW);
                        int h1 = Math.Min(h0 + 1, height - 1);
                        int w1 = Math.Min(w0 + 1, width - 1);

                        double hWeight = srcH - h0;
                        double wWeight = srcW - w0;

                        // Bilinear interpolation
                        T v00 = GetFlowValue(flow, b, c, h0, w0, hasBatch, height, width);
                        T v01 = GetFlowValue(flow, b, c, h0, w1, hasBatch, height, width);
                        T v10 = GetFlowValue(flow, b, c, h1, w0, hasBatch, height, width);
                        T v11 = GetFlowValue(flow, b, c, h1, w1, hasBatch, height, width);

                        T top = NumOps.Add(
                            NumOps.Multiply(v00, NumOps.FromDouble(1 - wWeight)),
                            NumOps.Multiply(v01, NumOps.FromDouble(wWeight)));
                        T bottom = NumOps.Add(
                            NumOps.Multiply(v10, NumOps.FromDouble(1 - wWeight)),
                            NumOps.Multiply(v11, NumOps.FromDouble(wWeight)));
                        T value = NumOps.Add(
                            NumOps.Multiply(top, NumOps.FromDouble(1 - hWeight)),
                            NumOps.Multiply(bottom, NumOps.FromDouble(hWeight)));

                        // Scale flow value by upsample factor
                        double flowScale = c == 0 ? scaleW : scaleH;
                        value = NumOps.Multiply(value, NumOps.FromDouble(flowScale));

                        int outIdx = hasBatch
                            ? b * 2 * targetH * targetW + c * targetH * targetW + h * targetW + w
                            : c * targetH * targetW + h * targetW + w;
                        output.Data[outIdx] = value;
                    }
                }
            }
        }

        return output;
    }

    private T GetFlowValue(Tensor<T> flow, int b, int c, int h, int w, bool hasBatch, int height, int width)
    {
        int idx = hasBatch
            ? b * 2 * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return flow.Data[idx];
    }

    /// <summary>
    /// Warps an image using optical flow via IEngine.GridSample.
    /// Returns both the warped image and the sampling grid for backward pass.
    /// </summary>
    /// <param name="image">Image to warp [batch, channels, height, width] or [channels, height, width]</param>
    /// <param name="flow">Optical flow [batch, 2, height, width] or [2, height, width] (dx, dy in pixels)</param>
    /// <param name="hasBatch">Whether tensors have batch dimension</param>
    /// <returns>Tuple of (warped image, sampling grid)</returns>
    private (Tensor<T> warped, Tensor<T> grid) WarpImageWithGrid(Tensor<T> image, Tensor<T> flow, bool hasBatch)
    {
        int batch = hasBatch ? image.Shape[0] : 1;
        int channels = hasBatch ? image.Shape[1] : image.Shape[0];
        int height = hasBatch ? image.Shape[2] : image.Shape[1];
        int width = hasBatch ? image.Shape[3] : image.Shape[2];

        // Create grid tensor: [batch, height, width, 2] where last dim is (x, y) in [-1, 1]
        var gridShape = new[] { batch, height, width, 2 };
        var grid = new Tensor<T>(gridShape);

        // Build sampling grid from flow
        // Grid coordinates are normalized to [-1, 1]
        // Flow is in pixel coordinates, need to convert:
        // grid_x = (x + flow_x) / (width - 1) * 2 - 1
        // grid_y = (y + flow_y) / (height - 1) * 2 - 1
        T widthNorm = NumOps.FromDouble(2.0 / (width - 1));
        T heightNorm = NumOps.FromDouble(2.0 / (height - 1));
        T one = NumOps.One;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Get flow at this position
                    int flowIdxX = hasBatch
                        ? b * 2 * height * width + 0 * height * width + h * width + w
                        : 0 * height * width + h * width + w;
                    int flowIdxY = hasBatch
                        ? b * 2 * height * width + 1 * height * width + h * width + w
                        : 1 * height * width + h * width + w;

                    T dx = flow.Data[flowIdxX];
                    T dy = flow.Data[flowIdxY];

                    // Source coordinates (pixel space)
                    T srcX = NumOps.Add(NumOps.FromDouble(w), dx);
                    T srcY = NumOps.Add(NumOps.FromDouble(h), dy);

                    // Convert to normalized coordinates [-1, 1]
                    T gridX = NumOps.Subtract(NumOps.Multiply(srcX, widthNorm), one);
                    T gridY = NumOps.Subtract(NumOps.Multiply(srcY, heightNorm), one);

                    // Store in grid
                    int gridIdx = b * height * width * 2 + h * width * 2 + w * 2;
                    grid.Data[gridIdx] = gridX;
                    grid.Data[gridIdx + 1] = gridY;
                }
            }
        }

        // Ensure image is in 4D format for GridSample
        var image4D = image;
        if (!hasBatch)
        {
            image4D = new Tensor<T>(new[] { 1, channels, height, width }, new Vector<T>(image.Data));
        }

        // Use IEngine.GridSample for hardware-accelerated bilinear sampling
        var warped = _engine.GridSample(image4D, grid);

        // Remove batch dimension if input didn't have it
        if (!hasBatch && warped.Rank == 4)
        {
            warped = new Tensor<T>(new[] { channels, height, width }, new Vector<T>(warped.Data));
        }

        return (warped, grid);
    }

    private Tensor<T> WarpImage(Tensor<T> image, Tensor<T> flow, bool hasBatch)
    {
        int batch = hasBatch ? image.Shape[0] : 1;
        int channels = hasBatch ? image.Shape[1] : image.Shape[0];
        int height = hasBatch ? image.Shape[2] : image.Shape[1];
        int width = hasBatch ? image.Shape[3] : image.Shape[2];

        var output = new Tensor<T>(image.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Get flow at this position
                    int flowIdxX = hasBatch
                        ? b * 2 * height * width + 0 * height * width + h * width + w
                        : 0 * height * width + h * width + w;
                    int flowIdxY = hasBatch
                        ? b * 2 * height * width + 1 * height * width + h * width + w
                        : 1 * height * width + h * width + w;

                    double dx = NumOps.ToDouble(flow.Data[flowIdxX]);
                    double dy = NumOps.ToDouble(flow.Data[flowIdxY]);

                    double srcX = w + dx;
                    double srcY = h + dy;

                    // Bilinear sample from source image
                    for (int c = 0; c < channels; c++)
                    {
                        T value = BilinearSample(image, b, c, srcY, srcX, hasBatch, height, width, channels);
                        int outIdx = hasBatch
                            ? b * channels * height * width + c * height * width + h * width + w
                            : c * height * width + h * width + w;
                        output.Data[outIdx] = value;
                    }
                }
            }
        }

        return output;
    }

    private T BilinearSample(Tensor<T> image, int b, int c, double h, double w, bool hasBatch, int height, int width, int channels)
    {
        int h0 = (int)Math.Floor(h);
        int w0 = (int)Math.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        // Clamp to valid range (compatible with .NET Framework 4.7.1)
        h0 = Math.Max(0, Math.Min(h0, height - 1));
        h1 = Math.Max(0, Math.Min(h1, height - 1));
        w0 = Math.Max(0, Math.Min(w0, width - 1));
        w1 = Math.Max(0, Math.Min(w1, width - 1));

        double hWeight = h - Math.Floor(h);
        double wWeight = w - Math.Floor(w);

        T v00 = GetImageValue(image, b, c, h0, w0, hasBatch, height, width, channels);
        T v01 = GetImageValue(image, b, c, h0, w1, hasBatch, height, width, channels);
        T v10 = GetImageValue(image, b, c, h1, w0, hasBatch, height, width, channels);
        T v11 = GetImageValue(image, b, c, h1, w1, hasBatch, height, width, channels);

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

    private T GetImageValue(Tensor<T> image, int b, int c, int h, int w, bool hasBatch, int height, int width, int channels)
    {
        int idx = hasBatch
            ? b * channels * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return image.Data[idx];
    }

    private Tensor<T> ConcatenateForModule(Tensor<T> img1, Tensor<T> warped2, Tensor<T> flow, bool hasBatch)
    {
        int batch = hasBatch ? img1.Shape[0] : 1;
        int channels = hasBatch ? img1.Shape[1] : img1.Shape[0];
        int height = hasBatch ? img1.Shape[2] : img1.Shape[1];
        int width = hasBatch ? img1.Shape[3] : img1.Shape[2];

        int outChannels = 2 * channels + 2;
        var outShape = hasBatch
            ? new[] { batch, outChannels, height, width }
            : new[] { outChannels, height, width };
        var output = new Tensor<T>(outShape);

        int pixelsPerChannel = height * width;

        var outputData = output.Data;
        var img1Data = img1.Data;
        var warped2Data = warped2.Data;
        var flowData = flow.Data;

        for (int b = 0; b < batch; b++)
        {
            int batchOffset = b * outChannels * pixelsPerChannel;

            // Copy img1
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch ? b * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = batchOffset + c * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    outputData[dstOffset + i] = img1Data[srcOffset + i];
                }
            }

            // Copy warped2
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch ? b * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = batchOffset + (channels + c) * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    outputData[dstOffset + i] = warped2Data[srcOffset + i];
                }
            }

            // Copy flow
            for (int c = 0; c < 2; c++)
            {
                int srcOffset = hasBatch ? b * 2 * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = batchOffset + (2 * channels + c) * pixelsPerChannel;
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    outputData[dstOffset + i] = flowData[srcOffset + i];
                }
            }
        }

        return output;
    }

    private Tensor<T> AddResidualFlow(Tensor<T> flow, Tensor<T> residual, bool hasBatch)
    {
        // Extract first 2 channels from residual (flow residual)
        int batch = hasBatch ? flow.Shape[0] : 1;
        int height = hasBatch ? flow.Shape[2] : flow.Shape[1];
        int width = hasBatch ? flow.Shape[3] : flow.Shape[2];

        var output = new Tensor<T>(flow.Shape);
        int pixelsPerChannel = height * width;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    int flowIdx = hasBatch
                        ? b * 2 * pixelsPerChannel + c * pixelsPerChannel + i
                        : c * pixelsPerChannel + i;
                    int residualIdx = hasBatch
                        ? b * residual.Shape[1] * pixelsPerChannel + c * pixelsPerChannel + i
                        : c * pixelsPerChannel + i;

                    output.Data[flowIdx] = NumOps.Add(flow.Data[flowIdx], residual.Data[residualIdx]);
                }
            }
        }

        return output;
    }

    #endregion

    #region Layer Properties

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    // GPU Caches
    private readonly Dictionary<(int batch, int height, int width), IGpuTensor<T>> _identityGridCache = new();
    private readonly Dictionary<(int batch, int channels, int height, int width), (IGpuBuffer idx1, IGpuBuffer idx2)> _sliceIndicesCache = new();

    /// <summary>
    /// Indicates whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0) throw new ArgumentException("SpyNetLayer requires an input tensor.");
        var input = inputs[0];

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        // Input: [B, 2*C, H, W]
        int batch = input.Shape[0];
        int doubleChannels = input.Shape[1];
        int channels = doubleChannels / 2;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var backend = gpuEngine.GetBackend()!;
        var cleanup = new List<IDisposable>();

        try
        {
            // Slice into frame1 and frame2 using cached indices
            var (frame1, frame2) = SliceChannelsGpu(input, batch, channels, height, width, gpuEngine, backend);
            cleanup.Add(frame1);
            cleanup.Add(frame2);

            if (IsTrainingMode)
            {
                _lastInput1 = frame1.ToTensor();
                _lastInput2 = frame2.ToTensor();
            }

            // Estimate Flow
            var flow = EstimateFlowGpu(frame1, frame2, batch, height, width, gpuEngine, backend);

            if (IsTrainingMode)
            {
                _lastFlow = flow.ToTensor();
            }

            return flow;
        }
        finally
        {
            foreach (var resource in cleanup)
            {
                resource.Dispose();
            }
        }
    }

    private (IGpuTensor<T> frame1, IGpuTensor<T> frame2) SliceChannelsGpu(
        IGpuTensor<T> input, int batch, int channels, int height, int width,
        DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        var key = (batch, channels, height, width);
        if (!_sliceIndicesCache.TryGetValue(key, out var indices))
        {
            int frameSize = channels * height * width;
            var frame1Indices = new int[batch * frameSize];
            var frame2Indices = new int[batch * frameSize];

            System.Threading.Tasks.Parallel.For(0, batch, b =>
            {
                int srcBase = b * 2 * frameSize;
                int dstBase = b * frameSize;
                for (int i = 0; i < frameSize; i++)
                {
                    frame1Indices[dstBase + i] = srcBase + i;
                    frame2Indices[dstBase + i] = srcBase + frameSize + i;
                }
            });

            var idx1 = backend.AllocateIntBuffer(frame1Indices);
            var idx2 = backend.AllocateIntBuffer(frame2Indices);
            indices = (idx1, idx2);
            _sliceIndicesCache[key] = indices;
        }

        int size = batch * channels * height * width;
        var f1 = engine.GatherGpu(input, indices.idx1, size, 1);
        var f2 = engine.GatherGpu(input, indices.idx2, size, 1);

        var shape = new[] { batch, channels, height, width };
        var f1Reshaped = engine.ReshapeGpu(f1, shape);
        var f2Reshaped = engine.ReshapeGpu(f2, shape);
        
        f1.Dispose();
        f2.Dispose();

        return (f1Reshaped, f2Reshaped);
    }

    private IGpuTensor<T> EstimateFlowGpu(IGpuTensor<T> frame1, IGpuTensor<T> frame2, int batch, int height, int width, 
        DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        var cleanup = new List<IDisposable>();
        try 
        {
            var pyramid1 = BuildPyramidGpu(frame1, engine);
            var pyramid2 = BuildPyramidGpu(frame2, engine);
            foreach (var t in pyramid1) if (t != frame1) cleanup.Add(t);
            foreach (var t in pyramid2) if (t != frame2) cleanup.Add(t);

            int coarseH = height >> (_numLevels - 1);
            int coarseW = width >> (_numLevels - 1);
            var flow = engine.ZerosGpu<T>([batch, 2, coarseH, coarseW]);

            for (int level = _numLevels - 1; level >= 0; level--)
            {
                var img1 = pyramid1[level];
                var img2 = pyramid2[level];
                int levelH = img1.Shape[2];
                int levelW = img1.Shape[3];

                if (level < _numLevels - 1)
                {
                    var upsampled = engine.UpsampleGpu(flow, 2);
                    var scaled = engine.ScaleGpu(upsampled, 2.0f);
                    
                    cleanup.Add(flow); 
                    cleanup.Add(upsampled);
                    flow = scaled;
                }

                var (warped2, grid) = WarpImageWithGridGpu(img2, flow, engine, backend);
                cleanup.Add(warped2);
                cleanup.Add(grid);
                
                var moduleInput = engine.ConcatGpu<T>([img1, warped2, flow], 1);
                cleanup.Add(moduleInput);

                var residualFlow = _basicModules[level].ForwardGpu(moduleInput);
                cleanup.Add(residualFlow);

                int sliceSize = batch * 2 * levelH * levelW;
                var sliceIndices = new int[sliceSize];
                int chStride = levelH * levelW;
                int batchStride = 32 * chStride;
                int outBatchStride = 2 * chStride;
                
                System.Threading.Tasks.Parallel.For(0, batch, b =>
                {
                    int srcBase = b * batchStride;
                    int dstBase = b * outBatchStride;
                    for(int i=0; i<chStride; i++) sliceIndices[dstBase + i] = srcBase + i;
                    for(int i=0; i<chStride; i++) sliceIndices[dstBase + chStride + i] = srcBase + chStride + i;
                });
                
                using var idxBuffer = backend.AllocateIntBuffer(sliceIndices);
                var slicedResidual = engine.GatherGpu(residualFlow, idxBuffer, sliceSize, 1);
                var reshapedResidual = engine.ReshapeGpu(slicedResidual, [batch, 2, levelH, levelW]);
                
                var newFlow = engine.AddGpu(flow, reshapedResidual);
                
                cleanup.Add(slicedResidual);
                cleanup.Add(reshapedResidual);
                cleanup.Add(flow);
                
                flow = newFlow;
            }
            
            return flow;
        }
        catch
        {
            foreach (var r in cleanup) r.Dispose();
            throw;
        }
    }

    private (IGpuTensor<T> warped, IGpuTensor<T> grid) WarpImageWithGridGpu(
        IGpuTensor<T> image, IGpuTensor<T> flow, DirectGpuTensorEngine engine, IDirectGpuBackend backend)
    {
        int batch = image.Shape[0];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var gridKey = (batch, height, width);
        if (!_identityGridCache.TryGetValue(gridKey, out var identityGrid))
        {
            var gridData = new float[batch * height * width * 2];
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * height * width * 2 + h * width * 2 + w * 2;
                        gridData[idx] = (float)(2.0 * w / (width - 1) - 1.0);
                        gridData[idx + 1] = (float)(2.0 * h / (height - 1) - 1.0);
                    }
                }
            }
            using var buffer = backend.AllocateBuffer(gridData);
            identityGrid = new GpuTensor<T>(backend, buffer, [batch, height, width, 2], GpuTensorRole.Constant, ownsBuffer: true);
            _identityGridCache[gridKey] = identityGrid;
        }

        var flowPermuted = engine.PermuteGpu(flow, [0, 2, 3, 1]);
        
        float scaleX = 2.0f / (width - 1);
        float scaleY = 2.0f / (height - 1);
        
        var scaleData = new float[] { scaleX, scaleY };
        using var scaleBuffer = backend.AllocateBuffer(scaleData);
        var scaleTensor = new GpuTensor<T>(backend, scaleBuffer, [1, 1, 1, 2], GpuTensorRole.Constant, ownsBuffer: false);
        
        var scaledFlow = engine.BroadcastMultiplyRowGpu(flowPermuted, scaleTensor);
        
        var grid = engine.AddGpu(identityGrid, scaledFlow);
        
        flowPermuted.Dispose();
        scaledFlow.Dispose();
        
        var warped = engine.GridSampleGpu(image, grid);
        
        return (warped, grid);
    }

    private List<IGpuTensor<T>> BuildPyramidGpu(IGpuTensor<T> image, DirectGpuTensorEngine engine)
    {
        var pyramid = new List<IGpuTensor<T>> { image };
        var current = image;
        for (int i = 1; i < _numLevels; i++)
        {
            var down = engine.AvgPool2DGpu(current, [2, 2], [2, 2]);
            pyramid.Add(down);
            current = down;
        }
        return pyramid;
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var grid in _identityGridCache.Values)
            {
                grid.Dispose();
            }
            _identityGridCache.Clear();

            foreach (var indices in _sliceIndicesCache.Values)
            {
                indices.idx1.Dispose();
                indices.idx2.Dispose();
            }
            _sliceIndicesCache.Clear();
        }
        base.Dispose(disposing);
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation
    {
        get
        {
            // SpyNet supports JIT if all basic modules support JIT
            foreach (var module in _basicModules)
            {
                if (!module.SupportsJitCompilation)
                    return false;
            }
            return _basicModules.Count > 0;
        }
    }

    #endregion

    #region IChainableComputationGraph Implementation

    /// <inheritdoc/>
    public override int[] GetInputShape() => [_inputChannels * 2, _inputHeight, _inputWidth];

    /// <summary>
    /// Gets the output shape for this layer (2 channels for optical flow: dx, dy).
    /// </summary>
    public new int[] GetOutputShape() => [2, _inputHeight, _inputWidth];

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null || inputNodes.Count == 0)
            throw new ArgumentException("Input nodes cannot be null or empty.", nameof(inputNodes));

        return BuildComputationGraph(inputNodes[0], "");
    }

    /// <inheritdoc/>
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        if (!SupportsJitCompilation)
            throw new InvalidOperationException("Layer modules not initialized for JIT compilation.");

        // Input is concatenated frames [batch, 2*channels, height, width]
        // Split into two frames
        var splitNodes = TensorOperations<T>.Split(inputNode, 2, axis: 1);
        var frame1Node = splitNodes[0];
        var frame2Node = splitNodes[1];

        // Build pyramid nodes for both frames
        var pyramid1 = BuildPyramidGraph(frame1Node, $"{namePrefix}pyr1_");
        var pyramid2 = BuildPyramidGraph(frame2Node, $"{namePrefix}pyr2_");

        // Initialize flow at coarsest level (zeros - constant node)
        int coarseH = _inputHeight >> (_numLevels - 1);
        int coarseW = _inputWidth >> (_numLevels - 1);
        var zeroFlow = new Tensor<T>(new[] { 1, 2, coarseH, coarseW });
        var flowNode = TensorOperations<T>.Constant(zeroFlow, $"{namePrefix}zero_flow");

        // Coarse-to-fine refinement
        for (int level = _numLevels - 1; level >= 0; level--)
        {
            var img1Node = pyramid1[level];
            var img2Node = pyramid2[level];

            // Upsample flow if not at coarsest level
            if (level < _numLevels - 1)
            {
                int targetH = _inputHeight >> level;
                int targetW = _inputWidth >> level;
                flowNode = TensorOperations<T>.Upsample(flowNode, 2);
                // Scale flow values by 2 for upsampling
                var scaleNode = TensorOperations<T>.Constant(
                    CreateScaleTensor(flowNode.Value.Shape, 2.0), $"{namePrefix}scale_{level}");
                flowNode = TensorOperations<T>.ElementwiseMultiply(flowNode, scaleNode);
            }

            // Warp img2 using flow via GridSample
            // First create sampling grid from flow
            var gridNode = CreateGridFromFlowGraph(flowNode, $"{namePrefix}grid_{level}_");
            var warpedNode = TensorOperations<T>.GridSample(img2Node, gridNode);

            // Concatenate [img1, warped2, flow] for basic module input
            var moduleInputNode = TensorOperations<T>.Concat(
                new List<ComputationNode<T>> { img1Node, warpedNode, flowNode }, axis: 1);

            // Get residual flow from basic module
            var residualFlowNode = _basicModules[level].ExportComputationGraph(
                new List<ComputationNode<T>> { moduleInputNode });

            // Extract first 2 channels as residual flow (if module outputs more)
            // Add residual to current flow
            flowNode = TensorOperations<T>.Add(flowNode, residualFlowNode);
        }

        return flowNode;
    }

    private List<ComputationNode<T>> BuildPyramidGraph(ComputationNode<T> imageNode, string namePrefix)
    {
        var pyramid = new List<ComputationNode<T>> { imageNode };
        var currentNode = imageNode;

        for (int i = 1; i < _numLevels; i++)
        {
            // Downsample by factor of 2 using average pooling
            currentNode = TensorOperations<T>.AvgPool2D(
                currentNode,
                poolSize: new[] { 2, 2 },
                strides: new[] { 2, 2 });
            pyramid.Add(currentNode);
        }

        return pyramid;
    }

    private ComputationNode<T> CreateGridFromFlowGraph(ComputationNode<T> flowNode, string namePrefix)
    {
        // Create identity grid and add flow to get sampling positions
        // Grid should be [batch, height, width, 2] in normalized coordinates [-1, 1]
        var flowShape = flowNode.Value.Shape;
        int batch = flowShape[0];
        int height = flowShape[2];
        int width = flowShape[3];

        // Create base identity grid
        var identityGrid = CreateIdentityGrid(batch, height, width);
        var identityNode = TensorOperations<T>.Constant(identityGrid, $"{namePrefix}identity");

        // Reshape flow from [B, 2, H, W] to [B, H, W, 2] and normalize
        var permutedFlow = TensorOperations<T>.Permute(flowNode, 0, 2, 3, 1);

        // Scale flow to normalized coordinates: flow / (dim - 1) * 2
        T widthScale = NumOps.FromDouble(2.0 / (width - 1));
        T heightScale = NumOps.FromDouble(2.0 / (height - 1));
        var scaleData = new T[batch * height * width * 2];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = b * height * width * 2 + h * width * 2 + w * 2;
                    scaleData[idx] = widthScale;     // x scale
                    scaleData[idx + 1] = heightScale; // y scale
                }
            }
        }
        var scaleTensor = new Tensor<T>(new[] { batch, height, width, 2 }, new Vector<T>(scaleData));
        var scaleNode = TensorOperations<T>.Constant(scaleTensor, $"{namePrefix}scale");

        var scaledFlow = TensorOperations<T>.ElementwiseMultiply(permutedFlow, scaleNode);
        var grid = TensorOperations<T>.Add(identityNode, scaledFlow);

        return grid;
    }

    private Tensor<T> CreateIdentityGrid(int batch, int height, int width)
    {
        var data = new T[batch * height * width * 2];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int idx = b * height * width * 2 + h * width * 2 + w * 2;
                    // Normalized coordinates [-1, 1]
                    data[idx] = NumOps.FromDouble(2.0 * w / (width - 1) - 1.0);     // x
                    data[idx + 1] = NumOps.FromDouble(2.0 * h / (height - 1) - 1.0); // y
                }
            }
        }
        return new Tensor<T>(new[] { batch, height, width, 2 }, new Vector<T>(data));
    }

    private Tensor<T> CreateScaleTensor(int[] shape, double scale)
    {
        int totalSize = 1;
        foreach (var dim in shape) totalSize *= dim;
        var data = new T[totalSize];
        T scaleVal = NumOps.FromDouble(scale);
        for (int i = 0; i < totalSize; i++)
            data[i] = scaleVal;
        return new Tensor<T>(shape, new Vector<T>(data));
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var module in _basicModules)
        {
            var moduleParams = module.GetParameters();
            for (int i = 0; i < moduleParams.Length; i++)
            {
                allParams.Add(moduleParams[i]);
            }
        }
        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var module in _basicModules)
        {
            var moduleParams = module.GetParameters();
            var newParams = new T[moduleParams.Length];
            for (int i = 0; i < moduleParams.Length; i++)
            {
                newParams[i] = parameters[offset + i];
            }
            module.SetParameters(new Vector<T>(newParams));
            offset += moduleParams.Length;
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var module in _basicModules)
        {
            module.UpdateParameters(learningRate);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();
        foreach (var module in _basicModules)
        {
            var grads = module.GetParameterGradients();
            for (int i = 0; i < grads.Length; i++)
            {
                allGrads.Add(grads[i]);
            }
        }
        return new Vector<T>([.. allGrads]);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput1 = null;
        _lastInput2 = null;
        _lastFlow = null;
        foreach (var module in _basicModules)
        {
            module.ResetState();
        }
    }

    #endregion
}
