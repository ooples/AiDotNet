using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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

    private readonly int _numLevels;
    private readonly int _inputChannels;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly List<ConvolutionalLayer<T>> _basicModules;
    private Tensor<T>? _lastInput1;
    private Tensor<T>? _lastInput2;
    private Tensor<T>? _lastFlow;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new SPyNet layer for optical flow estimation.
    /// </summary>
    /// <param name="inputHeight">Height of input frames.</param>
    /// <param name="inputWidth">Width of input frames.</param>
    /// <param name="inputChannels">Number of input channels (typically 3 for RGB).</param>
    /// <param name="numLevels">Number of pyramid levels (default: 5).</param>
    public SpyNetLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels = 3,
        int numLevels = 5)
        : base([inputChannels, inputHeight, inputWidth], [2, inputHeight, inputWidth])
    {
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

        // Build image pyramids
        var pyramid1 = BuildPyramid(frame1, hasBatch);
        var pyramid2 = BuildPyramid(frame2, hasBatch);

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

            // Warp img2 using current flow estimate
            var warped2 = WarpImage(img2, flow, hasBatch);

            // Concatenate inputs for basic module
            var moduleInput = ConcatenateForModule(img1, warped2, flow, hasBatch);

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
    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        // Simplified backward pass - propagate gradients through modules
        var gradient = gradOutput;

        for (int level = 0; level < _numLevels; level++)
        {
            gradient = _basicModules[level].Backward(gradient);
        }

        return gradient;
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

        for (int b = 0; b < batch; b++)
        {
            int batchOffset = b * 2 * channels * pixelsPerChannel;
            int outBatchOffset = b * channels * pixelsPerChannel;

            // Copy first frame channels
            Array.Copy(inputData, batchOffset, frame1.Data.ToArray(), outBatchOffset, channels * pixelsPerChannel);
            // Copy second frame channels
            Array.Copy(inputData, batchOffset + channels * pixelsPerChannel, frame2.Data.ToArray(), outBatchOffset, channels * pixelsPerChannel);
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

        for (int b = 0; b < batch; b++)
        {
            int batchOffset = b * outChannels * pixelsPerChannel;

            // Copy img1
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch ? b * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = batchOffset + c * pixelsPerChannel;
                Array.Copy(img1.Data.ToArray(), srcOffset, output.Data.ToArray(), dstOffset, pixelsPerChannel);
            }

            // Copy warped2
            for (int c = 0; c < channels; c++)
            {
                int srcOffset = hasBatch ? b * channels * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = batchOffset + (channels + c) * pixelsPerChannel;
                Array.Copy(warped2.Data.ToArray(), srcOffset, output.Data.ToArray(), dstOffset, pixelsPerChannel);
            }

            // Copy flow
            for (int c = 0; c < 2; c++)
            {
                int srcOffset = hasBatch ? b * 2 * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = batchOffset + (2 * channels + c) * pixelsPerChannel;
                Array.Copy(flow.Data.ToArray(), srcOffset, output.Data.ToArray(), dstOffset, pixelsPerChannel);
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

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

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
        // SPyNet is complex with multiple pyramid levels - return identity for now
        // Full JIT compilation support can be added later
        return inputNode;
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
