using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
public class E2FGVI<T> : VideoInpaintingBase<T>
{
    private readonly E2FGVIOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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

    public E2FGVI(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 128,
        E2FGVIOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new E2FGVIOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 432;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 240;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;

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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Ensure 4D input
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
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

        if (!hasBatch)
        {
            output = RemoveBatchDimension(output);
        }

        return output;
    }

    /// <summary>
    /// Trains the model on a masked frame and its ground truth.
    /// </summary>
    /// <param name="input">Input tensor containing masked frame [B, C, H, W].</param>
    /// <param name="expectedOutput">Ground truth unmasked frame [B, C, H, W].</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure 4D input
        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
            expectedOutput = AddBatchDimension(expectedOutput);
        }

        // Create a synthetic mask (non-zero differences indicate masked regions)
        var mask = CreateMaskFromDifference(input, expectedOutput);

        // Forward pass with mask
        var maskedInput = ConcatenateChannels(input, mask);
        var prediction = Predict(maskedInput);

        // Ensure prediction matches expected shape
        if (prediction.Rank != expectedOutput.Rank)
        {
            prediction = hasBatch ? prediction : AddBatchDimension(prediction);
        }

        // Compute MSE loss
        T loss = NumOps.Zero;
        for (int i = 0; i < expectedOutput.Length; i++)
        {
            T diff = NumOps.Subtract(prediction.Data.Span[i], expectedOutput.Data.Span[i]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }
        loss = NumOps.Divide(loss, NumOps.FromDouble(expectedOutput.Length));
        LastLoss = loss;

        // Compute gradient: d(MSE)/d(pred) = 2 * (pred - target) / N
        var gradient = new Tensor<T>(prediction.Shape);
        T scale = NumOps.FromDouble(2.0 / expectedOutput.Length);
        for (int i = 0; i < expectedOutput.Length; i++)
        {
            T diff = NumOps.Subtract(prediction.Data.Span[i], expectedOutput.Data.Span[i]);
            gradient.Data.Span[i] = NumOps.Multiply(diff, scale);
        }

        // Backpropagate
        BackpropagateGradient(gradient);

        // Update parameters
        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers) layer.UpdateParameters(lr);
    }

    /// <summary>
    /// Creates a mask based on differences between input and expected output.
    /// </summary>
    private Tensor<T> CreateMaskFromDifference(Tensor<T> input, Tensor<T> expected)
    {
        int batchSize = input.Shape[0];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var mask = new Tensor<T>([batchSize, 1, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double diff = 0;
                    int channels = input.Shape[1];
                    for (int c = 0; c < channels; c++)
                    {
                        double d = Math.Abs(Convert.ToDouble(input[b, c, h, w]) - Convert.ToDouble(expected[b, c, h, w]));
                        diff = Math.Max(diff, d);
                    }
                    // Threshold to create binary mask
                    mask[b, 0, h, w] = NumOps.FromDouble(diff > 0.01 ? 1.0 : 0.0);
                }
            }
        }

        return mask;
    }

    /// <summary>
    /// Backpropagates the gradient through all network layers.
    /// </summary>
    private void BackpropagateGradient(Tensor<T> gradient)
    {
        // Backpropagate through output head
        gradient = _outputHead.Backward(gradient);

        // Backpropagate through decoder
        for (int i = _decoder.Count - 1; i >= 0; i--)
        {
            gradient = _decoder[i].Backward(gradient);
        }

        // Backpropagate through transformer
        for (int i = _transformer.Count - 1; i >= 0; i--)
        {
            gradient = _transformer[i].Backward(gradient);
        }

        // Backpropagate through propagation layers
        for (int i = _propagation.Count - 1; i >= 0; i--)
        {
            gradient = _propagation[i].Backward(gradient);
        }

        // Backpropagate through encoder
        for (int i = _encoder.Count - 1; i >= 0; i--)
        {
            gradient = _encoder[i].Backward(gradient);
        }

        // Note: _flowNet and _flowHead are NOT used in the Predict forward pass.
        // They are only used in EstimateBidirectionalFlow for multi-frame inpainting.
        // Training the flow network requires a separate training routine with flow-specific
        // loss functions (e.g., endpoint error, photometric consistency).
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

        var forward = new Tensor<T>([batchSize, 2, height, width]);
        var backward = new Tensor<T>([batchSize, 2, height, width]);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    forward[b, 0, h, w] = flowOutput[b, 0, h, w];
                    forward[b, 1, h, w] = flowOutput[b, 1, h, w];
                    backward[b, 0, h, w] = flowOutput[b, 2, h, w];
                    backward[b, 1, h, w] = flowOutput[b, 3, h, w];
                }

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

        var accumulated = ZeroTensor(currentFeatures.Shape);

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

        var masked = new Tensor<T>(frame.Shape);
        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double m = 1.0 - Convert.ToDouble(mask[b, 0, h, w]); // Invert: 0 = fill
                        double f = Convert.ToDouble(frame[b, c, h, w]);
                        masked[b, c, h, w] = NumOps.FromDouble(f * m);
                    }

        return masked;
    }

    private Tensor<T> BlendWithOriginal(Tensor<T> original, Tensor<T> inpainted, Tensor<T> mask)
    {
        int batchSize = original.Shape[0];
        int channels = original.Shape[1];
        int height = original.Shape[2];
        int width = original.Shape[3];

        var blended = new Tensor<T>(original.Shape);
        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double m = Convert.ToDouble(mask[b, 0, h, w]);
                        double o = Convert.ToDouble(original[b, c, h, w]);
                        double i = Convert.ToDouble(inpainted[b, c, h, w]);
                        blended[b, c, h, w] = NumOps.FromDouble(o * (1 - m) + i * m);
                    }

        return blended;
    }

    private Tensor<T> WarpFeatures(Tensor<T> features, Tensor<T> flow)
    {
        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int height = features.Shape[2];
        int width = features.Shape[3];

        int flowH = flow.Shape[2];
        int flowW = flow.Shape[3];

        var warped = new Tensor<T>(features.Shape);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    // Scale flow to feature resolution
                    int fh = Math.Min((int)((double)h * flowH / height), flowH - 1);
                    int fw = Math.Min((int)((double)w * flowW / width), flowW - 1);

                    double flowX = Convert.ToDouble(flow[b, 0, fh, fw]) * width / flowW;
                    double flowY = Convert.ToDouble(flow[b, 1, fh, fw]) * height / flowH;

                    double srcX = w + flowX;
                    double srcY = h + flowY;

                    for (int c = 0; c < channels; c++)
                        warped[b, c, h, w] = BilinearSample(features, b, c, srcY, srcX);
                }

        return warped;
    }

    private T BilinearSample(Tensor<T> tensor, int batch, int channel, double y, double x)
    {
        int height = tensor.Shape[2];
        int width = tensor.Shape[3];

        int x0 = Math.Max(0, Math.Min((int)Math.Floor(x), width - 1));
        int y0 = Math.Max(0, Math.Min((int)Math.Floor(y), height - 1));
        int x1 = Math.Max(0, Math.Min(x0 + 1, width - 1));
        int y1 = Math.Max(0, Math.Min(y0 + 1, height - 1));

        double dx = x - Math.Floor(x);
        double dy = y - Math.Floor(y);

        double v00 = Convert.ToDouble(tensor[batch, channel, y0, x0]);
        double v01 = Convert.ToDouble(tensor[batch, channel, y0, x1]);
        double v10 = Convert.ToDouble(tensor[batch, channel, y1, x0]);
        double v11 = Convert.ToDouble(tensor[batch, channel, y1, x1]);

        double value = v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) +
                       v10 * (1 - dx) * dy + v11 * dx * dy;
        return NumOps.FromDouble(value);
    }

    private Tensor<T> UpsampleFlow(Tensor<T> flow, int targetH, int targetW)
    {
        int batchSize = flow.Shape[0];
        int srcH = flow.Shape[2];
        int srcW = flow.Shape[3];

        var upsampled = new Tensor<T>([batchSize, 2, targetH, targetW]);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < targetH; h++)
                for (int w = 0; w < targetW; w++)
                {
                    int srcY = Math.Min((int)((double)h * srcH / targetH), srcH - 1);
                    int srcX = Math.Min((int)((double)w * srcW / targetW), srcW - 1);
                    upsampled[b, 0, h, w] = NumOps.FromDouble(Convert.ToDouble(flow[b, 0, srcY, srcX]) * targetW / srcW);
                    upsampled[b, 1, h, w] = NumOps.FromDouble(Convert.ToDouble(flow[b, 1, srcY, srcX]) * targetH / srcH);
                }

        return upsampled;
    }

    private Tensor<T> ZeroTensor(int[] shape) => new Tensor<T>(shape);

    private Tensor<T> ScaleTensor(Tensor<T> tensor, double scale) =>
        tensor.Transform((v, _) => NumOps.FromDouble(Convert.ToDouble(v) * scale));

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        int batchSize = a.Shape[0];
        int channelsA = a.Shape[1];
        int channelsB = b.Shape[1];
        int height = a.Shape[2];
        int width = a.Shape[3];

        var output = new Tensor<T>([batchSize, channelsA + channelsB, height, width]);

        for (int batch = 0; batch < batchSize; batch++)
        {
            for (int c = 0; c < channelsA; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[batch, c, h, w] = a[batch, c, h, w];

            for (int c = 0; c < channelsB; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[batch, channelsA + c, h, w] = b[batch, c, h, w];
        }

        return output;
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, height * 2, width * 2]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        T val = input[b, c, h, w];
                        output[b, c, h * 2, w * 2] = val;
                        output[b, c, h * 2, w * 2 + 1] = val;
                        output[b, c, h * 2 + 1, w * 2] = val;
                        output[b, c, h * 2 + 1, w * 2 + 1] = val;
                    }

        return output;
    }

    private Tensor<T> ApplyReLU(Tensor<T> input) =>
        input.Transform((v, _) => NumOps.FromDouble(Math.Max(0, Convert.ToDouble(v))));

    private Tensor<T> ApplySigmoid(Tensor<T> input) =>
        input.Transform((v, _) => NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-Convert.ToDouble(v)))));

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

    protected override void InitializeLayers() => ClearLayers();

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
        ModelType = ModelType.VideoInpainting,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "E2FGVI" },
            { "Description", "End-to-End Flow-Guided Video Inpainting" },
            { "InputHeight", _height },
            { "InputWidth", _width }
        },
        ModelData = this.Serialize()
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFeatures);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new E2FGVI<T>(Architecture, _numFeatures);

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
