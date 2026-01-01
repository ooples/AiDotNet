using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// Real-time Intermediate Flow Estimation (RIFE) for video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> RIFE is a state-of-the-art model for generating intermediate frames between
/// two existing video frames. This is useful for:
/// - Increasing video frame rate (e.g., 24fps to 60fps)
/// - Creating slow-motion effects
/// - Smoothing video playback
/// - Reducing temporal aliasing
///
/// RIFE uses a privileged distillation approach with intermediate flow estimation
/// to create realistic frames at arbitrary positions between input frames.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Uses IFNet architecture for intermediate flow estimation
/// - Coarse-to-fine flow refinement across multiple scales
/// - Context-aware fusion with feature maps
/// - Supports arbitrary timestep interpolation (not just midpoint)
/// </para>
/// <para>
/// <b>Reference:</b> Huang et al., "RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation"
/// ECCV 2022.
/// </para>
/// </remarks>
public class RIFE<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _numFlowBlocks;

    // IFNet components - coarse to fine flow estimation
    private readonly List<ConvolutionalLayer<T>> _encoder;
    private readonly List<ConvolutionalLayer<T>> _flowDecoder;
    private readonly List<ConvolutionalLayer<T>> _contextEncoder;
    private ConvolutionalLayer<T>? _fusion;
    private ConvolutionalLayer<T>? _outputConv;

    // Flow refinement blocks
    private readonly List<ConvolutionalLayer<T>> _flowBlocks;

    private const int DefaultNumFeatures = 64;
    private const int DefaultNumFlowBlocks = 8;

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
    /// Initializes a new instance of the RIFE class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">The number of features in intermediate layers.</param>
    /// <param name="numFlowBlocks">The number of flow refinement blocks.</param>
    public RIFE(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = DefaultNumFeatures,
        int numFlowBlocks = DefaultNumFlowBlocks)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _numFlowBlocks = numFlowBlocks;

        _encoder = [];
        _flowDecoder = [];
        _contextEncoder = [];
        _flowBlocks = [];

        InitializeNativeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Interpolates frames between two input frames.
    /// </summary>
    /// <param name="frame1">The first frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="frame2">The second frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="timestep">The interpolation position (0.0 = frame1, 1.0 = frame2, 0.5 = midpoint).</param>
    /// <returns>The interpolated frame.</returns>
    public Tensor<T> Interpolate(Tensor<T> frame1, Tensor<T> frame2, double timestep = 0.5)
    {
        timestep = Math.Max(0.0, Math.Min(1.0, timestep));

        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        var concatenated = ConcatenateChannels(frame1, frame2);
        var result = ProcessInterpolation(concatenated, timestep);

        if (!hasBatch)
        {
            result = RemoveBatchDimension(result);
        }

        return result;
    }

    /// <summary>
    /// Interpolates multiple frames between input frames for frame rate upsampling.
    /// </summary>
    /// <param name="frames">List of input frames.</param>
    /// <param name="factor">The upsampling factor (e.g., 2 doubles frame rate).</param>
    /// <returns>List of interpolated frames including original frames.</returns>
    public List<Tensor<T>> UpsampleFrameRate(List<Tensor<T>> frames, int factor = 2)
    {
        if (frames.Count < 2)
        {
            return [.. frames];
        }

        var result = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count - 1; i++)
        {
            result.Add(frames[i]);

            for (int j = 1; j < factor; j++)
            {
                double timestep = (double)j / factor;
                var interpolated = Interpolate(frames[i], frames[i + 1], timestep);
                result.Add(interpolated);
            }
        }

        result.Add(frames[^1]);
        return result;
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Input should be concatenated frames [B, 2*C, H, W]
        return ProcessInterpolation(input, 0.5);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var predicted = Predict(input);

        // Calculate loss gradient
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data[idx]));

        // Backward pass
        BackwardPass(lossGradient);

        // Update parameters
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
        // Encoder: extract features from concatenated frames
        _encoder.Add(new ConvolutionalLayer<T>(
            _channels * 2, _height, _width, _numFeatures, 3, 1, 1));
        _encoder.Add(new ConvolutionalLayer<T>(
            _numFeatures, _height, _width, _numFeatures * 2, 3, 2, 1));
        _encoder.Add(new ConvolutionalLayer<T>(
            _numFeatures * 2, _height / 2, _width / 2, _numFeatures * 4, 3, 2, 1));

        // Context encoder
        _contextEncoder.Add(new ConvolutionalLayer<T>(
            _channels * 2, _height, _width, _numFeatures, 3, 1, 1));
        _contextEncoder.Add(new ConvolutionalLayer<T>(
            _numFeatures, _height, _width, _numFeatures, 3, 1, 1));

        // Flow decoder
        int decoderH = _height / 4;
        int decoderW = _width / 4;

        _flowDecoder.Add(new ConvolutionalLayer<T>(
            _numFeatures * 4, decoderH, decoderW, _numFeatures * 2, 3, 1, 1));
        _flowDecoder.Add(new ConvolutionalLayer<T>(
            _numFeatures * 2, decoderH * 2, decoderW * 2, _numFeatures, 3, 1, 1));
        _flowDecoder.Add(new ConvolutionalLayer<T>(
            _numFeatures, decoderH * 4, decoderW * 4, 4, 3, 1, 1));

        // Flow refinement blocks
        for (int i = 0; i < _numFlowBlocks; i++)
        {
            _flowBlocks.Add(new ConvolutionalLayer<T>(
                _numFeatures + 4, _height, _width, _numFeatures, 3, 1, 1));
        }

        // Fusion layer
        int fusionInputChannels = _channels * 2 + _numFeatures + 4;
        _fusion = new ConvolutionalLayer<T>(
            fusionInputChannels, _height, _width, _numFeatures, 3, 1, 1);

        // Output convolution
        _outputConv = new ConvolutionalLayer<T>(
            _numFeatures, _height, _width, _channels, 3, 1, 1);

        // Register all layers
        foreach (var layer in _encoder) Layers.Add(layer);
        foreach (var layer in _flowDecoder) Layers.Add(layer);
        foreach (var layer in _contextEncoder) Layers.Add(layer);
        foreach (var layer in _flowBlocks) Layers.Add(layer);
        Layers.Add(_fusion);
        Layers.Add(_outputConv);
    }

    private Tensor<T> ProcessInterpolation(Tensor<T> concatenatedFrames, double timestep)
    {
        var frame1 = SliceChannels(concatenatedFrames, 0, _channels);
        var frame2 = SliceChannels(concatenatedFrames, _channels, _channels * 2);

        // Encode features
        var features = concatenatedFrames;
        foreach (var encoder in _encoder)
        {
            features = encoder.Forward(features);
        }

        // Decode flow
        var flowFeatures = features;
        for (int i = 0; i < _flowDecoder.Count; i++)
        {
            flowFeatures = _flowDecoder[i].Forward(flowFeatures);
            if (i < _flowDecoder.Count - 1)
            {
                flowFeatures = BilinearUpsample(flowFeatures, 2);
            }
        }

        var flow = flowFeatures;
        var flow_0_1 = SliceChannels(flow, 0, 2);
        var flow_1_0 = SliceChannels(flow, 2, 4);

        var t = NumOps.FromDouble(timestep);
        var oneMinusT = NumOps.FromDouble(1.0 - timestep);

        // Scale flows correctly for interpolation at time t:
        // - frame1 warped toward t position uses flow_0_1 scaled by t
        // - frame2 warped toward t position uses flow_1_0 scaled by (1-t)
        var flow_t_0 = ScaleFlow(flow_0_1, t);
        var flow_t_1 = ScaleFlow(flow_1_0, oneMinusT);

        var frame1_warped = WarpImage(frame1, flow_t_0);
        var frame2_warped = WarpImage(frame2, flow_t_1);

        var context = concatenatedFrames;
        foreach (var contextEnc in _contextEncoder)
        {
            context = contextEnc.Forward(context);
        }

        var fusionInput = ConcatenateChannels(
            ConcatenateChannels(frame1_warped, frame2_warped),
            ConcatenateChannels(context, flow));

        var fused = _fusion!.Forward(fusionInput);

        for (int i = 0; i < _flowBlocks.Count; i++)
        {
            var blockInput = ConcatenateChannels(fused, flow);
            fused = _flowBlocks[i].Forward(blockInput);
        }

        return _outputConv!.Forward(fused);
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        gradient = _outputConv!.Backward(gradient);
        gradient = _fusion!.Backward(gradient);

        for (int i = _flowBlocks.Count - 1; i >= 0; i--)
        {
            gradient = _flowBlocks[i].Backward(gradient);
        }

        for (int i = _contextEncoder.Count - 1; i >= 0; i--)
        {
            gradient = _contextEncoder[i].Backward(gradient);
        }

        for (int i = _flowDecoder.Count - 1; i >= 0; i--)
        {
            gradient = _flowDecoder[i].Backward(gradient);
        }

        for (int i = _encoder.Count - 1; i >= 0; i--)
        {
            gradient = _encoder[i].Backward(gradient);
        }
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> t1, Tensor<T> t2)
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

    private Tensor<T> SliceChannels(Tensor<T> input, int startChannel, int endChannel)
    {
        int batchSize = input.Shape[0];
        int numChannels = endChannel - startChannel;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var result = new Tensor<T>([batchSize, numChannels, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numChannels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        result[b, c, h, w] = input[b, startChannel + c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ScaleFlow(Tensor<T> flow, T scale)
    {
        return flow.Transform((v, _) => NumOps.Multiply(v, scale));
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
        // Convert [C, H, W] to [1, C, H, W]
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        // Convert [1, C, H, W] to [C, H, W]
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([c, h, w]);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
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

        // Update encoder layers
        foreach (var layer in _encoder)
        {
            var layerParams = layer.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        // Update flow decoder layers
        foreach (var layer in _flowDecoder)
        {
            var layerParams = layer.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        // Update context encoder layers
        foreach (var layer in _contextEncoder)
        {
            var layerParams = layer.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        // Update flow blocks
        foreach (var layer in _flowBlocks)
        {
            var layerParams = layer.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        // Update fusion and output layers
        if (_fusion != null)
        {
            var layerParams = _fusion.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _fusion.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        if (_outputConv != null)
        {
            var layerParams = _outputConv.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _outputConv.SetParameters(newParams);
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "RIFE" },
            { "Description", "Real-time Intermediate Flow Estimation for Video Frame Interpolation" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumFeatures", _numFeatures },
            { "NumFlowBlocks", _numFlowBlocks },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            ModelType = ModelType.FrameInterpolation,
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
        writer.Write(_numFlowBlocks);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // height
        _ = reader.ReadInt32(); // width
        _ = reader.ReadInt32(); // channels
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // numFlowBlocks
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RIFE<T>(Architecture, _numFeatures, _numFlowBlocks);
    }

    #endregion
}
