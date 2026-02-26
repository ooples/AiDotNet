using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Motion;

/// <summary>
/// Recurrent All-pairs Field Transforms (RAFT) for optical flow estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> RAFT is a state-of-the-art optical flow estimation model that predicts
/// the motion between two consecutive video frames. Optical flow represents how pixels move
/// from one frame to the next, useful for:
/// - Motion analysis and tracking
/// - Video stabilization
/// - Action recognition
/// - Video compression
/// - Self-driving car perception
///
/// RAFT iteratively refines its flow estimate using a recurrent update mechanism,
/// making it very accurate while remaining efficient.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Feature extraction using CNN encoder
/// - 4D correlation volumes for all-pairs matching
/// - GRU-based iterative update operator
/// - Multi-scale feature pyramids
/// </para>
/// <para>
/// <b>Reference:</b> Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
/// ECCV 2020.
/// </para>
/// </remarks>
public class RAFT<T> : OpticalFlowBase<T>
{
    private readonly RAFTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _correlationLevels;
    private readonly int _correlationRadius;


    // Feature encoder
    private readonly List<ConvolutionalLayer<T>> _featureEncoder;

    // Context encoder
    private readonly List<ConvolutionalLayer<T>> _contextEncoder;

    // Correlation lookup
    private ConvolutionalLayer<T>? _correlationConv;

    // GRU update block
    private ConvolutionalLayer<T>? _gruConvZ;
    private ConvolutionalLayer<T>? _gruConvR;
    private ConvolutionalLayer<T>? _gruConvH;

    // Flow update heads
    private ConvolutionalLayer<T>? _flowHead;
    private ConvolutionalLayer<T>? _deltaFlowHead;

    // Upsampling
    private ConvolutionalLayer<T>? _upsampleConv;

    private const int DefaultNumFeatures = 256;
    private const int DefaultCorrelationLevels = 4;
    private const int DefaultCorrelationRadius = 4;
    private const int DefaultNumIterations = 12;

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

    /// <summary>
    /// Gets the number of refinement iterations.
    /// </summary>


    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the RAFT class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">The number of features in intermediate layers.</param>
    /// <param name="correlationLevels">The number of levels in the correlation pyramid.</param>
    /// <param name="correlationRadius">The search radius for correlation lookup.</param>
    /// <param name="numIterations">The number of GRU refinement iterations.</param>
    public RAFT(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = DefaultNumFeatures,
        int correlationLevels = DefaultCorrelationLevels,
        int correlationRadius = DefaultCorrelationRadius,
        int numIterations = DefaultNumIterations,
        RAFTOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new RAFTOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _correlationLevels = correlationLevels;
        _correlationRadius = correlationRadius;
        NumIterations = numIterations;

        _featureEncoder = [];
        _contextEncoder = [];

        InitializeNativeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Estimates optical flow between two frames.
    /// </summary>
    /// <param name="frame1">The first frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="frame2">The second frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The optical flow tensor [2, H, W] or [B, 2, H, W].</returns>
    public override Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        var concatenated = ConcatenateChannels(frame1, frame2);
        var result = Predict(concatenated);

        if (!hasBatch)
        {
            result = RemoveBatchDimension(result);
        }

        return result;
    }

    /// <summary>
    /// Estimates optical flow with intermediate flow predictions.
    /// </summary>
    /// <param name="frame1">The first frame tensor.</param>
    /// <param name="frame2">The second frame tensor.</param>
    /// <returns>List of flow predictions at each iteration.</returns>
    public List<Tensor<T>> EstimateFlowIterative(Tensor<T> frame1, Tensor<T> frame2)
    {
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        var flowIterations = ForwardIterative(frame1, frame2);

        if (!hasBatch)
        {
            for (int i = 0; i < flowIterations.Count; i++)
            {
                flowIterations[i] = RemoveBatchDimension(flowIterations[i]);
            }
        }

        return flowIterations;
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var frame1 = SliceChannels(input, 0, _channels);
        var frame2 = SliceChannels(input, _channels, _channels * 2);
        var flowIterations = ForwardIterative(frame1, frame2);
        return flowIterations[^1];
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
        // Check for user-provided custom layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateRAFTLayers(
                channels: _channels, height: _height, width: _width,
                numFeatures: _numFeatures, correlationLevels: _correlationLevels,
                correlationRadius: _correlationRadius).ToList();
            Layers.AddRange(layers);
        }

        // Distribute layers to sub-lists for forward pass
        int idx = 0;
        // Feature encoder (5 layers)
        for (int i = 0; i < 5; i++)
            _featureEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Context encoder (5 layers)
        for (int i = 0; i < 5; i++)
            _contextEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Correlation conv
        _correlationConv = (ConvolutionalLayer<T>)Layers[idx++];
        // GRU update block
        _gruConvZ = (ConvolutionalLayer<T>)Layers[idx++];
        _gruConvR = (ConvolutionalLayer<T>)Layers[idx++];
        _gruConvH = (ConvolutionalLayer<T>)Layers[idx++];
        // Flow heads
        _flowHead = (ConvolutionalLayer<T>)Layers[idx++];
        _deltaFlowHead = (ConvolutionalLayer<T>)Layers[idx++];
        // Upsample conv
        _upsampleConv = (ConvolutionalLayer<T>)Layers[idx++];
    }

    private List<Tensor<T>> ForwardIterative(Tensor<T> frame1, Tensor<T> frame2)
    {
        int batchSize = frame1.Shape[0];
        int featHeight = _height / 8;
        int featWidth = _width / 8;

        var fmap1 = ExtractFeatures(frame1);
        var fmap2 = ExtractFeatures(frame2);
        var context = ExtractContext(frame1);

        var flow = new Tensor<T>([batchSize, 2, featHeight, featWidth]);
        var hiddenState = context;

        var flowPredictions = new List<Tensor<T>>();

        for (int iter = 0; iter < NumIterations; iter++)
        {
            var correlation = ComputeCorrelation(fmap1, fmap2, flow);
            var corrFeatures = _correlationConv!.Forward(correlation);

            var gruInput = ConcatenateChannels(
                ConcatenateChannels(context, corrFeatures),
                flow);

            hiddenState = GRUUpdate(hiddenState, gruInput);

            var flowFeatures = _flowHead!.Forward(hiddenState);
            var deltaFlow = _deltaFlowHead!.Forward(flowFeatures);

            flow = AddTensors(flow, deltaFlow);

            var fullResFlow = UpsampleFlow(flow, 8);
            flowPredictions.Add(fullResFlow);
        }

        return flowPredictions;
    }

    private Tensor<T> ExtractFeatures(Tensor<T> frame)
    {
        var features = frame;
        foreach (var encoder in _featureEncoder)
        {
            features = encoder.Forward(features);
        }
        return features;
    }

    private Tensor<T> ExtractContext(Tensor<T> frame)
    {
        var context = frame;
        foreach (var encoder in _contextEncoder)
        {
            context = encoder.Forward(context);
        }
        return context;
    }

    private Tensor<T> ComputeCorrelation(Tensor<T> fmap1, Tensor<T> fmap2, Tensor<T> flow)
    {
        int batchSize = fmap1.Shape[0];
        int channels = fmap1.Shape[1];
        int height = fmap1.Shape[2];
        int width = fmap1.Shape[3];

        int corrDim = _correlationLevels * (2 * _correlationRadius + 1) * (2 * _correlationRadius + 1);
        var correlation = new Tensor<T>([batchSize, corrDim, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double dx = Convert.ToDouble(flow[b, 0, h, w]);
                    double dy = Convert.ToDouble(flow[b, 1, h, w]);

                    int corrIdx = 0;
                    for (int level = 0; level < _correlationLevels; level++)
                    {
                        int scale = 1 << level;
                        for (int dh = -_correlationRadius; dh <= _correlationRadius; dh++)
                        {
                            for (int dw = -_correlationRadius; dw <= _correlationRadius; dw++)
                            {
                                int h2 = (int)Math.Round(h / (double)scale + dy / scale + dh);
                                int w2 = (int)Math.Round(w / (double)scale + dx / scale + dw);

                                h2 = Math.Max(0, Math.Min(h2, height / scale - 1));
                                w2 = Math.Max(0, Math.Min(w2, width / scale - 1));

                                T corr = NumOps.Zero;
                                for (int c = 0; c < channels; c++)
                                {
                                    T v1 = fmap1[b, c, h, w];
                                    T v2 = fmap2[b, c, h2 * scale, w2 * scale];
                                    corr = NumOps.Add(corr, NumOps.Multiply(v1, v2));
                                }

                                correlation[b, corrIdx, h, w] = corr;
                                corrIdx++;
                            }
                        }
                    }
                }
            }
        }

        return correlation;
    }

    private Tensor<T> GRUUpdate(Tensor<T> hiddenState, Tensor<T> gruInput)
    {
        var z = ApplySigmoid(_gruConvZ!.Forward(gruInput));
        var r = ApplySigmoid(_gruConvR!.Forward(gruInput));
        var hNew = ApplyTanh(_gruConvH!.Forward(gruInput));

        var oneMinusZ = z.Transform((v, _) => NumOps.Subtract(NumOps.One, v));
        var term1 = hiddenState.Transform((v, idx) => NumOps.Multiply(oneMinusZ.Data.Span[idx], v));
        var term2 = hNew.Transform((v, idx) => NumOps.Multiply(z.Data.Span[idx], v));

        return AddTensors(term1, term2);
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
            return NumOps.FromDouble(sigmoid);
        });
    }

    private Tensor<T> ApplyTanh(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            return NumOps.FromDouble(Math.Tanh(x));
        });
    }

    private Tensor<T> UpsampleFlow(Tensor<T> flow, int factor)
    {
        int batchSize = flow.Shape[0];
        int channels = flow.Shape[1];
        int inHeight = flow.Shape[2];
        int inWidth = flow.Shape[3];

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
                        T value = BilinearSample(flow, b, c, srcH, srcW, inHeight, inWidth);
                        output[b, c, h, w] = NumOps.Multiply(value, NumOps.FromDouble(factor));
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

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        // Convert [C, H, W] to [1, C, H, W]
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        // Convert [1, C, H, W] to [C, H, W]
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        gradient = _upsampleConv!.Backward(gradient);
        gradient = _deltaFlowHead!.Backward(gradient);
        gradient = _flowHead!.Backward(gradient);
        gradient = _gruConvH!.Backward(gradient);
        gradient = _gruConvR!.Backward(gradient);
        gradient = _gruConvZ!.Backward(gradient);
        gradient = _correlationConv!.Backward(gradient);

        for (int i = _contextEncoder.Count - 1; i >= 0; i--)
        {
            gradient = _contextEncoder[i].Backward(gradient);
        }

        for (int i = _featureEncoder.Count - 1; i >= 0; i--)
        {
            gradient = _featureEncoder[i].Backward(gradient);
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
            { "ModelName", "RAFT" },
            { "Description", "Recurrent All-Pairs Field Transforms for Optical Flow" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumFeatures", _numFeatures },
            { "CorrelationLevels", _correlationLevels },
            { "CorrelationRadius", _correlationRadius },
            { "NumIterations", NumIterations },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            ModelType = ModelType.OpticalFlow,
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
        writer.Write(_correlationLevels);
        writer.Write(_correlationRadius);
        writer.Write(NumIterations);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RAFT<T>(Architecture, _numFeatures, _correlationLevels, _correlationRadius, NumIterations);
    }

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
