using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Motion;

/// <summary>
/// FlowFormer: A Transformer Architecture for Optical Flow.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FlowFormer estimates optical flow - the apparent motion of objects
/// between consecutive video frames. Unlike traditional methods, it uses transformers
/// to capture long-range dependencies in the cost volume.
///
/// Optical flow is useful for:
/// - Video stabilization
/// - Object tracking
/// - Action recognition
/// - Video editing and effects
///
/// Example usage:
/// <code>
/// var model = new FlowFormer&lt;double&gt;(arch);
/// var flow = model.EstimateFlow(frame1, frame2);
/// // flow[0] = horizontal motion, flow[1] = vertical motion
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Transformer-based cost volume aggregation
/// - Latent cost tokens for efficient memory
/// - Iterative flow refinement
/// - State-of-the-art accuracy on Sintel and KITTI benchmarks
/// </para>
/// <para>
/// <b>Reference:</b> "FlowFormer: A Transformer Architecture for Optical Flow" ECCV 2022
/// https://arxiv.org/abs/2203.16194
/// </para>
/// </remarks>
public class FlowFormer<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embedDim;
    private readonly int _numLayers;
    private readonly int _numIterations;
    private readonly int _imageHeight;
    private readonly int _imageWidth;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int EmbedDim => _embedDim;
    internal int NumLayers => _numLayers;
    internal int NumIterations => _numIterations;

    #endregion

    #region Constructors

    public FlowFormer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embedDim = 256,
        int numLayers = 6,
        int numIterations = 12)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _useNativeMode = true;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _numIterations = numIterations;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 448;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public FlowFormer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"FlowFormer ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _embedDim = 256;
        _numLayers = 6;
        _numIterations = 12;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 448;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Estimates optical flow between two frames.
    /// </summary>
    /// <param name="frame1">First frame [B, C, H, W] or [C, H, W].</param>
    /// <param name="frame2">Second frame with same shape.</param>
    /// <returns>Flow tensor [B, 2, H, W] where channel 0 is horizontal and channel 1 is vertical flow.</returns>
    public Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        if (frame1 is null) throw new ArgumentNullException(nameof(frame1));
        if (frame2 is null) throw new ArgumentNullException(nameof(frame2));

        var stacked = ConcatenateFrames(frame1, frame2);
        return _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);
    }

    /// <summary>
    /// Estimates bidirectional flow (forward and backward).
    /// </summary>
    public (Tensor<T> Forward, Tensor<T> Backward) EstimateBidirectionalFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        var forwardFlow = EstimateFlow(frame1, frame2);
        var backwardFlow = EstimateFlow(frame2, frame1);
        return (forwardFlow, backwardFlow);
    }

    /// <summary>
    /// Computes flow for all consecutive frame pairs in a video.
    /// </summary>
    public List<Tensor<T>> EstimateFlowForVideo(List<Tensor<T>> frames)
    {
        var flows = new List<Tensor<T>>();
        for (int i = 0; i < frames.Count - 1; i++)
        {
            flows.Add(EstimateFlow(frames[i], frames[i + 1]));
        }
        return flows;
    }

    /// <summary>
    /// Warps an image using the estimated flow.
    /// </summary>
    public Tensor<T> WarpWithFlow(Tensor<T> image, Tensor<T> flow)
    {
        int b = image.Rank == 4 ? image.Shape[0] : 1;
        int c = image.Rank == 4 ? image.Shape[1] : image.Shape[0];
        int h = image.Rank == 4 ? image.Shape[2] : image.Shape[1];
        int w = image.Rank == 4 ? image.Shape[3] : image.Shape[2];

        var warped = new Tensor<T>(image.Shape);

        for (int batch = 0; batch < b; batch++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    double fx = Convert.ToDouble(flow[batch, 0, y, x]);
                    double fy = Convert.ToDouble(flow[batch, 1, y, x]);

                    double srcX = x + fx;
                    double srcY = y + fy;

                    for (int ch = 0; ch < c; ch++)
                    {
                        warped[batch, ch, y, x] = BilinearSample(image, batch, ch, srcY, srcX, h, w);
                    }
                }
            }
        }

        return warped;
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers) result = layer.Forward(result);
        return result;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    private Tensor<T> ConcatenateFrames(Tensor<T> frame1, Tensor<T> frame2)
    {
        int c = frame1.Rank == 4 ? frame1.Shape[1] : frame1.Shape[0];
        int h = frame1.Rank == 4 ? frame1.Shape[2] : frame1.Shape[1];
        int w = frame1.Rank == 4 ? frame1.Shape[3] : frame1.Shape[2];

        var concat = new Tensor<T>([1, c * 2, h, w]);
        Array.Copy(frame1.Data, 0, concat.Data, 0, frame1.Data.Length);
        Array.Copy(frame2.Data, 0, concat.Data, frame1.Data.Length, frame2.Data.Length);
        return concat;
    }

    private T BilinearSample(Tensor<T> tensor, int b, int c, double y, double x, int h, int w)
    {
        int x0 = MathHelper.Clamp((int)Math.Floor(x), 0, w - 1);
        int x1 = MathHelper.Clamp(x0 + 1, 0, w - 1);
        int y0 = MathHelper.Clamp((int)Math.Floor(y), 0, h - 1);
        int y1 = MathHelper.Clamp(y0 + 1, 0, h - 1);

        double dx = x - Math.Floor(x);
        double dy = y - Math.Floor(y);

        double v00 = Convert.ToDouble(tensor[b, c, y0, x0]);
        double v01 = Convert.ToDouble(tensor[b, c, y0, x1]);
        double v10 = Convert.ToDouble(tensor[b, c, y1, x0]);
        double v11 = Convert.ToDouble(tensor[b, c, y1, x1]);

        double value = v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) + v10 * (1 - dx) * dy + v11 * dx * dy;
        return NumOps.FromDouble(value);
    }

    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var gradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var gradTensor = new Tensor<T>(prediction.Shape, gradient);

        for (int i = Layers.Count - 1; i >= 0; i--) gradTensor = Layers[i].Backward(gradTensor);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Serialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            Layers.AddRange(LayerHelper<T>.CreateDefaultFlowFormerLayers(ch, _imageHeight, _imageWidth, _embedDim, _numLayers));
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");
        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            if (p.Length > 0 && offset + p.Length <= parameters.Length)
            {
                var slice = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += p.Length;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.OpticalFlow,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "FlowFormer" }, { "EmbedDim", _embedDim },
            { "NumLayers", _numLayers }, { "NumIterations", _numIterations }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embedDim); writer.Write(_numLayers); writer.Write(_numIterations);
        writer.Write(_imageHeight); writer.Write(_imageWidth);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        for (int i = 0; i < 5; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new FlowFormer<T>(Architecture, _optimizer, _lossFunction, _embedDim, _numLayers, _numIterations);

    #endregion
}
