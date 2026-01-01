using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// FastDVDNet: Towards Real-Time Deep Video Denoising Without Flow Estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FastDVDNet removes noise from video while preserving details
/// and maintaining temporal consistency across frames. Unlike image denoisers,
/// it uses multiple frames to reduce noise more effectively.
///
/// Key advantages:
/// - Real-time video denoising
/// - No optical flow computation needed
/// - Handles various noise levels
/// - Preserves temporal consistency
///
/// Example usage:
/// <code>
/// var model = new FastDVDNet&lt;double&gt;(arch);
/// var denoisedFrames = model.Denoise(noisyFrames, noiseLevel: 25);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Two-stage denoising pipeline
/// - Stage 1: Denoise groups of 3 frames
/// - Stage 2: Fuse stage 1 outputs temporally
/// - Noise map as additional input for noise-level adaptation
/// </para>
/// <para>
/// <b>Reference:</b> "FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation"
/// https://arxiv.org/abs/1907.01361
/// </para>
/// </remarks>
public class FastDVDNet<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private int _numFeatures;
    private int _numInputFrames;
    private int _imageHeight;
    private int _imageWidth;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumFeatures => _numFeatures;
    internal int NumInputFrames => _numInputFrames;

    #endregion

    #region Constructors

    public FastDVDNet(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 32,
        int numInputFrames = 5)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _useNativeMode = true;
        _numFeatures = numFeatures;
        _numInputFrames = numInputFrames;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 854;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public FastDVDNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"FastDVDNet ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numFeatures = 32;
        _numInputFrames = 5;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 854;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Denoises a sequence of video frames.
    /// </summary>
    /// <param name="frames">Noisy video frames.</param>
    /// <param name="noiseLevel">Estimated noise standard deviation (sigma, typically 0-75).</param>
    public List<Tensor<T>> Denoise(List<Tensor<T>> frames, double noiseLevel = 25.0)
    {
        var denoised = new List<Tensor<T>>();
        int halfWindow = _numInputFrames / 2;

        for (int i = 0; i < frames.Count; i++)
        {
            var inputFrames = new List<Tensor<T>>();
            for (int j = -halfWindow; j <= halfWindow; j++)
            {
                int idx = MathHelper.Clamp(i + j, 0, frames.Count - 1);
                inputFrames.Add(frames[idx]);
            }

            var denoisedFrame = DenoiseFrame(inputFrames, noiseLevel);
            denoised.Add(denoisedFrame);
        }

        return denoised;
    }

    /// <summary>
    /// Denoises a single frame using neighboring frames.
    /// </summary>
    public Tensor<T> DenoiseFrame(List<Tensor<T>> neighborFrames, double noiseLevel = 25.0)
    {
        var noiseMap = CreateNoiseMap(neighborFrames[0], noiseLevel);
        var stacked = StackFramesWithNoiseMap(neighborFrames, noiseMap);

        return _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);
    }

    /// <summary>
    /// Estimates the noise level in a frame.
    /// </summary>
    public double EstimateNoiseLevel(Tensor<T> frame)
    {
        // Median Absolute Deviation (MAD) estimator
        var data = new List<double>();

        // Handle both 4D [N, C, H, W] and 3D [C, H, W] or 2D [H, W] tensors
        int h, w, stride;
        if (frame.Rank == 4)
        {
            // 4D tensor: [batch, channels, height, width]
            h = frame.Shape[2];
            w = frame.Shape[3];
            stride = frame.Shape[3]; // Width stride for NCHW layout
            // Use first batch, first channel for noise estimation
        }
        else if (frame.Rank == 3)
        {
            // 3D tensor: [channels, height, width]
            h = frame.Shape[1];
            w = frame.Shape[2];
            stride = frame.Shape[2];
        }
        else
        {
            // 2D tensor: [height, width]
            h = frame.Shape[0];
            w = frame.Shape[1];
            stride = w;
        }

        // Compute Laplacian for high-frequency noise estimation
        // For 4D tensors, use the first channel of the first batch element
        for (int y = 1; y < h - 1; y++)
        {
            for (int x = 1; x < w - 1; x++)
            {
                // Proper indexing for any rank tensor (use first batch/channel)
                double center = Convert.ToDouble(frame.Data[y * stride + x]);
                double laplacian = 4 * center
                    - Convert.ToDouble(frame.Data[(y - 1) * stride + x])
                    - Convert.ToDouble(frame.Data[(y + 1) * stride + x])
                    - Convert.ToDouble(frame.Data[y * stride + (x - 1)])
                    - Convert.ToDouble(frame.Data[y * stride + (x + 1)]);
                data.Add(Math.Abs(laplacian));
            }
        }

        if (data.Count == 0)
            return 0.0;

        data.Sort();
        double median = data[data.Count / 2];

        // Convert MAD to sigma estimate
        return median / 0.6745;
    }

    /// <summary>
    /// Adds synthetic noise to frames for testing.
    /// </summary>
    public Tensor<T> AddGaussianNoise(Tensor<T> frame, double sigma)
    {
        var random = RandomHelper.CreateSecureRandom();
        var noisy = new Tensor<T>(frame.Shape);

        for (int i = 0; i < frame.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2) * (sigma / 255.0);

            double value = Convert.ToDouble(frame.Data[i]) + noise;
            value = MathHelper.Clamp(value, 0.0, 1.0);
            noisy.Data[i] = NumOps.FromDouble(value);
        }

        return noisy;
    }

    #endregion

    #region Private Methods

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

    private Tensor<T> CreateNoiseMap(Tensor<T> frame, double noiseLevel)
    {
        int h = frame.Rank == 4 ? frame.Shape[2] : frame.Shape[1];
        int w = frame.Rank == 4 ? frame.Shape[3] : frame.Shape[2];

        var noiseMap = new Tensor<T>([1, 1, h, w]);
        double normalizedNoise = noiseLevel / 255.0;

        for (int i = 0; i < noiseMap.Length; i++)
        {
            noiseMap.Data[i] = NumOps.FromDouble(normalizedNoise);
        }

        return noiseMap;
    }

    private Tensor<T> StackFramesWithNoiseMap(List<Tensor<T>> frames, Tensor<T> noiseMap)
    {
        var first = frames[0];
        int c = first.Rank == 4 ? first.Shape[1] : first.Shape[0];
        int h = first.Rank == 4 ? first.Shape[2] : first.Shape[1];
        int w = first.Rank == 4 ? first.Shape[3] : first.Shape[2];

        int totalChannels = c * frames.Count + 1; // frames + noise map
        var stacked = new Tensor<T>([1, totalChannels, h, w]);

        int offset = 0;
        foreach (var frame in frames)
        {
            Array.Copy(frame.Data, 0, stacked.Data, offset, frame.Data.Length);
            offset += c * h * w;
        }

        // Add noise map
        Array.Copy(noiseMap.Data, 0, stacked.Data, offset, noiseMap.Data.Length);

        return stacked;
    }

    public override Tensor<T> Predict(Tensor<T> input) => Forward(input);

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFastDVDNetLayers(ch, _imageHeight, _imageWidth, _numFeatures, _numInputFrames));
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
        ModelType = ModelType.VideoDenoising,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "FastDVDNet" }, { "NumFeatures", _numFeatures },
            { "NumInputFrames", _numInputFrames }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures); writer.Write(_numInputFrames);
        writer.Write(_imageHeight); writer.Write(_imageWidth);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Restore serialized configuration values
        _numFeatures = reader.ReadInt32();
        _numInputFrames = reader.ReadInt32();
        _imageHeight = reader.ReadInt32();
        _imageWidth = reader.ReadInt32();

        // Reinitialize layers with restored configuration
        ClearLayers();
        InitializeLayers();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new FastDVDNet<T>(Architecture, _optimizer, _lossFunction, _numFeatures, _numInputFrames);

    #endregion
}
