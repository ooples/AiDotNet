using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Matting;

/// <summary>
/// RVM: Robust Video Matting for real-time human segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> RVM extracts people from video backgrounds in real-time.
/// Unlike simple background removal that creates hard edges, matting produces
/// a soft alpha matte that preserves hair details and semi-transparent regions.
///
/// Key capabilities:
/// - Real-time video matting without green screen
/// - High-quality alpha matte output
/// - Temporal consistency across frames
/// - Works with any background
///
/// Outputs:
/// - Alpha matte: Transparency at each pixel (0=background, 1=foreground)
/// - Foreground: RGB colors of the person with pre-multiplied alpha
///
/// Example usage:
/// <code>
/// var model = new RVM&lt;double&gt;(arch);
/// var (alpha, foreground) = model.MatteSingleFrame(frame);
/// var composite = model.CompositeWithBackground(foreground, alpha, newBackground);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - MobileNetV3 backbone for efficiency
/// - Recurrent architecture for temporal consistency
/// - Deep guided filter for detail refinement
/// - Multi-resolution processing
/// </para>
/// <para>
/// <b>Reference:</b> "Robust High-Resolution Video Matting with Temporal Guidance"
/// https://arxiv.org/abs/2108.11515
/// </para>
/// </remarks>
public class RVM<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _numFeatures;
    private readonly int _imageHeight;
    private readonly int _imageWidth;

    // Recurrent hidden state
    private Tensor<T>? _hiddenState;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumFeatures => _numFeatures;
    internal int ImageHeight => _imageHeight;
    internal int ImageWidth => _imageWidth;

    #endregion

    #region Constructors

    public RVM(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 32)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _useNativeMode = true;
        _numFeatures = numFeatures;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 512;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public RVM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"RVM ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numFeatures = 32;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Processes a video to extract alpha mattes and foregrounds.
    /// </summary>
    public List<(Tensor<T> Alpha, Tensor<T> Foreground)> MatteVideo(List<Tensor<T>> frames)
    {
        var results = new List<(Tensor<T>, Tensor<T>)>();
        ResetState();

        foreach (var frame in frames)
        {
            var (alpha, foreground) = MatteSingleFrame(frame);
            results.Add((alpha, foreground));
        }

        return results;
    }

    /// <summary>
    /// Mattes a single frame, maintaining temporal consistency with previous frames.
    /// </summary>
    public (Tensor<T> Alpha, Tensor<T> Foreground) MatteSingleFrame(Tensor<T> frame)
    {
        var output = _useNativeMode ? Forward(frame) : PredictOnnx(frame);

        // Split output into alpha (1 channel) and foreground RGB (3 channels)
        int c = output.Rank == 4 ? output.Shape[1] : output.Shape[0];
        int h = output.Rank == 4 ? output.Shape[2] : output.Shape[1];
        int w = output.Rank == 4 ? output.Shape[3] : output.Shape[2];

        var alpha = new Tensor<T>([1, 1, h, w]);
        var foreground = new Tensor<T>([1, 3, h, w]);

        // First channel is alpha
        output.Data.Span.Slice(0, h * w).CopyTo(alpha.Data.Span);

        // Remaining 3 channels are foreground RGB
        output.Data.Span.Slice(h * w, 3 * h * w).CopyTo(foreground.Data.Span);

        return (alpha, foreground);
    }

    /// <summary>
    /// Composites the foreground onto a new background.
    /// </summary>
    public Tensor<T> CompositeWithBackground(Tensor<T> foreground, Tensor<T> alpha, Tensor<T> background)
    {
        int h = foreground.Rank == 4 ? foreground.Shape[2] : foreground.Shape[1];
        int w = foreground.Rank == 4 ? foreground.Shape[3] : foreground.Shape[2];

        var composite = new Tensor<T>(foreground.Shape);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double a = Convert.ToDouble(alpha[0, 0, y, x]);

                for (int c = 0; c < 3; c++)
                {
                    double fg = Convert.ToDouble(foreground[0, c, y, x]);
                    double bg = Convert.ToDouble(background[0, c, y, x]);
                    double comp = a * fg + (1 - a) * bg;
                    composite[0, c, y, x] = NumOps.FromDouble(comp);
                }
            }
        }

        return composite;
    }

    /// <summary>
    /// Resets the recurrent state for a new video.
    /// </summary>
    public new void ResetState()
    {
        _hiddenState = null;
        base.ResetState();
    }

    /// <summary>
    /// Extracts just the alpha matte.
    /// </summary>
    public Tensor<T> GetAlpha(Tensor<T> frame)
    {
        var (alpha, _) = MatteSingleFrame(frame);
        return alpha;
    }

    /// <summary>
    /// Creates a green screen effect (extracts foreground).
    /// </summary>
    public Tensor<T> GreenScreenExtract(Tensor<T> frame)
    {
        var (alpha, foreground) = MatteSingleFrame(frame);

        int h = foreground.Rank == 4 ? foreground.Shape[2] : foreground.Shape[1];
        int w = foreground.Rank == 4 ? foreground.Shape[3] : foreground.Shape[2];

        // Create green background
        var greenBg = new Tensor<T>([1, 3, h, w]);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                greenBg[0, 0, y, x] = NumOps.Zero;
                greenBg[0, 1, y, x] = NumOps.FromDouble(1.0); // Green
                greenBg[0, 2, y, x] = NumOps.Zero;
            }
        }

        return CompositeWithBackground(foreground, alpha, greenBg);
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;

        // Use hidden state for temporal consistency if available
        if (_hiddenState != null && _hiddenState.Shape.SequenceEqual(input.Shape))
        {
            // Blend with previous state for temporal smoothing
            for (int i = 0; i < Math.Min(result.Length, _hiddenState.Length); i++)
            {
                double curr = Convert.ToDouble(result.Data.Span[i]);
                double prev = Convert.ToDouble(_hiddenState.Data.Span[i]);
                result.Data.Span[i] = NumOps.FromDouble(curr * 0.9 + prev * 0.1);
            }
        }

        foreach (var layer in Layers) result = layer.Forward(result);

        // Update hidden state for next frame
        _hiddenState = new Tensor<T>(result.Shape);
        result.Data.Span.CopyTo(_hiddenState.Data.Span);

        return result;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultRVMLayers(ch, _imageHeight, _imageWidth, _numFeatures));
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
        ModelType = ModelType.VideoMatting,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "RVM" }, { "NumFeatures", _numFeatures },
            { "ImageHeight", _imageHeight }, { "ImageWidth", _imageWidth }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures); writer.Write(_imageHeight); writer.Write(_imageWidth);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        for (int i = 0; i < 3; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new RVM<T>(Architecture, _optimizer, _lossFunction, _numFeatures);

    #endregion
}
