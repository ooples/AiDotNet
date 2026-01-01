using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FLAVR interpolates frames between existing video frames to create
/// smoother slow-motion effects or increase video frame rate. Unlike other methods that
/// explicitly estimate optical flow, FLAVR directly synthesizes intermediate frames.
///
/// Key advantages:
/// - No explicit optical flow computation (faster)
/// - Can generate multiple intermediate frames at once
/// - Uses 3D convolutions for spatiotemporal understanding
/// - Handles large motions better than flow-based methods
///
/// Example usage:
/// <code>
/// var model = new FLAVR&lt;double&gt;(arch);
/// var interpolatedFrames = model.Interpolate(frame1, frame2, numInterpolations: 3);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - 3D encoder-decoder architecture with skip connections
/// - Multi-scale feature extraction
/// - Direct frame synthesis without flow estimation
/// </para>
/// <para>
/// <b>Reference:</b> "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"
/// https://arxiv.org/abs/2012.08512
/// </para>
/// </remarks>
public class FLAVR<T> : NeuralNetworkBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private int _numFeatures;
    private int _numInputFrames;
    private int _numOutputFrames;
    private readonly int _imageSize;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumFeatures => _numFeatures;
    internal int NumInputFrames => _numInputFrames;
    internal int NumOutputFrames => _numOutputFrames;

    #endregion

    #region Constructors

    public FLAVR(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 64,
        int numInputFrames = 4,
        int numOutputFrames = 1)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _useNativeMode = true;
        _numFeatures = numFeatures;
        _numInputFrames = numInputFrames;
        _numOutputFrames = numOutputFrames;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 256;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public FLAVR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numOutputFrames = 1)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"FLAVR ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numFeatures = 64;
        _numInputFrames = 4;
        _numOutputFrames = numOutputFrames;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Interpolates frames between two input frames.
    /// </summary>
    public List<Tensor<T>> Interpolate(Tensor<T> frame1, Tensor<T> frame2, int numInterpolations = 1)
    {
        var results = new List<Tensor<T>>();
        var stacked = StackFrames([frame1, frame2]);

        // Get the model's mid-point interpolation
        var midInterpolated = _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);

        for (int i = 0; i < numInterpolations; i++)
        {
            // Compute temporal position t ∈ (0, 1)
            double t = (i + 1.0) / (numInterpolations + 1.0);

            // Blend between frames based on temporal position
            // t < 0.5: blend between frame1 and midInterpolated
            // t > 0.5: blend between midInterpolated and frame2
            Tensor<T> interpolated;
            if (Math.Abs(t - 0.5) < 1e-6)
            {
                // At midpoint, use the model output directly
                interpolated = midInterpolated;
            }
            else if (t < 0.5)
            {
                // Blend frame1 toward midInterpolated
                double blendFactor = t * 2.0; // 0 to 1 as t goes from 0 to 0.5
                interpolated = BlendFrames(frame1, midInterpolated, blendFactor);
            }
            else
            {
                // Blend midInterpolated toward frame2
                double blendFactor = (t - 0.5) * 2.0; // 0 to 1 as t goes from 0.5 to 1
                interpolated = BlendFrames(midInterpolated, frame2, blendFactor);
            }
            results.Add(interpolated);
        }
        return results;
    }

    /// <summary>
    /// Blends two frames together with a given factor.
    /// </summary>
    private Tensor<T> BlendFrames(Tensor<T> frameA, Tensor<T> frameB, double blendFactor)
    {
        var result = new Tensor<T>(frameA.Shape);
        T factorA = NumOps.FromDouble(1.0 - blendFactor);
        T factorB = NumOps.FromDouble(blendFactor);

        for (int i = 0; i < frameA.Length; i++)
        {
            result.Data[i] = NumOps.Add(
                NumOps.Multiply(frameA.Data[i], factorA),
                NumOps.Multiply(frameB.Data[i], factorB));
        }
        return result;
    }

    /// <summary>
    /// Interpolates frames using 4 input frames for better quality.
    /// </summary>
    public Tensor<T> InterpolateWith4Frames(Tensor<T> f0, Tensor<T> f1, Tensor<T> f2, Tensor<T> f3)
    {
        var stacked = StackFrames([f0, f1, f2, f3]);
        return _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);
    }

    /// <summary>
    /// Doubles the frame rate of a video.
    /// </summary>
    public List<Tensor<T>> DoubleFrameRate(List<Tensor<T>> frames)
    {
        var results = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count - 1; i++)
        {
            results.Add(frames[i]);
            var interpolated = Interpolate(frames[i], frames[i + 1], 1);
            results.AddRange(interpolated);
        }
        results.Add(frames[^1]);

        return results;
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

    private Tensor<T> StackFrames(List<Tensor<T>> frames)
    {
        var first = frames[0];
        int c = first.Rank == 4 ? first.Shape[1] : first.Shape[0];
        int h = first.Rank == 4 ? first.Shape[2] : first.Shape[1];
        int w = first.Rank == 4 ? first.Shape[3] : first.Shape[2];

        var stacked = new Tensor<T>([1, c * frames.Count, h, w]);
        for (int f = 0; f < frames.Count; f++)
            Array.Copy(frames[f].Data, 0, stacked.Data, f * c * h * w, c * h * w);
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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 256;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 256;
            Layers.AddRange(LayerHelper<T>.CreateDefaultFLAVRLayers(ch, h, w, _numFeatures, _numInputFrames));
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
        ModelType = ModelType.FrameInterpolation,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "FLAVR" }, { "NumFeatures", _numFeatures },
            { "NumInputFrames", _numInputFrames }, { "NumOutputFrames", _numOutputFrames }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures); writer.Write(_numInputFrames); writer.Write(_numOutputFrames);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numFeatures = reader.ReadInt32();
        _numInputFrames = reader.ReadInt32();
        _numOutputFrames = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new FLAVR<T>(Architecture, _optimizer, _lossFunction, _numFeatures, _numInputFrames, _numOutputFrames);

    #endregion
}
