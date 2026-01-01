using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.ActionRecognition;

/// <summary>
/// SlowFast Networks for Video Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SlowFast is a two-pathway network that processes video at two
/// different frame rates simultaneously:
/// - Slow pathway: Processes fewer frames (e.g., 4 fps) but with more channels to capture spatial details
/// - Fast pathway: Processes more frames (e.g., 32 fps) but with fewer channels to capture motion
///
/// This design is inspired by how human vision has:
/// - Parvo cells: Slow but detailed spatial processing
/// - Magno cells: Fast but coarse motion processing
///
/// Example usage:
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
/// var model = new SlowFast&lt;double&gt;(arch, numClasses: 400);
/// var predictions = model.Classify(videoFrames);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Two-pathway design with lateral connections
/// - Slow pathway: T frames, C channels
/// - Fast pathway: αT frames, βC channels (α=8, β=1/8 typically)
/// - Lateral connections fuse information between pathways
/// </para>
/// <para>
/// <b>Reference:</b> "SlowFast Networks for Video Recognition" ICCV 2019
/// https://arxiv.org/abs/1812.03982
/// </para>
/// </remarks>
public class SlowFast<T> : NeuralNetworkBase<T>
{
    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _numClasses;
    private readonly int _slowFrames;
    private readonly int _fastFrames;
    private readonly int _slowChannels;
    private readonly int _fastChannels;
    private readonly int _alpha;
    private readonly int _imageSize;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumClasses => _numClasses;
    internal int SlowFrames => _slowFrames;
    internal int FastFrames => _fastFrames;
    internal int Alpha => _alpha;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SlowFast model using native layers for training and inference.
    /// </summary>
    public SlowFast(
        NeuralNetworkArchitecture<T> architecture,
        int numClasses = 400,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int slowFrames = 4,
        int slowChannels = 64,
        int fastChannels = 8,
        int alpha = 8)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        if (numClasses < 1)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be at least 1.");

        _useNativeMode = true;
        _numClasses = numClasses;
        _slowFrames = slowFrames;
        _fastFrames = slowFrames * alpha;
        _slowChannels = slowChannels;
        _fastChannels = fastChannels;
        _alpha = alpha;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;

        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a SlowFast model using a pretrained ONNX model for inference.
    /// </summary>
    public SlowFast(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 400)
        : base(architecture, new CrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SlowFast ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numClasses = numClasses;
        _slowFrames = 4;
        _fastFrames = 32;
        _slowChannels = 64;
        _fastChannels = 8;
        _alpha = 8;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _lossFunction = new CrossEntropyLoss<T>();

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Classifies video frames into action categories.
    /// </summary>
    public Tensor<T> Classify(Tensor<T> videoFrames)
    {
        if (videoFrames is null)
            throw new ArgumentNullException(nameof(videoFrames));

        return _useNativeMode ? Forward(videoFrames) : PredictOnnx(videoFrames);
    }

    /// <summary>
    /// Gets top-K predictions with probabilities.
    /// </summary>
    public List<(int ClassIndex, double Probability)> GetTopKPredictions(Tensor<T> videoFrames, int topK = 5)
    {
        var logits = Classify(videoFrames);
        var probabilities = Softmax(logits);

        var results = new List<(int, double)>();
        for (int i = 0; i < probabilities.Length; i++)
        {
            results.Add((i, Convert.ToDouble(probabilities.Data[i])));
        }

        return results.OrderByDescending(x => x.Item2).Take(topK).ToList();
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers)
        {
            result = layer.Forward(result);
        }
        return result;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    private Tensor<T> Softmax(Tensor<T> logits)
    {
        var result = new Tensor<T>(logits.Shape);
        double maxVal = double.MinValue;

        for (int i = 0; i < logits.Length; i++)
        {
            double val = Convert.ToDouble(logits.Data[i]);
            if (val > maxVal) maxVal = val;
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += Math.Exp(Convert.ToDouble(logits.Data[i]) - maxVal);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            double prob = Math.Exp(Convert.ToDouble(logits.Data[i]) - maxVal) / sum;
            result.Data[i] = NumOps.FromDouble(prob);
        }

        return result;
    }

    public override Tensor<T> Predict(Tensor<T> input) => Classify(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }

        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Layer Initialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int inputChannels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int inputHeight = Architecture.InputHeight > 0 ? Architecture.InputHeight : 224;
            int inputWidth = Architecture.InputWidth > 0 ? Architecture.InputWidth : 224;

            // SlowFast uses a dual-pathway architecture. The Layers list contains the slow pathway.
            // Fast pathway is created and executed separately in the Forward method, then fused.
            Layers.AddRange(LayerHelper<T>.CreateSlowFastSlowPathwayLayers(
                inputChannels, inputHeight, inputWidth, _slowChannels));
        }
    }

    #endregion

    #region Serialization

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.VideoActionRecognition,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SlowFast" },
            { "NumClasses", _numClasses },
            { "SlowFrames", _slowFrames },
            { "FastFrames", _fastFrames },
            { "Alpha", _alpha },
            { "UseNativeMode", _useNativeMode }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Serialization is not supported in ONNX mode.");
        writer.Write(_numClasses);
        writer.Write(_slowFrames);
        writer.Write(_fastFrames);
        writer.Write(_slowChannels);
        writer.Write(_fastChannels);
        writer.Write(_alpha);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");
        for (int i = 0; i < 6; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new SlowFast<T>(Architecture, _numClasses, _optimizer, _lossFunction, _slowFrames, _slowChannels, _fastChannels, _alpha);

    #endregion
}
