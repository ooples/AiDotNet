using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Segmentation;

/// <summary>
/// XMem: Long-Term Video Object Segmentation with Atkinson-Shiffrin memory model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// XMem is designed for tracking objects in very long videos using a three-tier
/// memory system inspired by human memory.
/// </para>
/// <para>
/// <b>For Beginners:</b> XMem can track objects in hour-long videos without
/// running out of memory. It uses three types of memory:
/// - Sensory memory: Very recent frames (high detail, fast to forget)
/// - Working memory: Important recent frames (moderate detail)
/// - Long-term memory: Key historical frames (compressed, permanent)
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 480, inputWidth: 854, inputDepth: 3);
/// var model = new XMem&lt;double&gt;(arch);
/// var masks = model.TrackObjectLongTerm(videoFrames, initialMask);
/// </code>
///
/// Example usage (ONNX mode for inference):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 480, inputWidth: 854, inputDepth: 3);
/// var model = new XMem&lt;double&gt;(arch, "xmem.onnx");
/// var masks = model.TrackObjectLongTerm(videoFrames, initialMask);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model"
/// https://arxiv.org/abs/2207.07115
/// </para>
/// </remarks>
public class XMem<T> : NeuralNetworkBase<T>
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
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;
    private readonly int _numFeatures;
    private readonly int _sensoryMemorySize;
    private readonly int _workingMemorySize;
    private readonly int _longTermMemorySize;

    // Memory banks (three-tier system)
    private readonly List<Tensor<T>> _sensoryMemory;
    private readonly List<Tensor<T>> _workingMemory;
    private readonly List<Tensor<T>> _longTermMemory;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int InputHeight => _inputHeight;
    internal int InputWidth => _inputWidth;
    internal int SensoryMemoryCount => _sensoryMemory.Count;
    internal int WorkingMemoryCount => _workingMemory.Count;
    internal int LongTermMemoryCount => _longTermMemory.Count;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an XMem model using native layers for training and inference.
    /// </summary>
    public XMem(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 256,
        int sensoryMemorySize = 3,
        int workingMemorySize = 10,
        int longTermMemorySize = 100)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
    {
        _useNativeMode = true;
        _inputHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _inputWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 854;
        _inputChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _sensoryMemorySize = sensoryMemorySize;
        _workingMemorySize = workingMemorySize;
        _longTermMemorySize = longTermMemorySize;

        _sensoryMemory = [];
        _workingMemory = [];
        _longTermMemory = [];

        _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an XMem model using a pretrained ONNX model for inference.
    /// </summary>
    public XMem(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sensoryMemorySize = 3,
        int workingMemorySize = 10,
        int longTermMemorySize = 100)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"XMem ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _inputHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _inputWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 854;
        _inputChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = 256;
        _sensoryMemorySize = sensoryMemorySize;
        _workingMemorySize = workingMemorySize;
        _longTermMemorySize = longTermMemorySize;

        _sensoryMemory = [];
        _workingMemory = [];
        _longTermMemory = [];
        _lossFunction = new BinaryCrossEntropyLoss<T>();

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
    /// Tracks an object through a long video sequence.
    /// </summary>
    public List<Tensor<T>> TrackObjectLongTerm(List<Tensor<T>> frames, Tensor<T> initialMask)
    {
        if (frames is null || frames.Count == 0)
            throw new ArgumentException("Frames list cannot be null or empty.", nameof(frames));
        if (initialMask is null)
            throw new ArgumentNullException(nameof(initialMask));

        ClearAllMemory();
        var masks = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            bool hasBatch = frame.Rank == 4;
            if (!hasBatch) frame = AddBatchDimension(frame);

            Tensor<T> mask;
            if (i == 0)
            {
                mask = InitializeWithMask(frame, initialMask);
            }
            else
            {
                mask = SegmentWithMemory(frame);
            }

            UpdateMemoryHierarchy(frame, mask, i);

            if (!hasBatch) mask = RemoveBatchDimension(mask);
            masks.Add(mask);
        }

        return masks;
    }

    /// <summary>
    /// Segments a single frame using the memory hierarchy.
    /// </summary>
    public Tensor<T> SegmentFrame(Tensor<T> frame)
    {
        if (frame is null)
            throw new ArgumentNullException(nameof(frame));

        bool hasBatch = frame.Rank == 4;
        if (!hasBatch) frame = AddBatchDimension(frame);

        var mask = SegmentWithMemory(frame);

        if (!hasBatch) mask = RemoveBatchDimension(mask);
        return mask;
    }

    /// <summary>
    /// Clears all memory banks.
    /// </summary>
    public void ClearAllMemory()
    {
        _sensoryMemory.Clear();
        _workingMemory.Clear();
        _longTermMemory.Clear();
    }

    /// <summary>
    /// Gets memory statistics.
    /// </summary>
    public (int Sensory, int Working, int LongTerm) GetMemoryStats() =>
        (_sensoryMemory.Count, _workingMemory.Count, _longTermMemory.Count);

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
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

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

    public override Tensor<T> Predict(Tensor<T> input) => SegmentFrame(input);

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

    #region Private Methods

    private Tensor<T> InitializeWithMask(Tensor<T> frame, Tensor<T> mask)
    {
        if (mask.Rank == 3) mask = AddBatchDimension(mask);

        var features = EncodeFrame(frame);
        var maskFeatures = CreateMaskedFeatures(features, mask);

        _sensoryMemory.Add(maskFeatures);

        return UpsampleMask(DownsampleMask(mask, features.Shape[2], features.Shape[3]), _inputHeight, _inputWidth);
    }

    private Tensor<T> SegmentWithMemory(Tensor<T> frame)
    {
        if (_useNativeMode)
        {
            var queryFeatures = EncodeFrame(frame);
            var sensoryResponse = QueryMemory(_sensoryMemory, queryFeatures.Shape);
            var workingResponse = QueryMemory(_workingMemory, [queryFeatures.Shape[0], _numFeatures / 2, queryFeatures.Shape[2], queryFeatures.Shape[3]]);
            var longTermResponse = QueryMemory(_longTermMemory, [queryFeatures.Shape[0], _numFeatures / 4, queryFeatures.Shape[2], queryFeatures.Shape[3]]);

            var fused = FuseMemoryResponses(sensoryResponse, workingResponse, longTermResponse);
            return DecodeMask(fused);
        }
        else
        {
            return PredictOnnx(frame);
        }
    }

    private void UpdateMemoryHierarchy(Tensor<T> frame, Tensor<T> mask, int frameIndex)
    {
        var features = EncodeFrame(frame);
        var maskedFeatures = CreateMaskedFeatures(features, mask);

        _sensoryMemory.Add(maskedFeatures);

        while (_sensoryMemory.Count > _sensoryMemorySize)
        {
            var promoted = _sensoryMemory[0];
            _sensoryMemory.RemoveAt(0);
            _workingMemory.Add(CompressFeatures(promoted, _numFeatures / 2));
        }

        while (_workingMemory.Count > _workingMemorySize)
        {
            var promoted = _workingMemory[0];
            _workingMemory.RemoveAt(0);
            _longTermMemory.Add(CompressFeatures(promoted, _numFeatures / 4));
        }

        while (_longTermMemory.Count > _longTermMemorySize)
            _longTermMemory.RemoveAt(0);
    }

    private Tensor<T> EncodeFrame(Tensor<T> frame)
    {
        var features = frame;
        for (int i = 0; i < 4 && i < Layers.Count; i++)
        {
            features = Layers[i].Forward(features);
        }
        return features;
    }

    private Tensor<T> CreateMaskedFeatures(Tensor<T> features, Tensor<T> mask)
    {
        var downsampledMask = DownsampleMask(mask, features.Shape[2], features.Shape[3]);

        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int height = features.Shape[2];
        int width = features.Shape[3];

        var masked = new Tensor<T>(features.Shape);
        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double m = Convert.ToDouble(downsampledMask[b, 0, h, w]);
                        double f = Convert.ToDouble(features[b, c, h, w]);
                        masked[b, c, h, w] = NumOps.FromDouble(f * m);
                    }

        return masked;
    }

    private Tensor<T> QueryMemory(List<Tensor<T>> memory, int[] shape)
    {
        if (memory.Count == 0)
            return new Tensor<T>(shape);

        var result = new Tensor<T>(shape);
        foreach (var mem in memory)
        {
            int minLen = Math.Min(result.Length, mem.Length);
            for (int i = 0; i < minLen; i++)
            {
                result.Data.Span[i] = NumOps.Add(result.Data.Span[i], mem.Data.Span[i]);
            }
        }

        double scale = 1.0 / memory.Count;
        for (int i = 0; i < result.Length; i++)
        {
            result.Data.Span[i] = NumOps.FromDouble(Convert.ToDouble(result.Data.Span[i]) * scale);
        }

        return result;
    }

    private Tensor<T> CompressFeatures(Tensor<T> features, int targetChannels)
    {
        // Simple compression by averaging groups of channels
        int srcChannels = features.Shape[1];
        int ratio = srcChannels / targetChannels;
        if (ratio < 1) return features;

        int batchSize = features.Shape[0];
        int height = features.Shape[2];
        int width = features.Shape[3];

        var compressed = new Tensor<T>([batchSize, targetChannels, height, width]);

        for (int b = 0; b < batchSize; b++)
            for (int tc = 0; tc < targetChannels; tc++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double sum = 0;
                        for (int sc = tc * ratio; sc < (tc + 1) * ratio && sc < srcChannels; sc++)
                        {
                            sum += Convert.ToDouble(features[b, sc, h, w]);
                        }
                        compressed[b, tc, h, w] = NumOps.FromDouble(sum / ratio);
                    }

        return compressed;
    }

    private Tensor<T> FuseMemoryResponses(Tensor<T> sensory, Tensor<T> working, Tensor<T> longTerm)
    {
        var concat = ConcatenateChannels(sensory, working);
        concat = ConcatenateChannels(concat, longTerm);

        // Use memory fusion layer (index 12)
        if (Layers.Count > 12)
        {
            return Layers[12].Forward(concat);
        }

        return concat;
    }

    private Tensor<T> DecodeMask(Tensor<T> features)
    {
        var decoded = features;
        for (int i = 13; i < Layers.Count; i++)
        {
            if (i < Layers.Count - 1)
            {
                decoded = Upsample2x(decoded);
            }
            decoded = Layers[i].Forward(decoded);
        }

        while (decoded.Shape[2] < _inputHeight || decoded.Shape[3] < _inputWidth)
            decoded = Upsample2x(decoded);

        return decoded;
    }

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

    private Tensor<T> DownsampleMask(Tensor<T> mask, int targetH, int targetW)
    {
        int batchSize = mask.Shape[0];
        int srcH = mask.Shape[2];
        int srcW = mask.Shape[3];

        var downsampled = new Tensor<T>([batchSize, 1, targetH, targetW]);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < targetH; h++)
                for (int w = 0; w < targetW; w++)
                {
                    int srcY = Math.Min((int)((double)h * srcH / targetH), srcH - 1);
                    int srcX = Math.Min((int)((double)w * srcW / targetW), srcW - 1);
                    downsampled[b, 0, h, w] = mask[b, 0, srcY, srcX];
                }

        return downsampled;
    }

    private Tensor<T> UpsampleMask(Tensor<T> mask, int targetH, int targetW)
    {
        int batchSize = mask.Shape[0];
        int srcH = mask.Shape[2];
        int srcW = mask.Shape[3];

        var upsampled = new Tensor<T>([batchSize, 1, targetH, targetW]);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < targetH; h++)
                for (int w = 0; w < targetW; w++)
                {
                    int srcY = Math.Min((int)((double)h * srcH / targetH), srcH - 1);
                    int srcX = Math.Min((int)((double)w * srcW / targetW), srcW - 1);
                    upsampled[b, 0, h, w] = mask[b, 0, srcY, srcX];
                }

        return upsampled;
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

    #region Layer Initialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultXMemLayers(
                inputChannels: _inputChannels,
                inputHeight: _inputHeight,
                inputWidth: _inputWidth,
                numFeatures: _numFeatures));
        }
    }

    #endregion

    #region Serialization

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "XMem" },
            { "Description", "Long-Term Video Object Segmentation with Hierarchical Memory" },
            { "InputHeight", _inputHeight },
            { "InputWidth", _inputWidth },
            { "InputChannels", _inputChannels },
            { "NumFeatures", _numFeatures },
            { "SensoryMemorySize", _sensoryMemorySize },
            { "WorkingMemorySize", _workingMemorySize },
            { "LongTermMemorySize", _longTermMemorySize },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.VideoObjectSegmentation,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_inputHeight);
        writer.Write(_inputWidth);
        writer.Write(_inputChannels);
        writer.Write(_numFeatures);
        writer.Write(_sensoryMemorySize);
        writer.Write(_workingMemorySize);
        writer.Write(_longTermMemorySize);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        for (int i = 0; i < 7; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new XMem<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            _numFeatures,
            _sensoryMemorySize,
            _workingMemorySize,
            _longTermMemorySize);
    }

    #endregion
}
