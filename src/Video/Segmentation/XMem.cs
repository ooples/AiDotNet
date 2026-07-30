using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
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
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model",
    "https://arxiv.org/abs/2207.07115",
    Year = 2022,
    Authors = "Ho Kei Cheng, Alexander G. Schwing")]
public class XMem<T> : NeuralNetworkBase<T>
{
    private readonly XMemOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public XMem()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 1))
    {
    }

    public XMem(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 256,
        int sensoryMemorySize = 3,
        int workingMemorySize = 10,
        int longTermMemorySize = 100,
        XMemOptions? options = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>())
    {
        _options = options ?? new XMemOptions();
        Options = _options;
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
        // Cheng and Schwing train XMem with AdamW at 1e-5 and weight decay 0.05.
        // Keep the optimizer injectable, but make the native default reproduce those
        // settings instead of silently using the framework's generic Adam defaults.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
            });

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
        int longTermMemorySize = 100,
        XMemOptions? options = null)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        _options = options ?? new XMemOptions();
        Options = _options;
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

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
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

    protected override Tensor<T> PredictCore(Tensor<T> input) => SegmentFrame(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Routes training through XMem's grouped encoder, memory-read, and decoder graph.
    /// </summary>
    /// <remarks>
    /// XMem's layer list contains parallel memory projections, so the base class's flat
    /// sequential walk is not a valid XMem forward pass. Keeping these operations on the
    /// tensor engine also preserves the autodiff graph through every trainable layer group.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        EnsureLayerRandomSeedsWired();

        bool hasBatch = input.Rank == 4;
        var frame = hasBatch ? input : AddBatchDimension(input);
        var mask = SegmentWithMemory(frame);
        return hasBatch ? mask : RemoveBatchDimension(mask);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode)
        {
            return new Dictionary<string, Tensor<T>>
            {
                ["Output"] = PredictOnnx(input).Clone(),
            };
        }

        bool hasBatch = input.Rank == 4;
        var frame = hasBatch ? input : AddBatchDimension(input);
        var activations = new Dictionary<string, Tensor<T>>();

        void Record(int layerIndex, Tensor<T> value) =>
            activations[$"Layer_{layerIndex}_{Layers[layerIndex].GetType().Name}"] = value.Clone();

        var queryFeatures = frame;
        for (int i = 0; i < 4 && i < Layers.Count; i++)
        {
            queryFeatures = Layers[i].Forward(queryFeatures);
            Record(i, queryFeatures);
        }

        Tensor<T> RunMemoryBranch(List<Tensor<T>> memory, int startLayer, int endLayer)
        {
            var branch = queryFeatures;
            for (int i = startLayer; i < endLayer && i < Layers.Count; i++)
            {
                branch = Layers[i].Forward(branch);
                Record(i, branch);
            }

            return memory.Count == 0
                ? branch
                : Engine.TensorAdd(branch, QueryMemory(memory, branch._shape));
        }

        var sensory = RunMemoryBranch(_sensoryMemory, 4, 6);
        var working = RunMemoryBranch(_workingMemory, 6, 8);
        var longTerm = RunMemoryBranch(_longTermMemory, 8, 10);

        var current = ConcatenateChannels(ConcatenateChannels(sensory, working), longTerm);
        if (Layers.Count > 10)
        {
            current = Layers[10].Forward(current);
            Record(10, current);
        }

        for (int i = 11; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            Record(i, current);
        }

        return activations;
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
            // The paper decodes a memory readout together with query-encoder skip features.
            // In the compact native implementation each memory branch projects the current
            // query and adds its matching memory readout. This keeps an empty-memory forward
            // input-dependent and gives training a connected graph through all three stores.
            var sensoryResponse = ReadMemoryBranch(queryFeatures, _sensoryMemory, 4, 6);
            var workingResponse = ReadMemoryBranch(queryFeatures, _workingMemory, 6, 8);
            var longTermResponse = ReadMemoryBranch(queryFeatures, _longTermMemory, 8, 10);

            var fused = FuseMemoryResponses(sensoryResponse, workingResponse, longTermResponse);
            return DecodeMask(fused);
        }
        else
        {
            return PredictOnnx(frame);
        }
    }

    private Tensor<T> ReadMemoryBranch(
        Tensor<T> queryFeatures,
        List<Tensor<T>> memory,
        int startLayer,
        int endLayer)
    {
        var projectedQuery = queryFeatures;
        for (int i = startLayer; i < endLayer && i < Layers.Count; i++)
        {
            projectedQuery = Layers[i].Forward(projectedQuery);
        }

        if (memory.Count == 0)
            return projectedQuery;

        var memoryReadout = QueryMemory(memory, projectedQuery._shape);
        return Engine.TensorAdd(projectedQuery, memoryReadout);
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

        var masked = new Tensor<T>(features._shape);
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
            if (mem.Length != result.Length)
            {
                // Truncate or pad to match expected shape
                var aligned = new Tensor<T>(shape);
                int copyLen = Math.Min(mem.Length, aligned.Length);
                for (int i = 0; i < copyLen; i++)
                    aligned[i] = mem[i];
                result = Engine.TensorAdd(result, aligned);
            }
            else
            {
                result = Engine.TensorAdd(result, mem);
            }
        }

        return Engine.TensorDivideScalar(result, NumOps.FromDouble(memory.Count));
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

        // Encoder: 0-3; memory branches: 4-9; memory fusion: 10.
        if (Layers.Count > 10)
        {
            return Layers[10].Forward(concat);
        }

        return concat;
    }

    /// <summary>
    /// Runs the decoder over the fused readout to produce the mask.
    /// </summary>
    /// <remarks>
    /// The decoder stack built by CreateDefaultXMemLayers ALREADY carries its own UpsamplingLayer(2)
    /// between convolution stages — four of them, which is exactly the stride-16 to full-resolution
    /// climb the paper describes ("iteratively upsamples by 2x at a time"), ending in a 1-channel
    /// sigmoid mask head. This method additionally called Upsample2x once per layer on top of that, so
    /// the mask was upsampled twice over and a 32x32 clip decoded to 2048x2048. Let the layers do the
    /// upsampling they were built to do; the loop below only tops the mask up if a shallower decoder
    /// leaves it short of the frame.
    /// </remarks>
    private Tensor<T> DecodeMask(Tensor<T> features)
    {
        var decoded = features;
        // Decoder begins at 11. UpsamplingLayer instances in the layer list perform
        // the four 2x steps from stride 16 back to input resolution.
        for (int i = 11; i < Layers.Count; i++)
        {
            decoded = Layers[i].Forward(decoded);
        }

        return decoded;
    }
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate([a, b], axis: 1);
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

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, [1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, [tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]]);

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

            PinMemoryFusionInputDepth();
        }
    }

    /// <summary>
    /// Pins the memory-fusion layer's input depth to the width of the concatenated memory readout.
    /// </summary>
    /// <remarks>
    /// ConvolutionalLayer resolves its input depth lazily from the first tensor that reaches it, and
    /// this model does NOT use its layers in list order — the encoder, the three memory branches, the
    /// fusion layer and the decoder are wired by SegmentWithMemory. So the fusion layer was pinned by
    /// a sequential shape walk to its list predecessor's width and then rejected the real readout with
    /// "Expected input depth 128, but got 448". 448 is sensory + working + long-term, exactly the
    /// total the layer factory computes for this layer and then never applies, since there is no
    /// constructor overload that takes an input depth. Resolve it up front instead.
    /// </remarks>
    private void PinMemoryFusionInputDepth()
    {
        const int MemoryFusionLayerIndex = 12;
        if (Layers.Count <= MemoryFusionLayerIndex) return;
        if (Layers[MemoryFusionLayerIndex] is not LayerBase<T> fusion) return;

        int fusedChannels = _numFeatures + (_numFeatures / 2) + (_numFeatures / 4);

        // The encoder is four stride-2 convolutions, so the readout is the frame reduced by 16.
        int featureHeight = Math.Max(1, _inputHeight / 16);
        int featureWidth = Math.Max(1, _inputWidth / 16);

        fusion.ResolveShapesOnly(new[] { fusedChannels, featureHeight, featureWidth });
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
            int layerParameterCount = checked((int)layer.ParameterCount);
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
        var copiedOptions = new XMemOptions(_options);
        if (!_useNativeMode && _onnxModelPath is { } modelPath)
        {
            return new XMem<T>(
                Architecture,
                modelPath,
                _sensoryMemorySize,
                _workingMemorySize,
                _longTermMemorySize,
                copiedOptions);
        }

        return new XMem<T>(
            Architecture,
            optimizer: null,
            _lossFunction,
            _numFeatures,
            _sensoryMemorySize,
            _workingMemorySize,
            _longTermMemorySize,
            copiedOptions);
    }

    #endregion
}
