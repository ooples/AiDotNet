using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Motion;

/// <summary>
/// UFM unified flow and matching demonstrating unified training beats specialized models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "UFM: Unified Flow and Matching" (2025)</item>
/// </list></para>
/// <para><b>For Beginners:</b> UFM (Unified Flow Matching) provides a unified framework for optical flow that handles both forward and backward flow estimation in a single model pass.</para>
/// <para>
/// UFM is the first to demonstrate that unified training for optical flow and feature matching outperforms specialized models on both tasks.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a UFM model for unified flow and matching
/// var ufm = new UFM&lt;double&gt;();
///
/// // Or configure with custom parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new UFM&lt;double&gt;(architecture);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.OpticalFlow)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unifying Flow, Stereo and Depth Estimation",
    "https://arxiv.org/abs/2211.05783",
    Year = 2023,
    Authors = "Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, Andreas Geiger")]
public class UFM<T> : OpticalFlowBase<T>
{
    private readonly UFMOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numFeatures;
    private int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Creates a new UFM model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public UFM()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 2))
    {
    }

    public UFM(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        UFMOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new UFMOptions();
        Options = _options;

        _numFeatures = numFeatures;
        _numLayers = numLayers;
        _processingBlocks = [];

        InitializeNativeLayers(architecture);
    }

    private void InitializeNativeLayers(NeuralNetworkArchitecture<T> arch)
    {
        int height = arch.InputHeight > 0 ? arch.InputHeight : 64;
        int width = arch.InputWidth > 0 ? arch.InputWidth : 64;
        int channels = arch.InputDepth > 0 ? arch.InputDepth : 3;

        _featureExtract = new ConvolutionalLayer<T>(_numFeatures, 3, 1, 1);

        for (int i = 0; i < _numLayers; i++)
        {
            _processingBlocks.Add(new ConvolutionalLayer<T>(_numFeatures, 3, 1, 1));
        }

        _outputConv = new ConvolutionalLayer<T>(2, 3, 1, 1);

        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();

        if (_featureExtract is not null)
            Layers.Add(_featureExtract);
        foreach (var block in _processingBlocks)
            Layers.Add(block);
        if (_outputConv is not null)
            Layers.Add(_outputConv);
    }

    /// <summary>
    /// UFM consumes two RGB frames concatenated channel-wise — 2 ×
    /// Architecture.InputDepth = 6 channels — but Architecture.InputDepth
    /// itself reports the SINGLE-FRAME count (3) so it matches the
    /// architecture's per-frame metadata. Returning null suppresses the
    /// base class's ResolveLazyLayerShapes pre-walk, which would size the
    /// first ConvolutionalLayer (`_featureExtract`) for depth 3 and then
    /// every real Train()/Predict() with the [1, 6, H, W] concat would
    /// fail with "Expected input depth 3, but got 6". Same root-cause fix
    /// as MisGAN / AutoDiffTabGenerator / GOGGLE / MGTSD.
    /// </summary>
    protected override int[]? TryGetArchitectureInputShape() => null;

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

    /// <inheritdoc/>
    public override Tensor<T> EstimateFlow(Tensor<T> frame0, Tensor<T> frame1)
    {
        int channels = frame0.Shape[0];
        int height = frame0.Shape[1];
        int width = frame0.Shape[2];

        // Concatenate frames as input pair
        var concat = ConcatenateFeatures(frame0, frame1);
        if (_featureExtract is null || _outputConv is null)
            throw new InvalidOperationException("Model layers not initialized.");

        var feat = _featureExtract.Forward(concat);
        foreach (var block in _processingBlocks)
        {
            feat = block.Forward(feat);
        }
        var rawFlow = _outputConv.Forward(feat);

        // Extract 2-channel flow field
        var flow = new Tensor<T>([2, height, width]);
        for (int i = 0; i < Math.Min(rawFlow.Length, flow.Length); i++)
        {
            flow.Data.Span[i] = rawFlow.Data.Span[i];
        }

        return flow;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        if (_featureExtract is not null)
        {
            var p = _featureExtract.GetParameters();
            if (offset + p.Length <= parameters.Length)
            {
                var sub = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
                _featureExtract.SetParameters(sub);
                offset += p.Length;
            }
        }
        foreach (var block in _processingBlocks)
        {
            var p = block.GetParameters();
            if (offset + p.Length <= parameters.Length)
            {
                var sub = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
                block.SetParameters(sub);
                offset += p.Length;
            }
        }
        if (_outputConv is not null)
        {
            var p = _outputConv.GetParameters();
            if (offset + p.Length <= parameters.Length)
            {
                var sub = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
                _outputConv.SetParameters(sub);
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "UFM" },
                { "NumFeatures", _numFeatures },
                { "NumLayers", _numLayers }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures);
        writer.Write(_numLayers);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numFeatures = reader.ReadInt32();
        _numLayers = reader.ReadInt32();

        // Reconnect the typed field references (_featureExtract,
        // _processingBlocks, _outputConv) to the conv layers the base
        // class deserialized into the Layers collection. Without this
        // rewire, the freshly-constructed clone instance keeps pointing
        // its typed fields at the UNTRAINED conv layers from its own
        // InitializeNativeLayers call, while the trained weights live in
        // Layers[0..end] — and EstimateFlow's forward pass uses the
        // typed fields directly (not the Layers collection), so the
        // clone predicts as if untrained. InitializeLayers (called from
        // the ctor) emits the layers in stable order
        // [_featureExtract, _processingBlocks[0..N-1], _outputConv], so
        // we read them back in the same positions. Same fix pattern as
        // GOGGLEGenerator.DeserializeNetworkSpecificData.
        int expected = 1 + _numLayers + 1;
        // Require EXACT count: the rebind reads layers at fixed positions
        // [_featureExtract=0, _processingBlocks=1..N, _outputConv=1+N], so an
        // unexpected layer count means the deserialized graph doesn't match this
        // configuration and silently rebinding would point fields at wrong convs.
        if (Layers.Count == expected)
        {
            if (Layers[0] is ConvolutionalLayer<T> fe) _featureExtract = fe;
            _processingBlocks.Clear();
            for (int i = 0; i < _numLayers; i++)
            {
                if (Layers[1 + i] is ConvolutionalLayer<T> block)
                    _processingBlocks.Add(block);
            }
            // _outputConv is at its produced position 1 + _numLayers (not Count-1,
            // which would pick the wrong layer if the count ever differs).
            if (Layers[1 + _numLayers] is ConvolutionalLayer<T> oc) _outputConv = oc;
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new UFM<T>(Architecture, _numFeatures, _numLayers, _options);
    }
}
