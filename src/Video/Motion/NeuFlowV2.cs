using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Motion;

/// <summary>
/// NeuFlow v2 high-efficiency optical flow on edge devices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "NeuFlow v2: High-Efficiency Optical Flow Estimation on Edge Devices" (Zhang et al., 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> NeuFlow V2 is a fast, lightweight optical flow estimator designed for real-time applications. It achieves good accuracy with significantly reduced computation compared to transformer-based methods.</para>
/// <para>
/// NeuFlow v2 achieves high-efficiency optical flow estimation suitable for edge devices through a lightweight backbone and optimized inference.
/// </para>
/// </remarks>
public class NeuFlowV2<T> : OpticalFlowBase<T>
{
    private readonly NeuFlowV2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numFeatures;
    private int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Creates a new NeuFlowV2 model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    public NeuFlowV2(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        NeuFlowV2Options? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures, "Number of features must be positive.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be positive.");
        _options = options ?? new NeuFlowV2Options();
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

        _featureExtract = new ConvolutionalLayer<T>(2 * channels, height, width, _numFeatures, 3, 1, 1);

        for (int i = 0; i < _numLayers; i++)
        {
            _processingBlocks.Add(new ConvolutionalLayer<T>(_numFeatures, height, width, _numFeatures, 3, 1, 1));
        }

        _outputConv = new ConvolutionalLayer<T>(_numFeatures, height, width, 2, 3, 1, 1);

        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();
    }

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
        if (rawFlow.Length < flow.Length)
            throw new InvalidOperationException($"Raw flow output ({rawFlow.Length} elements) is smaller than expected flow field ({flow.Length} elements).");
        for (int i = 0; i < flow.Length; i++)
        {
            flow.Data.Span[i] = rawFlow.Data.Span[i];
        }

        return flow;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var output = Predict(input);
        var gradient = new Tensor<T>(output.Shape);
        for (int i = 0; i < output.Length; i++)
        {
            gradient.Data.Span[i] = NumOps.Subtract(output.Data.Span[i], expectedOutput.Data.Span[i]);
        }
        if (_outputConv is not null)
        {
            gradient = _outputConv.Backward(gradient);
        }
        for (int i = _processingBlocks.Count - 1; i >= 0; i--)
        {
            gradient = _processingBlocks[i].Backward(gradient);
        }
        if (_featureExtract is not null)
        {
            _featureExtract.Backward(gradient);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int required = 0;
        if (_featureExtract is not null) required += _featureExtract.GetParameters().Length;
        foreach (var block in _processingBlocks) required += block.GetParameters().Length;
        if (_outputConv is not null) required += _outputConv.GetParameters().Length;
        if (parameters.Length < required)
            throw new ArgumentException($"Parameter vector length {parameters.Length} is less than required {required}.", nameof(parameters));
        int offset = 0;
        if (_featureExtract is not null)
        {
            var p = _featureExtract.GetParameters();
            var sub = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
            _featureExtract.SetParameters(sub);
            offset += p.Length;
        }
        foreach (var block in _processingBlocks)
        {
            var p = block.GetParameters();
            var sub = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
            block.SetParameters(sub);
            offset += p.Length;
        }
        if (_outputConv is not null)
        {
            var p = _outputConv.GetParameters();
            var sub = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
            _outputConv.SetParameters(sub);
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "NeuFlowV2" },
                { "NumFeatures", _numFeatures },
                { "NumLayers", _numLayers }
            },
            ModelData = this.Serialize()
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
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuFlowV2<T>(Architecture, _numFeatures, _numLayers, _options);
    }
}
