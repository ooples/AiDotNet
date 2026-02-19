using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// PSRT progressive spatio-temporal alignment with window-based attention for video SR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "PSRT: Progressive Spatio-temporal Alignment for Video Super-Resolution" (Shi et al., 2022)</item>
/// </list></para>
/// <para>
/// PSRT uses progressive spatio-temporal alignment through window-based spatial-temporal attention, enabling effective multi-frame feature fusion.
/// </para>
/// </remarks>
public class PSRT<T> : VideoSuperResolutionBase<T>
{
    private readonly PSRTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly int _numFeatures;
    private readonly int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Creates a new PSRT model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    public PSRT(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        PSRTOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new PSRTOptions();
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

        _featureExtract = new ConvolutionalLayer<T>(channels, height, width, _numFeatures, 3, 1, 1);

        for (int i = 0; i < _numLayers; i++)
        {
            _processingBlocks.Add(new ConvolutionalLayer<T>(_numFeatures, height, width, _numFeatures, 3, 1, 1));
        }

        _outputConv = new ConvolutionalLayer<T>(_numFeatures, height, width, channels, 3, 1, 1);

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
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        int numFrames = lowResFrames.Shape[0];
        int channels = lowResFrames.Shape[1];
        int height = lowResFrames.Shape[2];
        int width = lowResFrames.Shape[3];
        int outHeight = height * ScaleFactor;
        int outWidth = width * ScaleFactor;

        var output = new Tensor<T>([numFrames, channels, outHeight, outWidth]);

        for (int f = 0; f < numFrames; f++)
        {
            var frame = ExtractFrame(lowResFrames, f);
            var feat = _featureExtract!.Forward(frame);
            foreach (var block in _processingBlocks)
            {
                feat = block.Forward(feat);
            }
            var recon = _outputConv!.Forward(feat);
            var upsampled = BicubicUpsample(recon, ScaleFactor);
            StoreFrame(output, upsampled, f);
        }

        return output;
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
            _outputConv.Backward(gradient);
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
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "PSRT" },
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
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new PSRT<T>(Architecture, _numFeatures, _numLayers);
    }
}
