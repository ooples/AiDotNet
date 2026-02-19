using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Inpainting;

/// <summary>
/// STTN spatial-temporal transformer network for video inpainting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Learning Joint Spatial-Temporal Transformations for Video Inpainting" (Zeng et al., ECCV 2020)</item>
/// </list></para>
/// <para>
/// STTN uses spatial-temporal transformers to learn joint transformations that fill missing regions with temporally coherent content.
/// </para>
/// </remarks>
public class STTN<T> : VideoInpaintingBase<T>
{
    private readonly STTNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly int _numFeatures;
    private readonly int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Creates a new STTN model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    public STTN(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        STTNOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new STTNOptions();
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
    public override Tensor<T> Inpaint(Tensor<T> frames, Tensor<T> masks)
    {
        int numFrames = frames.Shape[0];
        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        var output = new Tensor<T>(frames.Shape);

        for (int f = 0; f < numFrames; f++)
        {
            var frame = ExtractFrame(frames, f);
            var feat = _featureExtract!.Forward(frame);
            foreach (var block in _processingBlocks)
            {
                feat = block.Forward(feat);
            }
            var filled = _outputConv!.Forward(feat);

            // Blend: use original in non-masked regions, generated in masked regions
            int maskOffset = f * height * width;
            int frameOffset = f * channels * height * width;
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int pixIdx = frameOffset + c * height * width + h * width + w;
                        double maskVal = NumOps.ToDouble(masks.Data.Span[maskOffset + h * width + w]);
                        if (maskVal > 0.5)
                        {
                            output.Data.Span[pixIdx] = filled.Data.Span[c * height * width + h * width + w];
                        }
                        else
                        {
                            output.Data.Span[pixIdx] = frames.Data.Span[pixIdx];
                        }
                    }
                }
            }
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
                { "ModelName", "STTN" },
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
        return new STTN<T>(Architecture, _numFeatures, _numLayers);
    }
}
