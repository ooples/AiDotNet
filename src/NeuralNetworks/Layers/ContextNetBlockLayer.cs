using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A paper-faithful ContextNet convolution block over channels-first <c>[B, C, T]</c> features.
/// </summary>
/// <remarks>
/// Han et al. (INTERSPEECH 2020, arXiv:2005.03191, sections 2.2.1-2.2.4) define each
/// convolution as depthwise-separable Conv1D followed by batch normalization and Swish. The
/// stacked convolution path is recalibrated by utterance-wide squeeze-and-excitation, added to a
/// projected residual, and passed through Swish. C0 and C22 retain the main path and SE but omit
/// the residual projection, as specified by Table 1.
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 16", TestConstructorArgs = "8, 8, 3, 2")]
internal sealed partial class ContextNetBlockLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _numConvolutions;
    private readonly int _seReductionRatio;
    private readonly double _dropoutRate;
    private readonly int _stride;
    private readonly bool _useResidual;
    private readonly int _seed;

    private readonly DepthwiseConv1DLayer<T>[] _depthwise;
    private readonly Conv1DLayer<T>[] _pointwise;
    private readonly BatchNormalizationLayer<T>[] _batchNorm;
    private readonly SqueezeAndExcitationLayer<T> _squeezeExcitation;
    private readonly Conv1DLayer<T>? _residualProjection;
    private readonly BatchNormalizationLayer<T>? _residualBatchNorm;
    private readonly DropoutLayer<T>? _dropout;

    public ContextNetBlockLayer(
        [LayerState] int inputChannels,
        [LayerState] int outputChannels,
        [LayerState] int kernelSize,
        [LayerState] int numConvolutions = 5,
        [LayerState] int seReductionRatio = 8,
        [LayerState] double dropoutRate = 0.0,
        [LayerState] int stride = 1,
        [LayerState] bool useResidual = true,
        [LayerState] int seed = 2027)
        : base(
            new[] { inputChannels, -1 },
            new[] { outputChannels, -1 },
            (IActivationFunction<T>)new IdentityActivation<T>())
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (numConvolutions <= 0) throw new ArgumentOutOfRangeException(nameof(numConvolutions));
        if (seReductionRatio <= 0) throw new ArgumentOutOfRangeException(nameof(seReductionRatio));
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _numConvolutions = numConvolutions;
        _seReductionRatio = seReductionRatio;
        _dropoutRate = dropoutRate;
        _stride = stride;
        _useResidual = useResidual;
        _seed = seed;

        int initSeed = unchecked(seed + inputChannels * 131 + outputChannels * 37 + kernelSize * 17);
        IInitializationStrategy<T> Init(int offset) =>
            new HeInitializationStrategy<T>(RandomHelper.CreateSeededRandom(unchecked(initSeed + offset)));

        _depthwise = new DepthwiseConv1DLayer<T>[numConvolutions];
        _pointwise = new Conv1DLayer<T>[numConvolutions];
        _batchNorm = new BatchNormalizationLayer<T>[numConvolutions];

        int channels = inputChannels;
        for (int i = 0; i < numConvolutions; i++)
        {
            // ContextNet applies temporal downsampling on the LAST convolution in the block.
            int convolutionStride = i == numConvolutions - 1 ? stride : 1;
            _depthwise[i] = new DepthwiseConv1DLayer<T>(
                channels, kernelSize, multiplier: 1, stride: convolutionStride, padding: null,
                activation: new IdentityActivation<T>(), initializationStrategy: Init(i * 3));
            _pointwise[i] = new Conv1DLayer<T>(
                channels, outputChannels, kernelSize: 1, dilation: 1, stride: 1, padding: 0,
                activation: new IdentityActivation<T>(), initializationStrategy: Init(i * 3 + 1));
            _batchNorm[i] = new BatchNormalizationLayer<T>(outputChannels);
            channels = outputChannels;
        }

        _squeezeExcitation = new SqueezeAndExcitationLayer<T>(
            outputChannels, seReductionRatio,
            (IActivationFunction<T>?)new SwishActivation<T>(),
            (IActivationFunction<T>?)new SigmoidActivation<T>());

        if (useResidual)
        {
            _residualProjection = new Conv1DLayer<T>(
                inputChannels, outputChannels, kernelSize: 1, dilation: 1, stride: stride, padding: 0,
                activation: new IdentityActivation<T>(), initializationStrategy: Init(numConvolutions * 3 + 1));
            _residualBatchNorm = new BatchNormalizationLayer<T>(outputChannels);
        }

        _dropout = dropoutRate > 0.0 ? new DropoutLayer<T>(dropoutRate) : null;
    }

    private IEnumerable<ILayer<T>> TrainableSubLayers()
    {
        for (int i = 0; i < _numConvolutions; i++)
        {
            yield return _depthwise[i];
            yield return _pointwise[i];
            yield return _batchNorm[i];
        }

        yield return _squeezeExcitation;
        if (_residualProjection is not null) yield return _residualProjection;
        if (_residualBatchNorm is not null) yield return _residualBatchNorm;
    }

    private IEnumerable<BatchNormalizationLayer<T>> BatchNormLayers()
    {
        for (int i = 0; i < _batchNorm.Length; i++) yield return _batchNorm[i];
        if (_residualBatchNorm is not null) yield return _residualBatchNorm;
    }

    public override bool SupportsTraining => true;

    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var layer in TrainableSubLayers()) total += layer.ParameterCount;
            return total;
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 3)
        {
            throw new ArgumentException(
                $"ContextNetBlockLayer requires rank-3 [B, C, T] input; got rank {input.Rank}.",
                nameof(input));
        }

        var hidden = input;
        for (int i = 0; i < _numConvolutions; i++)
        {
            hidden = _depthwise[i].Forward(hidden);
            hidden = _pointwise[i].Forward(hidden);
            hidden = _batchNorm[i].Forward(hidden);
            hidden = Engine.Swish(hidden);
        }

        // SqueezeAndExcitationLayer's rank-3 contract is [B, T, C].
        hidden = Engine.TensorPermute(hidden, new[] { 0, 2, 1 });
        hidden = _squeezeExcitation.Forward(hidden);
        hidden = Engine.TensorPermute(hidden, new[] { 0, 2, 1 });

        if (_residualProjection is not null && _residualBatchNorm is not null)
        {
            var residual = _residualBatchNorm.Forward(_residualProjection.Forward(input));
            hidden = Engine.TensorAdd(hidden, residual);
        }

        hidden = Engine.Swish(hidden);
        return _dropout is null ? hidden : _dropout.Forward(hidden);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in TrainableSubLayers()) layer.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var parameters = Vector<T>.Empty();
        foreach (var layer in TrainableSubLayers())
            parameters = Vector<T>.Concatenate(parameters, layer.GetParameters());
        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        long expected = ParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters for ContextNetBlockLayer, but got {parameters.Length}.");
        }

        int offset = 0;
        foreach (var layer in TrainableSubLayers())
        {
            int count = (int)layer.ParameterCount;
            layer.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
    }

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in TrainableSubLayers()) layer.SetTrainingMode(isTraining);
        _dropout?.SetTrainingMode(isTraining);
    }

    public override void ResetState()
    {
        foreach (var layer in TrainableSubLayers()) layer.ResetState();
        _dropout?.ResetState();
    }

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            foreach (var batchNorm in BatchNormLayers())
                if (batchNorm is ILayerSerializationExtras<T> extras) count += extras.ExtraParameterCount;
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var values = new List<T>();
        foreach (var batchNorm in BatchNormLayers())
            if (batchNorm is ILayerSerializationExtras<T> extras) values.AddRange(extras.GetExtraParameters().ToArray());
        return new Vector<T>(values.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        int offset = 0;
        foreach (var batchNorm in BatchNormLayers())
        {
            if (batchNorm is not ILayerSerializationExtras<T> extras) continue;
            int count = extras.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
            {
                throw new ArgumentException(
                    $"Truncated ContextNetBlockLayer batch-normalization state: need {offset + count}, got {extraParameters.Length}.");
            }

            extras.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = _inputChannels.ToString();
        metadata["OutputChannels"] = _outputChannels.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["NumConvolutions"] = _numConvolutions.ToString();
        metadata["SeReductionRatio"] = _seReductionRatio.ToString();
        metadata["DropoutRate"] = _dropoutRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Stride"] = _stride.ToString();
        metadata["UseResidual"] = _useResidual.ToString();
        metadata["Seed"] = _seed.ToString();
        return metadata;
    }
}
