using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Video;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Native VideoGigaGAN generator with a style-modulated spatial backbone, temporal inflation,
/// anti-aliased flow-guided propagation, and a high-frequency shuttle.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This layer follows Xu et al., "VideoGigaGAN: Towards Detail-rich Video Super-Resolution"
/// (CVPR 2024). A GigaGAN-style residual generator is inflated with temporal convolutions, SPyNet
/// aligns recurrent features in both directions, a low-pass filter is applied before every warp to
/// suppress temporal aliasing, and a parallel high-frequency path injects genuine input detail at
/// each progressive 2x upsampling stage.
/// </para>
/// <para>
/// Input is <c>[frames, channels, height, width]</c> or
/// <c>[batch, frames, channels, height, width]</c>. Output preserves the leading dimensions and
/// enlarges height and width by the configured scale factor. Every transform uses engine or
/// registered-layer operations so the complete recurrent generator remains on the gradient tape.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerCategory(LayerCategory.Residual)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High,
    TestInputShape = "2, 3, 8, 8",
    TestConstructorArgs = "3, 8, 8, 8, 1, 1, 2, 2, 0.5")]
public partial class VideoGigaGANGeneratorLayer<T> : LayerBase<T>
{
    private readonly int _inputChannels;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _numFeatures;
    private readonly int _numResBlocks;
    private readonly int _numStyleLayers;
    private readonly int _scaleFactor;
    private readonly int _flowPyramidLevels;
    private readonly double _hfShuttleWeight;

    private readonly ConvolutionalLayer<T> _inputProjection;
    private readonly ConvolutionalLayer<T>[] _spatialConvolutions1;
    private readonly ConvolutionalLayer<T>[] _spatialConvolutions2;
    private readonly FullyConnectedLayer<T>[] _styleAffineLayers;
    private readonly Conv3DLayer<T>[] _temporalConvolutions;
    private readonly SpyNetLayer<T> _flowEstimator;
    private readonly ConvolutionalLayer<T> _propagationFusion;
    private readonly ConvolutionalLayer<T> _highFrequencyProjection;
    private readonly ConvolutionalLayer<T>[] _mainUpsampleConvolutions;
    private readonly ConvolutionalLayer<T>[] _shuttleUpsampleConvolutions;
    private readonly PixelShuffleLayer<T>[] _mainPixelShuffles;
    private readonly PixelShuffleLayer<T>[] _shuttlePixelShuffles;
    private readonly ConvolutionalLayer<T> _outputProjection;

    /// <summary>Creates the native VideoGigaGAN generator graph.</summary>
    public VideoGigaGANGeneratorLayer(
        int inputChannels = 3,
        int inputHeight = 64,
        int inputWidth = 64,
        int numFeatures = 128,
        int numResBlocks = 23,
        int numStyleLayers = 14,
        int scaleFactor = 4,
        int flowPyramidLevels = 5,
        double hfShuttleWeight = 0.5)
        : base(
            new[] { inputChannels, inputHeight, inputWidth },
            new[] { inputChannels, inputHeight * scaleFactor, inputWidth * scaleFactor })
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (inputHeight <= 0) throw new ArgumentOutOfRangeException(nameof(inputHeight));
        if (inputWidth <= 0) throw new ArgumentOutOfRangeException(nameof(inputWidth));
        if (numFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(numFeatures));
        if (numResBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numResBlocks));
        if (numStyleLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numStyleLayers));
        if (scaleFactor < 2 || scaleFactor > 8 || (scaleFactor & (scaleFactor - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "VideoGigaGAN supports 2x, 4x, or 8x upscaling.");
        if (flowPyramidLevels <= 0) throw new ArgumentOutOfRangeException(nameof(flowPyramidLevels));
        int minimumFlowExtent = 1 << (flowPyramidLevels - 1);
        if (inputHeight < minimumFlowExtent || inputWidth < minimumFlowExtent)
        {
            throw new ArgumentOutOfRangeException(
                nameof(flowPyramidLevels),
                $"A {flowPyramidLevels}-level flow pyramid requires H and W >= {minimumFlowExtent}.");
        }
        if (hfShuttleWeight < 0) throw new ArgumentOutOfRangeException(nameof(hfShuttleWeight));

        _inputChannels = inputChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _numFeatures = numFeatures;
        _numResBlocks = numResBlocks;
        _numStyleLayers = numStyleLayers;
        _scaleFactor = scaleFactor;
        _flowPyramidLevels = flowPyramidLevels;
        _hfShuttleWeight = hfShuttleWeight;

        var identity = new IdentityActivation<T>();
        var leakyRelu = new LeakyReLUActivation<T>();

        _inputProjection = CreateConv(
            inputChannels, numFeatures, 3, inputHeight, inputWidth, leakyRelu);
        RegisterSubLayer(_inputProjection);

        // The image GigaGAN backbone is retained as residual spatial blocks. The first
        // NumStyleLayers blocks receive an input-conditioned affine modulation, and the same
        // stage is inflated temporally with a residual 3D convolution.
        _spatialConvolutions1 = new ConvolutionalLayer<T>[numResBlocks];
        _spatialConvolutions2 = new ConvolutionalLayer<T>[numResBlocks];
        for (int block = 0; block < numResBlocks; block++)
        {
            _spatialConvolutions1[block] = CreateConv(
                numFeatures, numFeatures, 3, inputHeight, inputWidth, leakyRelu);
            _spatialConvolutions2[block] = CreateConv(
                numFeatures, numFeatures, 3, inputHeight, inputWidth, identity);
            RegisterSubLayer(_spatialConvolutions1[block]);
            RegisterSubLayer(_spatialConvolutions2[block]);
        }

        _styleAffineLayers = new FullyConnectedLayer<T>[numStyleLayers];
        _temporalConvolutions = new Conv3DLayer<T>[numStyleLayers];
        for (int stage = 0; stage < numStyleLayers; stage++)
        {
            _styleAffineLayers[stage] = new FullyConnectedLayer<T>(
                numFeatures, numFeatures * 2, identity);
            _temporalConvolutions[stage] = new Conv3DLayer<T>(
                numFeatures, kernelSize: 3, stride: 1, padding: 1, activationFunction: identity);
            _temporalConvolutions[stage].ResolveFromShape(
                new[] { numFeatures, 1, inputHeight, inputWidth });
            RegisterSubLayer(_styleAffineLayers[stage]);
            RegisterSubLayer(_temporalConvolutions[stage]);
        }

        _flowEstimator = new SpyNetLayer<T>(flowPyramidLevels);
        _flowEstimator.ResolveFromShape(new[] { inputChannels * 2, inputHeight, inputWidth });
        _propagationFusion = CreateConv(
            numFeatures * 3, numFeatures, 3, inputHeight, inputWidth, leakyRelu);
        RegisterSubLayer(_flowEstimator);
        RegisterSubLayer(_propagationFusion);

        _highFrequencyProjection = CreateConv(
            inputChannels, numFeatures, 3, inputHeight, inputWidth, leakyRelu);
        RegisterSubLayer(_highFrequencyProjection);

        // Math.Log2 is unavailable on net471; scaleFactor is already a power of two.
        int upsampleStages = (int)Math.Round(Math.Log(scaleFactor, 2.0));
        _mainUpsampleConvolutions = new ConvolutionalLayer<T>[upsampleStages];
        _shuttleUpsampleConvolutions = new ConvolutionalLayer<T>[upsampleStages];
        _mainPixelShuffles = new PixelShuffleLayer<T>[upsampleStages];
        _shuttlePixelShuffles = new PixelShuffleLayer<T>[upsampleStages];
        int height = inputHeight;
        int width = inputWidth;
        for (int stage = 0; stage < upsampleStages; stage++)
        {
            _mainUpsampleConvolutions[stage] = CreateConv(
                numFeatures, numFeatures * 4, 3, height, width, leakyRelu);
            _shuttleUpsampleConvolutions[stage] = CreateConv(
                numFeatures, numFeatures * 4, 3, height, width, leakyRelu);
            _mainPixelShuffles[stage] = new PixelShuffleLayer<T>(2);
            _shuttlePixelShuffles[stage] = new PixelShuffleLayer<T>(2);
            RegisterSubLayer(_mainUpsampleConvolutions[stage]);
            RegisterSubLayer(_shuttleUpsampleConvolutions[stage]);
            RegisterSubLayer(_mainPixelShuffles[stage]);
            RegisterSubLayer(_shuttlePixelShuffles[stage]);
            height *= 2;
            width *= 2;
        }

        _outputProjection = CreateConv(
            numFeatures, inputChannels, 3, height, width, identity);
        RegisterSubLayer(_outputProjection);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 4;
        if (!unbatched && input.Rank != 5)
        {
            throw new ArgumentException(
                "VideoGigaGANGeneratorLayer requires [F,C,H,W] or [B,F,C,H,W] input.",
                nameof(input));
        }

        int frameAxis = unbatched ? 0 : 1;
        int channelAxis = unbatched ? 1 : 2;
        int heightAxis = unbatched ? 2 : 3;
        int widthAxis = unbatched ? 3 : 4;
        if (input.Shape[channelAxis] != _inputChannels
            || input.Shape[heightAxis] != _inputHeight
            || input.Shape[widthAxis] != _inputWidth)
        {
            throw new ArgumentException(
                $"Expected [B,F,{_inputChannels},{_inputHeight},{_inputWidth}], got [{string.Join(",", input.Shape)}].",
                nameof(input));
        }

        int batch = unbatched ? 1 : input.Shape[0];
        int frames = input.Shape[frameAxis];
        if (frames <= 0) throw new ArgumentException("VideoGigaGAN requires at least one frame.", nameof(input));

        var video = unbatched
            ? Engine.Reshape(input, new[] { 1, frames, _inputChannels, _inputHeight, _inputWidth })
            : input;
        var rawFrames = Engine.Reshape(
            video, new[] { batch * frames, _inputChannels, _inputHeight, _inputWidth });

        var features = _inputProjection.Forward(rawFrames);
        features = PropagateBidirectionally(video, features, batch, frames);

        // Input-conditioned style vector shared over a clip's spatial positions. Each stage has
        // its own affine projection, matching GigaGAN's per-layer style modulation.
        var style = Engine.ReduceMean(features, new[] { 2, 3 }, keepDims: false);
        for (int block = 0; block < _numResBlocks; block++)
        {
            var residual = features;
            var blockOutput = _spatialConvolutions1[block].Forward(features);
            blockOutput = _spatialConvolutions2[block].Forward(blockOutput);

            if (block < _numStyleLayers)
            {
                var affine = _styleAffineLayers[block].Forward(style);
                var scale = Engine.TensorSlice(
                    affine,
                    new[] { 0, 0 },
                    new[] { batch * frames, _numFeatures });
                var bias = Engine.TensorSlice(
                    affine,
                    new[] { 0, _numFeatures },
                    new[] { batch * frames, _numFeatures });
                scale = Engine.Reshape(scale, new[] { batch * frames, _numFeatures, 1, 1 });
                bias = Engine.Reshape(bias, new[] { batch * frames, _numFeatures, 1, 1 });
                blockOutput = Engine.TensorBroadcastMultiply(
                    blockOutput,
                    Engine.TensorAddScalar(scale, NumOps.One));
                blockOutput = Engine.TensorBroadcastAdd(blockOutput, bias);
            }

            features = Engine.TensorAdd(
                residual,
                Engine.TensorMultiplyScalar(blockOutput, NumOps.FromDouble(0.2)));

            if (block < _numStyleLayers)
                features = InflateTemporal(features, batch, frames, _temporalConvolutions[block]);
        }

        // High-frequency shuttle: subtract a low-pass image, project the real input detail, then
        // carry and inject it through every progressive upsampling stage.
        var lowFrequency = AntiAlias(rawFrames);
        var highFrequency = Engine.TensorSubtract(rawFrames, lowFrequency);
        var shuttle = _highFrequencyProjection.Forward(highFrequency);

        for (int stage = 0; stage < _mainUpsampleConvolutions.Length; stage++)
        {
            features = _mainPixelShuffles[stage].Forward(
                _mainUpsampleConvolutions[stage].Forward(features));
            shuttle = _shuttlePixelShuffles[stage].Forward(
                _shuttleUpsampleConvolutions[stage].Forward(shuttle));
            features = Engine.TensorAdd(
                features,
                Engine.TensorMultiplyScalar(shuttle, NumOps.FromDouble(_hfShuttleWeight)));
        }

        var output = _outputProjection.Forward(features);
        var bicubicResidual = NearestUpsample(rawFrames, _scaleFactor);
        output = Engine.TensorAdd(output, bicubicResidual);

        int outputHeight = _inputHeight * _scaleFactor;
        int outputWidth = _inputWidth * _scaleFactor;
        return unbatched
            ? Engine.Reshape(output, new[] { frames, _inputChannels, outputHeight, outputWidth })
            : Engine.Reshape(output, new[] { batch, frames, _inputChannels, outputHeight, outputWidth });
    }

    private Tensor<T> PropagateBidirectionally(
        Tensor<T> rawVideo,
        Tensor<T> flatFeatures,
        int batch,
        int frames)
    {
        var featureVideo = Engine.Reshape(
            flatFeatures, new[] { batch, frames, _numFeatures, _inputHeight, _inputWidth });
        var forward = new Tensor<T>[frames];
        var backward = new Tensor<T>[frames];

        forward[0] = Engine.TensorSliceAxis(featureVideo, axis: 1, index: 0);
        for (int frame = 1; frame < frames; frame++)
        {
            var current = Engine.TensorSliceAxis(featureVideo, axis: 1, index: frame);
            var currentRaw = Engine.TensorSliceAxis(rawVideo, axis: 1, index: frame);
            var previousRaw = Engine.TensorSliceAxis(rawVideo, axis: 1, index: frame - 1);
            var flowInput = Engine.TensorConcatenate(new[] { currentRaw, previousRaw }, axis: 1);
            var flow = _flowEstimator.Forward(flowInput);
            var warped = FlowWarpHelper.Warp(Engine, AntiAlias(forward[frame - 1]), flow);
            forward[frame] = Engine.TensorAdd(current, warped);
        }

        backward[frames - 1] = Engine.TensorSliceAxis(featureVideo, axis: 1, index: frames - 1);
        for (int frame = frames - 2; frame >= 0; frame--)
        {
            var current = Engine.TensorSliceAxis(featureVideo, axis: 1, index: frame);
            var currentRaw = Engine.TensorSliceAxis(rawVideo, axis: 1, index: frame);
            var nextRaw = Engine.TensorSliceAxis(rawVideo, axis: 1, index: frame + 1);
            var flowInput = Engine.TensorConcatenate(new[] { currentRaw, nextRaw }, axis: 1);
            var flow = _flowEstimator.Forward(flowInput);
            var warped = FlowWarpHelper.Warp(Engine, AntiAlias(backward[frame + 1]), flow);
            backward[frame] = Engine.TensorAdd(current, warped);
        }

        var fusedFrames = new Tensor<T>[frames];
        for (int frame = 0; frame < frames; frame++)
        {
            var current = Engine.TensorSliceAxis(featureVideo, axis: 1, index: frame);
            var fusionInput = Engine.TensorConcatenate(
                new[] { current, forward[frame], backward[frame] }, axis: 1);
            fusedFrames[frame] = _propagationFusion.Forward(fusionInput);
        }

        var fusedVideo = Engine.TensorStack(fusedFrames, axis: 1);
        return Engine.Reshape(
            fusedVideo, new[] { batch * frames, _numFeatures, _inputHeight, _inputWidth });
    }

    private Tensor<T> InflateTemporal(
        Tensor<T> flatFeatures,
        int batch,
        int frames,
        Conv3DLayer<T> temporalConvolution)
    {
        var residual = flatFeatures;
        var video = Engine.Reshape(
            flatFeatures, new[] { batch, frames, _numFeatures, _inputHeight, _inputWidth });
        video = Engine.TensorPermute(video, new[] { 0, 2, 1, 3, 4 });
        video = temporalConvolution.Forward(video);
        video = Engine.TensorPermute(video, new[] { 0, 2, 1, 3, 4 });
        video = Engine.Reshape(
            video, new[] { batch * frames, _numFeatures, _inputHeight, _inputWidth });
        return Engine.TensorAdd(
            residual,
            Engine.TensorMultiplyScalar(video, NumOps.FromDouble(0.2)));
    }

    /// <summary>
    /// Fixed binomial BlurPool used immediately before flow warping. It is composed from
    /// tape-recorded padding, slicing, scaling, and addition instead of AvgPool2D: Tensors 0.120.5's
    /// padded AvgPool2D backward indexes beyond its input-gradient buffer. The separable
    /// [1,2,1] x [1,2,1] / 16 kernel is also the anti-aliasing filter used by BlurPool.
    /// </summary>
    private Tensor<T> AntiAlias(Tensor<T> input)
    {
        if (input.Rank != 4)
            throw new ArgumentException("VideoGigaGAN BlurPool requires [B,C,H,W] input.", nameof(input));

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        var padded = Engine.Pad(input, 1, 1, 1, 1, NumOps.Zero);
        int[] kernel = [1, 2, 1];
        Tensor<T>? blurred = null;
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                var shifted = Engine.TensorSlice(
                    padded,
                    new[] { 0, 0, y, x },
                    new[] { batch, channels, height, width });
                var weighted = Engine.TensorMultiplyScalar(
                    shifted,
                    NumOps.FromDouble(kernel[y] * kernel[x] / 16.0));
                blurred = blurred is null ? weighted : Engine.TensorAdd(blurred, weighted);
            }
        }

        return blurred ?? throw new InvalidOperationException("VideoGigaGAN BlurPool kernel is empty.");
    }

    private Tensor<T> NearestUpsample(Tensor<T> input, int scale)
    {
        var materialized = Engine.TensorMultiplyScalar(input, NumOps.One);
        var heightUpsampled = Engine.TensorRepeatInterleave(materialized, scale, dim: 2);
        return Engine.TensorRepeatInterleave(heightUpsampled, scale, dim: 3);
    }

    private static ConvolutionalLayer<T> CreateConv(
        int inputChannels,
        int outputChannels,
        int kernel,
        int inputHeight,
        int inputWidth,
        IActivationFunction<T> activation)
    {
        var layer = new ConvolutionalLayer<T>(
            outputChannels, kernel, stride: 1, padding: kernel / 2, activationFunction: activation);
        layer.ResolveFromShape(new[] { inputChannels, inputHeight, inputWidth });
        return layer;
    }

    private IEnumerable<ILayer<T>> OrderedLayers()
    {
        yield return _inputProjection;
        yield return _flowEstimator;
        yield return _propagationFusion;
        foreach (var layer in _spatialConvolutions1) yield return layer;
        foreach (var layer in _spatialConvolutions2) yield return layer;
        foreach (var layer in _styleAffineLayers) yield return layer;
        foreach (var layer in _temporalConvolutions) yield return layer;
        yield return _highFrequencyProjection;
        foreach (var layer in _mainUpsampleConvolutions) yield return layer;
        foreach (var layer in _shuttleUpsampleConvolutions) yield return layer;
        foreach (var layer in _mainPixelShuffles) yield return layer;
        foreach (var layer in _shuttlePixelShuffles) yield return layer;
        yield return _outputProjection;
    }

    /// <inheritdoc/>
    public override long ParameterCount => OrderedLayers().Sum(layer => layer.ParameterCount);

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var values = new List<T>((int)ParameterCount);
        foreach (var layer in OrderedLayers())
        {
            var parameters = layer.GetParameters();
            for (int i = 0; i < parameters.Length; i++) values.Add(parameters[i]);
        }
        return new Vector<T>(values.ToArray());
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        int offset = 0;
        foreach (var layer in OrderedLayers())
        {
            int count = (int)layer.ParameterCount;
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var values = new List<T>((int)ParameterCount);
        foreach (var layer in OrderedLayers())
        {
            var gradients = layer.GetParameterGradients();
            for (int i = 0; i < gradients.Length; i++) values.Add(gradients[i]);
        }
        return new Vector<T>(values.ToArray());
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in OrderedLayers()) layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in OrderedLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in OrderedLayers()) layer.ResetState();
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputChannels"] = _inputChannels.ToString(ci);
        metadata["InputHeight"] = _inputHeight.ToString(ci);
        metadata["InputWidth"] = _inputWidth.ToString(ci);
        metadata["NumFeatures"] = _numFeatures.ToString(ci);
        metadata["NumResBlocks"] = _numResBlocks.ToString(ci);
        metadata["NumStyleLayers"] = _numStyleLayers.ToString(ci);
        metadata["ScaleFactor"] = _scaleFactor.ToString(ci);
        metadata["FlowPyramidLevels"] = _flowPyramidLevels.ToString(ci);
        metadata["HFShuttleWeight"] = _hfShuttleWeight.ToString("R", ci);
        return metadata;
    }
}
