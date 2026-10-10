using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// Global response normalization (ConvNeXt v2, Woo et al. 2023; APNet2's reference <c>GRN</c>) over a channels-first
/// sequence <c>[1, C, T]</c>: <c>G = ‖x‖₂ over time</c> per channel, <c>N = G / (mean_c G + 1e-6)</c>,
/// <c>y = γ ⊙ (x ⊙ N) + β + x</c> with γ and β starting at zero.
/// </summary>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4, 6", TestConstructorArgs = "4")]
[ElementWiseShape(Note = "A per-channel response normalization; the shape is carried through.")]
[AutoParameters]
internal sealed partial class GlobalResponseNormLayer<T> : LayerBase<T>
{
    private readonly int _channels;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _gamma;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _beta;

    public override bool SupportsTraining => true;

    public GlobalResponseNormLayer([LayerState] int channels)
        : base(new[] { channels }, new[] { channels })
    {
        _channels = channels;
        _gamma = new Tensor<T>(new[] { channels });
        _beta = new Tensor<T>(new[] { channels });
        RegisterTrainableParameter(_gamma, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_beta, PersistentTensorRole.Biases);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int c = input.Shape[1], t = input.Shape[2];
        var g = Engine.TensorPow(Engine.ReduceSum(Engine.TensorMultiply(input, input), new[] { 2 }, keepDims: true), NumOps.FromDouble(0.5)); // [1, C, 1]
        var mean = Engine.TensorAddScalar(Engine.ReduceMean(g, new[] { 1 }, keepDims: true), NumOps.FromDouble(1e-6));                     // [1, 1, 1]
        var n = Engine.TensorMultiply(g, Engine.TensorTile(Engine.TensorReciprocal(mean), new[] { 1, c, 1 }));
        var scaled = Engine.TensorMultiply(input, Engine.TensorTile(n, new[] { 1, 1, t }));
        var gamma = Engine.TensorTile(Engine.Reshape(_gamma, new[] { 1, c, 1 }), new[] { 1, 1, t });
        var beta = Engine.TensorTile(Engine.Reshape(_beta, new[] { 1, c, 1 }), new[] { 1, 1, t });
        return Engine.TensorAdd(Engine.TensorAdd(Engine.TensorMultiply(gamma, scaled), beta), input);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}

/// <summary>
/// APNet2's spectrum predictor (Du et al. 2023, §3.1, Fig. 2; reference redmist328/APNet2 <c>Generator</c>): an input
/// convolution, layer norm, k ConvNeXt v2 blocks — a depth-wise convolution, layer norm, a point-wise expansion, GELU,
/// GRN, a point-wise projection and a residual connection — a final layer norm and one or more output convolutions.
/// </summary>
/// <remarks>Every convolution starts from N(0, 0.02²) (<c>trunc_normal_</c> at ±2, which an σ of 0.02 never reaches) with
/// zero biases, as the reference's <c>_init_weights</c>; layer norms use ε = 1e-6.</remarks>
internal sealed class ConvNeXtV2Predictor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly NormedConv1DLayer<T> _input;
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly List<(NormedConv1DLayer<T> Depthwise, LayerNormalizationLayer<T> Norm, NormedConv1DLayer<T> Expand,
        GlobalResponseNormLayer<T> Grn, NormedConv1DLayer<T> Project)> _blocks = new();
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly List<NormedConv1DLayer<T>> _outputs = new();

    public ConvNeXtV2Predictor(IEngine engine, Random initialization, int melChannels, int dim, int intermediate, int blocks, int depthwiseKernel,
        int inputKernel, int outputKernel, int bins, int outputCount, List<LayerBase<T>> owner)
    {
        _engine = engine;
        Func<double> normal = () =>
        {
            double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
            return 0.02 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        };
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int groups)
        {
            var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, 1, groups, (kernel - 1) / 2, false, ConvolutionNormalization.None);
            conv.Reinitialize(normal);
            owner.Add(conv);
            return conv;
        }
        LayerNormalizationLayer<T> Norm()
        {
            var norm = new LayerNormalizationLayer<T>(dim, 1e-6);
            owner.Add(norm);
            return norm;
        }
        _input = Conv(melChannels, dim, inputKernel, 1);
        _norm = Norm();
        for (int i = 0; i < blocks; i++)
        {
            var depthwise = Conv(dim, dim, depthwiseKernel, dim);
            var norm = Norm();
            var expand = Conv(dim, intermediate, 1, 1);
            var grn = new GlobalResponseNormLayer<T>(intermediate);
            owner.Add(grn);
            var project = Conv(intermediate, dim, 1, 1);
            _blocks.Add((depthwise, norm, expand, grn, project));
        }
        _finalNorm = Norm();
        for (int i = 0; i < outputCount; i++) _outputs.Add(Conv(dim, bins, outputKernel, 1));
    }

    // Layer norm over the channels of [1, C, T].
    private Tensor<T> ChannelNorm(LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = norm.Forward(_engine.TensorTranspose(_engine.Reshape(x, new[] { c, t })));
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, c, t });
    }

    // GELU(x) = x · Φ(x) = ½ x erfc(−x / √2).
    private Tensor<T> Gelu(Tensor<T> x)
        => _engine.TensorMultiplyScalar(_engine.TensorMultiply(x, _engine.TensorErfc(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(-1 / Math.Sqrt(2))))),
            NumOps.FromDouble(0.5));

    /// <summary>The output convolutions' maps <c>[1, bins, frames]</c> of a mel spectrogram <c>[1, mel, frames]</c>.</summary>
    public IReadOnlyList<Tensor<T>> Forward(Tensor<T> mel)
    {
        var x = ChannelNorm(_norm, _input.Forward(mel));
        foreach (var (depthwise, norm, expand, grn, project) in _blocks)
            x = _engine.TensorAdd(x, project.Forward(grn.Forward(Gelu(expand.Forward(ChannelNorm(norm, depthwise.Forward(x)))))));
        var h = ChannelNorm(_finalNorm, x);
        return _outputs.Select(o => o.Forward(h)).ToList();
    }
}

/// <summary>
/// APNet2's multi-resolution discriminator (Du et al. 2023, §3.2, Fig. 3; reference <c>DiscriminatorR</c>): per
/// resolution the magnitude of a centred, unwindowed STFT laid out as frequency × time, five weight-normalized strided
/// 2-D convolutions of 64 channels ((7, 5)/(2, 2), (5, 3)/(2, 1), (5, 3)/(2, 2), 3/(2, 1), 3/(2, 2)) with leaky ReLU 0.1,
/// and a (3, 3) convolution to one channel. Each sub-discriminator returns its score and its feature maps.
/// </summary>
internal sealed class ApNet2ResolutionDiscriminators<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly List<(CenteredComplexStft<T> Stft, List<Conv2DLayer<T>> Convs, Conv2DLayer<T> Post)> _discriminators = new();

    public ApNet2ResolutionDiscriminators(IEngine engine, int[] fftSizes, int[] hopSizes, int[] windowSizes, int channels)
    {
        if (fftSizes.Length != hopSizes.Length || fftSizes.Length != windowSizes.Length)
            throw new ArgumentException("Every resolution needs an FFT size, a hop and a window.");
        _engine = engine;
        for (int r = 0; r < fftSizes.Length; r++)
        {
            var convs = new List<Conv2DLayer<T>>
            {
                Add(new Conv2DLayer<T>(1, channels, 7, 5, 2, 2, 3, 2, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 5, 3, 2, 1, 2, 1, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 5, 3, 2, 2, 2, 1, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 3, 2, 1, 1, 1, true, ConvolutionNormalization.Weight)),
                Add(new Conv2DLayer<T>(channels, channels, 3, 3, 2, 2, 1, 1, true, ConvolutionNormalization.Weight)),
            };
            var post = Add(new Conv2DLayer<T>(channels, 1, 3, 3, 1, 1, 1, 1, true, ConvolutionNormalization.Weight));
            _discriminators.Add((new CenteredComplexStft<T>(engine, fftSizes[r], hopSizes[r], windowSizes[r], hannWindow: false), convs, post));
        }
    }

    private Conv2DLayer<T> Add(Conv2DLayer<T> layer)
    {
        _layers.Add(layer);
        return layer;
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>Each sub-discriminator's score and feature maps for a waveform <c>[samples]</c>.</summary>
    public List<(Tensor<T> Score, List<Tensor<T>> Features)> Forward(Tensor<T> audio)
    {
        var results = new List<(Tensor<T>, List<Tensor<T>>)>();
        foreach (var (stft, convs, post) in _discriminators)
        {
            var (re, im) = stft.Forward(audio);
            // |S| (the 1e-12 keeps the square root's gradient finite at an exact zero).
            var magnitude = _engine.TensorPow(_engine.TensorAddScalar(_engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im)),
                NumOps.FromDouble(1e-12)), NumOps.FromDouble(0.5));
            var h = _engine.Reshape(magnitude, new[] { 1, 1, magnitude.Shape[1], magnitude.Shape[2] });
            var features = new List<Tensor<T>>();
            foreach (var conv in convs)
            {
                h = VocoderOps.LeakyRelu(_engine, conv.Forward(h), 0.1);
                features.Add(h);
            }
            h = post.Forward(h);
            features.Add(h);
            results.Add((h, features));
        }
        return results;
    }
}
