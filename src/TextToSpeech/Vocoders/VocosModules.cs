using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>A per-channel scale γ ⊙ x of <c>[1, channels, time]</c> (ConvNeXt's layer scale), γ starting at a constant.</summary>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 3, 4", TestConstructorArgs = "3, 0.5")]
[ElementWiseShape(Note = "A per-channel scale; the shape is carried through.")]
[AutoParameters]
internal sealed partial class ChannelScaleLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly double _initial;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _scale;

    public override bool SupportsTraining => true;

    /// <summary>γ, for loading released weights.</summary>
    internal Tensor<T> Scale => _scale;

    public ChannelScaleLayer([LayerState] int channels, [LayerState] double initial)
        : base(new[] { channels }, new[] { channels })
    {
        _channels = channels;
        _initial = initial;
        _scale = new Tensor<T>(new[] { channels });
        for (int c = 0; c < channels; c++) _scale[c] = NumOps.FromDouble(initial);
        RegisterTrainableParameter(_scale, PersistentTensorRole.Weights);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => Engine.TensorMultiply(input, Engine.TensorTile(Engine.Reshape(_scale, new[] { 1, _channels, 1 }), new[] { input.Shape[0], 1, input.Shape[2] }));

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Channels"] = _channels.ToString(inv);
        metadata["Initial"] = _initial.ToString("R", inv);
        return metadata;
    }
}

/// <summary>
/// Vocos's generator (Siuzdak 2024, §3.2; reference gemelo-ai/vocos <c>VocosBackbone</c>, <c>ISTFTHead</c>): a 7-wide
/// convolution embedding the mel spectrogram into d channels and a layer norm; ConvNeXt blocks at frame rate — a 7-wide
/// depthwise convolution, layer norm, a pointwise expansion to the intermediate width, GELU, a pointwise projection, a
/// layer scale starting at 1/blocks and a residual; a final layer norm; a linear map to n_fft + 2 channels split into
/// m and p; magnitude <c>M = min(exp(m), 100)</c>, phase on the unit circle <c>(cos p, sin p)</c>; and the inverse STFT
/// (Hann window, centred).
/// </summary>
/// <remarks>The backbone's convolutions and linear maps draw N(0, 0.02) weights with zero biases (reference
/// <c>_init_weights</c>, a truncated normal whose ±2 bounds are 100σ away); layer norms use ε = 1e-6; GELU is exact.</remarks>
internal sealed class VocosGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _dim;
    private readonly int _fft;
    private readonly NormedConv1DLayer<T> _embed;
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly List<(NormedConv1DLayer<T> Depthwise, LayerNormalizationLayer<T> Norm, NormedConv1DLayer<T> Expand,
        NormedConv1DLayer<T> Project, ChannelScaleLayer<T> Scale)> _blocks = new();
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly NormedConv1DLayer<T> _head;
    private readonly InverseStft<T> _istft;
    // AdaLayerNorm (conditional generators): per-class scale and shift of an affine-free layer norm, one pair for the
    // embedding's norm and one per block.
    private readonly List<(TiedEmbeddingLayer<T> Scale, TiedEmbeddingLayer<T> Shift)> _adaptive = new();

    /// <param name="adaptiveClasses">Classes of Vocos's <c>AdaLayerNorm</c> (the EnCodec model's bandwidths); 0 for plain
    /// layer norms.</param>
    /// <param name="samePadding">The ISTFT head's <c>padding="same"</c> (the EnCodec model) instead of <c>center</c>.</param>
    public VocosGenerator(IEngine engine, Random initialization, int melChannels, int dim, int intermediate, int blocks, int fftSize, int hopSize,
        int adaptiveClasses = 0, bool samePadding = false)
    {
        _engine = engine;
        _dim = dim;
        _fft = fftSize;
        TiedEmbeddingLayer<T> Table(double value)
        {
            var table = new TiedEmbeddingLayer<T>(adaptiveClasses, dim);
            table.Reinitialize(() => value);
            _layers.Add(table);
            return table;
        }
        Func<double> normal = () =>
        {
            double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
            return 0.02 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        };
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int groups, int padding)
        {
            var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, 1, groups, padding, false, ConvolutionNormalization.None);
            conv.Reinitialize(normal);
            _layers.Add(conv);
            return conv;
        }
        _embed = Conv(melChannels, dim, 7, 1, 3);
        _norm = new LayerNormalizationLayer<T>(dim, 1e-6);
        if (adaptiveClasses > 0) _adaptive.Add((Table(1.0), Table(0.0)));
        else _layers.Add(_norm);
        for (int i = 0; i < blocks; i++)
        {
            var depthwise = Conv(dim, dim, 7, dim, 3);
            var norm = new LayerNormalizationLayer<T>(dim, 1e-6);
            if (adaptiveClasses > 0) _adaptive.Add((Table(1.0), Table(0.0)));
            else _layers.Add(norm);
            var expand = Conv(dim, intermediate, 1, 1, 0);
            var project = Conv(intermediate, dim, 1, 1, 0);
            var scale = new ChannelScaleLayer<T>(dim, 1.0 / blocks);
            _layers.Add(scale);
            _blocks.Add((depthwise, norm, expand, project, scale));
        }
        _layers.Add(_finalNorm = new LayerNormalizationLayer<T>(dim, 1e-6));
        // The head's linear map keeps PyTorch's default initialization (the backbone's init does not reach it).
        _layers.Add(_head = new NormedConv1DLayer<T>(dim, fftSize + 2, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
        _istft = new InverseStft<T>(engine, fftSize, hopSize, fftSize, samePadding);
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    // Layer norm over channels of [1, C, T].
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

    // AdaLayerNorm over channels of [1, C, T]: layer_norm(x) · scale[class] + shift[class] (ε = 1e-6, no affine).
    private Tensor<T> AdaptiveNorm(int index, int condition, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = _engine.TensorTranspose(_engine.Reshape(x, new[] { c, t }));                             // [T, C]
        var mean = _engine.TensorTile(_engine.ReduceMean(rows, new[] { 1 }, keepDims: true), new[] { 1, c });
        var centered = _engine.TensorSubtract(rows, mean);
        var variance = _engine.ReduceMean(_engine.TensorSquare(centered), new[] { 1 }, keepDims: true);
        var deviation = _engine.TensorTile(_engine.TensorSqrt(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-6))), new[] { 1, c });
        var normalized = _engine.TensorDivide(centered, deviation);
        var id = new Tensor<T>(new[] { 1 });
        id[0] = NumOps.FromDouble(condition);
        var (scaleTable, shiftTable) = _adaptive[index];
        var scale = _engine.TensorTile(scaleTable.Forward(id), new[] { t, 1 });
        var shift = _engine.TensorTile(shiftTable.Forward(id), new[] { t, 1 });
        var output = _engine.TensorAdd(_engine.TensorMultiply(normalized, scale), shift);
        return _engine.Reshape(_engine.TensorTranspose(output), new[] { 1, c, t });
    }

    private Tensor<T> Norm(int index, LayerNormalizationLayer<T> norm, Tensor<T> x, int condition) =>
        _adaptive.Count > 0 ? AdaptiveNorm(index, condition, x) : ChannelNorm(norm, x);

    /// <summary>A waveform <c>[(frames − 1) · hop]</c> (<c>[frames · hop]</c> with "same" padding) from features
    /// <c>[1, channels, frames]</c>; <paramref name="condition"/> is the adaptive norms' class.</summary>
    public Tensor<T> Forward(Tensor<T> mel, int condition = 0)
    {
        var x = Norm(0, _norm, _embed.Forward(mel), condition);
        int block = 1;
        foreach (var (depthwise, norm, expand, project, scale) in _blocks)
            x = _engine.TensorAdd(x, scale.Forward(project.Forward(Gelu(expand.Forward(Norm(block++, norm, depthwise.Forward(x), condition))))));
        var h = _head.Forward(ChannelNorm(_finalNorm, x));                                               // [1, n_fft + 2, frames]
        int bins = _fft / 2 + 1, frames = h.Shape[2];
        var m = _engine.TensorSlice(h, new[] { 0, 0, 0 }, new[] { 1, bins, frames });
        var p = _engine.TensorSlice(h, new[] { 0, bins, 0 }, new[] { 1, bins, frames });
        // min(exp(m), 100) = 100 − relu(100 − exp(m))
        var magnitude = _engine.TensorAddScalar(_engine.TensorNegate(_engine.ReLU(_engine.TensorAddScalar(_engine.TensorNegate(_engine.TensorExp(m)),
            NumOps.FromDouble(100)))), NumOps.FromDouble(100));
        return _istft.Forward(magnitude, p);
    }

    /// <summary>Loads the reference's <c>backbone.*</c> and <c>head.*</c> parameters (gemelo-ai/vocos).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, int inputChannels, int intermediate)
    {
        int dim = _dim, kernel = 7;
        _embed.LoadTorchWeights(read("backbone.embed.weight", new[] { dim, inputChannels, kernel }), null, read("backbone.embed.bias", new[] { dim }));
        void LoadNorm(int index, LayerNormalizationLayer<T> norm, string name)
        {
            if (_adaptive.Count > 0)
            {
                _adaptive[index].Scale.LoadTable(read(name + ".scale.weight", new[] { _adaptive[index].Scale.VocabularySize, dim }));
                _adaptive[index].Shift.LoadTable(read(name + ".shift.weight", new[] { _adaptive[index].Shift.VocabularySize, dim }));
            }
            else
            {
                CodecBased.TorchParameters.LayerNorm(_engine, norm, read(name + ".weight", new[] { dim }), read(name + ".bias", new[] { dim }));
            }
        }
        LoadNorm(0, _norm, "backbone.norm");
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (depthwise, norm, expand, project, scale) = _blocks[i];
            string p = $"backbone.convnext.{i}";
            depthwise.LoadTorchWeights(read(p + ".dwconv.weight", new[] { dim, 1, kernel }), null, read(p + ".dwconv.bias", new[] { dim }));
            LoadNorm(i + 1, norm, p + ".norm");
            // pwconv1 / pwconv2 are nn.Linear over channels; the 1-wide convolutions take them as [out, in, 1].
            expand.LoadTorchWeights(read(p + ".pwconv1.weight", new[] { intermediate, dim }), null, read(p + ".pwconv1.bias", new[] { intermediate }));
            project.LoadTorchWeights(read(p + ".pwconv2.weight", new[] { dim, intermediate }), null, read(p + ".pwconv2.bias", new[] { dim }));
            var gamma = read(p + ".gamma", new[] { dim });
            for (int c = 0; c < dim; c++) scale.Scale[c] = NumOps.FromDouble(gamma[c]);
            _engine.InvalidatePersistentTensor(scale.Scale);
        }
        CodecBased.TorchParameters.LayerNorm(_engine, _finalNorm, read("backbone.final_layer_norm.weight", new[] { dim }),
            read("backbone.final_layer_norm.bias", new[] { dim }));
        _head.LoadTorchWeights(read("head.out.weight", new[] { _fft + 2, dim }), null, read("head.out.bias", new[] { _fft + 2 }));
    }
}
