using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Kaiming-normal initialization and the layer-level SGD step shared by Matcha-TTS's decoder layers.</summary>
internal static class KaimingInit
{
    /// <summary><c>kaiming_normal_(nonlinearity='relu')</c>: N(0, 2 / fan_in).</summary>
    public static Tensor<T> Normal<T>(int[] shape, int fanIn, int? seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = seed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        double std = Math.Sqrt(2.0 / fanIn);
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            tensor[i] = ops.FromDouble(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return tensor;
    }

    /// <summary>Plain SGD on the weight then the bias, for the layer-level <c>UpdateParameters</c> contract.</summary>
    public static void Step<T>(LayerBase<T> layer, T learningRate, Tensor<T> weight, Tensor<T> bias)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var gradients = layer.GetParameterGradients();
        if (gradients.Length != weight.Length + bias.Length) return;
        int k = 0;
        foreach (var tensor in new[] { weight, bias })
        {
            for (int i = 0; i < tensor.Length; i++, k++)
                tensor[i] = ops.Subtract(tensor[i], ops.Multiply(learningRate, gradients[k]));
            AiDotNetEngine.Current.InvalidatePersistentTensor(tensor);
        }
    }
}

/// <summary>A linear layer <c>y = x W + b</c> with Kaiming-normal (ReLU) weights and zero bias, as Matcha-TTS's decoder
/// initializes every <c>nn.Linear</c>.</summary>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "2, 4", TestConstructorArgs = "4, 3, true")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class KaimingLinearLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _in;
    private readonly int _out;
    private readonly bool _useBias;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weight;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    public override bool SupportsTraining => true;

    public KaimingLinearLayer([LayerState] int inFeatures, [LayerState] int outFeatures, [LayerState] bool useBias)
        : base(new[] { inFeatures }, new[] { outFeatures })
    {
        _in = inFeatures;
        _out = outFeatures;
        _useBias = useBias;
        _weight = KaimingInit.Normal<T>(new[] { inFeatures, outFeatures }, inFeatures, RandomSeed);
        _bias = new Tensor<T>(new[] { outFeatures });
        RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_out)),
            }
            : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var y = Engine.TensorMatMul(input, _weight);
        if (!_useBias) return y;
        return Engine.TensorAdd(y, Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _out }), new[] { y.Shape[0], 1 }));
    }

    public override void UpdateParameters(T learningRate) => KaimingInit.Step(this, learningRate, _weight, _bias);

    internal int InFeatures => _in;
    internal int OutFeatures => _out;

    /// <summary>Loads PyTorch's <c>weight [out, in]</c> and, when the layer has one, <c>bias [out]</c>.</summary>
    internal void Load(double[] weight, double[]? bias)
    {
        if (weight.Length != _in * _out) throw new ArgumentException($"Expected {_in * _out} weights, got {weight.Length}.", nameof(weight));
        for (int o = 0; o < _out; o++)
            for (int i = 0; i < _in; i++) _weight[i, o] = NumOps.FromDouble(weight[o * _in + i]);
        for (int o = 0; o < _out; o++) _bias[o] = NumOps.FromDouble(bias is null ? 0.0 : bias[o]);
        Engine.InvalidatePersistentTensor(_weight);
        Engine.InvalidatePersistentTensor(_bias);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InFeatures"] = _in.ToString(inv);
        metadata["OutFeatures"] = _out.ToString(inv);
        metadata["UseBias"] = _useBias.ToString(inv);
        return metadata;
    }
}

/// <summary>A 1-D convolution with PyTorch <c>Conv1d</c>/<c>ConvTranspose1d</c> semantics, Kaiming-normal (ReLU) weights
/// and zero bias, as Matcha-TTS's decoder initializes its convolutions.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 2, 6", TestConstructorArgs = "2, 3, 3, 1, 1, false")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class KaimingConv1DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _in;
    private readonly int _out;
    private readonly int _kernel;
    private readonly int _stride;
    private readonly int _padding;
    private readonly bool _transposed;
    private readonly int _dilation;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weight;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    public override bool SupportsTraining => true;

    public KaimingConv1DLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int kernelSize,
        [LayerState] int stride, [LayerState] int padding, [LayerState] bool transposed, [LayerState] int dilation = 1)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        if (dilation <= 0 || (transposed && dilation != 1)) throw new ArgumentOutOfRangeException(nameof(dilation));
        _dilation = dilation;
        _in = inChannels;
        _out = outChannels;
        _kernel = kernelSize;
        _stride = stride;
        _padding = padding;
        _transposed = transposed;
        // PyTorch's fan_in is weight.size(1) · k, which is out_channels for a transposed convolution.
        _weight = KaimingInit.Normal<T>(
            transposed ? new[] { inChannels, outChannels, 1, kernelSize } : new[] { outChannels, inChannels, 1, kernelSize },
            (transposed ? outChannels : inChannels) * kernelSize, RandomSeed);
        _bias = new Tensor<T>(new[] { outChannels });
        RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int batch = input.Shape[0], time = input.Shape[2];
        var x4 = Engine.Reshape(input, new[] { batch, _in, 1, time });
        var y = _transposed
            ? Engine.ConvTranspose2D(x4, _weight, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 0, 0 })
            : Engine.Conv2D(x4, _weight, new[] { 1, _stride }, new[] { 0, _padding }, new[] { 1, _dilation });
        int outTime = y.Shape[3];
        var bias = Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _out, 1, 1 }), new[] { batch, 1, 1, outTime });
        return Engine.Reshape(Engine.TensorAdd(y, bias), new[] { batch, _out, outTime });
    }

    public override void UpdateParameters(T learningRate) => KaimingInit.Step(this, learningRate, _weight, _bias);

    /// <summary>Loads PyTorch's weight (<c>[out, in, k]</c>, or <c>[in, out, k]</c> transposed) and bias.</summary>
    internal void Load(double[] weight, double[] bias)
    {
        if (weight.Length != _weight.Length) throw new ArgumentException($"Expected {_weight.Length} weights, got {weight.Length}.", nameof(weight));
        if (bias.Length != _out) throw new ArgumentException($"Expected {_out} biases, got {bias.Length}.", nameof(bias));
        for (int i = 0; i < weight.Length; i++) _weight[i] = NumOps.FromDouble(weight[i]);
        for (int o = 0; o < _out; o++) _bias[o] = NumOps.FromDouble(bias[o]);
        Engine.InvalidatePersistentTensor(_weight);
        Engine.InvalidatePersistentTensor(_bias);
    }

    internal int InChannels => _in;
    internal int OutChannels => _out;
    internal int KernelSize => _kernel;
    internal bool Transposed => _transposed;

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _in.ToString(inv);
        metadata["OutChannels"] = _out.ToString(inv);
        metadata["KernelSize"] = _kernel.ToString(inv);
        metadata["Stride"] = _stride.ToString(inv);
        metadata["Padding"] = _padding.ToString(inv);
        metadata["Transposed"] = _transposed.ToString(inv);
        metadata["Dilation"] = _dilation.ToString(inv);
        return metadata;
    }
}

/// <summary>SnakeBeta (Ziyin et al. 2020, as Matcha-TTS uses it): a linear projection, then
/// <c>x + 1/(e^β + 1e-9) · sin²(e^α x)</c> with per-channel log-scale α and β initialized to 0.</summary>
[LayerCategory(LayerCategory.Activation)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class SnakeBetaLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inFeatures;
    private readonly int _outFeatures;
    private readonly KaimingLinearLayer<T> _projection;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logAlpha;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logBeta;

    public override bool SupportsTraining => true;

    public SnakeBetaLayer([LayerState] int inFeatures, [LayerState] int outFeatures)
        : base(new[] { inFeatures }, new[] { outFeatures })
    {
        _inFeatures = inFeatures;
        _outFeatures = outFeatures;
        _projection = new KaimingLinearLayer<T>(inFeatures, outFeatures, true);
        _logAlpha = new Tensor<T>(new[] { outFeatures });
        _logBeta = new Tensor<T>(new[] { outFeatures });
        RegisterSubLayer(_projection);
        RegisterTrainableParameter(_logAlpha, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_logBeta, PersistentTensorRole.Weights);
    }

    internal KaimingLinearLayer<T> Projection => _projection;

    /// <summary>Loads Matcha's SnakeBeta: the projection (<c>proj.weight [out, in]</c>, <c>proj.bias</c>) and the log-scale
    /// <c>alpha</c> and <c>beta</c>.</summary>
    internal void Load(double[] weight, double[] bias, double[] alpha, double[] beta)
    {
        _projection.Load(weight, bias);
        if (alpha.Length != _outFeatures || beta.Length != _outFeatures)
            throw new ArgumentException($"Expected {_outFeatures} alphas and betas.");
        for (int i = 0; i < _outFeatures; i++)
        {
            _logAlpha[i] = NumOps.FromDouble(alpha[i]);
            _logBeta[i] = NumOps.FromDouble(beta[i]);
        }
        Engine.InvalidatePersistentTensor(_logAlpha);
        Engine.InvalidatePersistentTensor(_logBeta);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outFeatures)),
            }
            : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var x = _projection.Forward(input);
        int rows = x.Shape[0];
        var alpha = Engine.TensorTile(Engine.Reshape(Engine.TensorExp(_logAlpha), new[] { 1, _outFeatures }), new[] { rows, 1 });
        var beta = Engine.TensorTile(Engine.Reshape(Engine.TensorExp(_logBeta), new[] { 1, _outFeatures }), new[] { rows, 1 });
        var sine = Engine.TensorSin(Engine.TensorMultiply(x, alpha));
        var periodic = Engine.TensorDivide(Engine.TensorMultiply(sine, sine), Engine.TensorAddScalar(beta, NumOps.FromDouble(1e-9)));
        return Engine.TensorAdd(x, periodic);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InFeatures"] = _inFeatures.ToString(inv);
        metadata["OutFeatures"] = _outFeatures.ToString(inv);
        return metadata;
    }
}

/// <summary>
/// Matcha-TTS's flow-matching vector field estimator v_θ(x_t, μ, t) (reference <c>matcha/models/components/decoder.py</c>):
/// a 1-D U-Net over <c>[x; μ]</c> with a sinusoidal time embedding (scale 1000) and a SiLU MLP, ResNet blocks
/// (conv–GroupNorm(8)–Mish with a Mish–Linear time shift), pre-norm Transformer blocks at each level (SnakeBeta
/// feed-forward ×4, dropout), a strided-convolution down path, mid blocks, and a transposed-convolution up path with
/// skip connections. Every convolution and linear layer is Kaiming-normal initialized with zero bias.
/// </summary>
/// <remarks>
/// <para>The reference passes the 0/1 frame mask to diffusers' <c>Attention</c>, which ADDS it to the attention
/// logits rather than masking with −∞; that behaviour is reproduced.</para>
/// <para>CosyVoice's <c>ConditionalDecoder</c> is the same network with more conditioning channels (see
/// <see cref="Estimate(Tensor{T}, double, Tensor{T})"/>), several Transformer blocks per level, a GELU feed-forward and
/// the up path cropped to each skip connection's length, so any frame count works.</para>
/// </remarks>
internal sealed class MatchaDecoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _inChannels;
    private readonly KaimingLinearLayer<T> _time1;
    private readonly KaimingLinearLayer<T> _time2;
    private readonly List<(ResnetBlock Resnet, List<TransformerBlock> Transformers, KaimingConv1DLayer<T> Down)> _downs = new();
    private readonly List<(ResnetBlock Resnet, List<TransformerBlock> Transformers)> _mids = new();
    private readonly List<(ResnetBlock Resnet, List<TransformerBlock> Transformers, KaimingConv1DLayer<T> Up)> _ups = new();
    private readonly Block _finalBlock;
    private readonly KaimingConv1DLayer<T> _finalProjection;

    public MatchaDecoder(IEngine engine, int inChannels, int outChannels, int[] channels, double dropout, int headDim,
        int heads, int midBlocks, int transformerBlocks = 1, bool geluFeedForward = false)
    {
        _engine = engine;
        _inChannels = inChannels;
        int timeDim = channels[0] * 4;
        _time1 = Add(new KaimingLinearLayer<T>(inChannels, timeDim, true));
        _time2 = Add(new KaimingLinearLayer<T>(timeDim, timeDim, true));
        List<TransformerBlock> Transformers(int dim) =>
            Enumerable.Range(0, transformerBlocks).Select(_ => new TransformerBlock(this, dim, heads, headDim, dropout, geluFeedForward)).ToList();
        int output = inChannels;
        for (int i = 0; i < channels.Length; i++)
        {
            int input = output;
            output = channels[i];
            bool last = i == channels.Length - 1;
            var resnet = new ResnetBlock(this, input, output, timeDim);
            var transformers = Transformers(output);
            _downs.Add((resnet, transformers, Add(new KaimingConv1DLayer<T>(output, output, 3, last ? 1 : 2, 1, false))));
        }
        for (int i = 0; i < midBlocks; i++)
        {
            var resnet = new ResnetBlock(this, channels[^1], output, timeDim);
            _mids.Add((resnet, Transformers(output)));
        }
        // Enumerable.Reverse explicitly: on net471 an array's .Reverse() binds to the in-place MemoryExtensions.Reverse.
        var up = Enumerable.Reverse(channels).Append(channels[0]).ToArray();
        for (int i = 0; i < up.Length - 1; i++)
        {
            bool last = i == up.Length - 2;
            var resnet = new ResnetBlock(this, 2 * up[i], up[i + 1], timeDim);
            var transformers = Transformers(up[i + 1]);
            _ups.Add((resnet, transformers,
                last
                    ? Add(new KaimingConv1DLayer<T>(up[i + 1], up[i + 1], 3, 1, 1, false))
                    : Add(new KaimingConv1DLayer<T>(up[i + 1], up[i + 1], 4, 2, 1, true))));
        }
        _finalBlock = new Block(this, up[^1], up[^1]);
        _finalProjection = Add(new KaimingConv1DLayer<T>(up[^1], outChannels, 1, 1, 0, false));
    }

    /// <summary>Every trainable layer, for registration with the owning model.</summary>
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>The vector field at <paramref name="x"/> <c>[frames, mel]</c> given the aligned encoder output
    /// <paramref name="mu"/> <c>[frames, mel]</c> and time <paramref name="t"/>; <paramref name="mask"/> <c>[frames]</c>
    /// is 1 on real frames. <c>frames</c> must be divisible by 4.</summary>
    public Tensor<T> Estimate(Tensor<T> x, Tensor<T> mu, double t, Tensor<T> mask)
    {
        int frames = x.Shape[0];
        var input = _engine.Reshape(_engine.TensorTranspose(_engine.TensorConcatenate(new[] { x, mu }, 1)), new[] { 1, _inChannels, frames });
        var output = Estimate(input, t, mask);                                                  // [1, mel, frames]
        return _engine.TensorTranspose(_engine.Reshape(output, new[] { output.Shape[1], frames }));
    }

    /// <summary>The vector field <c>[1, out, frames]</c> for a channels-first input <c>[1, in, frames]</c> that already
    /// stacks x and its conditions (CosyVoice: x, μ, the speaker embedding and the prompt mel).</summary>
    public Tensor<T> Estimate(Tensor<T> input, double t, Tensor<T> mask)
    {
        var h = input;
        var time = _time2.Forward(_engine.Swish(_time1.Forward(TimeEmbedding(t))));

        var masks = new List<Tensor<T>> { mask };
        var hiddens = new List<Tensor<T>>();
        foreach (var (resnet, transformers, down) in _downs)
        {
            var m = masks[^1];
            h = resnet.Forward(h, m, time);
            foreach (var transformer in transformers) h = transformer.Forward(h, m);
            hiddens.Add(h);
            h = down.Forward(Mask(h, m));
            masks.Add(Every2nd(m));
        }
        masks.RemoveAt(masks.Count - 1);
        var midMask = masks[^1];
        foreach (var (resnet, transformers) in _mids)
        {
            h = resnet.Forward(h, midMask, time);
            foreach (var transformer in transformers) h = transformer.Forward(h, midMask);
        }
        var upMask = mask;
        foreach (var (resnet, transformers, up) in _ups)
        {
            upMask = masks[^1];
            masks.RemoveAt(masks.Count - 1);
            var skip = hiddens[^1];
            hiddens.RemoveAt(hiddens.Count - 1);
            // The reference crops the up path to the skip connection's length (x[:, :, :skip.shape[-1]]).
            if (h.Shape[2] != skip.Shape[2]) h = _engine.TensorSlice(h, new[] { 0, 0, 0 }, new[] { h.Shape[0], h.Shape[1], skip.Shape[2] });
            h = resnet.Forward(_engine.TensorConcatenate(new[] { h, skip }, 1), upMask, time);
            foreach (var transformer in transformers) h = transformer.Forward(h, upMask);
            h = up.Forward(Mask(h, upMask));
        }
        h = _finalBlock.Forward(h, upMask);
        return Mask(_finalProjection.Forward(Mask(h, upMask)), mask);
    }

    // SinusoidalPosEmb(in_channels) of 1000·t: [sin, cos] over in_channels / 2 log-spaced frequencies.
    private Tensor<T> TimeEmbedding(double t)
    {
        int half = _inChannels / 2;
        double step = Math.Log(10000) / (half - 1);
        var embedding = new Tensor<T>(new[] { 1, _inChannels });
        for (int i = 0; i < half; i++)
        {
            double angle = 1000.0 * t * Math.Exp(-step * i);
            embedding[0, i] = NumOps.FromDouble(Math.Sin(angle));
            embedding[0, half + i] = NumOps.FromDouble(Math.Cos(angle));
        }
        return embedding;
    }

    // mask [frames] broadcast over [1, channels, frames].
    private Tensor<T> Mask(Tensor<T> x, Tensor<T> mask)
        => _engine.TensorMultiply(x, _engine.TensorTile(_engine.Reshape(mask, new[] { 1, 1, mask.Length }), new[] { x.Shape[0], x.Shape[1], 1 }));

    private static Tensor<T> Every2nd(Tensor<T> mask)
    {
        int half = (mask.Length + 1) / 2;
        var result = new Tensor<T>(new[] { half });
        for (int i = 0; i < half; i++) result[i] = mask[2 * i];
        return result;
    }

    /// <summary>
    /// The reference's initialization (<c>ConditionalDecoder.initialize_weights</c>): every <c>nn.Conv1d</c> and
    /// <c>nn.Linear</c> Kaiming-normal (ReLU) with zero bias, GroupNorm and LayerNorm ones and zeros; the transposed
    /// convolutions are not <c>nn.Conv1d</c>, so they keep PyTorch's default (uniform on ±1/√fan_in, fan_in = out·k).
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal()
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        foreach (var layer in _layers)
        {
            switch (layer)
            {
                case KaimingLinearLayer<T> linear:
                {
                    int inputs = linear.InFeatures, outputs = linear.OutFeatures;
                    double std = Math.Sqrt(2.0 / inputs);
                    linear.Load(Enumerable.Range(0, inputs * outputs).Select(_ => std * Normal()).ToArray(), new double[outputs]);
                    break;
                }
                case KaimingConv1DLayer<T> conv when conv.Transposed:
                {
                    double bound = 1.0 / Math.Sqrt(conv.OutChannels * conv.KernelSize);
                    int count = conv.InChannels * conv.OutChannels * conv.KernelSize;
                    conv.Load(Enumerable.Range(0, count).Select(_ => (random.NextDouble() * 2 - 1) * bound).ToArray(),
                        Enumerable.Range(0, conv.OutChannels).Select(_ => (random.NextDouble() * 2 - 1) * bound).ToArray());
                    break;
                }
                case KaimingConv1DLayer<T> conv:
                {
                    double std = Math.Sqrt(2.0 / (conv.InChannels * conv.KernelSize));
                    int count = conv.InChannels * conv.OutChannels * conv.KernelSize;
                    conv.Load(Enumerable.Range(0, count).Select(_ => std * Normal()).ToArray(), new double[conv.OutChannels]);
                    break;
                }
            }
        }
    }

    /// <summary>Loads a CosyVoice <c>ConditionalDecoder</c> (or Matcha <c>Decoder</c>) state dict under <paramref name="prefix"/>.</summary>
    public void LoadTorch(IReadOnlyDictionary<string, double[]> state, string prefix)
    {
        _time1.Load(state[prefix + "time_mlp.linear_1.weight"], state[prefix + "time_mlp.linear_1.bias"]);
        _time2.Load(state[prefix + "time_mlp.linear_2.weight"], state[prefix + "time_mlp.linear_2.bias"]);
        void Level(string p, ResnetBlock resnet, List<TransformerBlock> transformers)
        {
            resnet.Load(state, p + "0.");
            for (int j = 0; j < transformers.Count; j++) transformers[j].Load(state, $"{p}1.{j}.");
        }
        // Downsample1D and Upsample1D wrap their convolution as ".conv"; the last level's plain Conv1d has none.
        void Conv(KaimingConv1DLayer<T> conv, string name)
        {
            string key = state.ContainsKey(name + ".weight") ? name : name + ".conv";
            conv.Load(state[key + ".weight"], state[key + ".bias"]);
        }
        for (int i = 0; i < _downs.Count; i++)
        {
            Level($"{prefix}down_blocks.{i}.", _downs[i].Resnet, _downs[i].Transformers);
            Conv(_downs[i].Down, $"{prefix}down_blocks.{i}.2");
        }
        for (int i = 0; i < _mids.Count; i++) Level($"{prefix}mid_blocks.{i}.", _mids[i].Resnet, _mids[i].Transformers);
        for (int i = 0; i < _ups.Count; i++)
        {
            Level($"{prefix}up_blocks.{i}.", _ups[i].Resnet, _ups[i].Transformers);
            Conv(_ups[i].Up, $"{prefix}up_blocks.{i}.2");
        }
        _finalBlock.Load(state, prefix + "final_block.");
        Conv(_finalProjection, prefix + "final_proj");
    }

    /// <summary>Conv(3) → GroupNorm(8) → Mish, with the input and output masked.</summary>
    private sealed class Block
    {
        private readonly MatchaDecoder<T> _owner;
        private readonly KaimingConv1DLayer<T> _conv;
        private readonly GroupNormalizationLayer<T> _norm;

        public Block(MatchaDecoder<T> owner, int dim, int dimOut)
        {
            _owner = owner;
            _conv = owner.Add(new KaimingConv1DLayer<T>(dim, dimOut, 3, 1, 1, false));
            _norm = owner.Add(new GroupNormalizationLayer<T>(8, dimOut, 1e-5));
        }

        public Tensor<T> Forward(Tensor<T> x, Tensor<T> mask)
        {
            var engine = _owner._engine;
            var h = _conv.Forward(_owner.Mask(x, mask));
            int channels = h.Shape[1], frames = h.Shape[2];
            h = engine.Reshape(_norm.Forward(engine.Reshape(h, new[] { 1, channels, 1, frames })), new[] { 1, channels, frames });
            return _owner.Mask(engine.Mish(h), mask);
        }

        public void Load(IReadOnlyDictionary<string, double[]> state, string p)
        {
            _conv.Load(state[p + "block.0.weight"], state[p + "block.0.bias"]);
            var affine = state[p + "block.1.weight"].Concat(state[p + "block.1.bias"]).Select(v => NumOps.FromDouble(v)).ToArray();
            _norm.SetParameters(new Vector<T>(affine));
        }
    }

    /// <summary>Block → + Linear(Mish(time)) → Block, plus a 1×1 convolution of the masked input.</summary>
    private sealed class ResnetBlock
    {
        private readonly MatchaDecoder<T> _owner;
        private readonly KaimingLinearLayer<T> _timeProjection;
        private readonly Block _block1;
        private readonly Block _block2;
        private readonly KaimingConv1DLayer<T> _residual;
        private readonly int _dimOut;

        public ResnetBlock(MatchaDecoder<T> owner, int dim, int dimOut, int timeDim)
        {
            _owner = owner;
            _dimOut = dimOut;
            _timeProjection = owner.Add(new KaimingLinearLayer<T>(timeDim, dimOut, true));
            _block1 = new Block(owner, dim, dimOut);
            _block2 = new Block(owner, dimOut, dimOut);
            _residual = owner.Add(new KaimingConv1DLayer<T>(dim, dimOut, 1, 1, 0, false));
        }

        public Tensor<T> Forward(Tensor<T> x, Tensor<T> mask, Tensor<T> time)
        {
            var engine = _owner._engine;
            var h = _block1.Forward(x, mask);
            var shift = engine.Reshape(_timeProjection.Forward(engine.Mish(time)), new[] { 1, _dimOut, 1 });
            h = engine.TensorAdd(h, engine.TensorTile(shift, new[] { 1, 1, h.Shape[2] }));
            h = _block2.Forward(h, mask);
            return engine.TensorAdd(h, _residual.Forward(_owner.Mask(x, mask)));
        }

        public void Load(IReadOnlyDictionary<string, double[]> state, string p)
        {
            _timeProjection.Load(state[p + "mlp.1.weight"], state[p + "mlp.1.bias"]);
            _block1.Load(state, p + "block1.");
            _block2.Load(state, p + "block2.");
            _residual.Load(state[p + "res_conv.weight"], state[p + "res_conv.bias"]);
        }
    }

    /// <summary>diffusers' BasicTransformerBlock as Matcha configures it: pre-norm self-attention (no QKV bias, an
    /// output projection then dropout) and a pre-norm feed-forward (SnakeBeta, or a GELU projection, to 4·dim, dropout,
    /// linear), each residual.</summary>
    private sealed class TransformerBlock
    {
        private readonly MatchaDecoder<T> _owner;
        private readonly int _dim;
        private readonly int _heads;
        private readonly int _headDim;
        private readonly LayerNormalizationLayer<T> _norm1;
        private readonly KaimingLinearLayer<T> _query;
        private readonly KaimingLinearLayer<T> _key;
        private readonly KaimingLinearLayer<T> _value;
        private readonly KaimingLinearLayer<T> _out;
        private readonly LayerNormalizationLayer<T> _norm3;
        private readonly SnakeBetaLayer<T>? _ffSnake;
        private readonly KaimingLinearLayer<T>? _ffGelu;
        private readonly KaimingLinearLayer<T> _ffOut;
        private readonly DropoutLayer<T>? _dropout;

        public TransformerBlock(MatchaDecoder<T> owner, int dim, int heads, int headDim, double dropout, bool gelu)
        {
            _owner = owner;
            _dim = dim;
            _heads = heads;
            _headDim = headDim;
            int inner = heads * headDim;
            _norm1 = owner.Add(new LayerNormalizationLayer<T>(dim, 1e-5));
            _query = owner.Add(new KaimingLinearLayer<T>(dim, inner, false));
            _key = owner.Add(new KaimingLinearLayer<T>(dim, inner, false));
            _value = owner.Add(new KaimingLinearLayer<T>(dim, inner, false));
            _out = owner.Add(new KaimingLinearLayer<T>(inner, dim, true));
            _norm3 = owner.Add(new LayerNormalizationLayer<T>(dim, 1e-5));
            if (gelu) _ffGelu = owner.Add(new KaimingLinearLayer<T>(dim, 4 * dim, true));
            else _ffSnake = owner.Add(new SnakeBetaLayer<T>(dim, 4 * dim));
            _ffOut = owner.Add(new KaimingLinearLayer<T>(4 * dim, dim, true));
            _dropout = dropout > 0 ? owner.Add(new DropoutLayer<T>(dropout)) : null;
        }

        // x [1, dim, frames] channels-first; returns the same layout.
        public Tensor<T> Forward(Tensor<T> x, Tensor<T> mask)
        {
            var engine = _owner._engine;
            int frames = x.Shape[2];
            var h = engine.TensorTranspose(engine.Reshape(x, new[] { _dim, frames }));           // [frames, dim]

            var normed = _norm1.Forward(h);
            var q = _query.Forward(normed);
            var k = _key.Forward(normed);
            var v = _value.Forward(normed);
            // diffusers adds the float 0/1 key mask to the logits.
            var additiveMask = engine.TensorTile(engine.Reshape(mask, new[] { 1, frames }), new[] { frames, 1 });
            var heads = new Tensor<T>[_heads];
            for (int i = 0; i < _heads; i++)
            {
                var qh = engine.TensorSlice(q, new[] { 0, i * _headDim }, new[] { frames, _headDim });
                var kh = engine.TensorSlice(k, new[] { 0, i * _headDim }, new[] { frames, _headDim });
                var vh = engine.TensorSlice(v, new[] { 0, i * _headDim }, new[] { frames, _headDim });
                var logits = engine.TensorAdd(engine.TensorMultiplyScalar(engine.TensorMatMul(qh, engine.TensorTranspose(kh)),
                    NumOps.FromDouble(1.0 / Math.Sqrt(_headDim))), additiveMask);
                heads[i] = engine.TensorMatMul(engine.TensorSoftmax(logits, axis: 1), vh);
            }
            var attended = _out.Forward(_heads == 1 ? heads[0] : engine.TensorConcatenate(heads, 1));
            if (_dropout is not null) attended = _dropout.Forward(attended);
            h = engine.TensorAdd(h, attended);

            var normed3 = _norm3.Forward(h);
            var ff = _ffGelu is not null
                ? AiDotNet.Helpers.ActivationHelper.ExactGelu(_ffGelu.Forward(normed3), engine)
                : _ffSnake!.Forward(normed3);
            if (_dropout is not null) ff = _dropout.Forward(ff);
            h = engine.TensorAdd(h, _ffOut.Forward(ff));
            return engine.Reshape(engine.TensorTranspose(h), new[] { 1, _dim, frames });
        }

        public void Load(IReadOnlyDictionary<string, double[]> state, string p)
        {
            var engine = _owner._engine;
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.LayerNorm(engine, _norm1, state[p + "norm1.weight"], state[p + "norm1.bias"]);
            _query.Load(state[p + "attn1.to_q.weight"], null);
            _key.Load(state[p + "attn1.to_k.weight"], null);
            _value.Load(state[p + "attn1.to_v.weight"], null);
            _out.Load(state[p + "attn1.to_out.0.weight"], state[p + "attn1.to_out.0.bias"]);
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.LayerNorm(engine, _norm3, state[p + "norm3.weight"], state[p + "norm3.bias"]);
            if (_ffGelu is not null)
                _ffGelu.Load(state[p + "ff.net.0.proj.weight"], state[p + "ff.net.0.proj.bias"]);
            else
                _ffSnake!.Load(state[p + "ff.net.0.proj.weight"], state[p + "ff.net.0.proj.bias"], state[p + "ff.net.0.alpha"], state[p + "ff.net.0.beta"]);
            _ffOut.Load(state[p + "ff.net.2.weight"], state[p + "ff.net.2.bias"]);
        }
    }
}
