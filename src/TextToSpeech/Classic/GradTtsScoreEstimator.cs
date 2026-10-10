using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>A 2-D convolution or transposed convolution with PyTorch's <c>Conv2d</c>/<c>ConvTranspose2d</c> semantics
/// and default initialization, on <c>[batch, channels, height, width]</c>.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 2, 4, 4", TestConstructorArgs = "2, 3, 3, 1, 1, false, true")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class TorchConv2DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly bool _transposed;
    private readonly bool _useBias;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weight;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    public override bool SupportsTraining => true;

    public TorchConv2DLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int kernelSize,
        [LayerState] int stride, [LayerState] int padding, [LayerState] bool transposed, [LayerState] bool useBias)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _transposed = transposed;
        _useBias = useBias;
        // PyTorch default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)); a transposed convolution's fan_in is out_channels * k * k.
        int fanIn = (transposed ? outChannels : inChannels) * kernelSize * kernelSize;
        double bound = 1.0 / Math.Sqrt(fanIn);
        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _weight = transposed
            ? new Tensor<T>(new[] { inChannels, outChannels, kernelSize, kernelSize })
            : new Tensor<T>(new[] { outChannels, inChannels, kernelSize, kernelSize });
        for (int i = 0; i < _weight.Length; i++) _weight[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        _bias = new Tensor<T>(new[] { outChannels });
        if (useBias)
            for (int i = 0; i < outChannels; i++) _bias[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        var y = _transposed
            ? Engine.ConvTranspose2D(input, _weight, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 0, 0 })
            : Engine.Conv2D(input, _weight, new[] { _stride, _stride }, new[] { _padding, _padding }, new[] { 1, 1 });
        if (!_useBias) return y;
        var bias = Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _outChannels, 1, 1 }),
            new[] { y.Shape[0], 1, y.Shape[2], y.Shape[3] });
        return Engine.TensorAdd(y, bias);
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _weight.Length + _bias.Length) return;
        int k = 0;
        foreach (var tensor in new[] { _weight, _bias })
        {
            for (int i = 0; i < tensor.Length; i++, k++)
                tensor[i] = NumOps.Subtract(tensor[i], NumOps.Multiply(learningRate, gradients[k]));
            Engine.InvalidatePersistentTensor(tensor);
        }
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _inChannels.ToString(inv);
        metadata["OutChannels"] = _outChannels.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["Stride"] = _stride.ToString(inv);
        metadata["Padding"] = _padding.ToString(inv);
        metadata["Transposed"] = _transposed.ToString(inv);
        metadata["UseBias"] = _useBias.ToString(inv);
        return metadata;
    }
}

/// <summary>A learned scalar gate starting at zero (ReZero, Bachlechner et al. 2020): <c>y = g · x</c>.</summary>
[LayerCategory(LayerCategory.Residual)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4", TestConstructorArgs = "")]
[ElementWiseShape(Note = "Scales its input; the shape is carried through.")]
[AutoParameters]
internal sealed partial class RezeroGateLayer<T> : LayerBase<T>
{
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _gate;

    public override bool SupportsTraining => true;

    public RezeroGateLayer() : base(new[] { 1 }, new[] { 1 })
    {
        _gate = new Tensor<T>(new[] { 1 });
        RegisterTrainableParameter(_gate, PersistentTensorRole.Weights);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => Engine.TensorMultiply(input, Engine.Reshape(Engine.TensorTile(_gate, new[] { input.Length }), input._shape));

    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != 1) return;
        _gate[0] = NumOps.Subtract(_gate[0], NumOps.Multiply(learningRate, gradients[0]));
        Engine.InvalidatePersistentTensor(_gate);
    }

    public override void ResetState()
    {
    }
}

/// <summary>
/// Grad-TTS's score network <c>s_θ(X_t, μ, t)</c> (Popov et al. 2021, §3; reference implementation
/// <c>model/diffusion.py</c>, <c>GradLogPEstimator2d</c>): a U-Net over the mel spectrogram as a 2-channel image
/// (<c>[μ, X_t]</c>) with three resolutions (64, 128, 256 channels), each of two ResNet blocks conditioned on the
/// diffusion time (sinusoidal embedding scaled by 1000 and a two-layer MLP with Mish), a ReZero linear-attention
/// residual, and a strided-convolution downsampling; a middle of ResNet, attention, ResNet; two upsampling levels with
/// skip connections and transposed convolutions; a final convolution block and a 1×1 convolution to one channel.
/// GroupNorm uses 8 groups; activations are Mish. A time mask zeroes padded frames.
/// </summary>
internal sealed class GradTtsScoreEstimator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly double _positionScale;
    private readonly List<LayerBase<T>> _layers = new();

    private readonly DenseLayer<T> _timeMlp1;
    private readonly DenseLayer<T> _timeMlp2;
    private readonly List<(ResnetBlock R1, ResnetBlock R2, LinearAttention A, TorchConv2DLayer<T>? Down)> _downs = new();
    private readonly ResnetBlock _mid1;
    private readonly LinearAttention _midAttention;
    private readonly ResnetBlock _mid2;
    private readonly List<(ResnetBlock R1, ResnetBlock R2, LinearAttention A, TorchConv2DLayer<T> Up)> _ups = new();
    private readonly Block _finalBlock;
    private readonly TorchConv2DLayer<T> _finalConv;

    public GradTtsScoreEstimator(IEngine engine, int dim, int[] dimMults, double positionScale)
    {
        _engine = engine;
        _dim = dim;
        _positionScale = positionScale;
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _timeMlp1 = Add(new DenseLayer<T>(dim * 4, identity));
        _timeMlp2 = Add(new DenseLayer<T>(dim, identity));

        var dims = new List<int> { 2 };
        dims.AddRange(dimMults.Select(m => dim * m));
        var inOut = dims.Zip(dims.Skip(1), (a, b) => (In: a, Out: b)).ToList();
        for (int i = 0; i < inOut.Count; i++)
        {
            bool last = i >= inOut.Count - 1;
            _downs.Add((new ResnetBlock(this, inOut[i].In, inOut[i].Out, dim), new ResnetBlock(this, inOut[i].Out, inOut[i].Out, dim),
                new LinearAttention(this, inOut[i].Out), last ? null : Add(new TorchConv2DLayer<T>(inOut[i].Out, inOut[i].Out, 3, 2, 1, false, true))));
        }
        int mid = dims[^1];
        _mid1 = new ResnetBlock(this, mid, mid, dim);
        _midAttention = new LinearAttention(this, mid);
        _mid2 = new ResnetBlock(this, mid, mid, dim);
        foreach (var (dimIn, dimOut) in Enumerable.Reverse(inOut.Skip(1)))
            _ups.Add((new ResnetBlock(this, dimOut * 2, dimIn, dim), new ResnetBlock(this, dimIn, dimIn, dim),
                new LinearAttention(this, dimIn), Add(new TorchConv2DLayer<T>(dimIn, dimIn, 4, 2, 1, true, true))));
        _finalBlock = new Block(this, dim, dim);
        _finalConv = Add(new TorchConv2DLayer<T>(dim, 1, 1, 1, 0, false, true));
    }

    /// <summary>Every trainable layer, for registration with the owning model.</summary>
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>Estimates the score for <paramref name="xt"/> <c>[frames, mel]</c> given the aligned prior mean
    /// <paramref name="mu"/> and diffusion time <paramref name="t"/>; <paramref name="mask"/> is 1 on real frames.
    /// <c>frames</c> must be divisible by 4.</summary>
    public Tensor<T> Estimate(Tensor<T> xt, Tensor<T> mu, double t, Tensor<T> mask)
    {
        int frames = xt.Shape[0], bins = xt.Shape[1];
        // [frames, mel] -> [1, 2, mel, frames] image with channels (mu, x).
        var image = _engine.TensorConcatenate(new[]
        {
            _engine.Reshape(_engine.TensorTranspose(mu), new[] { 1, 1, bins, frames }),
            _engine.Reshape(_engine.TensorTranspose(xt), new[] { 1, 1, bins, frames }),
        }, 1);
        var time = TimeEmbedding(t);

        var masks = new List<Tensor<T>> { mask };
        var hiddens = new List<Tensor<T>>();
        var x = image;
        foreach (var (r1, r2, attention, down) in _downs)
        {
            var m = masks[^1];
            x = r1.Forward(x, m, time);
            x = r2.Forward(x, m, time);
            x = attention.Forward(x);
            hiddens.Add(x);
            if (down is not null) x = down.Forward(Mask(x, m));
            masks.Add(Every2nd(m));
        }
        masks.RemoveAt(masks.Count - 1);
        var midMask = masks[^1];
        x = _mid1.Forward(x, midMask, time);
        x = _midAttention.Forward(x);
        x = _mid2.Forward(x, midMask, time);
        foreach (var (r1, r2, attention, up) in _ups)
        {
            var m = masks[^1];
            masks.RemoveAt(masks.Count - 1);
            x = _engine.TensorConcatenate(new[] { x, hiddens[^1] }, 1);
            hiddens.RemoveAt(hiddens.Count - 1);
            x = r1.Forward(x, m, time);
            x = r2.Forward(x, m, time);
            x = attention.Forward(x);
            x = up.Forward(Mask(x, m));
        }
        x = _finalBlock.Forward(x, mask);
        var output = Mask(_finalConv.Forward(Mask(x, mask)), mask);              // [1, 1, mel, frames]
        return _engine.TensorTranspose(_engine.Reshape(output, new[] { bins, frames }));
    }

    // Sinusoidal embedding of t (scaled by the position scale), then Linear -> Mish -> Linear.
    private Tensor<T> TimeEmbedding(double t)
    {
        int half = _dim / 2;
        double step = Math.Log(10000) / (half - 1);
        var embedding = new Tensor<T>(new[] { 1, _dim });
        for (int i = 0; i < half; i++)
        {
            double angle = _positionScale * t * Math.Exp(-step * i);
            embedding[0, i] = NumOps.FromDouble(Math.Sin(angle));
            embedding[0, half + i] = NumOps.FromDouble(Math.Cos(angle));
        }
        return _timeMlp2.Forward(_engine.Mish(_timeMlp1.Forward(embedding)));
    }

    // mask is [1, 1, 1, frames]; broadcast over channels and mel bins.
    private Tensor<T> Mask(Tensor<T> x, Tensor<T> mask)
        => _engine.TensorMultiply(x, _engine.TensorTile(mask, new[] { x.Shape[0], x.Shape[1], x.Shape[2], 1 }));

    private Tensor<T> Every2nd(Tensor<T> mask)
    {
        int frames = mask.Shape[3], half = (frames + 1) / 2;
        var result = new Tensor<T>(new[] { 1, 1, 1, half });
        for (int i = 0; i < half; i++) result[0, 0, 0, i] = mask[0, 0, 0, 2 * i];
        return result;
    }

    /// <summary>Conv 3x3 -> GroupNorm(8) -> Mish, with the input and output masked.</summary>
    private sealed class Block
    {
        private readonly GradTtsScoreEstimator<T> _owner;
        private readonly TorchConv2DLayer<T> _conv;
        private readonly GroupNormalizationLayer<T> _norm;

        public Block(GradTtsScoreEstimator<T> owner, int dim, int dimOut)
        {
            _owner = owner;
            _conv = owner.Add(new TorchConv2DLayer<T>(dim, dimOut, 3, 1, 1, false, true));
            _norm = owner.Add(new GroupNormalizationLayer<T>(8, dimOut, 1e-5));
        }

        public Tensor<T> Forward(Tensor<T> x, Tensor<T> mask)
            => _owner.Mask(_owner._engine.Mish(_norm.Forward(_conv.Forward(_owner.Mask(x, mask)))), mask);
    }

    /// <summary>Block -> + MLP(Mish(time)) -> Block, plus a (1x1 if needed) residual of the masked input.</summary>
    private sealed class ResnetBlock
    {
        private readonly GradTtsScoreEstimator<T> _owner;
        private readonly DenseLayer<T> _timeProjection;
        private readonly Block _block1;
        private readonly Block _block2;
        private readonly TorchConv2DLayer<T>? _residual;
        private readonly int _dimOut;

        public ResnetBlock(GradTtsScoreEstimator<T> owner, int dim, int dimOut, int timeDim)
        {
            _owner = owner;
            _dimOut = dimOut;
            _timeProjection = owner.Add(new DenseLayer<T>(dimOut, new IdentityActivation<T>() as IActivationFunction<T>));
            _block1 = new Block(owner, dim, dimOut);
            _block2 = new Block(owner, dimOut, dimOut);
            if (dim != dimOut) _residual = owner.Add(new TorchConv2DLayer<T>(dim, dimOut, 1, 1, 0, false, true));
        }

        public Tensor<T> Forward(Tensor<T> x, Tensor<T> mask, Tensor<T> time)
        {
            var engine = _owner._engine;
            var h = _block1.Forward(x, mask);
            var shift = engine.Reshape(_timeProjection.Forward(engine.Mish(time)), new[] { 1, _dimOut, 1, 1 });
            h = engine.TensorAdd(h, engine.TensorTile(shift, new[] { h.Shape[0], 1, h.Shape[2], h.Shape[3] }));
            h = _block2.Forward(h, mask);
            var masked = _owner.Mask(x, mask);
            return engine.TensorAdd(h, _residual is null ? masked : _residual.Forward(masked));
        }
    }

    /// <summary>Residual(Rezero(linear attention)): 4 heads of 32, keys soft-maxed over positions.</summary>
    private sealed class LinearAttention
    {
        private const int Heads = 4, HeadDim = 32;
        private readonly GradTtsScoreEstimator<T> _owner;
        private readonly TorchConv2DLayer<T> _toQkv;
        private readonly TorchConv2DLayer<T> _toOut;
        private readonly RezeroGateLayer<T> _gate;

        public LinearAttention(GradTtsScoreEstimator<T> owner, int dim)
        {
            _owner = owner;
            _toQkv = owner.Add(new TorchConv2DLayer<T>(dim, Heads * HeadDim * 3, 1, 1, 0, false, false));
            _toOut = owner.Add(new TorchConv2DLayer<T>(Heads * HeadDim, dim, 1, 1, 0, false, true));
            _gate = owner.Add(new RezeroGateLayer<T>());
        }

        public Tensor<T> Forward(Tensor<T> x)
        {
            var engine = _owner._engine;
            int batch = x.Shape[0], height = x.Shape[2], width = x.Shape[3], n = height * width;
            if (batch != 1) throw new ArgumentException("The score network runs one utterance at a time.", nameof(x));
            var qkv = engine.Reshape(_toQkv.Forward(x), new[] { batch, 3, Heads, HeadDim, n });
            var heads = new Tensor<T>[Heads];
            for (int h = 0; h < Heads; h++)
            {
                Tensor<T> Part(int which) => engine.Reshape(
                    engine.TensorSlice(qkv, new[] { 0, which, h, 0, 0 }, new[] { 1, 1, 1, HeadDim, n }), new[] { HeadDim, n });
                var q = Part(0);
                var k = engine.TensorSoftmax(Part(1), axis: 1);
                var v = Part(2);
                var context = engine.TensorMatMul(k, engine.TensorTranspose(v));         // [d, e] = sum_n k_dn v_en
                heads[h] = engine.TensorMatMul(engine.TensorTranspose(context), q);      // [e, n] = sum_d context_de q_dn
            }
            var merged = engine.Reshape(engine.TensorConcatenate(heads, 0), new[] { 1, Heads * HeadDim, height, width });
            return engine.TensorAdd(x, _gate.Forward(_toOut.Forward(merged)));
        }
    }
}
