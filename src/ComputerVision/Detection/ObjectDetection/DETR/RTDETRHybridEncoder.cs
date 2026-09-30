using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// The reference <c>ConvNormLayer</c>: a convolution with padding <c>(k - 1) / 2</c>, then BatchNorm
/// (PyTorch defaults: eps 1e-5, momentum 0.1), then an optional SiLU.
/// </summary>
internal sealed class RtdetrConvNorm<T> : CvParameterModule<T>
{
    private readonly Conv2D<T> _conv;
    private readonly BatchNorm2D<T> _norm;
    private readonly SiLUActivation<T>? _activation;

    public RtdetrConvNorm(int inChannels, int outChannels, int kernelSize, int stride = 1, bool silu = false)
    {
        _conv = new Conv2D<T>(inChannels, outChannels, kernelSize, stride, (kernelSize - 1) / 2);
        _norm = new BatchNorm2D<T>(outChannels);
        _activation = silu ? new SiLUActivation<T>() : null;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var y = _norm.Forward(_conv.Forward(input));
        return _activation is null ? y : _activation.Activate(y);
    }

    public void SetTrainingMode(bool training) => _norm.SetTrainingMode(training);

    /// <summary>The live BatchNorm shift (test access for controlled-weight fixtures).</summary>
    internal Tensor<T> NormBeta => _norm.Beta;

    public void Write(BinaryWriter writer) { _conv.WriteParameters(writer); _norm.WriteParameters(writer); }

    public void Read(BinaryReader reader) { _conv.ReadParameters(reader); _norm.ReadParameters(reader); }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() { yield return _conv; yield return _norm; }
}

/// <summary>The reference <c>RepVggBlock</c> in training form: <c>SiLU(ConvNorm3x3(x) + ConvNorm1x1(x))</c>.</summary>
internal sealed class RtdetrRepVggBlock<T> : CvParameterModule<T>
{
    private readonly RtdetrConvNorm<T> _conv3;
    private readonly RtdetrConvNorm<T> _conv1;
    private readonly SiLUActivation<T> _activation = new();

    public RtdetrRepVggBlock(int channels)
    {
        _conv3 = new RtdetrConvNorm<T>(channels, channels, 3);
        _conv1 = new RtdetrConvNorm<T>(channels, channels, 1);
    }

    public Tensor<T> Forward(Tensor<T> input)
        => _activation.Activate(AiDotNetEngine.Current.TensorAdd(_conv3.Forward(input), _conv1.Forward(input)));

    public void SetTrainingMode(bool training) { _conv3.SetTrainingMode(training); _conv1.SetTrainingMode(training); }

    public void Write(BinaryWriter writer) { _conv3.Write(writer); _conv1.Write(writer); }

    public void Read(BinaryReader reader) { _conv3.Read(reader); _conv1.Read(reader); }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() { yield return _conv3; yield return _conv1; }
}

/// <summary>
/// The reference <c>CSPRepLayer</c>: two 1x1 ConvNorm+SiLU branches of width <c>out * expansion</c>, one
/// through <c>numBlocks</c> RepVGG blocks. Their SUM goes through a final 1x1 ConvNorm+SiLU, which is the
/// identity when the hidden width equals the output width.
/// </summary>
internal sealed class RtdetrCspRepLayer<T> : CvParameterModule<T>
{
    private readonly RtdetrConvNorm<T> _conv1;
    private readonly RtdetrConvNorm<T> _conv2;
    private readonly List<RtdetrRepVggBlock<T>> _bottlenecks = new();
    private readonly RtdetrConvNorm<T>? _conv3;

    public RtdetrCspRepLayer(int inChannels, int outChannels, int numBlocks, double expansion)
    {
        int hidden = (int)(outChannels * expansion);
        _conv1 = new RtdetrConvNorm<T>(inChannels, hidden, 1, silu: true);
        _conv2 = new RtdetrConvNorm<T>(inChannels, hidden, 1, silu: true);
        for (int i = 0; i < numBlocks; i++) _bottlenecks.Add(new RtdetrRepVggBlock<T>(hidden));
        _conv3 = hidden != outChannels ? new RtdetrConvNorm<T>(hidden, outChannels, 1, silu: true) : null;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x1 = _conv1.Forward(input);
        foreach (var block in _bottlenecks) x1 = block.Forward(x1);
        var sum = AiDotNetEngine.Current.TensorAdd(x1, _conv2.Forward(input));
        return _conv3 is null ? sum : _conv3.Forward(sum);
    }

    public void SetTrainingMode(bool training)
    {
        _conv1.SetTrainingMode(training);
        _conv2.SetTrainingMode(training);
        foreach (var block in _bottlenecks) block.SetTrainingMode(training);
        _conv3?.SetTrainingMode(training);
    }

    public void Write(BinaryWriter writer)
    {
        _conv1.Write(writer); _conv2.Write(writer);
        foreach (var block in _bottlenecks) block.Write(writer);
        _conv3?.Write(writer);
    }

    public void Read(BinaryReader reader)
    {
        _conv1.Read(reader); _conv2.Read(reader);
        foreach (var block in _bottlenecks) block.Read(reader);
        _conv3?.Read(reader);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _conv1;
        yield return _conv2;
        foreach (var block in _bottlenecks) yield return block;
        yield return _conv3;
    }
}

/// <summary>
/// AIFI, the reference <c>TransformerEncoderLayer</c>: post-norm, GELU, no dropout. Self-attention uses
/// <c>q = k = src + pos</c> and <c>v = src</c>. Initialization is <c>nn.MultiheadAttention</c>'s: Xavier
/// in-projections with zero bias, and an out-projection with zero bias.
/// </summary>
internal sealed class RtdetrAifiLayer<T> : CvParameterModule<T>
{
    private readonly int _dModel;
    private readonly int _numHeads;
    private readonly DetrLinear<T> _query;
    private readonly DetrLinear<T> _key;
    private readonly DetrLinear<T> _value;
    private readonly DetrLinear<T> _output;
    private readonly LayerNorm<T> _norm1;
    private readonly DetrLinear<T> _linear1;
    private readonly DetrLinear<T> _linear2;
    private readonly LayerNorm<T> _norm2;

    public RtdetrAifiLayer(int dModel, int numHeads, int feedForward)
    {
        if (dModel % numHeads != 0) throw new ArgumentException($"{dModel} is not divisible by {numHeads} heads.", nameof(numHeads));
        _dModel = dModel;
        _numHeads = numHeads;
        _query = new DetrLinear<T>(dModel, dModel);
        _key = new DetrLinear<T>(dModel, dModel);
        _value = new DetrLinear<T>(dModel, dModel);
        _query.XavierUniformZeroBias();
        _key.XavierUniformZeroBias();
        _value.XavierUniformZeroBias();
        _output = new DetrLinear<T>(dModel, dModel);
        _output.Bias.Fill(MathHelper.GetNumericOperations<T>().Zero);
        _norm1 = new LayerNorm<T>(dModel, 1e-5);
        _linear1 = new DetrLinear<T>(dModel, feedForward);
        _linear2 = new DetrLinear<T>(feedForward, dModel);
        _norm2 = new LayerNorm<T>(dModel, 1e-5);
    }

    public Tensor<T> Forward(Tensor<T> source, Tensor<T> position)
    {
        var engine = AiDotNetEngine.Current;
        int batch = source.Shape[0], tokens = source.Shape[1], headDim = _dModel / _numHeads;
        var withPosition = engine.TensorAdd(source, position);
        Tensor<T> Heads(Tensor<T> x) => engine.TensorPermute(engine.Reshape(x, new[] { batch, tokens, _numHeads, headDim }), new[] { 0, 2, 1, 3 });
        var context = engine.ScaledDotProductAttention(
            Heads(_query.Forward(withPosition)), Heads(_key.Forward(withPosition)), Heads(_value.Forward(source)),
            null, 1.0 / Math.Sqrt(headDim), out _);
        var merged = engine.Reshape(engine.TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { batch, tokens, _dModel });
        var x = _norm1.Forward(engine.TensorAdd(source, _output.Forward(merged)));
        return _norm2.Forward(engine.TensorAdd(x, _linear2.Forward(engine.GELU(_linear1.Forward(x)))));
    }

    public void Write(BinaryWriter writer)
    {
        _query.Write(writer); _key.Write(writer); _value.Write(writer); _output.Write(writer);
        _norm1.WriteParameters(writer); _linear1.Write(writer); _linear2.Write(writer); _norm2.WriteParameters(writer);
    }

    public void Read(BinaryReader reader)
    {
        _query.Read(reader); _key.Read(reader); _value.Read(reader); _output.Read(reader);
        _norm1.ReadParameters(reader); _linear1.Read(reader); _linear2.Read(reader); _norm2.ReadParameters(reader);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _query; yield return _key; yield return _value; yield return _output;
        yield return _norm1; yield return _linear1; yield return _linear2; yield return _norm2;
    }

    /// <summary>
    /// <c>build_2d_sincos_position_embedding(w, h, dim, temperature)</c>, as written in the reference. It is
    /// built from <c>meshgrid(arange(w), arange(h), indexing='ij')</c>, so token <c>t</c>'s w-coordinate is
    /// <c>t / h</c> and its h-coordinate is <c>t % h</c>. The features are flattened as <c>y * w + x</c>,
    /// and this quirk is kept. Returns <c>[h * w, dim]</c> as (w.sin, w.cos, h.sin, h.cos).
    /// </summary>
    public static double[] SinCosPositionalEncoding(int width, int height, int dim, double temperature)
    {
        if (dim % 4 != 0) throw new ArgumentException("The 2-D sin-cos encoding needs a dimension divisible by 4.", nameof(dim));
        int posDim = dim / 4;
        var result = new double[width * height * dim];
        for (int t = 0; t < width * height; t++)
        {
            double w = t / height, h = t % height;
            for (int i = 0; i < posDim; i++)
            {
                double omega = 1.0 / Math.Pow(temperature, (double)i / posDim);
                result[(t * dim) + i] = Math.Sin(w * omega);
                result[(t * dim) + posDim + i] = Math.Cos(w * omega);
                result[(t * dim) + (2 * posDim) + i] = Math.Sin(h * omega);
                result[(t * dim) + (3 * posDim) + i] = Math.Cos(h * omega);
            }
        }
        return result;
    }
}

/// <summary>
/// The reference <c>HybridEncoder</c>:
/// <list type="bullet">
/// <item>A 1x1 ConvNorm projection of S3-S5.</item>
/// <item>AIFI (one intra-scale transformer layer) on S5 only.</item>
/// <item>CCFM: a top-down FPN (1x1 lateral ConvNorm+SiLU, 2x nearest upsample, concat, CSPRepLayer),
/// then a bottom-up PAN (3x3 stride-2 ConvNorm+SiLU, concat, CSPRepLayer).</item>
/// </list>
/// </summary>
internal sealed class RtdetrHybridEncoder<T> : CvParameterModule<T>
{
    private readonly int _hidden;
    private readonly double _temperature;
    private readonly List<RtdetrConvNorm<T>> _inputProjections = new();
    private readonly RtdetrAifiLayer<T> _aifi;
    private readonly List<RtdetrConvNorm<T>> _lateral = new();
    private readonly List<RtdetrCspRepLayer<T>> _fpn = new();
    private readonly List<RtdetrConvNorm<T>> _downsample = new();
    private readonly List<RtdetrCspRepLayer<T>> _pan = new();

    public RtdetrHybridEncoder(IReadOnlyList<int> inChannels, int hidden, int numHeads, int feedForward, double expansion, double temperature)
    {
        if (inChannels.Count != 3) throw new ArgumentException("The hybrid encoder takes S3, S4 and S5.", nameof(inChannels));
        _hidden = hidden;
        _temperature = temperature;
        foreach (int channels in inChannels) _inputProjections.Add(new RtdetrConvNorm<T>(channels, hidden, 1));
        _aifi = new RtdetrAifiLayer<T>(hidden, numHeads, feedForward);
        for (int i = 0; i < 2; i++)
        {
            _lateral.Add(new RtdetrConvNorm<T>(hidden, hidden, 1, silu: true));
            _fpn.Add(new RtdetrCspRepLayer<T>(2 * hidden, hidden, 3, expansion));
            _downsample.Add(new RtdetrConvNorm<T>(hidden, hidden, 3, 2, silu: true));
            _pan.Add(new RtdetrCspRepLayer<T>(2 * hidden, hidden, 3, expansion));
        }
    }

    public int Hidden => _hidden;

    public List<Tensor<T>> Forward(IReadOnlyList<Tensor<T>> features)
    {
        var engine = AiDotNetEngine.Current;
        var projected = features.Select((feature, i) => _inputProjections[i].Forward(feature)).ToList();

        // AIFI on S5: flatten [B, C, H, W] -> [B, H*W, C], attend, restore.
        var s5 = projected[2];
        int batch = s5.Shape[0], h = s5.Shape[2], w = s5.Shape[3];
        var tokens = engine.TensorPermute(engine.Reshape(s5, new[] { batch, _hidden, h * w }), new[] { 0, 2, 1 });
        var numOps = MathHelper.GetNumericOperations<T>();
        var pe = RtdetrAifiLayer<T>.SinCosPositionalEncoding(w, h, _hidden, _temperature);
        var position = engine.TensorBroadcastTo(
            new Tensor<T>(pe.Select(v => numOps.FromDouble(v)).ToArray(), new[] { 1, h * w, _hidden }), new[] { batch, h * w, _hidden });
        var attended = _aifi.Forward(tokens, position);
        projected[2] = engine.Reshape(engine.TensorPermute(attended, new[] { 0, 2, 1 }), new[] { batch, _hidden, h, w });

        // Top-down: inner = [S5']; for S4 then S3: lateral the coarse map, upsample, concat, fuse.
        var inner = new List<Tensor<T>> { projected[2] };
        for (int idx = 2; idx >= 1; idx--)
        {
            var high = _lateral[2 - idx].Forward(inner[0]);
            inner[0] = high;
            var low = projected[idx - 1];
            var up = CvTensorOps<T>.ResizeNearest(high, low.Shape[2], low.Shape[3]);
            inner.Insert(0, _fpn[2 - idx].Forward(engine.TensorConcatenate(new[] { up, low }, 1)));
        }

        // Bottom-up: downsample, concat with the next top-down map, fuse.
        var outs = new List<Tensor<T>> { inner[0] };
        for (int idx = 0; idx < 2; idx++)
        {
            var down = _downsample[idx].Forward(outs[outs.Count - 1]);
            outs.Add(_pan[idx].Forward(engine.TensorConcatenate(new[] { down, inner[idx + 1] }, 1)));
        }
        return outs;
    }

    public void SetTrainingMode(bool training)
    {
        foreach (var p in _inputProjections) p.SetTrainingMode(training);
        for (int i = 0; i < 2; i++)
        {
            _lateral[i].SetTrainingMode(training);
            _fpn[i].SetTrainingMode(training);
            _downsample[i].SetTrainingMode(training);
            _pan[i].SetTrainingMode(training);
        }
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var p in _inputProjections) p.Write(writer);
        _aifi.Write(writer);
        for (int i = 0; i < 2; i++) { _lateral[i].Write(writer); _fpn[i].Write(writer); _downsample[i].Write(writer); _pan[i].Write(writer); }
    }

    public void Read(BinaryReader reader)
    {
        foreach (var p in _inputProjections) p.Read(reader);
        _aifi.Read(reader);
        for (int i = 0; i < 2; i++) { _lateral[i].Read(reader); _fpn[i].Read(reader); _downsample[i].Read(reader); _pan[i].Read(reader); }
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        foreach (var p in _inputProjections) yield return p;
        yield return _aifi;
        foreach (var l in _lateral) yield return l;
        foreach (var f in _fpn) yield return f;
        foreach (var d in _downsample) yield return d;
        foreach (var p in _pan) yield return p;
    }
}
