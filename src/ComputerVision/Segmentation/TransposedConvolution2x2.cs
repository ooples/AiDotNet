using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation;

/// <summary>
/// A 2×2 transposed convolution with stride 2 (PyTorch <c>ConvTranspose2d(in, out, kernel_size=2, stride=2)</c>): every
/// input pixel writes its own 2×2 output block, so it is exactly a per-pixel projection to <c>out·4</c> values and a
/// pixel shuffle, plus a bias per output channel.
/// </summary>
internal sealed class TransposedConvolution2x2<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public TransposedConvolution2x2(IEngine engine, Func<LayerBase<T>, LayerBase<T>> own, int inputs, int outputs)
    {
        _engine = engine;
        Inputs = inputs;
        Outputs = outputs;
        Weight = (BiasFreeLinearLayer<T>)own(new BiasFreeLinearLayer<T>(inputs, 4 * outputs));
        Bias = (BiasFreeLinearLayer<T>)own(new BiasFreeLinearLayer<T>(1, outputs));
    }

    public int Inputs { get; }
    public int Outputs { get; }
    public BiasFreeLinearLayer<T> Weight { get; }
    public BiasFreeLinearLayer<T> Bias { get; }

    /// <summary><c>[B, in, H, W]</c> → <c>[B, out, 2H, 2W]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x)
    {
        int b = x.Shape[0], h = x.Shape[2], w = x.Shape[3], o = Outputs;
        var pixels = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 3, 1 }), new[] { b * h * w, Inputs });
        var blocks = _engine.Reshape(Weight.Forward(pixels), new[] { b, h, w, o, 2, 2 });
        var y = _engine.Reshape(_engine.TensorPermute(blocks, new[] { 0, 3, 1, 4, 2, 5 }), new[] { b, o, 2 * h, 2 * w });
        var bias = Bias.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));
        return _engine.TensorAdd(y, _engine.TensorBroadcastTo(_engine.Reshape(bias, new[] { 1, o, 1, 1 }), new[] { b, o, 2 * h, 2 * w }));
    }

    /// <summary>Loads PyTorch's <c>[in, out, 2, 2]</c> weight and <c>[out]</c> bias.</summary>
    public void Load(double[] weight, double[] bias)
    {
        int i0 = Inputs, o0 = Outputs;
        // The projection's row (o, a, b) reads every input channel.
        var rows = new double[4 * o0 * i0];
        for (int i = 0; i < i0; i++)
            for (int o = 0; o < o0; o++)
                for (int a = 0; a < 4; a++)
                    rows[(o * 4 + a) * i0 + i] = weight[(i * o0 + o) * 4 + a];
        TorchParameters.Linear(Weight, i0, 4 * o0, rows);
        TorchParameters.Linear(Bias, 1, o0, bias);
    }

    /// <summary>PyTorch's default: U(±1/√fan-in) for the weight and the bias, fan-in = out·4 for a transposed
    /// convolution's <c>[in, out, 2, 2]</c> weight.</summary>
    public void Initialize(Random random)
    {
        double bound = 1.0 / Math.Sqrt(4.0 * Outputs);
        double[] Uniform(int count)
        {
            var a = new double[count];
            for (int i = 0; i < count; i++) a[i] = (2 * random.NextDouble() - 1) * bound;
            return a;
        }
        Load(Uniform(4 * Inputs * Outputs), Uniform(Outputs));
    }
}
