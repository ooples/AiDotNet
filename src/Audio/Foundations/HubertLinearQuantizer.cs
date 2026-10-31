using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// SpiritLM's HuBERT unit quantizer (the release's <c>LinearQuantizerModel</c>, <c>L11_quantizer_500.pt</c>): three linear
/// layers with leaky ReLU between them map each layer-11 frame to <c>units + 1</c> logits, the last a CTC blank; the frame's
/// unit is the argmax, and blank frames take the unit on their left (repeated until none is left, the first frame taking the
/// last's, as <c>torch.roll</c> does, for at most 10000 rounds).
/// </summary>
/// <remarks>The widths narrow from the input to the logits in two even steps: with <c>m = upstream − (units + 1)</c> the
/// hidden widths are <c>dim − m/4</c> and <c>dim − m/2</c> (integer division), 702 and 635 for the release's 768 and 500.</remarks>
internal sealed class HubertLinearQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int MaxRounds = 10000;
    private readonly IEngine _engine;

    public HubertLinearQuantizer(IEngine engine, List<LayerBase<T>> layers, int dim, int upstreamDim, int units)
    {
        if (units <= 0) throw new ArgumentOutOfRangeException(nameof(units));
        _engine = engine;
        Dim = dim;
        Units = units;
        int middle = upstreamDim - (units + 1);
        First = dim - middle / 4;
        Second = dim - middle / 2;
        if (First <= 0 || Second <= 0) throw new ArgumentException($"Widths {dim} and {upstreamDim} leave no hidden layer for {units} units.");
        DenseLayer<T> Linear(int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            return layer;
        }
        Hidden1 = Linear(First);
        Hidden2 = Linear(Second);
        Output = Linear(units + 1);
    }

    public int Dim { get; }

    /// <summary>The units (the blank is <see cref="Units"/>).</summary>
    public int Units { get; }

    public int First { get; }
    public int Second { get; }
    public DenseLayer<T> Hidden1 { get; }
    public DenseLayer<T> Hidden2 { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>The logits <c>[frames, units + 1]</c> of features <c>[frames, dim]</c>.</summary>
    public Tensor<T> Logits(Tensor<T> features)
    {
        var slope = NumOps.FromDouble(0.01);
        var h = _engine.LeakyReLU(Hidden1.Forward(features), slope);
        h = _engine.LeakyReLU(Hidden2.Forward(h), slope);
        return Output.Forward(h);
    }

    /// <summary>The unit of each frame of <paramref name="features"/> <c>[frames, dim]</c>, blanks filled from the left.</summary>
    public int[] Codes(Tensor<T> features)
    {
        var logits = Logits(features);
        int frames = logits.Shape[0], classes = logits.Shape[1];
        var code = new int[frames];
        for (int f = 0; f < frames; f++)
        {
            int best = 0;
            double bestValue = NumOps.ToDouble(logits[f, 0]);
            for (int c = 1; c < classes; c++)
            {
                double v = NumOps.ToDouble(logits[f, c]);
                if (v > bestValue) { bestValue = v; best = c; }
            }
            code[f] = best;
        }
        return FillBlanks(code, Units);
    }

    /// <summary>The release's blank filling: <c>code[blank] = roll(code, 1)[blank]</c>, every blank from the previous
    /// round's codes at once, until none is left or 10000 rounds have run (all blanks stay blank).</summary>
    public static int[] FillBlanks(int[] codes, int blank)
    {
        var code = (int[])codes.Clone();
        int frames = code.Length;
        for (int round = 0; round < MaxRounds && Array.IndexOf(code, blank) >= 0; round++)
        {
            var previous = (int[])code.Clone();
            for (int f = 0; f < frames; f++)
                if (previous[f] == blank) code[f] = previous[(f - 1 + frames) % frames];
        }
        return code;
    }

    /// <summary>Loads the release's <c>model_weight</c> (<c>0.</c>, <c>2.</c>, <c>4.</c> of the encoder Sequential).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        double[] Read(string name) => take(name) ?? throw new InvalidDataException($"The quantizer checkpoint has no tensor '{name}'.");
        TorchParameters.Linear(_engine, Hidden1, Dim, First, Read("0.weight"), Read("0.bias"));
        TorchParameters.Linear(_engine, Hidden2, First, Second, Read("2.weight"), Read("2.bias"));
        TorchParameters.Linear(_engine, Output, Second, Units + 1, Read("4.weight"), Read("4.bias"));
    }

    /// <summary><c>nn.Linear</c>'s default initialization, U(±1/√in) for weights and biases.</summary>
    public void InitializeLikeReference(Random random)
    {
        void Init(DenseLayer<T> layer, int inputs, int outputs)
        {
            double bound = 1.0 / Math.Sqrt(inputs);
            var w = new double[outputs * inputs];
            var b = new double[outputs];
            for (int i = 0; i < w.Length; i++) w[i] = (2 * random.NextDouble() - 1) * bound;
            for (int i = 0; i < b.Length; i++) b[i] = (2 * random.NextDouble() - 1) * bound;
            TorchParameters.Linear(_engine, layer, inputs, outputs, w, b);
        }
        Init(Hidden1, Dim, First);
        Init(Hidden2, First, Second);
        Init(Output, Second, Units + 1);
    }
}
