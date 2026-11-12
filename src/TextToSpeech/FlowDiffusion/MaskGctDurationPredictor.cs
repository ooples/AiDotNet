using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// MaskGCT's total-duration predictor (App. A.5): flow matching over each phone's log(duration + 1), its velocity estimated
/// by a Transformer like the generators' (12 layers, 12 heads, width 768, the step conditioning every norm), the phones'
/// embeddings conditioning each position. A prefix of the phones and their durations is the prompt, given un-noised
/// (dropped with probability 0.15 in training); the target's durations are generated from Gaussian noise by a midpoint
/// ODE solver in 4 steps, and their sum is the target's length.
/// </summary>
/// <remarks>
/// The paper leaves the release unspecified beyond that; this port fills it as the generators are built: the phone
/// embedding enters as <c>DiffLlama</c>'s per-position condition, a linear map lifts the scalar log-duration to the width,
/// another reads the velocity out, and the uniform-step time is the generators' sinusoidal step. Durations count
/// semantic frames (50 per second).
/// </remarks>
internal sealed class MaskGctDurationPredictor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _hidden;

    public MaskGctDurationPredictor(IEngine engine, List<LayerBase<T>> owner, MaskGctTransformerConfiguration backbone, int phones, Random initialization)
    {
        _engine = engine;
        _hidden = backbone.Hidden;
        Phones = phones;
        Phone = new TiedEmbeddingLayer<T>(phones, _hidden);
        Phone.Reinitialize(() => 0.02 * MaskGctTextToSemantic<T>.Normal(initialization), phones - 1);
        owner.Add(Phone);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            owner.Add(layer);
            TorchParameters.Linear(engine, layer, inputs, outputs,
                TorchParameters.Gaussian<T>(initialization, inputs * outputs, 0.02).Select(v => NumOps.ToDouble(v)).ToArray(), new double[outputs]);
            return layer;
        }
        Input = Linear(1, _hidden);
        Output = Linear(_hidden, 1);
        Backbone = new MaskGctTransformer<T>(engine, owner, backbone with { PrefixCondition = false }, initialization);
    }

    public int Phones { get; }
    public TiedEmbeddingLayer<T> Phone { get; }
    public DenseLayer<T> Input { get; }
    public DenseLayer<T> Output { get; }
    public MaskGctTransformer<T> Backbone { get; }

    private static Tensor<T> Column(IReadOnlyList<double> values)
    {
        var x = new Tensor<T>(new[] { values.Count, 1 });
        for (int i = 0; i < values.Count; i++) x[i, 0] = NumOps.FromDouble(values[i]);
        return x;
    }

    /// <summary>The velocity <c>[phones, 1]</c> at step <paramref name="t"/> of the log-durations <paramref name="x"/> (the
    /// prompt's clean, the target's at x_t).</summary>
    public Tensor<T> Velocity(IReadOnlyList<int> phones, Tensor<T> x, double t)
    {
        var condition = Phone.Forward(MaskGctSampling.Ids<T>(phones));
        return Output.Forward(Backbone.Forward(Input.Forward(x), t, condition));
    }

    /// <summary>
    /// The flow-matching loss <c>E‖v(x_t, t) − (x1 − x0)‖²</c> over the target phones (those past the prompt), x1 the
    /// phones' log(duration + 1), x0 <paramref name="noise"/>.
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<int> phones, IReadOnlyList<int> durations, int promptLength, double t, IReadOnlyList<double> noise)
    {
        int n = phones.Count;
        var xt = new double[n];
        var target = new Tensor<T>(new[] { n - promptLength, 1 });
        for (int i = 0; i < n; i++)
        {
            double x1 = Math.Log(durations[i] + 1.0);
            if (i < promptLength) xt[i] = x1;
            else
            {
                xt[i] = (1 - t) * noise[i] + t * x1;
                target[i - promptLength, 0] = NumOps.FromDouble(x1 - noise[i]);
            }
        }
        var v = Velocity(phones, Column(xt), t);
        if (promptLength > 0) v = _engine.TensorSlice(v, new[] { promptLength, 0 }, new[] { n - promptLength, 1 });
        var difference = _engine.TensorSubtract(v, target);
        return _engine.TensorDivideScalar(_engine.ReduceSum(_engine.TensorSquare(difference), new[] { 0, 1 }, keepDims: false), NumOps.FromDouble(n - promptLength));
    }

    /// <summary>
    /// The target phones' durations in frames, solved from Gaussian <paramref name="noise"/> by the midpoint method in
    /// <paramref name="steps"/> steps after a prompt of phones with known durations (none to run without one), the
    /// velocity guided toward the prompted one by <paramref name="guidance"/> when there is a prompt (1: no guidance).
    /// </summary>
    public double[] Generate(IReadOnlyList<int> promptPhones, IReadOnlyList<int> promptDurations, IReadOnlyList<int> targetPhones, int steps,
        double guidance, IReadOnlyList<double> noise)
    {
        int lp = promptPhones.Count, n = targetPhones.Count;
        var phones = promptPhones.Concat(targetPhones).ToArray();
        var prompt = promptDurations.Select(d => Math.Log(d + 1.0)).ToArray();
        var x = noise.Take(n).ToArray();
        double[] VelocityAt(double[] target, double t)
        {
            var v = Velocity(phones, Column(prompt.Concat(target).ToArray()), t).ToArray().Skip(lp).Select(NumOps.ToDouble).ToArray();
            if (lp == 0 || guidance == 1) return v;
            var unconditional = Velocity(targetPhones, Column(target), t).ToArray().Select(NumOps.ToDouble).ToArray();
            for (int i = 0; i < n; i++) v[i] = unconditional[i] + guidance * (v[i] - unconditional[i]);
            return v;
        }
        double h = 1.0 / steps;
        for (int s = 0; s < steps; s++)
        {
            double t = s * h;
            var k1 = VelocityAt(x, t);
            var mid = x.Select((v, i) => v + 0.5 * h * k1[i]).ToArray();
            var k2 = VelocityAt(mid, t + 0.5 * h);
            for (int i = 0; i < n; i++) x[i] += h * k2[i];
        }
        return x.Select(v => Math.Max(0, Math.Exp(v) - 1)).ToArray();
    }
}
