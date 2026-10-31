using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The gated activation in <see cref="MaskGctTransformer{T}"/>'s feed-forward.</summary>
public enum MaskGctActivation
{
    /// <summary>SiLU, transformers' <c>LlamaConfig</c> default, which the released MaskGCT models were built and trained
    /// with.</summary>
    SiLU,

    /// <summary>GELU (exact), as the paper writes (§3.2.2: "gated linear units with GELU activation").</summary>
    Gelu,
}

/// <summary>The shape of a <see cref="MaskGctTransformer{T}"/>.</summary>
internal sealed record MaskGctTransformerConfiguration
{
    public int Hidden { get; init; } = 1024;
    public int Layers { get; init; } = 16;
    public int Heads { get; init; } = 16;
    public double RmsEpsilon { get; init; } = 1e-6;
    public double RopeTheta { get; init; } = 10000;
    public MaskGctActivation Activation { get; init; } = MaskGctActivation.SiLU;

    /// <summary>Whether the input carries a prefix projected by <c>cond_mlp</c> (<c>DiffLlamaPrefix</c>'s phone
    /// condition) rather than a per-frame condition added to the input (<c>DiffLlama</c>'s).</summary>
    public bool PrefixCondition { get; init; }
}

/// <summary>
/// MaskGCT's non-autoregressive Llama (Amphion <c>llama_nar.py</c>: <c>DiffLlama</c> for the S2A models,
/// <c>DiffLlamaPrefix</c> for T2S): transformers' Llama decoder layers (bias-free attention with rotary positions,
/// SwiGLU with four times the width) run bidirectionally, each RMSNorm made adaptive — its scale a linear map of the
/// step's embedding (zero weight, unit bias at initialization) — the step embedded sinusoidally and through a SiLU MLP,
/// and a final adaptive RMSNorm. <c>DiffLlama</c> adds an MLP of a per-frame condition to its input; <c>DiffLlamaPrefix</c>
/// puts an MLP of the phone embeddings in front of its input and drops their outputs.
/// </summary>
/// <remarks>Works on one sequence, <c>[length, hidden]</c>; a key mask hides padding as transformers' additive mask does.</remarks>
internal sealed class MaskGctTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly MaskGctTransformerConfiguration _c;

    internal sealed record Layer(DenseLayer<T> InputNorm, BiasFreeLinearLayer<T> Query, BiasFreeLinearLayer<T> Key, BiasFreeLinearLayer<T> Value,
        BiasFreeLinearLayer<T> Output, DenseLayer<T> PostNorm, BiasFreeLinearLayer<T> Gate, BiasFreeLinearLayer<T> Up, BiasFreeLinearLayer<T> Down);

    public MaskGctTransformer(IEngine engine, List<LayerBase<T>> owner, MaskGctTransformerConfiguration configuration, Random initialization)
    {
        _engine = engine;
        _c = configuration;
        int h = configuration.Hidden;
        if (h % configuration.Heads != 0 || (h / configuration.Heads) % 2 != 0)
            throw new ArgumentException("The width must split into heads of even width.", nameof(configuration));
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            owner.Add(layer);
            return layer;
        }
        // transformers' _init_weights: N(0, 0.02) linear weights, zero biases (LlamaConfig.initializer_range).
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Own(new BiasFreeLinearLayer<T>(inputs, outputs));
            layer.SetParameters(TorchParameters.Gaussian<T>(initialization, inputs * outputs, 0.02));
            return layer;
        }
        DenseLayer<T> Dense(int inputs, int outputs, double weightStd, double bias)
        {
            var layer = Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            var w = TorchParameters.Gaussian<T>(initialization, inputs * outputs, weightStd);
            TorchParameters.Linear(_engine, layer, inputs, outputs, w.Select(v => NumOps.ToDouble(v)).ToArray(), Enumerable.Repeat(bias, outputs).ToArray());
            return layer;
        }
        // LlamaAdaptiveRMSNorm: to_weight zero-initialized with a bias of ones, so it starts as a plain RMSNorm.
        DenseLayer<T> Adaptive() => Dense(h, h, 0.0, 1.0);
        for (int i = 0; i < configuration.Layers; i++)
            Layers.Add(new Layer(Adaptive(), Linear(h, h), Linear(h, h), Linear(h, h), Linear(h, h), Adaptive(),
                Linear(h, 4 * h), Linear(h, 4 * h), Linear(4 * h, h)));
        FinalNorm = Adaptive();
        StepMlp = (Dense(h, 4 * h, 0.02, 0), Dense(4 * h, h, 0.02, 0));
        ConditionMlp = (Dense(h, 4 * h, 0.02, 0), Dense(4 * h, h, 0.02, 0));
    }

    public List<Layer> Layers { get; } = new();
    public DenseLayer<T> FinalNorm { get; }
    public (DenseLayer<T> In, DenseLayer<T> Out) StepMlp { get; }
    public (DenseLayer<T> In, DenseLayer<T> Out) ConditionMlp { get; }

    private Tensor<T> Silu(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Sigmoid(x));

    private Tensor<T> Mlp((DenseLayer<T> In, DenseLayer<T> Out) mlp, Tensor<T> x) => mlp.Out.Forward(Silu(mlp.In.Forward(x)));

    /// <summary>The step's embedding <c>[1, hidden]</c>: <c>SinusoidalPosEmb</c> (sin then cos of t · 10000^(−i/(d/2 − 1)))
    /// through the step MLP.</summary>
    private Tensor<T> Step(double t)
    {
        int half = _c.Hidden / 2;
        double scale = Math.Log(10000) / (half - 1);
        var e = new Tensor<T>(new[] { 1, _c.Hidden });
        for (int i = 0; i < half; i++)
        {
            double a = t * Math.Exp(-scale * i);
            e[0, i] = NumOps.FromDouble(Math.Sin(a));
            e[0, half + i] = NumOps.FromDouble(Math.Cos(a));
        }
        return Mlp(StepMlp, e);
    }

    // LlamaAdaptiveRMSNorm: x / sqrt(mean(x²) + ε) scaled by to_weight(step).
    private Tensor<T> Norm(DenseLayer<T> scale, Tensor<T> x, Tensor<T> step)
    {
        int l = x.Shape[0], h = x.Shape[1];
        var meanSquare = _engine.TensorDivideScalar(_engine.ReduceSum(_engine.TensorSquare(x), new[] { 1 }, keepDims: true), NumOps.FromDouble(h));
        var inverse = _engine.TensorReciprocal(_engine.TensorSqrt(_engine.TensorAddScalar(meanSquare, NumOps.FromDouble(_c.RmsEpsilon))));
        var normed = _engine.TensorMultiply(x, _engine.TensorTile(inverse, new[] { 1, h }));
        return _engine.TensorMultiply(normed, _engine.TensorTile(scale.Forward(step), new[] { l, 1 }));
    }

    private Tensor<T> Rotate(Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        int heads = x.Shape[0], l = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var first = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { heads, l, half });
        var second = _engine.TensorSlice(x, new[] { 0, 0, half }, new[] { heads, l, half });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 2);
        var shape = new[] { heads, l, hd };
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(cos, shape)), _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(sin, shape)));
    }

    private Tensor<T> Run(Layer layer, Tensor<T> x, Tensor<T> step, Tensor<T> cos, Tensor<T> sin, Tensor<T>? bias)
    {
        int l = x.Shape[0], heads = _c.Heads, hd = _c.Hidden / heads;
        var normed = Norm(layer.InputNorm, x, step);
        Tensor<T> Heads(BiasFreeLinearLayer<T> projection) =>
            _engine.TensorPermute(_engine.Reshape(projection.Forward(normed), new[] { l, heads, hd }), new[] { 1, 0, 2 });
        var q = Rotate(Heads(layer.Query), cos, sin);
        var k = Rotate(Heads(layer.Key), cos, sin);
        var v = Heads(layer.Value);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        if (bias is not null) scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(bias, new[] { heads, l, l }));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        x = _engine.TensorAdd(x, layer.Output.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { l, _c.Hidden })));
        var post = Norm(layer.PostNorm, x, step);
        var gate = layer.Gate.Forward(post);
        var activated = _c.Activation == MaskGctActivation.SiLU ? Silu(gate) : ActivationHelper.ExactGelu(gate, _engine);
        return _engine.TensorAdd(x, layer.Down.Forward(_engine.TensorMultiply(activated, layer.Up.Forward(post))));
    }

    /// <summary>
    /// The hidden states <c>[length, hidden]</c> of <paramref name="x"/> at step <paramref name="t"/>. With
    /// <see cref="MaskGctTransformerConfiguration.PrefixCondition"/>, <paramref name="condition"/> is the phone embeddings
    /// <c>[phones, hidden]</c> (or null), put in front and dropped from the output; otherwise it is the per-frame condition
    /// <c>[length, hidden]</c> added through its MLP. <paramref name="keyMask"/> (over the whole sequence, the prefix first)
    /// is false for padding; null attends everywhere.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, double t, Tensor<T>? condition, bool[]? keyMask = null)
    {
        int prefix = 0;
        if (_c.PrefixCondition)
        {
            if (condition is not null && condition.Shape[0] > 0)
            {
                prefix = condition.Shape[0];
                x = _engine.TensorConcatenate(new[] { Mlp(ConditionMlp, condition), x }, 0);
            }
        }
        else if (condition is not null)
        {
            x = _engine.TensorAdd(x, Mlp(ConditionMlp, condition));
        }
        int l = x.Shape[0], hd = _c.Hidden / _c.Heads, half = hd / 2;
        if (keyMask is not null && keyMask.Length != l) throw new ArgumentException($"The key mask covers {keyMask.Length} positions; the sequence has {l}.", nameof(keyMask));
        var step = Step(t);
        var cos = new Tensor<T>(new[] { 1, l, hd });
        var sin = new Tensor<T>(new[] { 1, l, hd });
        for (int j = 0; j < l; j++)
            for (int k = 0; k < half; k++)
            {
                double angle = j / Math.Pow(_c.RopeTheta, 2.0 * k / hd);
                cos[0, j, k] = cos[0, j, k + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, j, k] = sin[0, j, k + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        Tensor<T>? bias = null;
        if (keyMask is not null && keyMask.Any(m => !m))
        {
            bias = new Tensor<T>(new[] { 1, l, l });
            var blocked = NumOps.FromDouble(typeof(T) == typeof(float) ? float.MinValue : double.MinValue);
            for (int q = 0; q < l; q++)
                for (int k = 0; k < l; k++)
                    if (!keyMask[k]) bias[0, q, k] = blocked;
        }
        foreach (var layer in Layers) x = Run(layer, x, step, cos, sin, bias);
        var hidden = Norm(FinalNorm, x, step);
        return prefix == 0 ? hidden : _engine.TensorSlice(hidden, new[] { prefix, 0 }, new[] { l - prefix, _c.Hidden });
    }

    /// <summary>Loads the transformer's parameters (transformers' Llama names under <paramref name="prefix"/>, Amphion's
    /// <c>diff_step_mlp</c>, <c>cond_mlp</c> and adaptive norms' <c>to_weight</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]?> read, string prefix)
    {
        int h = _c.Hidden;
        double[] Need(string name, int[] shape) => read(prefix + name, shape) ?? throw new InvalidDataException($"The checkpoint has no tensor '{prefix}{name}'.");
        void Lin(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs) => TorchParameters.Linear(layer, inputs, outputs, Need(name + ".weight", new[] { outputs, inputs }));
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, Need(name + ".weight", new[] { outputs, inputs }), Need(name + ".bias", new[] { outputs }));
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"layers.{i}";
            Dense(l.InputNorm, $"{p}.input_layernorm.to_weight", h, h);
            Lin(l.Query, $"{p}.self_attn.q_proj", h, h);
            Lin(l.Key, $"{p}.self_attn.k_proj", h, h);
            Lin(l.Value, $"{p}.self_attn.v_proj", h, h);
            Lin(l.Output, $"{p}.self_attn.o_proj", h, h);
            Dense(l.PostNorm, $"{p}.post_attention_layernorm.to_weight", h, h);
            Lin(l.Gate, $"{p}.mlp.gate_proj", h, 4 * h);
            Lin(l.Up, $"{p}.mlp.up_proj", h, 4 * h);
            Lin(l.Down, $"{p}.mlp.down_proj", 4 * h, h);
        }
        Dense(FinalNorm, "norm.to_weight", h, h);
        Dense(StepMlp.In, "diff_step_mlp.0", h, 4 * h);
        Dense(StepMlp.Out, "diff_step_mlp.2", 4 * h, h);
        Dense(ConditionMlp.In, "cond_mlp.0", h, 4 * h);
        Dense(ConditionMlp.Out, "cond_mlp.2", 4 * h, h);
    }
}
