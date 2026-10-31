using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// What training Mimi adds to the codec (Défossez et al. 2024 §3.3): the multi-scale STFT discriminator of Audiocraft's
/// compression solver (filters 32, windows 1024, 2048, 512, 256, 128 with hop w/4, weight norm, LeakyReLU 0.3) and its
/// adversarial-only objective — the hinge losses and the L1 feature-matching loss, averaged over the sub-discriminators,
/// balanced on the output (adversarial 4, feature 4) — plus the semantic distillation: WavLM's embeddings of the input
/// resampled to 16 kHz, average-pooled non-causally (kernel 8, stride 4) to 12.5 Hz, against a linear projection of the
/// first quantizer's output, by cosine distance.
/// </summary>
/// <remarks>The non-causal pooling pads two frames on each side and averages the frames present (the paper fixes the kernel
/// and stride, not the edges). The distillation's weight and WavLM's layer are not stated: weight 1 and the last hidden
/// state.</remarks>
internal sealed class MimiTraining<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int PoolKernel = 8, PoolStride = 4;
    private readonly IEngine _engine;
    private readonly int _sampleRate;
    private readonly double _distillationWeight;

    public MimiTraining(IEngine engine, int sampleRate, int latentDim, WavLmConfiguration? wavLm, double adversarialWeight, double featureWeight,
        double distillationWeight, int[] discriminatorWindows, int discriminatorFilters, List<LayerBase<T>> discriminatorLayers,
        List<LayerBase<T>> distillationLayers, List<LayerBase<T>> wavLmLayers)
    {
        _engine = engine;
        _sampleRate = sampleRate;
        _distillationWeight = distillationWeight;
        Discriminator = new MultiScaleStftDiscriminator<T>(engine, discriminatorWindows, discriminatorFilters, new[] { 1, 2, 4 }, 3, 9, 0.3,
            discriminatorLayers);
        Balancer = new LossBalancer<T>(engine, new Dictionary<string, double> { ["adv"] = adversarialWeight, ["feat"] = featureWeight });
        if (wavLm is not null)
        {
            WavLm = new WavLmNetwork<T>(engine, wavLmLayers, wavLm);
            Projection = new DenseLayer<T>(wavLm.HiddenSize, (IActivationFunction<T>)new IdentityActivation<T>());
            Projection.ResolveFromShape(new[] { 1, latentDim });
            distillationLayers.Add(Projection);
        }
    }

    public MultiScaleStftDiscriminator<T> Discriminator { get; }

    public LossBalancer<T> Balancer { get; }

    /// <summary>WavLM, frozen, for the distillation targets; null trains without distillation.</summary>
    public WavLmNetwork<T>? WavLm { get; }

    /// <summary>The projection of the first quantizer's output to WavLM's width.</summary>
    public DenseLayer<T>? Projection { get; }

    private Tensor<T> Mean(Tensor<T> x) => _engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    private static Tensor<T> Detached(Tensor<T> x) => new(x.Shape.ToArray(), x.ToVector());

    /// <summary>The discriminator's hinge loss (1/K) Σ_k mean(relu(1 − D_k(x))) + mean(relu(1 + D_k(x̂))) on waveforms
    /// <c>[samples]</c>, x̂ detached.</summary>
    public Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var onFake = Discriminator.Forward(Detached(generated));
        var onReal = Discriminator.Forward(Detached(real));
        Tensor<T>? total = null;
        for (int k = 0; k < onReal.Count; k++)
        {
            var term = _engine.TensorAdd(Mean(_engine.ReLU(_engine.TensorAddScalar(_engine.TensorNegate(onReal[k].Logits), NumOps.One))),
                Mean(_engine.ReLU(_engine.TensorAddScalar(onFake[k].Logits, NumOps.One))));
            total = total is null ? term : _engine.TensorAdd(total, term);
        }
        return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(1.0 / onReal.Count));
    }

    /// <summary>The generator's adversarial loss (1/K) Σ_k −mean D_k(x̂).</summary>
    private Tensor<T> AdversarialLoss(Tensor<T> generated)
    {
        var onFake = Discriminator.Forward(generated);
        Tensor<T>? total = null;
        foreach (var d in onFake)
        {
            var term = _engine.TensorNegate(Mean(d.Logits));
            total = total is null ? term : _engine.TensorAdd(total, term);
        }
        return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(1.0 / onFake.Count));
    }

    /// <summary>The feature-matching loss (1/K) Σ_k (1/L) Σ_l mean|D_k^l(x̂) − D_k^l(x)|.</summary>
    private Tensor<T> FeatureLoss(IReadOnlyList<List<Tensor<T>>> real, Tensor<T> generated)
    {
        var onFake = Discriminator.Forward(generated);
        Tensor<T>? total = null;
        for (int k = 0; k < onFake.Count; k++)
        {
            Tensor<T>? sum = null;
            for (int l = 0; l < onFake[k].Features.Count; l++)
            {
                var term = Mean(_engine.TensorAbs(_engine.TensorSubtract(onFake[k].Features[l], real[k][l])));
                sum = sum is null ? term : _engine.TensorAdd(sum, term);
            }
            var averaged = _engine.TensorMultiplyScalar(sum!, NumOps.FromDouble(1.0 / onFake[k].Features.Count));
            total = total is null ? averaged : _engine.TensorAdd(total, averaged);
        }
        return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(1.0 / onFake.Count));
    }

    /// <summary>
    /// The distillation targets <c>[frames, width]</c> of a waveform <c>[samples]</c> at the codec's rate: WavLM's last hidden
    /// state of the 16 kHz resampling, average-pooled to the code rate. Null without WavLM.
    /// </summary>
    public Tensor<T>? DistillationTargets(Tensor<T> audio)
    {
        if (WavLm is null) return null;
        using var _ = new NoGradScope<T>();
        var wave = _engine.Reshape(audio, new[] { audio.Length });
        var resampled = _sampleRate == 16000 ? wave : AudioHelper<T>.ResampleBandLimited(_engine, wave, _sampleRate, 16000);
        var hidden = WavLm.HiddenStates(resampled);
        var last = hidden[hidden.Count - 1];                                                            // [L, width]
        int length = last.Shape[0], width = last.Shape[1], pad = (PoolKernel - PoolStride) / 2;
        int frames = (length + 2 * pad - PoolKernel) / PoolStride + 1;
        if (frames <= 0) throw new ArgumentException("The recording is too short for a distillation target.", nameof(audio));
        var pooled = new Tensor<T>(new[] { frames, width });
        for (int f = 0; f < frames; f++)
        {
            int start = Math.Max(0, f * PoolStride - pad), end = Math.Min(length, f * PoolStride - pad + PoolKernel);
            for (int j = 0; j < width; j++)
            {
                double sum = 0;
                for (int t = start; t < end; t++) sum += NumOps.ToDouble(last[t, j]);
                pooled[f, j] = NumOps.FromDouble(sum / (end - start));
            }
        }
        return pooled;
    }

    /// <summary>The cosine distance (1/T) Σ_t 1 − cos(P q_t, w_t) between the projected first-quantizer output
    /// <c>[1, dim, frames]</c> and the targets <c>[frames', width]</c>, over the frames both have.</summary>
    public Tensor<T> DistillationLoss(Tensor<T> semantic, Tensor<T> targets)
    {
        int dim = semantic.Shape[1], frames = Math.Min(semantic.Shape[2], targets.Shape[0]), width = targets.Shape[1];
        var q = _engine.TensorPermute(_engine.Reshape(semantic, new[] { dim, semantic.Shape[2] }), new[] { 1, 0 });
        q = _engine.TensorSlice(q, new[] { 0, 0 }, new[] { frames, dim });
        var p = Projection!.Forward(q);                                                                 // [T, width]
        var w = _engine.TensorSlice(targets, new[] { 0, 0 }, new[] { frames, width });
        var dot = _engine.ReduceSum(_engine.TensorMultiply(p, w), new[] { 1 }, false);
        var pn = _engine.TensorSqrt(_engine.TensorAddScalar(_engine.ReduceSum(_engine.TensorMultiply(p, p), new[] { 1 }, false), NumOps.FromDouble(1e-16)));
        var wn = _engine.TensorSqrt(_engine.TensorAddScalar(_engine.ReduceSum(_engine.TensorMultiply(w, w), new[] { 1 }, false), NumOps.FromDouble(1e-16)));
        var cosine = _engine.TensorDivide(dot, _engine.TensorMultiply(pn, wn));
        return _engine.TensorAddScalar(_engine.TensorNegate(Mean(cosine)), NumOps.One);
    }

    /// <summary>
    /// The generator's objective for a step: the balanced adversarial and feature losses on x̂ (a surrogate whose gradient
    /// on x̂ is the balanced sum), plus the commitment penalty and the weighted distillation. Returns the objective and the
    /// reported loss (the weighted losses' value).
    /// </summary>
    public (Tensor<T> Objective, double Reported) GeneratorObjective(Tensor<T> real, Tensor<T> generated, Tensor<T> semantic, Tensor<T> penalty,
        Tensor<T>? targets)
    {
        List<List<Tensor<T>>> realFeatures;
        using (new NoGradScope<T>())
            realFeatures = Discriminator.Forward(Detached(real)).Select(d => d.Features.Select(Detached).ToList()).ToList();
        var surrogate = Balancer.Surrogate(generated, new Dictionary<string, Func<Tensor<T>, Tensor<T>>>
        {
            ["adv"] = AdversarialLoss,
            ["feat"] = x => FeatureLoss(realFeatures, x),
        });
        var objective = _engine.TensorAdd(surrogate, penalty);
        double reported = Balancer.LastWeightedLoss + NumOps.ToDouble(penalty.GetFlat(0));
        if (targets is not null && Projection is not null && _distillationWeight > 0)
        {
            var distillation = _engine.TensorMultiplyScalar(DistillationLoss(semantic, targets), NumOps.FromDouble(_distillationWeight));
            objective = _engine.TensorAdd(objective, distillation);
            reported += NumOps.ToDouble(distillation.GetFlat(0));
        }
        return (objective, reported);
    }
}
