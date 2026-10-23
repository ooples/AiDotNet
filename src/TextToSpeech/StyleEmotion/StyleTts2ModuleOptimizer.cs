using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// One AdamW per module, as StyleTTS 2's reference trains (<c>optimizers.py</c>'s <c>MultiOptimizer</c>): each module keeps
/// its own moments and step count, so a module that starts training late (the diffusion model, the aligner) begins with
/// Adam's bias correction at step 1, and modules can differ in rate, betas and decay (PL-BERT's 1e-5, β₁ 0.9, decay 0.01).
/// A step's parameters are split by the module that owns them and each module's optimizer steps its share.
/// </summary>
/// <remarks>The configured options are the default module's; <see cref="GradientTransform"/>, when set, rewrites the
/// gradients before any module steps (the SLM adversarial step's gradient scaling).</remarks>
internal sealed class StyleTts2ModuleOptimizer<T> : AdamWOptimizer<T, Tensor<T>, Tensor<T>>
{
    private readonly Func<Tensor<T>, string?> _moduleOf;
    private readonly Func<string, AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>> _optionsFor;
    private readonly IFullModel<T, Tensor<T>, Tensor<T>> _model;
    private readonly Dictionary<string, AdamWOptimizer<T, Tensor<T>, Tensor<T>>> _modules = new();

    public StyleTts2ModuleOptimizer(IFullModel<T, Tensor<T>, Tensor<T>> model, AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> defaults,
        Func<Tensor<T>, string?> moduleOf, Func<string, AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>> optionsFor)
        : base(model, defaults)
    {
        _model = model;
        _moduleOf = moduleOf;
        _optionsFor = optionsFor;
    }

    /// <summary>Rewrites a step's gradients (keyed by parameter) before the modules step; null leaves them.</summary>
    public Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? GradientTransform { get; set; }

    /// <summary>The steps each module's optimizer has taken.</summary>
    internal IReadOnlyDictionary<string, AdamWOptimizer<T, Tensor<T>, Tensor<T>>> Modules => _modules;

    /// <inheritdoc />
    protected override void StepCore(TapeStepContext<T> context)
    {
        GradientTransform?.Invoke(context.Parameters, context.Gradients);
        foreach (var group in context.Parameters.GroupBy(p => _moduleOf(p) ?? throw new InvalidOperationException(
                     "A stepped parameter belongs to no StyleTTS 2 module.")))
        {
            if (!_modules.TryGetValue(group.Key, out var optimizer))
                _modules[group.Key] = optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(_model, _optionsFor(group.Key));
            optimizer.Step(new TapeStepContext<T>(group.ToList(), context.Gradients, context.Loss));
        }
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _modules.Clear();
    }
}
