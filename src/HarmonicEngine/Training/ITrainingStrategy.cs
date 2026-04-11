using AiDotNet.HarmonicEngine.Models;

namespace AiDotNet.HarmonicEngine.Training;

/// <summary>
/// Plugin interface for no-backpropagation training strategies applied to
/// <see cref="HRELanguageModel{T}"/>. Each concrete implementation defines a
/// different DSP-native or biologically-plausible alternative to gradient
/// descent. The paper compares the four implementations head-to-head as its
/// central empirical study.
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional neural networks train with backpropagation —
/// errors at the output are propagated backward through the network via the chain
/// rule of calculus, producing gradients that update the weights. This works but
/// has downsides: it requires storing intermediate activations, it's not
/// biologically plausible, and it forces the "all layers learn simultaneously
/// from global error" pattern.
/// </para>
/// <para>
/// HRE's paper explores whether you can train a deep architecture without any
/// backpropagation, using only local / spectral / DSP-native learning rules.
/// This interface is the abstraction for plugging in different such rules:
/// </para>
/// <list type="bullet">
/// <item><description><b>SpectralTargetPropagation</b> — propagates targets backward through
/// spectral inverse filters; each layer does a local Hebbian update. <i>Primary, novel.</i></description></item>
/// <item><description><b>LayerwiseHebbianCascade</b> — trains layers one at a time bottom-up,
/// each with an auto-associative target. Greedy, fully local.</description></item>
/// <item><description><b>PredictiveCoding</b> — each layer learns to predict the next layer's
/// activations, local error drives Hebbian updates.</description></item>
/// <item><description><b>ForwardForward</b> — Hinton 2022's two-forward-pass method with
/// local goodness objectives per layer.</description></item>
/// </list>
/// </remarks>
public interface ITrainingStrategy<T>
{
    /// <summary>
    /// Gets the display name of the strategy, used for logging and ablation tables.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets a brief one-line description of what this strategy does.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Performs a single training step on the given batch.
    /// </summary>
    /// <param name="model">The HRE language model to update. The strategy may read and write
    /// the model's internal Hebbian filters, LayerNorm parameters, embeddings, etc.</param>
    /// <param name="batch">A batch of input/target token pairs to learn from.</param>
    void TrainStep(HRELanguageModel<T> model, TrainingBatch<T> batch);

    /// <summary>
    /// Gets current per-strategy metrics (e.g., layer-local losses, step count,
    /// convergence rates). Used for the paper's ablation tables and training curves.
    /// </summary>
    /// <returns>A dictionary of metric name → value.</returns>
    IReadOnlyDictionary<string, double> GetMetrics();

    /// <summary>
    /// Resets all internal counters and metrics. Call between training runs.
    /// </summary>
    void Reset();
}
