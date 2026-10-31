using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// The <see cref="LayerBase{T}"/> save-for-backward decision, for code that is not a layer: whole
/// models with a hand-written backward, and standalone components that keep their own activation
/// caches.
/// </summary>
/// <typeparam name="T">The numeric type the model computes in.</typeparam>
/// <remarks>
/// <para>
/// A layer routes every manual-backward cache write through <c>LayerBase.SaveForBackward</c>, so the
/// decision "will a backward read this?" lives in one place. A model or component with its own
/// <c>_lastX = value</c> fields had no equivalent and wrote them unconditionally, so every forward
/// (inference included) left its activations reachable until the next one. This is the same policy
/// with the same terms, so a model's caches follow the layers' rules exactly.
/// </para>
/// <para>
/// A value is kept only when a manual backward can read it: not inside an
/// <see cref="InferenceMode"/> scope, and not while a <see cref="GradientTape{T}"/> is recording
/// (the tape keeps what its own backward needs) unless
/// <see cref="LayerBase{T}.KeepActivationCacheUnderTape"/> forces it. Callers that have a train/eval
/// flag also pass it, matching <c>LayerBase.ShouldCacheForBackward</c>.
/// </para>
/// </remarks>
internal static class ManualBackwardCache<T>
{
    /// <summary>
    /// True when a manual backward may read activation caches written now. The variant for code with
    /// no train/eval flag, matching <c>LayerBase.ShouldCacheActivationsForManualBackward</c>.
    /// </summary>
    internal static bool ShouldCacheActivationsForManualBackward
        => !InferenceMode.IsActive
        && (GradientTape<T>.Current is null || LayerBase<T>.KeepActivationCacheUnderTape);

    /// <summary>
    /// True when a manual backward may read activation caches written now, for code with a train/eval
    /// flag. Matches <c>LayerBase.ShouldCacheForBackward</c>.
    /// </summary>
    /// <param name="isTrainingMode">The caller's training-mode flag.</param>
    internal static bool ShouldCacheForBackward(bool isTrainingMode)
        => isTrainingMode && ShouldCacheActivationsForManualBackward;

    /// <summary>
    /// Saves an activation for a manual backward, or clears the field when no backward will read it.
    /// For code with no train/eval flag.
    /// </summary>
    /// <typeparam name="TValue">The cached value's type.</typeparam>
    /// <param name="field">The backing field, passed by reference.</param>
    /// <param name="value">The activation to save.</param>
    internal static void SaveForBackward<TValue>(ref TValue? field, TValue value) where TValue : class
    {
        field = ShouldCacheActivationsForManualBackward ? value : null;
    }

    /// <summary>
    /// Saves an activation for a manual backward, or clears the field when no backward will read it.
    /// For code with a train/eval flag.
    /// </summary>
    /// <typeparam name="TValue">The cached value's type.</typeparam>
    /// <param name="isTrainingMode">The caller's training-mode flag.</param>
    /// <param name="field">The backing field, passed by reference.</param>
    /// <param name="value">The activation to save.</param>
    internal static void SaveForBackward<TValue>(bool isTrainingMode, ref TValue? field, TValue value)
        where TValue : class
    {
        field = ShouldCacheForBackward(isTrainingMode) ? value : null;
    }
}
