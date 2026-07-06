using System;
using System.Threading;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Ambient "no backward pass will run on this flow" scope — the analog of
/// PyTorch's <c>torch.inference_mode()</c> / <c>torch.no_grad()</c>. While a scope
/// is active, layers skip populating their manual-backward activation caches
/// (the per-layer <c>_lastInput</c> / pre-activation fields that an eager
/// <c>Backward()</c> reads).
/// </summary>
/// <remarks>
/// <para>
/// Two reasons to skip those caches during inference:
/// </para>
/// <list type="number">
///   <item><description><b>Memory.</b> No backward will read them, so storing one
///   tensor reference per layer per forward is pure waste — it pins the whole
///   forward activation set alive for the duration of a prediction.</description></item>
///   <item><description><b>Correctness under the inference caching allocator.</b>
///   The multi-forward inference path (e.g. a diffusion denoise loop) runs inside a
///   <c>TensorArena</c> and calls <c>Reset()</c> between forwards to recycle that
///   step's scratch. A layer field that still references a recycled scratch buffer
///   would alias whatever the next step's allocation hands out of the same slot,
///   corrupting its contents/shape. Not caching means nothing survives the
///   <c>Reset()</c>, so the recycle is safe (issue #1668).</description></item>
/// </list>
/// <para><b>For Beginners:</b> When a model trains, every layer must remember the
/// inputs it saw so it can later compute how to improve (the "backward pass").
/// When the model is only making predictions there is no backward pass, so
/// remembering those inputs just wastes memory. This scope is a way to tell every
/// layer at once: "we're only predicting right now — don't bother remembering."</para>
/// <para>
/// Backed by <see cref="AsyncLocal{T}"/> so entering inference on one async flow
/// (say, a prediction request) never disables caching for a training loop running
/// in parallel on another thread. Scopes nest correctly: an inner scope restores
/// the enclosing scope's state on dispose rather than unconditionally clearing it.
/// </para>
/// </remarks>
internal static class InferenceMode
{
    private static readonly AsyncLocal<bool> _active = new();

    /// <summary>
    /// True when the current async flow is inside an <see cref="Enter"/> scope, i.e.
    /// no backward pass will run and manual-backward activation caches can be skipped.
    /// </summary>
    internal static bool IsActive => _active.Value;

    /// <summary>
    /// Enters an inference scope. Dispose the returned token (idiomatically with a
    /// <c>using</c> statement) to restore the previous state. Nesting is supported.
    /// </summary>
    /// <returns>A token whose disposal restores the prior inference-scope state.</returns>
    internal static IDisposable Enter()
    {
        bool previous = _active.Value;
        _active.Value = true;
        return new Scope(previous);
    }

    private sealed class Scope : IDisposable
    {
        private readonly bool _previous;
        private bool _disposed;

        internal Scope(bool previous) => _previous = previous;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _active.Value = _previous;
        }
    }
}
