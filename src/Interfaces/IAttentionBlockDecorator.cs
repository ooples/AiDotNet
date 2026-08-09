namespace AiDotNet.Interfaces;

/// <summary>
/// A layer that WRAPS an attention block, adding something to its output without replacing how the
/// block itself is invoked.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This exists because invoking an attention block is not uniform: self-attention takes
/// <c>Forward(x)</c>, while cross-attention needs its conditioning passed through a type-specific
/// overload (<c>ForwardWithContext</c> or a two-argument <c>Forward</c>). A decorator that only
/// overrides <c>Forward(x)</c> therefore matches none of those type tests and falls through to the
/// single-argument path, which SILENTLY DROPS the conditioning — the wrapped model still runs, and
/// produces plausible-looking output computed without the text embedding at all.
/// </para>
/// <para>
/// Splitting the wrapper into "how do I call the inner block" (the caller's existing dispatch, applied
/// to <see cref="Inner"/>) and "what do I add afterwards" (<see cref="PostProcess"/>) keeps the inner
/// invocation exactly as it was. Callers should prefer this over a type test against any concrete
/// decorator, so new adapters need no changes at the call site.
/// </para>
/// </remarks>
public interface IAttentionBlockDecorator<T> : ILayer<T>
{
    /// <summary>
    /// Gets the wrapped block, which should be invoked using whatever dispatch the caller would have
    /// used had it not been wrapped.
    /// </summary>
    ILayer<T> Inner { get; }

    /// <summary>
    /// Transforms the wrapped block's output. Must be a pure function of its argument and the
    /// decorator's own state.
    /// </summary>
    /// <param name="innerOutput">Whatever the inner block produced.</param>
    /// <returns>The decorated output.</returns>
    Tensor<T> PostProcess(Tensor<T> innerOutput);
}
