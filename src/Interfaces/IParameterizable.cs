using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that have optimizable parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("Parameterizable")]
public interface IParameterizable<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the parameters that can be optimized.
    /// </summary>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">The parameter vector to set.</param>
    /// <remarks>
    /// This method allows direct modification of the model's internal parameters.
    /// This is useful for optimization algorithms that need to update parameters iteratively.
    /// If the length of <paramref name="parameters"/> does not match <see cref="ParameterCount"/>,
    /// an <see cref="ArgumentException"/> should be thrown.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when the length of <paramref name="parameters"/> does not match <see cref="ParameterCount"/>.
    /// </exception>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the total number of trainable parameters in the model. Return
    /// type is <see cref="long"/> (int64), matching PyTorch's
    /// <c>c10::TensorImpl::numel()</c> int64_t convention so foundation-scale
    /// models like Sora, HiDream Full, SD3.5 Large, HunyuanVideo, Flux 2,
    /// etc. (all &gt;2.1 B parameters) report accurately without overflow.
    /// </summary>
    /// <remarks>
    /// Per-tensor weights still fit in <see cref="int"/> (Vector{T}.Length
    /// is int and a single tensor doesn't exceed int.MaxValue elements in
    /// any current architecture). The aggregate uses a long accumulator,
    /// computed by summing each chunk's <c>Length</c> from
    /// <see cref="GetParameterChunks"/>.
    /// </remarks>
    long ParameterCount { get; }

    /// <summary>
    /// Yields the model's trainable weight tensors as references — zero-copy,
    /// streaming. Callers iterate per-tensor without ever materializing a
    /// flat <c>Vector&lt;T&gt;</c> of all parameters. Mirrors PyTorch's
    /// <c>nn.Module.parameters()</c> generator: foundation-scale models
    /// (Sora 5 B+, HiDream 8 B+, GPT-3-class 175 B+) cannot fit a single
    /// flat vector but each individual weight tensor is well below
    /// <see cref="int"/>.MaxValue elements.
    /// </summary>
    /// <returns>An enumerable of trainable weight tensors (no copy).</returns>
    /// <remarks>
    /// Default implementation yields nothing — concrete implementations
    /// should override to yield each layer's <c>GetTrainableParameters</c>
    /// or equivalent per-tensor weight references. Used by:
    /// <list type="bullet">
    /// <item>Foundation-scale parameter counting (sum lengths as
    /// <see cref="long"/> to count past <see cref="int"/>.MaxValue)</item>
    /// <item>Streaming serialization without flat-vector allocation</item>
    /// <item>PyTorch-compatibility shims (state_dict-style export)</item>
    /// </list>
    /// </remarks>
#if !NETFRAMEWORK
    // Chunked-API contract is .NET-Standard-2.1+ / .NET 10 only. Default
    // interface methods need runtime dispatch support that .NET Framework
    // 4.7.1 doesn't provide, so we omit this from the IParameterizable
    // contract on net471 entirely. Concrete types (e.g., NeuralNetworkBase,
    // ModelBase) still expose the same `GetParameterChunks()` method as
    // a regular virtual on both targets — net471 callers just access it
    // through the concrete type instead of the interface.
    IEnumerable<Tensor<T>> GetParameterChunks() => System.Linq.Enumerable.Empty<Tensor<T>>();

    /// <summary>
    /// Streaming counterpart to <see cref="SetParameters"/>: assigns the model's trainable
    /// weight tensors from a sequence of per-tensor chunks supplied in the SAME order
    /// <see cref="GetParameterChunks"/> yields them, WITHOUT ever materializing a flat
    /// <c>Vector&lt;T&gt;</c> of all parameters. Foundation-scale models (&gt;2.1 B params) cannot
    /// round-trip through the flat <see cref="SetParameters"/> path — the aggregate overflows
    /// <c>Vector.Length</c>'s <see cref="int"/> contract and OOMs the host — so they override this
    /// to consume one chunk at a time.
    /// </summary>
    /// <remarks>
    /// Default implementation buffers the chunks into a single flat <c>Vector&lt;T&gt;</c> and
    /// delegates to <see cref="SetParameters"/>. This is correct and back-compatible for tractable
    /// models; only foundation-scale types need to override it to stay flat-free.
    /// </remarks>
    void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        var buffered = new List<Tensor<T>>();
        long total = 0;
        foreach (var chunk in chunks)
        {
            buffered.Add(chunk);
            total += chunk.Length;
        }

        var flat = new Vector<T>(checked((int)total));
        int offset = 0;
        foreach (var chunk in buffered)
        {
            var v = chunk.ToVector();
            for (int i = 0; i < v.Length; i++) flat[offset++] = v[i];
        }

        SetParameters(flat);
    }
#endif

    /// <summary>
    /// Gets whether this model supports direct parameter-based initialization.
    /// </summary>
    /// <remarks>
    /// Models that learn their structure during training (decision trees, ensemble methods, clustering)
    /// may not support having random parameters injected before training. The optimizer uses this
    /// property to decide whether to call <see cref="SetParameters"/> during random initialization.
    /// The default implementation returns <c>true</c> when <see cref="ParameterCount"/> is greater than zero.
    /// </remarks>
#if NETFRAMEWORK
    bool SupportsParameterInitialization { get; }
#else
    bool SupportsParameterInitialization => ParameterCount > 0;
#endif

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    /// <summary>
    /// Sanitizes random parameters to satisfy model-specific constraints.
    /// Called by the optimizer after generating random parameter vectors.
    /// </summary>
    /// <param name="parameters">The randomly generated parameter vector.</param>
    /// <returns>A parameter vector that satisfies model constraints (e.g., sorted thresholds for ordinal models).</returns>
    /// <remarks>
    /// The default implementation returns the parameters unchanged. Override this in models
    /// that have structural constraints on their parameters (e.g., monotonically increasing
    /// thresholds in ordinal regression, non-negative weights in NMF, etc.).
    /// </remarks>
#if NETFRAMEWORK
    Vector<T> SanitizeParameters(Vector<T> parameters);
#else
    Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;
#endif
}
