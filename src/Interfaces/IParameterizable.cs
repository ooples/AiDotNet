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
    /// Gets the number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// This property returns the total count of trainable parameters in the model.
    /// It's useful for understanding model complexity and memory requirements.
    /// <para>
    /// <b>Limitation for foundation-scale models:</b> the return type is
    /// <see cref="int"/>, capped at ~2.1 B. Models like Sora, HiDream, SD3.5
    /// Large, GPT-3-class etc. exceed this limit. For accurate per-tensor
    /// counting on those models, iterate <see cref="GetParameterChunks"/>
    /// and sum lengths into a <see cref="long"/> accumulator. Mirrors
    /// PyTorch's <c>nn.Module.parameters()</c> generator pattern.
    /// </para>
    /// </remarks>
    int ParameterCount { get; }

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
