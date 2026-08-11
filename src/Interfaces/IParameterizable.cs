using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that have optimizable parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("Parameterizable")]
/// <remarks>
/// GetParameters, SetParameters and ParameterCount come from
/// <see cref="IParameterSource{T}"/> -- the minimal contract shared with components that
/// own parameters without being full models (encoders, projectors, layers).
/// </remarks>
public interface IParameterizable<T, TInput, TOutput> : IParameterSource<T>
{
    /// <summary>
    /// Yields the model's persistent state as bounded tensors, in the exact order used by
    /// <see cref="IParameterSource{T}.GetParameters"/>. Callers iterate without materializing one
    /// aggregate <c>Vector&lt;T&gt;</c>. Tensor-backed sources normally return live references; classical
    /// sources may return payload chunks that are committed through <see cref="SetParameterChunks"/>.
    /// Foundation-scale models
    /// (Sora 5 B+, HiDream 8 B+, GPT-3-class 175 B+) cannot fit a single
    /// flat vector but each individual weight tensor is well below
    /// <see cref="int"/>.MaxValue elements.
    /// </summary>
    /// <returns>An enumerable of ordered model-state payload tensors.</returns>
    /// <remarks>
    /// Implementations deriving from the framework model bases receive this surface from the
    /// generated parameter registry. Use <see cref="AiDotNet.Models.Parameters.IParameterChunkSource{T}"/>
    /// when semantic roles are required: it distinguishes optimizer-owned trainable tensors from
    /// learned or frozen persistent state. Used by:
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
    IEnumerable<Tensor<T>> GetParameterChunks()
    {
        // Universal bounded-model fallback. Tensor-backed framework bases override this with
        // zero-copy, role-aware chunks; a classical model whose state lives in scalars, arrays,
        // matrices, trees, or tables still receives one exact payload chunk automatically.
        // This keeps chunk parity a property of the interface rather than boilerplate repeated by
        // every non-neural base hierarchy.
        var flat = GetParameters();
        if (flat.Length == 0) yield break;
        yield return new Tensor<T>(new[] { flat.Length }, flat);
    }

    /// <summary>
    /// Streaming counterpart to <see cref="SetParameters"/>: assigns the model's persistent
    /// state from a sequence of chunks supplied in the SAME order
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
        if (chunks is null) throw new System.ArgumentNullException(nameof(chunks));
        var buffered = new List<Tensor<T>>();
        long total = 0;
        foreach (var chunk in chunks)
        {
            if (chunk is null)
                throw new System.ArgumentException("Chunk sequence contains a null tensor.", nameof(chunks));
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
