namespace AiDotNet.Helpers;

/// <summary>
/// Centralizes the int-narrowing guard used wherever a model's <c>long</c>
/// <c>ParameterCount</c> needs to fit into a flat <see cref="Vector{T}"/>-backed
/// buffer (which is bounded to <c>int.MaxValue</c> elements).
/// </summary>
/// <remarks>
/// <para>
/// The public <c>ParameterCount</c> on <c>NeuralNetworkBase&lt;T&gt;</c>,
/// <c>LoRAAdapterBase&lt;T&gt;</c>, and similar bases returns <c>long</c> so
/// modern parameter counts (PaLM-E, GPT-3-class, &gt;2.1B params) can be
/// reported without overflow. But the project still uses
/// <see cref="LinearAlgebra.Vector{T}"/> for the flat parameter buffer, and
/// that type's <c>Length</c> is <c>int</c>. Many model classes need to
/// allocate / size that flat buffer: those call sites need to narrow the
/// long count to int.
/// </para>
/// <para>
/// A naked unchecked cast at those sites would silently truncate for models
/// above the int boundary, producing a wrong-sized <c>Vector&lt;T&gt;</c>
/// that mis-slices the parameter buffer downstream (corrupted parameter
/// writes, layer-by-layer offset drift, hard-to-diagnose numerical bugs at
/// inference / training time). A blanket <c>checked</c> cast would surface
/// the issue at runtime as a generic <c>OverflowException</c> with no
/// actionable context. This helper produces an actionable
/// <see cref="System.InvalidOperationException"/> pointing the caller at
/// weight streaming / model splitting as the right fix, with the call-site
/// location preserved in the stack trace.
/// </para>
/// <para>
/// Usage pattern at the call site:
/// <code>
/// // Before (silent truncation hazard):
/// //   var parameters = new Vector&lt;T&gt;((int)ParameterCount);
/// // After (this helper):
/// var parameters = new Vector&lt;T&gt;(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
/// </code>
/// Centralizing the guard means every flat-buffer allocation across the
/// codebase hits the same error path and can be updated in one place if the
/// flat-Vector limit ever changes.
/// </para>
/// </remarks>
public static class ParameterCountHelper
{
    /// <summary>
    /// Narrows a <see cref="long"/> parameter count to <see cref="int"/> for
    /// allocating or sizing a flat <see cref="LinearAlgebra.Vector{T}"/>-backed
    /// parameter buffer. Throws <see cref="System.InvalidOperationException"/>
    /// with an actionable message if the model has more than
    /// <see cref="int.MaxValue"/> parameters, since the flat-vector path
    /// cannot represent that many in a single contiguous buffer.
    /// </summary>
    /// <param name="parameterCount">The model's long-typed parameter count.</param>
    /// <returns>The same count narrowed to <see cref="int"/>, when in range.</returns>
    /// <exception cref="System.InvalidOperationException">
    /// Thrown when <paramref name="parameterCount"/> exceeds
    /// <see cref="int.MaxValue"/>. The message identifies weight streaming
    /// and model splitting as the supported escape hatches; the call-site
    /// is identified by the stack trace.
    /// </exception>
    public static int ToFlatVectorSize(long parameterCount)
    {
        if (parameterCount < 0)
        {
            throw new System.ArgumentOutOfRangeException(
                nameof(parameterCount),
                parameterCount,
                "Parameter count cannot be negative.");
        }

        if (parameterCount > int.MaxValue)
        {
            throw new System.InvalidOperationException(
                $"Parameter count ({parameterCount:N0}) exceeds int32 capacity " +
                $"({int.MaxValue:N0}); cannot allocate a flat Vector<T>-backed " +
                $"parameter buffer this large. For models above this threshold, " +
                $"either enable weight streaming via " +
                $"AiModelBuilder.ConfigureWeightStreaming(...) (so parameters " +
                $"are loaded on demand from disk and never materialized as a " +
                $"single flat vector) or split the model across multiple " +
                $"network instances. The stack trace identifies the call site " +
                $"that hit the limit.");
        }

        return (int)parameterCount;
    }
}
