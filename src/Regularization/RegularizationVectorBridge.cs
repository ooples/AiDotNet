using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Regularization;

/// <summary>
/// Shared Vector↔TOutput bridge for invoking
/// <see cref="IRegularization{T, TInput, TOutput}.Regularize(TOutput, TOutput)"/>
/// from a hot-path call site that holds <see cref="Vector{T}"/>-shaped
/// gradient and parameter buffers.
/// </summary>
/// <remarks>
/// <para>
/// Used both by
/// <see cref="RegularizationBase{T, TInput, TOutput}.Regularize(Vector{T}, Vector{T})"/>'s
/// default fallback and by
/// <see cref="Optimizers.GradientBasedOptimizerBase{T, TInput, TOutput}"/>'s
/// external-IRegularization branch. Centralised here so a future
/// <c>TOutput</c> shape needs only one update site instead of two
/// independently-evolving copies.
/// </para>
/// <para>
/// <see cref="Tensor{T}.FromVector(Vector{T})"/> is zero-copy (wraps
/// the underlying Vector), so wrapping is allocation-free.
/// <see cref="Tensor{T}.ToVector"/> on unwrap allocates a new
/// <see cref="Vector{T}"/> and copies element-by-element, so callers
/// that own a Vector-direct regularizer implementation should prefer
/// that path (e.g., <see cref="RegularizationBase{T, TInput, TOutput}"/>'s
/// concrete subclasses all override
/// <see cref="RegularizationBase{T, TInput, TOutput}.Regularize(Vector{T}, Vector{T})"/>
/// to skip this bridge entirely). The bridge is the correct-but-slower
/// fallback for external <c>IRegularization</c> implementations that
/// don't extend <c>RegularizationBase</c>.
/// </para>
/// </remarks>
internal static class RegularizationVectorBridge<T, TInput, TOutput>
{
    /// <summary>
    /// Invokes <paramref name="regularizer"/>'s TOutput-typed
    /// <see cref="IRegularization{T, TInput, TOutput}.Regularize(TOutput, TOutput)"/>
    /// overload against <see cref="Vector{T}"/>-typed inputs, returning
    /// the regularized gradient as a flat <see cref="Vector{T}"/>.
    /// </summary>
    public static Vector<T> Invoke(
        IRegularization<T, TInput, TOutput> regularizer,
        Vector<T> gradient,
        Vector<T> coefficients)
    {
        TOutput gradientOut, coefficientsOut;
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            gradientOut = (TOutput)(object)gradient;
            coefficientsOut = (TOutput)(object)coefficients;
        }
        else if (typeof(TOutput) == typeof(Tensor<T>))
        {
            gradientOut = (TOutput)(object)Tensor<T>.FromVector(gradient);
            coefficientsOut = (TOutput)(object)Tensor<T>.FromVector(coefficients);
        }
        else
        {
            throw new System.InvalidOperationException(
                $"RegularizationVectorBridge<{typeof(T).Name}, {typeof(TInput).Name}, " +
                $"{typeof(TOutput).Name}>: supported TOutput shapes are " +
                $"Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>. Either " +
                "override Regularize(Vector<T>, Vector<T>) on the regularizer for " +
                "a direct Vector path, or contribute a bridge for this TOutput type.");
        }

        var result = regularizer.Regularize(gradientOut, coefficientsOut);
        if (result is Vector<T> vec) return vec;
        if (result is Tensor<T> tensor) return tensor.ToVector();
        throw new System.InvalidOperationException(
            $"RegularizationVectorBridge: regularizer returned unexpected type " +
            $"({result?.GetType().Name ?? "null"}); expected Vector<T> or Tensor<T>.");
    }
}
