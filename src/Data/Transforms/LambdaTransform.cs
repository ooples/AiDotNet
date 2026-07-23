namespace AiDotNet.Data.Transforms;

/// <summary>
/// Wraps a <see cref="Func{TInput, TOutput}"/> delegate as an <see cref="ITransform{TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="TInput">The type of the input data.</typeparam>
/// <typeparam name="TOutput">The type of the output data.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Use this when you need a quick, inline transform
/// without creating a full class:
/// <code>
/// var squareTransform = new LambdaTransform&lt;double, double&gt;(x => x * x);
/// double result = squareTransform.Apply(5.0); // 25.0
/// </code>
/// </para>
/// </remarks>
public class LambdaTransform<TInput, TOutput> : ITransform<TInput, TOutput>
{
    private readonly Func<TInput, TOutput> _func;

    /// <summary>
    /// Creates a new lambda transform from the given function.
    /// </summary>
    /// <param name="func">The function to wrap as a transform.</param>
    public LambdaTransform(Func<TInput, TOutput> func)
    {
        Guard.NotNull(func);
        _func = func;
    }

    /// <inheritdoc/>
    public TOutput Apply(TInput input)
    {
        return _func(input);
    }
}
