namespace AiDotNet.Data.Transforms;

/// <summary>
/// A no-op transform that passes input through unchanged.
/// </summary>
/// <typeparam name="T">The data type.</typeparam>
/// <remarks>
/// <para>
/// Useful as a default or placeholder in transform pipelines where a transform
/// slot exists but no actual transformation is needed.
/// </para>
/// </remarks>
public class IdentityTransform<T> : ITransform<T, T>
{
    /// <inheritdoc/>
    public T Apply(T input)
    {
        return input;
    }
}
