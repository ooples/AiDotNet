using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Defines core operations for a Lie group with vector-valued algebra coordinates.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TElement">The group element type.</typeparam>
public interface ILieGroup<T, TElement>
{
    TElement Identity { get; }

    TElement Compose(TElement a, TElement b);

    TElement Inverse(TElement value);

    TElement Exp(Vector<T> tangent);

    Vector<T> Log(TElement value);

    Matrix<T> Adjoint(TElement value);
}
