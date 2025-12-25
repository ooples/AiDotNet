using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Represents a Lie algebra element backed by coordinate vectors.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface ILieAlgebra<T>
{
    Vector<T> Coordinates { get; }
}
