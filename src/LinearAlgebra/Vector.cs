
namespace AiDotNet.LinearAlgebra;

public class Vector<T> : VectorBase<T>
{
    public Vector(IEnumerable<T> values) : base(values)
    {
    }
}
