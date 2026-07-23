namespace AiDotNet.Tests.UnitTests.Serialization;

using AiDotNet.Tensors.LinearAlgebra;

internal sealed class NoVectorConstructorTensor<T> : Tensor<T>
{
    public NoVectorConstructorTensor(int[] dimensions) : base(dimensions)
    {
    }
}

