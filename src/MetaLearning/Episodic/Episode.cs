namespace AiDotNet.MetaLearning.Episodic;

using AiDotNet.LinearAlgebra;

public sealed class Episode<T>
{
    public Tensor<T> SupportInputs { get; }
    public Tensor<T> SupportLabels { get; }
    public Tensor<T> QueryInputs { get; }
    public Tensor<T> QueryLabels { get; }

    public Episode(Tensor<T> sX, Tensor<T> sY, Tensor<T> qX, Tensor<T> qY)
    {
        SupportInputs = sX; SupportLabels = sY; QueryInputs = qX; QueryLabels = qY;
    }
}

