namespace AiDotNet.Models.Generative.Diffusion;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

public interface IDiffusionModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
}

