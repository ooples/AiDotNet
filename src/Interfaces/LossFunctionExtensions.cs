using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

public static class LossFunctionExtensions
{
    public static T ComputeLoss<T>(
        this ILossFunction<T> lossFunction,
        Tensor<T> predicted,
        Tensor<T> actual)
    {
        if (lossFunction == null)
        {
            throw new ArgumentNullException(nameof(lossFunction));
        }
        if (predicted == null)
        {
            throw new ArgumentNullException(nameof(predicted));
        }
        if (actual == null)
        {
            throw new ArgumentNullException(nameof(actual));
        }

        return lossFunction.CalculateLoss(predicted.ToVector(), actual.ToVector());
    }

    public static Tensor<T> ComputeGradient<T>(
        this ILossFunction<T> lossFunction,
        Tensor<T> predicted,
        Tensor<T> actual)
    {
        if (lossFunction == null)
        {
            throw new ArgumentNullException(nameof(lossFunction));
        }
        if (predicted == null)
        {
            throw new ArgumentNullException(nameof(predicted));
        }
        if (actual == null)
        {
            throw new ArgumentNullException(nameof(actual));
        }

        var derivative = lossFunction.CalculateDerivative(predicted.ToVector(), actual.ToVector());
        return new Tensor<T>(predicted.Shape, derivative);
    }
}
