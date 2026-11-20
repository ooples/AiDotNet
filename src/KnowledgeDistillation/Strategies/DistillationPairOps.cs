using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>Direct pair differences with bounded row and feature working buffers.</summary>
internal static class DistillationPairOps
{
    internal const int RowBlockSize = 32;
    internal const int FeatureBlockSize = 256;
    private static IEngine Engine => AiDotNetEngine.Current;

    // Callers pass at most RowBlockSize rows from either side. Accumulate feature chunks in double.
    internal static Tensor<double> DistanceBlock(Tensor<double> a, Tensor<double> b, bool manhattan = false)
    {
        int n = a.Shape[0], m = b.Shape[0], dimensions = a.Shape[1];
        var sum = new Tensor<double>([n, m]);
        for (int f = 0; f < dimensions; f += FeatureBlockSize)
        {
            int width = Math.Min(FeatureBlockSize, dimensions - f);
            var left = Engine.TensorNarrow(a, 1, f, width);
            var right = Engine.TensorNarrow(b, 1, f, width);
            var difference = Engine.TensorSubtract(
                Engine.TensorBroadcastTo(Engine.Reshape(left, [n, 1, width]), [n, m, width]),
                Engine.TensorBroadcastTo(Engine.Reshape(right, [1, m, width]), [n, m, width]));
            var terms = manhattan ? Engine.TensorAbs(difference) : Engine.TensorMultiply(difference, difference);
            sum = Engine.TensorAdd(sum, Engine.ReduceSum(terms, [2], keepDims: false));
        }
        return sum;
    }

    internal static Tensor<double> WeightedDifferences(Tensor<double> a, Tensor<double> b, Tensor<double> weights, bool manhattan)
    {
        int n = a.Shape[0], m = b.Shape[0], dimensions = a.Shape[1];
        var blocks = new List<Tensor<double>>();
        for (int f = 0; f < dimensions; f += FeatureBlockSize)
        {
            int width = Math.Min(FeatureBlockSize, dimensions - f);
            var left = Engine.TensorNarrow(a, 1, f, width);
            var right = Engine.TensorNarrow(b, 1, f, width);
            var difference = Engine.TensorSubtract(
                Engine.TensorBroadcastTo(Engine.Reshape(left, [n, 1, width]), [n, m, width]),
                Engine.TensorBroadcastTo(Engine.Reshape(right, [1, m, width]), [n, m, width]));
            if (manhattan) difference = Engine.TensorSign(difference);
            var terms = Engine.TensorMultiply(difference,
                Engine.TensorBroadcastTo(Engine.Reshape(weights, [n, m, 1]), [n, m, width]));
            blocks.Add(Engine.ReduceSum(terms, [1], keepDims: false));
        }
        return Engine.TensorConcatenate(blocks.ToArray(), 1);
    }
}
