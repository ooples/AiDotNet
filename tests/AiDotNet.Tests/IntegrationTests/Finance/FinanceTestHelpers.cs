using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNet.Tests.IntegrationTests.Finance;

internal static class FinanceTestHelpers
{
    internal static NeuralNetworkArchitecture<T> CreateArchitecture<T>(int inputSize, int outputSize)
    {
        int safeInputSize = Math.Max(1, inputSize);
        int safeOutputSize = Math.Max(1, outputSize);

        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: safeInputSize,
            outputSize: safeOutputSize);
    }

    internal static Tensor<T> CreateRandomTensor<T>(int[] shape, int seed = 42)
    {
        var random = new Random(seed);
        int length = shape.Aggregate(1, (acc, dim) => acc * dim);
        var data = new T[length];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < length; i++)
        {
            data[i] = numOps.FromDouble((random.NextDouble() * 2.0) - 1.0);
        }

        return new Tensor<T>(shape, new Vector<T>(data));
    }

    internal static Tensor<T> CreateTimeSeriesInput<T>(int batchSize, int sequenceLength, int numFeatures, int seed = 42)
    {
        int safeBatch = Math.Max(1, batchSize);
        int safeSequence = Math.Max(1, sequenceLength);
        int safeFeatures = Math.Max(1, numFeatures);

        return CreateRandomTensor<T>(new[] { safeBatch, safeSequence, safeFeatures }, seed);
    }

    internal static Tensor<T> CreateTokenTensor<T>(int batchSize, int sequenceLength, int vocabularySize, int seed = 42)
    {
        int safeBatch = Math.Max(1, batchSize);
        int safeSequence = Math.Max(1, sequenceLength);
        int safeVocab = Math.Max(2, vocabularySize);

        var random = new Random(seed);
        var data = new T[safeBatch * safeSequence];
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = numOps.FromDouble(random.Next(safeVocab));
        }

        return new Tensor<T>(new[] { safeBatch, safeSequence }, new Vector<T>(data));
    }

    internal static Vector<T> CreateRandomVector<T>(int length, int seed = 42)
    {
        int safeLength = Math.Max(1, length);
        var tensor = CreateRandomTensor<T>(new[] { safeLength }, seed);
        return tensor.ToVector();
    }
}
