using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Data;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

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
        var random = RandomHelper.CreateSeededRandom(seed);
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

        var random = RandomHelper.CreateSeededRandom(seed);
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

    internal static Tensor<T> CreateReturnsMatrix<T>(int samples, int assets, int seed = 42)
    {
        int safeSamples = Math.Max(1, samples);
        int safeAssets = Math.Max(1, assets);
        return CreateRandomTensor<T>(new[] { safeSamples, safeAssets }, seed);
    }

    internal static Tensor<T> CreateReturnsSeries<T>(int batchSize, int sequenceLength, int assets, int seed = 42)
    {
        int safeBatch = Math.Max(1, batchSize);
        int safeSequence = Math.Max(1, sequenceLength);
        int safeAssets = Math.Max(1, assets);
        return CreateRandomTensor<T>(new[] { safeBatch, safeSequence, safeAssets }, seed);
    }

    internal static Tensor<T> CreateFactorSeries<T>(int batchSize, int sequenceLength, int factors, int seed = 42)
    {
        int safeBatch = Math.Max(1, batchSize);
        int safeSequence = Math.Max(1, sequenceLength);
        int safeFactors = Math.Max(1, factors);
        return CreateRandomTensor<T>(new[] { safeBatch, safeSequence, safeFactors }, seed);
    }

    internal static Tensor<T> CreateFactorExposure<T>(int batchSize, int factors, int seed = 42)
    {
        int safeBatch = Math.Max(1, batchSize);
        int safeFactors = Math.Max(1, factors);
        return CreateRandomTensor<T>(new[] { safeBatch, safeFactors }, seed);
    }

    internal static Tensor<T> CreateScenarioMatrix<T>(int scenarios, int assets, int seed = 42)
    {
        int safeScenarios = Math.Max(1, scenarios);
        int safeAssets = Math.Max(1, assets);
        return CreateRandomTensor<T>(new[] { safeScenarios, safeAssets }, seed);
    }

    internal static Tensor<T> CreateExpectedReturns<T>(int assets, int seed = 42)
    {
        int safeAssets = Math.Max(1, assets);
        return CreateRandomTensor<T>(new[] { safeAssets }, seed);
    }

    internal static Tensor<T> CreateUniformWeights<T>(int assets)
    {
        int safeAssets = Math.Max(1, assets);
        var numOps = MathHelper.GetNumericOperations<T>();
        T weight = numOps.FromDouble(1.0 / safeAssets);
        var data = new T[safeAssets];

        for (int i = 0; i < safeAssets; i++)
        {
            data[i] = weight;
        }

        return new Tensor<T>(new[] { safeAssets }, new Vector<T>(data));
    }

    internal static Tensor<T> CreateDiagonalMatrix<T>(int size, double diagonalValue = 0.1)
    {
        int safeSize = Math.Max(1, size);
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = new T[safeSize * safeSize];
        T diag = numOps.FromDouble(diagonalValue);

        for (int i = 0; i < safeSize; i++)
        {
            data[(i * safeSize) + i] = diag;
        }

        return new Tensor<T>(new[] { safeSize, safeSize }, new Vector<T>(data));
    }

    internal static Tensor<T> CreatePriceTensor<T>(int steps, int assets, double startPrice = 100.0)
    {
        int safeSteps = Math.Max(1, steps);
        int safeAssets = Math.Max(1, assets);
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = new T[safeSteps * safeAssets];

        for (int t = 0; t < safeSteps; t++)
        {
            for (int a = 0; a < safeAssets; a++)
            {
                data[(t * safeAssets) + a] = numOps.FromDouble(startPrice + t + a);
            }
        }

        return new Tensor<T>(new[] { safeSteps, safeAssets }, new Vector<T>(data));
    }

    internal static System.Collections.Generic.List<MarketDataPoint<T>> CreateMarketSeries<T>(int count, DateTime? start = null)
    {
        int safeCount = Math.Max(1, count);
        var numOps = MathHelper.GetNumericOperations<T>();
        var startTime = start ?? DateTime.UtcNow;
        var series = new System.Collections.Generic.List<MarketDataPoint<T>>(safeCount);

        for (int i = 0; i < safeCount; i++)
        {
            T price = numOps.FromDouble(100 + i);
            T volume = numOps.FromDouble(1000 + (i * 10));
            series.Add(new MarketDataPoint<T>(startTime.AddMinutes(i), price, price, price, price, volume));
        }

        return series;
    }
}
