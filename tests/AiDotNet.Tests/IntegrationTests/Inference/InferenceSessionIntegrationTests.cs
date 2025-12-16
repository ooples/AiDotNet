using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Normalizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

public class InferenceSessionIntegrationTests
{
    private const float Tolerance = 1e-4f;
    private const int SequenceLength = 1;
    private const int EmbeddingDimension = 8;
    private const int HeadCount = 2;
    private const int FlatSize = SequenceLength * EmbeddingDimension;

    [Fact]
    public void PredictionModelResult_Predict_IsStateless_WhenInferenceOptimizationsConfigured()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = false,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        var token = CreateTokenTensor(1.0f);

        var y1 = result.Predict(token);
        var y2 = result.Predict(token);

        AssertTensorsEqual(y1, y2, Tolerance);
    }

    [Fact]
    public void BeginInferenceSession_SequencesAreIndependent()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = false,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        var token = CreateTokenTensor(0.75f);
        var tokenForB = CreateTokenTensor(0.75f);
        var tokenFresh = CreateTokenTensor(0.75f);

        using var session = result.BeginInferenceSession();

        var seqA = session.CreateSequence();
        var seqB = session.CreateSequence();
        var seqFresh = session.CreateSequence();

        var a1 = seqA.Predict(token);
        var statsAfterFirst = seqA.GetInferenceStatistics();
        var lengthsAfterFirst = (int[])statsAfterFirst["KVCache_SequenceLengths"];
        int lenAfterFirst = lengthsAfterFirst[0];

        var b1 = seqB.Predict(tokenForB);
        var fresh1 = seqFresh.Predict(tokenFresh);

        AssertTensorsEqual(a1, b1, Tolerance);
        AssertTensorsEqual(a1, fresh1, Tolerance);

        var freshStatsAfterFirst = seqFresh.GetInferenceStatistics();
        var freshLengthsAfterFirst = (int[])freshStatsAfterFirst["KVCache_SequenceLengths"];
        int freshLenAfterFirst = freshLengthsAfterFirst[0];
        Assert.Equal(lenAfterFirst, freshLenAfterFirst);

        _ = seqA.Predict(CreateTokenTensor(-0.25f));

        var statsAfterSecond = seqA.GetInferenceStatistics();
        var lengthsAfterSecond = (int[])statsAfterSecond["KVCache_SequenceLengths"];
        Assert.True(lengthsAfterSecond[0] > lenAfterFirst, $"Expected KV-cache length to grow, but got {lenAfterFirst} -> {lengthsAfterSecond[0]}");

        // Fresh sequence should grow independently when it advances.
        _ = seqFresh.Predict(CreateTokenTensor(-0.25f));
        var freshStatsAfterSecond = seqFresh.GetInferenceStatistics();
        var freshLengthsAfterSecond = (int[])freshStatsAfterSecond["KVCache_SequenceLengths"];
        Assert.True(
            freshLengthsAfterSecond[0] > freshLenAfterFirst,
            $"Expected fresh KV-cache length to grow, but got {freshLenAfterFirst} -> {freshLengthsAfterSecond[0]}");
    }

    [Fact]
    public void BeginInferenceSession_ResetRestoresInitialSequenceState()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = false,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        var token1 = CreateTokenTensor(0.25f);
        var token2 = CreateTokenTensor(0.5f);

        using var session = result.BeginInferenceSession();
        var seq = session.CreateSequence();

        var y1 = seq.Predict(token1);
        _ = seq.Predict(token2);

        seq.Reset();

        var y1AfterReset = seq.Predict(token1);
        AssertTensorsEqual(y1, y1AfterReset, Tolerance);
    }

    [Fact]
    public void NeuralNetworkBase_Clone_DoesNotShareParameters()
    {
        var model = CreateDeterministicAttentionOnlyModel();
        var clone = (NeuralNetworkBase<float>)model.Clone();

        // Clone should preserve parameters exactly (deep copy via serialization/deserialization).
        Assert.Equal(model.GetParameters().Length, clone.GetParameters().Length);
        for (int i = 0; i < model.GetParameters().Length; i++)
        {
            Assert.True(
                Math.Abs(model.GetParameters()[i] - clone.GetParameters()[i]) <= Tolerance,
                $"Parameter mismatch at {i}: {model.GetParameters()[i]} != {clone.GetParameters()[i]}");
        }

        var cloneParams = clone.GetParameters();
        cloneParams[0] += 1.0f;
        clone.UpdateParameters(cloneParams);

        Assert.NotEqual(model.GetParameters()[0], clone.GetParameters()[0]);
    }

    private static PredictionModelResult<float, Tensor<float>, Tensor<float>> CreateDeterministicResult(InferenceOptimizationConfig config)
    {
        var model = CreateDeterministicAttentionOnlyModel();

        var optimization = new OptimizationResult<float, Tensor<float>, Tensor<float>>
        {
            BestSolution = model
        };

        var normalization = new NormalizationInfo<float, Tensor<float>, Tensor<float>>
        {
            Normalizer = new NoNormalizer<float, Tensor<float>, Tensor<float>>(),
            YParams = new NormalizationParameters<float> { Method = NormalizationMethod.None }
        };

        return new PredictionModelResult<float, Tensor<float>, Tensor<float>>(
            optimization,
            normalization,
            inferenceOptimizationConfig: config);
    }

    private static NeuralNetworkBase<float> CreateDeterministicAttentionOnlyModel()
    {
        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(FlatSize),
            new ReshapeLayer<float>(new[] { FlatSize }, new[] { SequenceLength, EmbeddingDimension }),
            new MultiHeadAttentionLayer<float>(
                sequenceLength: SequenceLength,
                embeddingDimension: EmbeddingDimension,
                headCount: HeadCount,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
            ,
            new FlattenLayer<float>(new[] { SequenceLength, EmbeddingDimension }),
            new DenseLayer<float>(FlatSize, FlatSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: FlatSize,
            outputSize: FlatSize,
            layers: layers);

        var model = new NeuralNetwork<float>(architecture);

        var p = model.GetParameters();
        var deterministic = new float[p.Length];
        for (int i = 0; i < deterministic.Length; i++)
        {
            deterministic[i] = ((i % 23) - 11) / 11.0f;
        }
        model.UpdateParameters(new Vector<float>(deterministic));

        return model;
    }

    private static Tensor<float> CreateTokenTensor(float scalar)
    {
        var t = new Tensor<float>(new[] { 1, FlatSize });
        for (int i = 0; i < t.Length; i++)
        {
            t[i] = scalar + (i * 0.01f);
        }
        return t;
    }

    private static void AssertTensorsEqual(Tensor<float> a, Tensor<float> b, float tolerance)
    {
        Assert.Equal(a.Shape, b.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(Math.Abs(a[i] - b[i]) <= tolerance, $"Index {i}: {a[i]} != {b[i]}");
        }
    }

    private static void AssertTensorsNotEqual(Tensor<float> a, Tensor<float> b, float minAbsDiff)
    {
        Assert.Equal(a.Shape, b.Shape);

        float maxAbs = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float abs = Math.Abs(a[i] - b[i]);
            if (abs > maxAbs)
            {
                maxAbs = abs;
            }
        }

        Assert.True(maxAbs >= minAbsDiff, $"Expected tensors to differ by at least {minAbsDiff}, but max diff was {maxAbs}");
    }
}
