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

    [Fact]
    public void PredictionModelResult_Predict_IsStateless_WhenInferenceOptimizationsConfigured()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = true,
                EnableKVCache = true,
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
                EnableFlashAttention = true,
                EnableKVCache = true,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        var token1 = CreateTokenTensor(1.0f);
        var token2 = CreateTokenTensor(-0.5f);

        using var session = result.BeginInferenceSession();

        var seqA = session.CreateSequence();
        var seqB = session.CreateSequence();
        var seqFresh = session.CreateSequence();

        var a1 = seqA.Predict(token1);
        var a2 = seqA.Predict(token2);

        var b1 = seqB.Predict(token1);
        var fresh2 = seqFresh.Predict(token2);

        AssertTensorsEqual(a1, b1, Tolerance);
        AssertTensorsNotEqual(fresh2, a2, minAbsDiff: 1e-6f);
    }

    [Fact]
    public void BeginInferenceSession_ResetRestoresInitialSequenceState()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = true,
                EnableKVCache = true,
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
            new MultiHeadAttentionLayer<float>(
                sequenceLength: 8,
                embeddingDimension: 8,
                headCount: 2,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 8,
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
        var t = new Tensor<float>(new[] { 1, 1, 8 });
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
