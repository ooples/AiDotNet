using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Normalizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

[Collection(AiDotNet.Tests.TestInfrastructure.DiagnosticsEnvironmentCollection.Name)]
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
    public void PredictionModelResult_SerializeDeserialize_PreservesInferenceOptimizationConfig()
    {
        var config = new InferenceOptimizationConfig
        {
            EnableFlashAttention = false,
            EnableKVCache = true,
            EnablePagedKVCache = false,
            AttentionMasking = AttentionMaskingMode.Auto
        };

        var original = CreateDeterministicResult(config);
        var bytes = original.Serialize();

        var loaded = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = true,
                EnableKVCache = false,
                EnablePagedKVCache = true,
                AttentionMasking = AttentionMaskingMode.Causal
            });

        loaded.Deserialize(bytes);

        var loadedConfig = loaded.GetInferenceOptimizationConfigForServing();
        Assert.NotNull(loadedConfig);
        Assert.Equal(config.EnableFlashAttention, loadedConfig!.EnableFlashAttention);
        Assert.Equal(config.EnableKVCache, loadedConfig.EnableKVCache);
        Assert.Equal(config.EnablePagedKVCache, loadedConfig.EnablePagedKVCache);
        Assert.Equal(config.AttentionMasking, loadedConfig.AttentionMasking);
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
    public async Task BeginInferenceSession_ConcurrentPredict_MultipleSequences_DoesNotThrow()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = true,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        using var session = result.BeginInferenceSession();
        var seqA = session.CreateSequence();
        var seqB = session.CreateSequence();

        var tasks = Enumerable.Range(0, 20)
            .Select(i => Task.Run(() =>
            {
                var t = CreateTokenTensor(0.1f + (i * 0.01f));
                _ = (i % 2 == 0 ? seqA : seqB).Predict(t);
            }))
            .ToArray();

        await Task.WhenAll(tasks);

        var statsA = seqA.GetInferenceStatistics();
        var statsB = seqB.GetInferenceStatistics();
        Assert.True((int)statsA["PagedAttentionLayerCount"] > 0);
        Assert.True((int)statsB["PagedAttentionLayerCount"] > 0);
    }

    [Fact]
    public void BeginInferenceSession_KVCacheQuantization_Int8_UsesQuantizedStorage()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = false,
                KVCacheQuantization = KVCacheQuantizationMode.Int8,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        using var session = result.BeginInferenceSession();
        var seq = session.CreateSequence();

        _ = seq.Predict(CreateTokenTensor(0.1f));

        var stats = seq.GetInferenceStatistics();
        Assert.True(stats.TryGetValue("KVCache_DataType", out var dataType));
        Assert.Equal("Int8", dataType);
        Assert.True(stats.TryGetValue("KVCache_UseInt8Storage", out var useInt8));
        Assert.True((bool)useInt8);
    }

    [Fact]
    public void BeginInferenceSession_KVCachePrecision_Auto_UsesFloat16Storage_ForFloatModel()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = false,
                KVCachePrecision = KVCachePrecisionMode.Auto,
                KVCacheQuantization = KVCacheQuantizationMode.None,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        using var session = result.BeginInferenceSession();
        var seq = session.CreateSequence();

        _ = seq.Predict(CreateTokenTensor(0.1f));

        var stats = seq.GetInferenceStatistics();
        Assert.True(stats.TryGetValue("KVCache_DataType", out var dataType));
        Assert.Equal("Float16", dataType);
        Assert.True(stats.TryGetValue("KVCache_UseFp16Storage", out var useFp16));
        Assert.True((bool)useFp16);
        Assert.True(stats.TryGetValue("KVCache_UseInt8Storage", out var useInt8));
        Assert.False((bool)useInt8);
    }

    [Fact]
    public void BeginInferenceSession_SpeculativeDecoding_Configured_DoesNotRunDuringPredict()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = false,
                EnablePagedKVCache = false,
                EnableSpeculativeDecoding = true,
                DraftModelType = DraftModelType.NGram,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        using var session = result.BeginInferenceSession();
        var seq = session.CreateSequence();

        _ = seq.Predict(CreateTokenTensor(0.1f));

        var stats = seq.GetInferenceStatistics();
        Assert.True(stats.TryGetValue("SpeculativeDecodingEnabled", out var enabled));
        Assert.True((bool)enabled);
        Assert.False(stats.ContainsKey("DraftModelType"));
        Assert.False(stats.ContainsKey("SpeculationDepth"));
    }

    [Fact]
    public void BeginInferenceSession_PagedKVCache_IsInitialized_WhenEnabled()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = true,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        using var session = result.BeginInferenceSession();
        var seq = session.CreateSequence();

        _ = seq.Predict(CreateTokenTensor(0.1f));

        var stats = seq.GetInferenceStatistics();
        Assert.True(stats.TryGetValue("PagedKVCacheInitialized", out var initialized));
        Assert.True((bool)initialized);
        Assert.True(stats.TryGetValue("PagedAttentionLayerCount", out var count));
        Assert.True((int)count > 0);
    }

    [Fact]
    public void BeginInferenceSession_PagedAttention_WOQ_IsEnabled_WhenConfigured()
    {
        var result = CreateDeterministicResult(
            new InferenceOptimizationConfig
            {
                EnableFlashAttention = false,
                EnableKVCache = true,
                EnablePagedKVCache = true,
                EnableWeightOnlyQuantization = true,
                AttentionMasking = AttentionMaskingMode.Auto
            });

        using var session = result.BeginInferenceSession();
        var seq = session.CreateSequence();

        _ = seq.Predict(CreateTokenTensor(0.2f));

        var stats = seq.GetInferenceStatistics();
        Assert.True(stats.TryGetValue("PagedAttentionWeightOnlyQuantizationEnabled", out var enabled));
        Assert.True((bool)enabled);
    }

    [Fact]
    public void BeginInferenceSession_MultiLoRA_TaskSelection_IsIsolatedPerSequence()
    {
        var config = new InferenceOptimizationConfig
        {
            EnableFlashAttention = false,
            EnableKVCache = false,
            EnablePagedKVCache = false,
            EnableSpeculativeDecoding = false,
            EnableBatching = false
        };

        var model = CreateDeterministicMultiLoRAModel();
        var result = CreateDeterministicResultWithModel(config, model);

        var token = CreateTokenTensor(0.25f);

        using var session = result.BeginInferenceSession();
        var seqA = session.CreateSequence("taskA");
        var seqB = session.CreateSequence("taskB");

        var yA = seqA.Predict(token);
        var yB = seqB.Predict(token);

        AssertTensorsNotEqual(yA, yB, minAbsDiff: 1e-3f);

        seqA.SetMultiLoRATask("taskB");
        var yA2 = seqA.Predict(token);
        AssertTensorsNotEqual(yA, yA2, minAbsDiff: 1e-3f);
    }

    [Fact]
    public void BeginInferenceSession_MultiLoRA_TaskSwitch_ResetsKVCacheState_ForSameSequence()
    {
        var originalDiagnostics = Environment.GetEnvironmentVariable("AIDOTNET_DIAGNOSTICS");

        var config = new InferenceOptimizationConfig
        {
            EnableFlashAttention = false,
            EnableKVCache = true,
            EnablePagedKVCache = false,
            AttentionMasking = AttentionMaskingMode.Auto
        };

        var model = CreateDeterministicAttentionWithMultiLoRAModel();
        var result = CreateDeterministicResultWithModel(config, model);

        try
        {
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", "1");
            AiDotNet.Helpers.InferenceDiagnostics.Clear();

            using var session = result.BeginInferenceSession();
            var seq = session.CreateSequence("taskA");

            var token1 = CreateTokenTensor(0.25f);
            var token2 = CreateTokenTensor(0.5f);

            _ = seq.Predict(token1);
            var statsAfterFirst = seq.GetInferenceStatistics();
            var lenAfterFirst = ((int[])statsAfterFirst["KVCache_SequenceLengths"])[0];

            _ = seq.Predict(token2);
            var statsAfterSecond = seq.GetInferenceStatistics();
            var lenAfterSecond = ((int[])statsAfterSecond["KVCache_SequenceLengths"])[0];
            Assert.True(lenAfterSecond > lenAfterFirst, $"Expected KV-cache length to grow, but got {lenAfterFirst} -> {lenAfterSecond}");

            seq.SetMultiLoRATask("taskB");
            _ = seq.Predict(token1);
            var statsAfterSwitch = seq.GetInferenceStatistics();
            var lenAfterSwitch = ((int[])statsAfterSwitch["KVCache_SequenceLengths"])[0];

            Assert.True(lenAfterSwitch <= lenAfterFirst, $"Expected KV-cache to reset after task switch, but got {lenAfterFirst} -> {lenAfterSwitch}");

            var entries = AiDotNet.Helpers.InferenceDiagnostics.Snapshot();
            Assert.Contains(entries, e => e.Area == "InferenceSession" && e.Feature == "MultiLoRA" && e.Reason.Contains("Task=taskB"));
        }
        finally
        {
            AiDotNet.Helpers.InferenceDiagnostics.Clear();
            Environment.SetEnvironmentVariable("AIDOTNET_DIAGNOSTICS", originalDiagnostics);
        }
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
        return CreateDeterministicResultWithModel(config, model);
    }

    private static PredictionModelResult<float, Tensor<float>, Tensor<float>> CreateDeterministicResultWithModel(
        InferenceOptimizationConfig config,
        NeuralNetworkBase<float> model)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var optimization = new OptimizationResult<float, Tensor<float>, Tensor<float>>
        {
            BestSolution = model
        };

        var normalization = new NormalizationInfo<float, Tensor<float>, Tensor<float>>
        {
            Normalizer = new NoNormalizer<float, Tensor<float>, Tensor<float>>(),
            YParams = new NormalizationParameters<float> { Method = NormalizationMethod.None }
        };

        var options = new PredictionModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = optimization,
            NormalizationInfo = normalization,
            InferenceOptimizationConfig = config
        };

        return new PredictionModelResult<float, Tensor<float>, Tensor<float>>(options);
    }

    private static NeuralNetworkBase<float> CreateDeterministicMultiLoRAModel()
    {
        const int inputSize = FlatSize;
        const int outputSize = FlatSize;

        var baseDense = new DenseLayer<float>(inputSize, outputSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>());
        var multi = new AiDotNet.LoRA.Adapters.MultiLoRAAdapter<float>(baseDense, defaultTaskName: "taskA", defaultRank: 1, alpha: 1.0, freezeBaseLayer: true);
        multi.AddTask("taskB", rank: 1, alpha: 1.0);

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(inputSize),
            multi,
            new DenseLayer<float>(outputSize, outputSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);

        var model = new NeuralNetwork<float>(architecture);

        // Deterministic base weights across the whole model.
        var p = model.GetParameters();
        var deterministic = new float[p.Length];
        for (int i = 0; i < deterministic.Length; i++)
        {
            deterministic[i] = ((i % 19) - 9) / 9.0f;
        }
        model.UpdateParameters(new Vector<float>(deterministic));

        // Make taskB differ from taskA by setting distinct LoRA parameters.
        // (Both A and B must be non-zero for the low-rank delta to have an effect.)
        var taskA = multi.GetTaskAdapter("taskA");
        var taskB = multi.GetTaskAdapter("taskB");

        var aParams = taskA.GetParameters();
        var bParams = taskB.GetParameters();

        var a = new float[aParams.Length]; // all zeros => no delta
        var b = new float[bParams.Length];
        for (int i = 0; i < b.Length; i++)
        {
            b[i] = 0.05f;
        }

        taskA.UpdateParameters(new Vector<float>(a));
        taskB.UpdateParameters(new Vector<float>(b));

        return model;
    }

    private static NeuralNetworkBase<float> CreateDeterministicAttentionWithMultiLoRAModel()
    {
        const int inputSize = FlatSize;
        const int outputSize = FlatSize;

        var baseDense = new DenseLayer<float>(outputSize, outputSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>());
        var multi = new AiDotNet.LoRA.Adapters.MultiLoRAAdapter<float>(baseDense, defaultTaskName: "taskA", defaultRank: 1, alpha: 1.0, freezeBaseLayer: true);
        multi.AddTask("taskB", rank: 1, alpha: 1.0);

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(inputSize),
            new ReshapeLayer<float>(new[] { FlatSize }, new[] { SequenceLength, EmbeddingDimension }),
            new MultiHeadAttentionLayer<float>(
                sequenceLength: SequenceLength,
                embeddingDimension: EmbeddingDimension,
                headCount: HeadCount,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new FlattenLayer<float>(new[] { SequenceLength, EmbeddingDimension }),
            multi,
            new DenseLayer<float>(outputSize, outputSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);

        var model = new NeuralNetwork<float>(architecture);

        var p = model.GetParameters();
        var deterministic = new float[p.Length];
        for (int i = 0; i < deterministic.Length; i++)
        {
            deterministic[i] = ((i % 23) - 11) / 11.0f;
        }
        model.UpdateParameters(new Vector<float>(deterministic));

        var taskA = multi.GetTaskAdapter("taskA");
        var taskB = multi.GetTaskAdapter("taskB");

        var aParams = taskA.GetParameters();
        var bParams = taskB.GetParameters();

        var a = new float[aParams.Length];
        var b = new float[bParams.Length];
        for (int i = 0; i < b.Length; i++)
        {
            b[i] = 0.05f;
        }

        taskA.UpdateParameters(new Vector<float>(a));
        taskB.UpdateParameters(new Vector<float>(b));

        return model;
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
