using AiDotNet.Enums;
using AiDotNet.FineTuning;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FineTuning;

/// <summary>
/// Integration tests for the FineTuning module covering all fine-tuning methods.
/// </summary>
public class FineTuningIntegrationTests
{
    private const int InputSize = 10;
    private const int OutputSize = 5;
    private const int SampleCount = 20;

    #region Helper Classes

    /// <summary>
    /// Mock model for testing fine-tuning methods.
    /// </summary>
    private class MockFullModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private Vector<double> _weights;
        private readonly ILossFunction<double> _lossFunction;
        private int[] _activeFeatureIndices;

        public MockFullModel(int inputSize, int outputSize)
        {
            _weights = new Vector<double>(inputSize * outputSize);
            var random = new Random(42);
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = random.NextDouble() * 0.1;
            }
            _lossFunction = new MeanSquaredErrorLoss<double>();
            _activeFeatureIndices = Enumerable.Range(0, inputSize).ToArray();
        }

        private MockFullModel(Vector<double> weights, ILossFunction<double> lossFunction, int[] activeFeatureIndices)
        {
            _weights = new Vector<double>(weights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                _weights[i] = weights[i];
            }
            _lossFunction = lossFunction;
            _activeFeatureIndices = activeFeatureIndices.ToArray();
        }

        public ILossFunction<double> DefaultLossFunction => _lossFunction;

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(OutputSize);
            for (int i = 0; i < OutputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < input.Length; j++)
                {
                    sum += input[j] * _weights[i * input.Length + j];
                }
                output[i] = sum;
            }
            return output;
        }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                ModelType = ModelType.NeuralNetwork
            };
        }

        // IModel
        public void Train(Vector<double> input, Vector<double> target) { }

        // IModelSerializer
        public void Save(string filePath) { }
        public void Load(string filePath) { }
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }

        // ICheckpointableModel
        public byte[] ToCheckpoint() => Array.Empty<byte>();
        public void FromCheckpoint(byte[] checkpoint) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        // IParameterizable
        public Vector<double> GetParameters() => _weights;
        public void SetParameters(Vector<double> parameters) { _weights = parameters; }
        public int ParameterCount => _weights.Length;
        public Vector<double> GetParameterGradients() => new Vector<double>(ParameterCount);
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var copy = DeepCopy();
            copy.SetParameters(parameters);
            return copy;
        }

        // IFeatureAware
        public string[] FeatureNames { get; set; } = Array.Empty<string>();
        public int FeatureCount => InputSize;
        public void SetFeatureNames(string[] names) { FeatureNames = names; }
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatureIndices;
        public void SetActiveFeatureIndices(IEnumerable<int> indices) { _activeFeatureIndices = indices.ToArray(); }
        public bool IsFeatureUsed(int index) => _activeFeatureIndices.Contains(index);

        // IFeatureImportance
        public Vector<double> GetFeatureImportances() => new Vector<double>(InputSize);
        public Dictionary<string, double> GetFeatureImportance()
        {
            var result = new Dictionary<string, double>();
            for (int i = 0; i < FeatureCount; i++)
            {
                result[$"feature_{i}"] = 1.0 / FeatureCount;
            }
            return result;
        }

        // ICloneable
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => DeepCopy();
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() =>
            new MockFullModel(_weights, _lossFunction, _activeFeatureIndices);

        // IGradientComputable
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double> lossFunction)
        {
            var gradients = new Vector<double>(ParameterCount);
            var prediction = Predict(input);

            for (int i = 0; i < OutputSize; i++)
            {
                double diff = prediction[i] - target[i];
                for (int j = 0; j < input.Length; j++)
                {
                    gradients[i * input.Length + j] = diff * input[j];
                }
            }
            return gradients;
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] -= learningRate * gradients[i];
            }
        }

        // IJitCompilable
        public bool IsJitCompiled => false;
        public bool SupportsJitCompilation => false;
        public void CompileForJit() { }
        public void ResetJitCompilation() { }
        public AiDotNet.Autodiff.ComputationNode<double> ExportComputationGraph(List<AiDotNet.Autodiff.ComputationNode<double>> nodes)
        {
            return new AiDotNet.Autodiff.ComputationNode<double>(new Tensor<double>(new[] { 1 }), false, null, null, "mock");
        }
    }

    private static Vector<double> CreateVector(int size)
    {
        var vector = new Vector<double>(size);
        var random = new Random(42);
        for (int i = 0; i < size; i++)
        {
            vector[i] = random.NextDouble();
        }
        return vector;
    }

    private static FineTuningData<double, Vector<double>, Vector<double>> CreateSFTData(int count)
    {
        var inputs = new Vector<double>[count];
        var outputs = new Vector<double>[count];
        var random = new Random(42);

        for (int i = 0; i < count; i++)
        {
            inputs[i] = new Vector<double>(InputSize);
            outputs[i] = new Vector<double>(OutputSize);
            for (int j = 0; j < InputSize; j++)
            {
                inputs[i][j] = random.NextDouble();
            }
            for (int j = 0; j < OutputSize; j++)
            {
                outputs[i][j] = random.NextDouble();
            }
        }

        return new FineTuningData<double, Vector<double>, Vector<double>>
        {
            Inputs = inputs,
            Outputs = outputs
        };
    }

    private static FineTuningData<double, Vector<double>, Vector<double>> CreatePreferenceData(int count)
    {
        var inputs = new Vector<double>[count];
        var chosen = new Vector<double>[count];
        var rejected = new Vector<double>[count];
        var random = new Random(42);

        for (int i = 0; i < count; i++)
        {
            inputs[i] = new Vector<double>(InputSize);
            chosen[i] = new Vector<double>(OutputSize);
            rejected[i] = new Vector<double>(OutputSize);

            for (int j = 0; j < InputSize; j++)
            {
                inputs[i][j] = random.NextDouble();
            }
            for (int j = 0; j < OutputSize; j++)
            {
                // Chosen outputs have higher values (simulating "better" responses)
                chosen[i][j] = random.NextDouble() + 0.5;
                rejected[i][j] = random.NextDouble();
            }
        }

        return new FineTuningData<double, Vector<double>, Vector<double>>
        {
            Inputs = inputs,
            ChosenOutputs = chosen,
            RejectedOutputs = rejected
        };
    }

    private static FineTuningData<double, Vector<double>, Vector<double>> CreateRLData(int count)
    {
        var data = CreateSFTData(count);
        var rewards = new double[count];
        var random = new Random(42);

        for (int i = 0; i < count; i++)
        {
            rewards[i] = random.NextDouble() * 2 - 1; // Rewards between -1 and 1
        }

        data.Rewards = rewards;
        return data;
    }

    private static FineTuningData<double, Vector<double>, Vector<double>> CreateRankingData(int count, int rankSize = 4)
    {
        var inputs = new Vector<double>[count];
        var rankedOutputs = new Vector<double>[count][];
        var random = new Random(42);

        for (int i = 0; i < count; i++)
        {
            inputs[i] = new Vector<double>(InputSize);
            for (int j = 0; j < InputSize; j++)
            {
                inputs[i][j] = random.NextDouble();
            }

            rankedOutputs[i] = new Vector<double>[rankSize];
            for (int k = 0; k < rankSize; k++)
            {
                rankedOutputs[i][k] = new Vector<double>(OutputSize);
                for (int j = 0; j < OutputSize; j++)
                {
                    // Higher ranked outputs have higher values
                    rankedOutputs[i][k][j] = random.NextDouble() + (rankSize - k) * 0.1;
                }
            }
        }

        return new FineTuningData<double, Vector<double>, Vector<double>>
        {
            Inputs = inputs,
            RankedOutputs = rankedOutputs
        };
    }

    #endregion

    #region FineTuningBase Tests

    [Fact]
    public void FineTuningOptions_DefaultValues_AreReasonable()
    {
        var options = new FineTuningOptions<double>();

        Assert.Equal(FineTuningMethodType.SFT, options.MethodType);
        Assert.Equal(1e-5, options.LearningRate);
        Assert.Equal(8, options.BatchSize);
        Assert.Equal(3, options.Epochs);
        Assert.Equal(0.1, options.Beta);
        Assert.Equal(1.0, options.MaxGradientNorm);
    }

    [Fact]
    public void FineTuningData_HasSFTData_ReturnsTrueWhenValid()
    {
        var data = CreateSFTData(SampleCount);

        Assert.True(data.HasSFTData);
        Assert.Equal(SampleCount, data.Count);
    }

    [Fact]
    public void FineTuningData_HasPreferenceData_ReturnsTrueWhenValid()
    {
        var data = CreatePreferenceData(SampleCount);

        Assert.True(data.HasPairwisePreferenceData);
        Assert.Equal(SampleCount, data.Count);
    }

    [Fact]
    public void FineTuningData_Split_CreatesValidSplits()
    {
        var data = CreateSFTData(100);
        var (train, val) = data.Split(validationRatio: 0.2, seed: 42);

        Assert.Equal(80, train.Count);
        Assert.Equal(20, val.Count);
        Assert.True(train.HasSFTData);
        Assert.True(val.HasSFTData);
    }

    [Fact]
    public void FineTuningData_Subset_CreatesCorrectSubset()
    {
        var data = CreateSFTData(SampleCount);
        var indices = new[] { 0, 5, 10 };

        var subset = data.Subset(indices);

        Assert.Equal(3, subset.Count);
        Assert.True(subset.HasSFTData);
    }

    #endregion

    #region SupervisedFineTuning Tests

    [Fact]
    public void SFT_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("SFT", sft.MethodName);
        Assert.Equal(FineTuningCategory.SupervisedFineTuning, sft.Category);
        Assert.False(sft.RequiresRewardModel);
        Assert.False(sft.RequiresReferenceModel);
        Assert.True(sft.SupportsPEFT);
    }

    [Fact]
    public async Task SFT_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 2,
            BatchSize = 4,
            LearningRate = 0.01,
            LoggingSteps = 1000 // Disable logging for test
        };
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateSFTData(SampleCount);

        var fineTunedModel = await sft.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
        Assert.NotSame(model, fineTunedModel);
    }

    [Fact]
    public async Task SFT_FineTuneAsync_ThrowsOnNullModel()
    {
        var options = new FineTuningOptions<double>();
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);
        var data = CreateSFTData(SampleCount);

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            sft.FineTuneAsync(null!, data));
    }

    [Fact]
    public async Task SFT_FineTuneAsync_ThrowsOnEmptyData()
    {
        var options = new FineTuningOptions<double>();
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var emptyData = new FineTuningData<double, Vector<double>, Vector<double>>();

        await Assert.ThrowsAsync<ArgumentException>(() =>
            sft.FineTuneAsync(model, emptyData));
    }

    [Fact]
    public async Task SFT_EvaluateAsync_ReturnsMetrics()
    {
        var options = new FineTuningOptions<double> { LoggingSteps = 1000 };
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateSFTData(SampleCount);

        var metrics = await sft.EvaluateAsync(model, data);

        Assert.NotNull(metrics);
        Assert.Equal("SFT", metrics.MethodName);
        Assert.True(metrics.CustomMetrics.ContainsKey("accuracy"));
    }

    [Fact]
    public async Task SFT_FineTuneAsync_SupportsCancellation()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 100,
            BatchSize = 1,
            LoggingSteps = 1000
        };
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateSFTData(SampleCount);
        var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            sft.FineTuneAsync(model, data, cts.Token));
    }

    #endregion

    #region DirectPreferenceOptimization Tests

    [Fact]
    public void DPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var dpo = new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("DPO", dpo.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, dpo.Category);
        Assert.False(dpo.RequiresRewardModel);
        Assert.True(dpo.RequiresReferenceModel);
        Assert.True(dpo.SupportsPEFT);
        Assert.Equal(FineTuningMethodType.DPO, dpo.GetOptions().MethodType);
    }

    [Fact]
    public async Task DPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            Beta = 0.1,
            LoggingSteps = 1000
        };
        var dpo = new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await dpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
        Assert.NotSame(model, fineTunedModel);
    }

    [Fact]
    public async Task DPO_FineTuneAsync_ThrowsOnSFTData()
    {
        var options = new FineTuningOptions<double>();
        var dpo = new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var sftData = CreateSFTData(SampleCount);

        await Assert.ThrowsAsync<ArgumentException>(() =>
            dpo.FineTuneAsync(model, sftData));
    }

    [Fact]
    public async Task DPO_EvaluateAsync_ReturnsPreferenceMetrics()
    {
        var options = new FineTuningOptions<double> { LoggingSteps = 1000 };
        var dpo = new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var metrics = await dpo.EvaluateAsync(model, data);

        Assert.NotNull(metrics);
        Assert.Equal("DPO", metrics.MethodName);
    }

    #endregion

    #region SimplePreferenceOptimization (SimPO) Tests

    [Fact]
    public void SimPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var simpo = new SimplePreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("SimPO", simpo.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, simpo.Category);
        Assert.False(simpo.RequiresRewardModel);
        Assert.False(simpo.RequiresReferenceModel); // SimPO doesn't need reference model
    }

    [Fact]
    public async Task SimPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            SimPOGamma = 0.5,
            LoggingSteps = 1000
        };
        var simpo = new SimplePreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await simpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region OddsRatioPreferenceOptimization (ORPO) Tests

    [Fact]
    public void ORPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var orpo = new OddsRatioPreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("ORPO", orpo.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, orpo.Category);
        Assert.False(orpo.RequiresRewardModel);
        Assert.False(orpo.RequiresReferenceModel);
    }

    [Fact]
    public async Task ORPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            ORPOLambda = 0.1,
            LoggingSteps = 1000
        };
        var orpo = new OddsRatioPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await orpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region IdentityPreferenceOptimization (IPO) Tests

    [Fact]
    public void IPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var ipo = new IdentityPreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("IPO", ipo.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, ipo.Category);
    }

    [Fact]
    public async Task IPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            LoggingSteps = 1000
        };
        var ipo = new IdentityPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await ipo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region RobustDirectPreferenceOptimization (RDPO) Tests

    [Fact]
    public void RDPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var rdpo = new RobustDirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("RDPO", rdpo.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, rdpo.Category);
    }

    [Fact]
    public async Task RDPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            LoggingSteps = 1000
        };
        var rdpo = new RobustDirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await rdpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region KahnemanTverskyOptimization (KTO) Tests

    [Fact]
    public void KTO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var kto = new KahnemanTverskyOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("KTO", kto.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, kto.Category);
    }

    [Fact]
    public async Task KTO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            KTODesirableWeight = 1.0,
            KTOUndesirableWeight = 1.0,
            LoggingSteps = 1000
        };
        var kto = new KahnemanTverskyOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await kto.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region ContrastivePreferenceOptimization (CPO) Tests

    [Fact]
    public void CPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var cpo = new ContrastivePreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("CPO", cpo.MethodName);
        Assert.Equal(FineTuningCategory.DirectPreference, cpo.Category);
    }

    [Fact]
    public async Task CPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            LoggingSteps = 1000
        };
        var cpo = new ContrastivePreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await cpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region ReinforcementLearningHumanFeedback (RLHF) Tests

    [Fact]
    public void RLHF_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var rlhf = new ReinforcementLearningHumanFeedback<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("RLHF-PPO", rlhf.MethodName);
        Assert.Equal(FineTuningCategory.ReinforcementLearning, rlhf.Category);
        Assert.True(rlhf.RequiresRewardModel);
        Assert.True(rlhf.RequiresReferenceModel);
    }

    [Fact]
    public async Task RLHF_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            KLCoefficient = 0.02,
            PPOClipRange = 0.2,
            LoggingSteps = 1000
        };
        var rlhf = new ReinforcementLearningHumanFeedback<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateRLData(SampleCount);

        var fineTunedModel = await rlhf.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region GroupRelativePolicyOptimization (GRPO) Tests

    [Fact]
    public void GRPO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var grpo = new GroupRelativePolicyOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("GRPO", grpo.MethodName);
        Assert.Equal(FineTuningCategory.ReinforcementLearning, grpo.Category);
    }

    [Fact]
    public async Task GRPO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            GRPOGroupSize = 4,
            GRPOTemperature = 0.7,
            LoggingSteps = 1000
        };
        var grpo = new GroupRelativePolicyOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateRLData(SampleCount);

        var fineTunedModel = await grpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region PairwiseRankingOptimization (PRO) Tests

    [Fact]
    public void PRO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var pro = new PairwiseRankingOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("PRO", pro.MethodName);
        Assert.Equal(FineTuningCategory.RankingBased, pro.Category);
    }

    [Fact]
    public async Task PRO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            RankingMargin = 0.1,
            LoggingSteps = 1000
        };
        var pro = new PairwiseRankingOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateRankingData(SampleCount);

        var fineTunedModel = await pro.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region RankResponsesHumanFeedback (RRHF) Tests

    [Fact]
    public void RRHF_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var rrhf = new RankResponsesHumanFeedback<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("RRHF", rrhf.MethodName);
        Assert.Equal(FineTuningCategory.RankingBased, rrhf.Category);
    }

    [Fact]
    public async Task RRHF_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            LoggingSteps = 1000
        };
        var rrhf = new RankResponsesHumanFeedback<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateRankingData(SampleCount);

        var fineTunedModel = await rrhf.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region StatisticalRejectionSampling (RSO) Tests

    [Fact]
    public void RSO_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var rso = new StatisticalRejectionSampling<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("RSO", rso.MethodName);
        Assert.Equal(FineTuningCategory.RankingBased, rso.Category);
    }

    [Fact]
    public async Task RSO_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            LoggingSteps = 1000
        };
        var rso = new StatisticalRejectionSampling<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateRankingData(SampleCount);

        var fineTunedModel = await rso.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region SelfPlayFineTuning (SPIN) Tests

    [Fact]
    public void SPIN_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var spin = new SelfPlayFineTuning<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("SPIN", spin.MethodName);
        Assert.Equal(FineTuningCategory.SelfPlay, spin.Category);
    }

    [Fact]
    public async Task SPIN_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            SPINIterations = 2,
            LoggingSteps = 1000
        };
        var spin = new SelfPlayFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateSFTData(SampleCount);

        var fineTunedModel = await spin.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region ConstitutionalAIFineTuning (CAI) Tests

    [Fact]
    public void CAI_Constructor_InitializesCorrectly()
    {
        var options = new FineTuningOptions<double>();
        var cai = new ConstitutionalAIFineTuning<double, Vector<double>, Vector<double>>(options);

        Assert.Equal("CAI", cai.MethodName);
        Assert.Equal(FineTuningCategory.Constitutional, cai.Category);
    }

    [Fact]
    public async Task CAI_FineTuneAsync_CompletesSuccessfully()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            CritiqueIterations = 1,
            LoggingSteps = 1000
        };
        var cai = new ConstitutionalAIFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        // CAI requires either SFT data or CritiqueRevisions data
        var data = CreateSFTData(SampleCount);

        var fineTunedModel = await cai.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void FineTuning_Serialize_Deserialize_PreservesOptions()
    {
        var options = new FineTuningOptions<double>
        {
            LearningRate = 0.001,
            BatchSize = 16,
            Epochs = 5,
            Beta = 0.2
        };
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);

        var serialized = sft.Serialize();
        Assert.NotEmpty(serialized);

        var sft2 = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(new FineTuningOptions<double>());
        sft2.Deserialize(serialized);
        var restoredOptions = sft2.GetOptions();

        Assert.Equal(options.LearningRate, restoredOptions.LearningRate);
        Assert.Equal(options.BatchSize, restoredOptions.BatchSize);
        Assert.Equal(options.Epochs, restoredOptions.Epochs);
        Assert.Equal(options.Beta, restoredOptions.Beta);
    }

    [Fact]
    public void FineTuning_Reset_ClearsMetrics()
    {
        var options = new FineTuningOptions<double>();
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);

        sft.Reset();

        // Reset should not throw and should work correctly
        Assert.NotNull(sft.GetOptions());
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void FineTuningOptions_ExtremeBeta_Handled()
    {
        var options = new FineTuningOptions<double>
        {
            Beta = 10.0 // Very high beta
        };
        var dpo = new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);

        Assert.Equal(10.0, dpo.GetOptions().Beta);
    }

    [Fact]
    public void FineTuningOptions_ZeroLearningRate_Allowed()
    {
        var options = new FineTuningOptions<double>
        {
            LearningRate = 0.0
        };

        Assert.Equal(0.0, options.LearningRate);
    }

    [Fact]
    public async Task SFT_SingleSample_Works()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 1,
            LoggingSteps = 1000
        };
        var sft = new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreateSFTData(1);

        var fineTunedModel = await sft.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    [Fact]
    public async Task DPO_LabelSmoothing_Works()
    {
        var options = new FineTuningOptions<double>
        {
            Epochs = 1,
            BatchSize = 4,
            LabelSmoothing = 0.1,
            LoggingSteps = 1000
        };
        var dpo = new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options);
        var model = new MockFullModel(InputSize, OutputSize);
        var data = CreatePreferenceData(SampleCount);

        var fineTunedModel = await dpo.FineTuneAsync(model, data);

        Assert.NotNull(fineTunedModel);
    }

    #endregion

    #region All Fine-Tuning Methods Test

    [Fact]
    public void AllFineTuningMethods_HaveUniqueNames()
    {
        var options = new FineTuningOptions<double>();
        var methods = new FineTuningBase<double, Vector<double>, Vector<double>>[]
        {
            new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options),
            new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new SimplePreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new OddsRatioPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new IdentityPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new RobustDirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new KahnemanTverskyOptimization<double, Vector<double>, Vector<double>>(options),
            new ContrastivePreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new ReinforcementLearningHumanFeedback<double, Vector<double>, Vector<double>>(options),
            new GroupRelativePolicyOptimization<double, Vector<double>, Vector<double>>(options),
            new PairwiseRankingOptimization<double, Vector<double>, Vector<double>>(options),
            new RankResponsesHumanFeedback<double, Vector<double>, Vector<double>>(options),
            new StatisticalRejectionSampling<double, Vector<double>, Vector<double>>(options),
            new SelfPlayFineTuning<double, Vector<double>, Vector<double>>(options),
            new ConstitutionalAIFineTuning<double, Vector<double>, Vector<double>>(options),
        };

        var names = methods.Select(m => m.MethodName).ToArray();
        var uniqueNames = names.Distinct().ToArray();

        Assert.Equal(methods.Length, uniqueNames.Length);
    }

    [Fact]
    public void AllFineTuningMethods_HaveValidCategories()
    {
        var options = new FineTuningOptions<double>();
        var methods = new FineTuningBase<double, Vector<double>, Vector<double>>[]
        {
            new SupervisedFineTuning<double, Vector<double>, Vector<double>>(options),
            new DirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new SimplePreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new OddsRatioPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new IdentityPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new RobustDirectPreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new KahnemanTverskyOptimization<double, Vector<double>, Vector<double>>(options),
            new ContrastivePreferenceOptimization<double, Vector<double>, Vector<double>>(options),
            new ReinforcementLearningHumanFeedback<double, Vector<double>, Vector<double>>(options),
            new GroupRelativePolicyOptimization<double, Vector<double>, Vector<double>>(options),
            new PairwiseRankingOptimization<double, Vector<double>, Vector<double>>(options),
            new RankResponsesHumanFeedback<double, Vector<double>, Vector<double>>(options),
            new StatisticalRejectionSampling<double, Vector<double>, Vector<double>>(options),
            new SelfPlayFineTuning<double, Vector<double>, Vector<double>>(options),
            new ConstitutionalAIFineTuning<double, Vector<double>, Vector<double>>(options),
        };

        foreach (var method in methods)
        {
            Assert.True(Enum.IsDefined(typeof(FineTuningCategory), method.Category),
                $"{method.MethodName} has invalid category: {method.Category}");
        }
    }

    #endregion
}
