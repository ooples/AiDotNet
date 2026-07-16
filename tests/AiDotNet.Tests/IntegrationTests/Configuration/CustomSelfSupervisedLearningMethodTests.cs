using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Covers the enum-to-interface migration for the SSL method: <c>SelfSupervisedLearningConfig&lt;T&gt;.Method</c> is an
/// <see cref="ISelfSupervisedLearningMethod{T}"/>, nullable, defaulting to SimCLR — the standard contrastive baseline and
/// what this pipeline has always defaulted to. <c>SelfSupervisedLearningMethodType</c> is gone; a closed enum could only
/// ever name the methods the library ships.
///
/// <para>Assertions read <see cref="SelfSupervisedLearningResult{T}.MethodName"/>, which the session takes from the
/// method that actually ran — observable state, not "it constructed".</para>
/// </summary>
public class CustomSSLMethodTests
{
    private const int EncoderOutputDim = 4;

    private static NeuralNetwork<double> BuildEncoder()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(EncoderOutputDim, activationFunction: (IActivationFunction<double>?)null),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 5,
            outputSize: EncoderOutputDim,
            layers: layers);
        return new NeuralNetwork<double>(arch);
    }

    private static Func<IEnumerable<Tensor<double>>> BuildDataLoader()
    {
        return () =>
        {
            var batch = new Tensor<double>(new[] { 4, 5 });
            var s = batch.AsWritableSpan();
            for (int i = 0; i < s.Length; i++) s[i] = Math.Sin(i * 0.31);
            return new[] { batch };
        };
    }

    private static SelfSupervisedLearningResult<double> RunPipeline(Action<SelfSupervisedLearningConfig<double>> configure)
    {
        var pipeline = new SelfSupervisedLearningPretrainingPipeline<double>(BuildEncoder(), EncoderOutputDim)
            .WithConfig(cfg =>
            {
                cfg.PretrainingEpochs = 1;
                cfg.BatchSize = 4;
                cfg.EnableLinearEvaluation = false;
                cfg.EnableKNNEvaluation = false;
                cfg.CheckpointFrequency = 0;
                cfg.Seed = 7;
                configure(cfg);
            });

        return pipeline.Train(BuildDataLoader());
    }

    /// <summary>An SSL method the library does not ship — the point of the interface door.</summary>
    private sealed class SentinelSSLMethod : ISelfSupervisedLearningMethod<double>
    {
        private readonly INeuralNetwork<double> _encoder;

        public SentinelSSLMethod(INeuralNetwork<double> encoder) => _encoder = encoder;

        public string Name => "sentinel";

        public SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.Contrastive;

        public bool RequiresMemoryBank => false;

        public bool UsesMomentumEncoder => false;

        public long ParameterCount => _encoder.GetParameters().Length;

        public INeuralNetwork<double> GetEncoder() => _encoder;

        public SelfSupervisedLearningStepResult<double> TrainStep(
            Tensor<double> batch, SelfSupervisedLearningAugmentationContext<double>? augmentationContext = null)
            => new SelfSupervisedLearningStepResult<double> { Loss = 0.0 };

        public Tensor<double> Encode(Tensor<double> input) => _encoder.Predict(input);

        public void Reset() { }

        public Vector<double> GetParameters() => _encoder.GetParameters();

        public void SetParameters(Vector<double> parameters) => _encoder.UpdateParameters(parameters);

        public void OnEpochStart(int epochNumber) { }

        public void OnEpochEnd(int epochNumber) { }
    }

    [Fact(Timeout = 120000)]
    public async Task NullMethod_UsesTheIndustryStandardDefault()
    {
        // The parameter is nullable and defaults to the standard method, so callers who do not care
        // are not forced to name one.
        var result = RunPipeline(_ => { });

        Assert.True(result.IsSuccess, result.ErrorMessage);
        Assert.Equal("SimCLR", result.MethodName);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 120000)]
    public async Task ShippedMethod_IsUsedWhenSupplied()
    {
        var encoder = BuildEncoder();
        var result = RunPipeline(cfg =>
            cfg.Method = MoCoV2<double>.Create(
                encoder,
                _ => BuildEncoder(),
                EncoderOutputDim,
                projectionDim: 4,
                hiddenDim: 8,
                queueSize: 8));

        Assert.True(result.IsSuccess, result.ErrorMessage);
        Assert.Equal("MoCo v2", result.MethodName);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 120000)]
    public async Task CustomMethod_TheLibraryDoesNotShip_IsAccepted()
    {
        // The point of replacing the enum: a closed enum could only ever name the methods we ship.
        var result = RunPipeline(cfg => cfg.Method = new SentinelSSLMethod(BuildEncoder()));

        Assert.True(result.IsSuccess, result.ErrorMessage);
        Assert.Equal("sentinel", result.MethodName);
        await Task.CompletedTask;
    }

    [Fact(Timeout = 120000)]
    public async Task SuppliedMethod_IsRecordedByNameInTheConfigurationDictionary()
    {
        // An enum ordinal only worked while the choice was closed; the name works for any method.
        var config = new SelfSupervisedLearningConfig<double> { Method = new SentinelSSLMethod(BuildEncoder()) };

        Assert.Equal("sentinel", config.GetConfiguration()["method"]);
        await Task.CompletedTask;
    }
}
