using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression test for AiDotNet#1361 — <c>ConfigureFineTuning</c> was
/// stored but never consumed by any Build path. Calling
/// <c>ConfigureFineTuning(enabled-config-with-impl)</c> followed by
/// <c>BuildAsync</c> previously had no observable effect on the resulting
/// model; the fine-tuning implementation's <c>FineTuneAsync</c> was never
/// invoked.
///
/// <para>The fix wires fine-tuning into <c>BuildAsync</c> immediately
/// after the optimizer's main-training pass completes (right after the
/// <c>optimizationResult = finalOptimizer.Optimize(...)</c> line) and
/// before metric finalization. The fine-tuned model replaces
/// <c>optimizationResult.BestSolution</c> so all downstream
/// consumers — checkpoint manager, model registry, JIT-compiled predict
/// function, the returned <c>AiModelResult.Model</c> — see the
/// post-fine-tune weights.</para>
///
/// <para>These tests use a recording stub fine-tuner that registers its
/// invocation and returns the input model unchanged. That isolates the
/// wire-up from any specific fine-tuning algorithm's behaviour — if the
/// stub's <c>FineTuneCalls</c> increments after <c>BuildAsync</c>, the
/// wire-up is live.</para>
/// </summary>
public class ConfigureFineTuningWiringTests
{
    private sealed class RecordingFineTuner :
        IFineTuning<double, Matrix<double>, Vector<double>>
    {
        public int FineTuneCalls;
        public int EvaluateCalls;
        public IFullModel<double, Matrix<double>, Vector<double>>? LastBaseModel;
        public FineTuningData<double, Matrix<double>, Vector<double>>? LastTrainingData;

        public string MethodName => "RecordingStub";
        public FineTuningCategory Category => FineTuningCategory.SupervisedFineTuning;
        public bool RequiresRewardModel => false;
        public bool RequiresReferenceModel => false;
        public bool SupportsPEFT => false;

        public Task<IFullModel<double, Matrix<double>, Vector<double>>> FineTuneAsync(
            IFullModel<double, Matrix<double>, Vector<double>> baseModel,
            FineTuningData<double, Matrix<double>, Vector<double>> trainingData,
            CancellationToken cancellationToken = default)
        {
            FineTuneCalls++;
            LastBaseModel = baseModel;
            LastTrainingData = trainingData;
            return Task.FromResult(baseModel);
        }

        public Task<FineTuningMetrics<double>> EvaluateAsync(
            IFullModel<double, Matrix<double>, Vector<double>> model,
            FineTuningData<double, Matrix<double>, Vector<double>> evaluationData,
            CancellationToken cancellationToken = default)
        {
            EvaluateCalls++;
            return Task.FromResult(new FineTuningMetrics<double>());
        }

        public FineTuningOptions<double> GetOptions() => new();
        public void Reset()
        {
            FineTuneCalls = 0;
            EvaluateCalls = 0;
            LastBaseModel = null;
            LastTrainingData = null;
        }

        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
    }

    private static (Matrix<double> x, Vector<double> y) BuildDataset(int rows = 30, int features = 3)
    {
        var rng = new Random(42);
        var xData = new double[rows, features];
        var yData = new double[rows];
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < features; c++)
            {
                xData[r, c] = rng.NextDouble() * 2 - 1;
                sum += xData[r, c];
            }
            yData[r] = sum + rng.NextDouble() * 0.05;
        }
        return (new Matrix<double>(xData), new Vector<double>(yData));
    }

    private static FineTuningData<double, Matrix<double>, Vector<double>> BuildSFTData()
    {
        var (x, y) = BuildDataset(rows: 8, features: 3);
        return new FineTuningData<double, Matrix<double>, Vector<double>>
        {
            Inputs = new[] { x },
            Outputs = new[] { y }
        };
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureFineTuning_Enabled_InvokesFineTuneAsync_OnBuildAsync()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var stubFt = new RecordingFineTuner();

        var ftConfig = new FineTuningConfiguration<double, Matrix<double>, Vector<double>>
        {
            Enabled = true,
            Implementation = stubFt,
            TrainingData = BuildSFTData(),
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureFineTuning(ftConfig)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(1, stubFt.FineTuneCalls);
        Assert.NotNull(stubFt.LastBaseModel);
        Assert.NotNull(stubFt.LastTrainingData);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureFineTuning_Disabled_DoesNotInvokeFineTuneAsync()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var stubFt = new RecordingFineTuner();

        var ftConfig = new FineTuningConfiguration<double, Matrix<double>, Vector<double>>
        {
            Enabled = false, // explicitly disabled
            Implementation = stubFt,
            TrainingData = BuildSFTData(),
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureFineTuning(ftConfig)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(0, stubFt.FineTuneCalls);
    }

    [Fact(Timeout = 120000)]
    public async Task BuildAsync_WithoutConfigureFineTuning_DoesNotInvokeFineTuneAsync()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);
        var stubFt = new RecordingFineTuner();
        // Stub is not attached to the builder — so it must remain unused.
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(0, stubFt.FineTuneCalls);
    }

    [Fact(Timeout = 60000)]
    public async Task ConfigureFineTuning_Enabled_NoImplementation_ThrowsInvalidOperation()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var ftConfig = new FineTuningConfiguration<double, Matrix<double>, Vector<double>>
        {
            Enabled = true,
            Implementation = null,
            TrainingData = BuildSFTData(),
        };

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureFineTuning(ftConfig);

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await builder.BuildAsync());
    }

    [Fact(Timeout = 60000)]
    public async Task ConfigureFineTuning_Enabled_NoTrainingData_ThrowsInvalidOperation()
    {
        var (x, y) = BuildDataset();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var ftConfig = new FineTuningConfiguration<double, Matrix<double>, Vector<double>>
        {
            Enabled = true,
            Implementation = new RecordingFineTuner(),
            TrainingData = null,
        };

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .ConfigureFineTuning(ftConfig);

        await Assert.ThrowsAsync<InvalidOperationException>(async () => await builder.BuildAsync());
    }
}
