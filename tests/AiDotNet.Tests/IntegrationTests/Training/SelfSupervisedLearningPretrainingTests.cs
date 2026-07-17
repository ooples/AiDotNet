using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that a self-supervised method configured via ConfigureSelfSupervisedLearningMethod runs a
/// default pretraining loop and surfaces the pretraining report (with collapse detection + linear probe).
/// </summary>
public class SelfSupervisedLearningPretrainingTests
{
    /// <summary>A minimal SSL method: decreasing loss, and a configurable encoder output for the tests.</summary>
    private sealed class FakeMethod : ISelfSupervisedLearningMethod<double>
    {
        private readonly bool _collapse;
        private int _step;
        public FakeMethod(bool collapse) => _collapse = collapse;
        public string Name => "fake-ssl";
        public SelfSupervisedLearningMethodCategory Category => default;
        public bool RequiresMemoryBank => false;
        public bool UsesMomentumEncoder => false;
        public INeuralNetwork<double> GetEncoder() => throw new NotImplementedException();
        public SelfSupervisedLearningStepResult<double> TrainStep(
            Tensor<double> batch, SelfSupervisedLearningAugmentationContext<double>? augmentationContext = null)
        {
            _step++;
            return new SelfSupervisedLearningStepResult<double> { Loss = 1.0 / _step };
        }
        public Tensor<double> Encode(Tensor<double> input)
        {
            if (!_collapse) return input; // identity → varied representations
            int rows = input.Shape[0];
            var collapsed = new Tensor<double>(new[] { rows, input.Length / rows });
            for (int i = 0; i < collapsed.Length; i++) collapsed[i] = 0.5; // all inputs → same point
            return collapsed;
        }
        public void Reset() { }
        public Vector<double> GetParameters() => new Vector<double>(0);
        public void SetParameters(Vector<double> parameters) { }
        public long ParameterCount => 0;
        public void OnEpochStart(int epoch) { }
        public void OnEpochEnd(int epoch) { }
    }

    private static NeuralNetwork<double> BuildModel(int inputSize, int outputSize)
    {
        var layers = new System.Collections.Generic.List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: inputSize, outputSize: outputSize, layers: layers);
        return new NeuralNetwork<double>(arch);
    }

    private static (Tensor<double> X, Tensor<double> Y) BuildData(int rows = 40, int cols = 4, int outs = 2)
    {
        var x = new Tensor<double>(new[] { rows, cols });
        var y = new Tensor<double>(new[] { rows, outs });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            for (int o = 0; o < outs; o++) y[i, o] = Math.Cos((i + o) * 0.2);
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureSelfSupervisedLearningMethod_RunsDefaultPretraining_AndSurfacesReport()
    {
        // Single-output targets so the linear probe (one target per sample) runs.
        var (x, y) = BuildData(outs: 1);

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildModel(4, 1))
            .ConfigureSelfSupervisedLearningMethod(new FakeMethod(collapse: false))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        var report = result.SelfSupervisedLearningPretrainingResult;
        Assert.NotNull(report);
        Assert.Equal("fake-ssl", report?.MethodName);
        Assert.True(report.StepsRun > 0, "no pretraining steps ran");
        Assert.False(report.CollapseDetected, "varied representations should not read as collapse");
        Assert.NotNull(report.LinearProbeR2); // one target per sample, so a probe runs
    }

    [Fact(Timeout = 120000)]
    public async Task DefaultPretraining_DetectsRepresentationCollapse()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildModel(4, 2))
            .ConfigureSelfSupervisedLearningMethod(new FakeMethod(collapse: true))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result.SelfSupervisedLearningPretrainingResult);
        Assert.True(result.SelfSupervisedLearningPretrainingResult?.CollapseDetected == true,
            "constant representations should be flagged as collapse");
    }

    [Fact(Timeout = 120000)]
    public async Task NoSelfSupervisedMethod_LeavesReportNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildModel(4, 2))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.SelfSupervisedLearningPretrainingResult);
    }
}
