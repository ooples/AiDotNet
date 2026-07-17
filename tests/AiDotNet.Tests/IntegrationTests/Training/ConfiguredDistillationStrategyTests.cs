using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that the strategy passed to ConfigureDistillationStrategy actually drives distillation
/// training (both call orderings relative to ConfigureKnowledgeDistillation), and that a strategy
/// configured without a teacher fails clearly rather than silently training normally.
/// </summary>
public class ConfiguredDistillationStrategyTests
{
    /// <summary>Wraps a real strategy and records whether its gradient was requested during training.</summary>
    private sealed class SentinelStrategy : IDistillationStrategy<double>
    {
        private readonly IDistillationStrategy<double> _inner;
        public bool GradientComputed { get; private set; }
        public SentinelStrategy(IDistillationStrategy<double> inner) => _inner = inner;
        public string Name => "sentinel-strategy";
        public double Temperature { get => _inner.Temperature; set => _inner.Temperature = value; }
        public double Alpha { get => _inner.Alpha; set => _inner.Alpha = value; }
        public double ComputeLoss(Matrix<double> s, Matrix<double> t, Matrix<double>? y = null)
            => _inner.ComputeLoss(s, t, y);
        public Matrix<double> ComputeGradient(Matrix<double> s, Matrix<double> t, Matrix<double>? y = null)
        {
            GradientComputed = true;
            return _inner.ComputeGradient(s, t, y);
        }
    }

    private static NeuralNetwork<double> BuildStudent(int inputSize, int outputSize)
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

    private static (Tensor<double> X, Tensor<double> Y) BuildData(int rows, int cols, int outs)
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

    private static Tensor<double> ConstantTeacher(Tensor<double> input)
    {
        var o = new Tensor<double>(new[] { 1, 2 });
        o[0, 0] = 0.5; o[0, 1] = 0.5;
        return o;
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredStrategy_DrivesTraining_StrategyBeforeKD()
    {
        var (x, y) = BuildData(30, 4, 2);
        var sentinel = new SentinelStrategy(DistillationStrategyFactory<double>.CreateResponseBasedStrategy(2.0, 0.0));

        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildStudent(4, 2))
            .ConfigureDistillationStrategy(sentinel)
            .ConfigureKnowledgeDistillation(new KnowledgeDistillationOptions<double, Tensor<double>, Tensor<double>>
            {
                TeacherForward = ConstantTeacher,
                Temperature = 2.0,
                Alpha = 0.0,
            })
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.True(sentinel.GradientComputed, "the configured distillation strategy did not drive training");
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredStrategy_DrivesTraining_KDBeforeStrategy()
    {
        var (x, y) = BuildData(30, 4, 2);
        var sentinel = new SentinelStrategy(DistillationStrategyFactory<double>.CreateResponseBasedStrategy(2.0, 0.0));

        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildStudent(4, 2))
            .ConfigureKnowledgeDistillation(new KnowledgeDistillationOptions<double, Tensor<double>, Tensor<double>>
            {
                TeacherForward = ConstantTeacher,
                Temperature = 2.0,
                Alpha = 0.0,
            })
            .ConfigureDistillationStrategy(sentinel)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.True(sentinel.GradientComputed,
            "the configured distillation strategy did not drive training when configured after KD");
    }

    [Fact(Timeout = 120000)]
    public async Task StrategyWithoutTeacher_FailsClearly()
    {
        var (x, y) = BuildData(20, 4, 2);
        var strategy = DistillationStrategyFactory<double>.CreateResponseBasedStrategy(2.0, 0.0);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
                .ConfigureModel(BuildStudent(4, 2))
                .ConfigureDistillationStrategy(strategy)
                .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
                .BuildAsync());

        Assert.Contains("teacher", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
