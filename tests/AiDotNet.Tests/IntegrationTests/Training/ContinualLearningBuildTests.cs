using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureContinualLearning routes training through a continual learner built around the
/// configured model, and surfaces the task result plus the all-tasks retention report on the result.
/// </summary>
public class ContinualLearningBuildTests
{
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

    private static (Tensor<double> X, Tensor<double> Y) BuildData(int rows = 30, int cols = 4, int outs = 2)
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
    public async Task ConfigureContinualLearning_DefaultHybrid_SurfacesResultAndReport()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildModel(4, 2))
            .ConfigureContinualLearning() // null strategy → EWC + experience-replay hybrid default
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result.ContinualLearningResult);
        // One task has been learned this Build.
        Assert.Equal(0, result.ContinualLearningResult!.TaskId);
        // The retention report re-evaluates every task learned so far (here, one).
        Assert.NotNull(result.ContinualLearningReport);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureContinualLearning_ExplicitStrategy_IsHonored()
    {
        var (x, y) = BuildData();
        var ewc = new ElasticWeightConsolidation<double, Tensor<double>, Tensor<double>>(
            new MeanSquaredErrorLoss<double>());

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildModel(4, 2))
            .ConfigureContinualLearning(ewc)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result.ContinualLearningResult);
    }

    [Fact(Timeout = 120000)]
    public async Task NoContinualLearning_LeavesResultNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildModel(4, 2))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.ContinualLearningResult);
        Assert.Null(result.ContinualLearningReport);
    }
}
