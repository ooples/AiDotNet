using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureKnowledgeDistillation actually trains the student through Build, rather than
/// throwing "not yet integrated with the tape-based training flow" as it used to.
/// </summary>
/// <remarks>
/// The distillation build path threw for every model. These tests confirm it now runs, and — the
/// point of distillation — that training moves the student's outputs toward the teacher's, not merely
/// that Build returns without error.
/// </remarks>
public class KnowledgeDistillationBuildTests
{
    private static NeuralNetwork<double> BuildStudent(int inputSize, int outputSize)
    {
        var layers = new System.Collections.Generic.List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: inputSize, outputSize: outputSize, layers: layers);
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

    [Fact(Timeout = 120000)]
    public async Task ConfigureKnowledgeDistillation_TrainsInsteadOfThrowing()
    {
        const int cols = 4, outs = 2, rows = 40;
        var (x, y) = BuildData(rows, cols, outs);

        // Teacher: a fixed target function distinct from the labels, so "matched the teacher" is a
        // real signal, not an artifact of also fitting the labels.
        Tensor<double> Teacher(Tensor<double> input)
        {
            var o = new Tensor<double>(new[] { 1, outs });
            for (int k = 0; k < outs; k++) o[0, k] = 0.5; // constant teacher logits
            return o;
        }

        // Measure the ACTUAL distillation objective (the strategy's own loss between student and
        // teacher outputs), so "it trained" is unambiguous — distillation matches softmax
        // distributions, which a raw-output gap would not capture.
        var strategy = DistillationStrategyFactory<double>.CreateResponseBasedStrategy(2.0, 0.0);
        var student = BuildStudent(cols, outs);
        double beforeLoss = DistillLoss(strategy, student, x, Teacher, outs);

        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.05,
                MaxIterations = 30,
            });

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(student)
            .ConfigureOptimizer(adam)
            .ConfigureKnowledgeDistillation(new KnowledgeDistillationOptions<double, Tensor<double>, Tensor<double>>
            {
                TeacherForward = Teacher,
                Temperature = 2.0,
                Alpha = 0.0, // pure distillation: match the teacher only, so the assertion is unambiguous
            })
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result);
        double afterLoss = DistillLoss(strategy, student, x, Teacher, outs);

        // Distillation must reduce its own objective — the student's outputs move toward the teacher's.
        Assert.True(
            afterLoss < beforeLoss,
            $"distillation loss did not decrease (before={beforeLoss:F5}, after={afterLoss:F5})");
    }

    [Fact(Timeout = 60000)]
    public async Task KnowledgeDistillation_WithoutATeacher_FailsClearly()
    {
        var (x, y) = BuildData(20, 4, 2);
        var student = BuildStudent(4, 2);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
                .ConfigureModel(student)
                .ConfigureKnowledgeDistillation(new KnowledgeDistillationOptions<double, Tensor<double>, Tensor<double>>
                {
                    Temperature = 2.0,
                })
                .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
                .BuildAsync());

        Assert.Contains("teacher", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static double DistillLoss(
        IDistillationStrategy<double> strategy, NeuralNetwork<double> student,
        Tensor<double> x, Func<Tensor<double>, Tensor<double>> teacher, int outs)
    {
        int rows = x.Shape[0];
        double total = 0;
        for (int i = 0; i < rows; i++)
        {
            var row = new Tensor<double>(new[] { 1, x.Shape[1] });
            for (int j = 0; j < x.Shape[1]; j++) row[0, j] = x[i, j];
            var pred = student.Predict(row);
            var teach = teacher(row);
            var sm = new Matrix<double>(1, outs);
            var tm = new Matrix<double>(1, outs);
            for (int k = 0; k < outs; k++) { sm[0, k] = pred[0, k]; tm[0, k] = teach[0, k]; }
            total += Convert.ToDouble(strategy.ComputeLoss(sm, tm, null));
        }

        return total / rows;
    }
}
