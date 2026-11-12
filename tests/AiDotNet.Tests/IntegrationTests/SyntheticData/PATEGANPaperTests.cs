using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Jordon, Yoon and van der Schaar 2019, PATE-GAN: teachers on disjoint partitions, a student trained on PATE_λ
/// labels of generated rows, a generator trained through the student, the moments accountant that stops training at
/// ε (Algorithm 1), and the appendix's networks (#2397).
/// </summary>
public class PATEGANPaperTests
{
    private static (Matrix<double> Data, List<ColumnMetadata> Columns) Table(int rows, int seed, double favouredShare)
    {
        // A continuous column and a three-way category whose first value holds favouredShare of the rows.
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new Matrix<double>(rows, 2);
        for (int i = 0; i < rows; i++)
        {
            data[i, 0] = 5.0 + random.NextDouble();
            data[i, 1] = random.NextDouble() < favouredShare ? 0 : 1 + random.Next(2);
        }
        var columns = new List<ColumnMetadata>
        {
            new("Value", ColumnDataType.Continuous, columnIndex: 0),
            new("Category", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 1),
        };
        return (data, columns);
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 4);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new PATEGANOptions<double>();
        Assert.Equal((64, 5, 5), (o.BatchSize, o.TeacherSteps, o.StudentSteps));
        Assert.Equal(1e-4, o.LearningRate);
        Assert.Equal(1e-5, o.Delta);
        Assert.Equal((10, 1.0, 1.0, 20), (o.NumTeachers, o.LaplaceScale, o.Epsilon, o.MomentOrders));
        // Depth-1 teachers and the d, d/2 hidden widths derived from the data.
        Assert.Empty(o.TeacherDimensions);
        Assert.Null(o.GeneratorDimensions);
        Assert.Null(o.StudentDimensions);
        Assert.Null(o.EmbeddingDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task Networks_AreTheAppendixDepths()
    {
        await Task.Yield();
        var (data, columns) = Table(60, 1, 0.5);
        var generator = new PATEGANGenerator<double>(Architecture(), new PATEGANOptions<double> { Seed = 2, NumTeachers = 3, BatchSize = 16 });
        generator.Fit(data, columns, epochs: 1);
        // [0, 1]^d: one min-max value and a three-way one-hot block.
        int d = generator.EncodedWidth, h = d / 2;
        Assert.Equal(4, d);
        Assert.Equal(3, generator.TeacherCount);
        // z ∈ [0, 1]^d -> d -> d/2 -> d; logistic-regression teachers d -> 1; student d -> d -> d/2 -> 1.
        int generatorCount = (d * d + d) + (d * h + h) + (h * d + d);
        int teachers = 3 * (d + 1);
        int student = (d * d + d) + (d * h + h) + (h + 1);
        Assert.Equal(generatorCount + teachers + student, generator.GetParameters().Length);
        // Sigmoid outputs keep every generated encoding inside [0, 1]^d.
        var z = new Tensor<double>(new[] { d });
        for (int i = 0; i < d; i++) z[i] = 0.25 * i;
        var output = generator.Predict(z);
        Assert.All(Enumerable.Range(0, output.Length), i => Assert.InRange(output[i], 0.0, 1.0));
    }

    [Theory(Timeout = 60000)]
    [InlineData(1.0, 10, 1, 0.001739620162026765)]
    [InlineData(1.0, 10, 3, 0.05583749401078813)]
    [InlineData(1.0, 0, 2, 12.0)]
    [InlineData(0.5, 4, 2, 0.9623606688040371)]
    public async Task MomentsAccountant_IsAlgorithmOnesBound(double gamma, int gap, int order, double expected)
    {
        await Task.Yield();
        // Lines 17-19: min{2γ²l(l+1), log((1−q)((1−q)/(1−e^{2γ}q))^l + q e^{2γl})}, q = (2 + γ|n0−n1|)/(4 e^{γ|n0−n1|}).
        // A split vote (gap 0) only gets the data-independent 2γ²l(l+1); a confident one far less.
        Assert.Equal(expected, PATEGANGenerator<double>.MomentBound(gamma, gap, order), 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_StopsWhenThePrivacyBudgetIsSpent()
    {
        await Task.Yield();
        var (data, columns) = Table(60, 3, 0.5);
        int Iterations(double noise, double epsilon, out double spent)
        {
            var generator = new PATEGANGenerator<double>(Architecture(), new PATEGANOptions<double>
            {
                Seed = 4, NumTeachers = 3, BatchSize = 8, LaplaceScale = noise, Epsilon = epsilon,
            });
            generator.Fit(data, columns, epochs: 1000);
            spent = generator.PrivacySpent;
            return generator.IterationsTrained;
        }

        // Every iteration spends privacy; training runs until ε̂ reaches ε, well before the iteration cap.
        int quiet = Iterations(4.0, 30.0, out double quietSpent);
        Assert.True(quietSpent >= 30.0 && quiet < 1000, $"{quiet} iterations, ε̂ = {quietSpent}");
        // More noise per vote buys more iterations of the same budget.
        int loud = Iterations(8.0, 30.0, out _);
        Assert.True(loud > quiet, $"λ = 8: {loud} iterations, λ = 4: {quiet}");
    }

    [Fact(Timeout = 300000)]
    public async Task Iterations_TrainTeachersAgainstTheGeneratorAndTheGeneratorThroughTheStudent()
    {
        await Task.Yield();
        // Lines 5-10: each teacher learns to tell its partition from generated rows (Eq. 6); line 24 moves the
        // generator, which only ever sees the student.
        var (data, columns) = Table(200, 5, 0.8);
        var options = new PATEGANOptions<double> { Seed = 6, NumTeachers = 4, BatchSize = 32, LaplaceScale = 0.05, Epsilon = 1e12, LearningRate = 2e-3 };
        var untrained = new PATEGANGenerator<double>(Architecture(), options);
        untrained.Fit(data, columns, epochs: 0);
        var trained = new PATEGANGenerator<double>(Architecture(), options);
        trained.Fit(data, columns, epochs: 100);
        Assert.Equal(100, trained.IterationsTrained);

        double real = trained.MeanTeacherScore(trained.EncodeRows(data));
        double generated = trained.MeanTeacherScore(trained.GenerateEncoded(200));
        Assert.True(real > generated + 0.2, $"teachers: real {real:F3}, generated {generated:F3}");

        // Same seed, same initialization: the 100 iterations moved the generator.
        var z = new Tensor<double>(new[] { trained.EncodedWidth });
        for (int i = 0; i < z.Length; i++) z[i] = 0.2 + 0.15 * i;
        var before = untrained.Predict(z);
        var after = trained.Predict(z);
        Assert.True(Enumerable.Range(0, z.Length).Max(i => Math.Abs(after[i] - before[i])) > 1e-3);
    }
}
