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
/// Kim et al. 2021, OCT-GAN: a neural-ODE generator over z ⊕ c, a discriminator that scores an ODE trajectory read at
/// learnable time points, the conditional matching loss, and Algorithm 1's schedule (sections 3-4) (#2397).
/// </summary>
public class OCTGANPaperTests
{
    private static (Matrix<double> Data, List<ColumnMetadata> Columns) Table(int rows, int seed)
    {
        // A continuous column whose mean depends on a three-way category.
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new Matrix<double>(rows, 2);
        for (int i = 0; i < rows; i++)
        {
            int category = random.Next(3);
            data[i, 0] = 10.0 * category + random.NextDouble();
            data[i, 1] = category;
        }
        var columns = new List<ColumnMetadata>
        {
            new("Value", ColumnDataType.Continuous, columnIndex: 0),
            new("Category", ColumnDataType.Categorical, new[] { "A", "B", "C" }, columnIndex: 1),
        };
        return (data, columns);
    }

    private static OCTGANOptions<double> Small(int seed) => new()
    {
        Seed = seed, EmbeddingDimension = 8, GeneratorDimensions = [32, 32], DiscriminatorDimensions = [16, 16],
        TimePoints = 3, OdeSolverSteps = 1, BatchSize = 50, VGMModes = 3, LearningRate = 2e-3,
        DiscriminatorDropout = 0.0,
    };

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 8, outputSize: 2);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new OCTGANOptions<double>();
        Assert.Equal(new[] { 256, 256 }, o.GeneratorDimensions);
        Assert.Equal(new[] { 256, 256 }, o.DiscriminatorDimensions);
        Assert.Equal((3, 300, 500, 10), (o.TimePoints, o.Epochs, o.BatchSize, o.VGMModes));
        Assert.Equal((0.97, 2), (o.LearningRateDecay, o.LearningRateDecayPeriod));
        Assert.Equal(2e-4, o.LearningRate);
        Assert.Equal(1, o.DiscriminatorSteps);
        // t_0 = 0 and t_m = 1 are fixed; before training the interior points split [0, 1] evenly (OCT-GAN(fixed)).
        var generator = new OCTGANGenerator<double>();
        Assert.Equal(new[] { 1.0 / 3, 2.0 / 3 }, generator.TimePointValues, new DoubleTolerance(1e-12));
    }

    [Fact(Timeout = 300000)]
    public async Task TimePoints_AreDifferentiableBoundsOfTheTrajectory()
    {
        await Task.Yield();
        // Proposition 3.1: dL/dt_i exists and flows through the integral bounds; through the solver it must match
        // central differences of the score.
        var (data, columns) = Table(100, 1);
        var generator = new OCTGANGenerator<double>(Architecture(), Small(2));
        generator.Fit(data, columns, epochs: 1);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var rows = new Tensor<double>(new[] { 8, generator.DiscriminatorInputWidth });
        for (int i = 0; i < rows.Length; i++) rows[i] = rng.NextDouble();
        for (int index = 0; index < 2; index++)
        {
            var (analytic, numeric) = generator.TimePointGradientCheck(rows, index, 1e-5);
            Assert.True(Math.Abs(analytic) > 1e-9, $"dD/dt_{index + 1} = {analytic}");
            Assert.Equal(numeric, analytic, 6);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Training_MovesTheTimePointsInsideTheTrajectoryAndAnnealsTheRates()
    {
        await Task.Yield();
        var (data, columns) = Table(100, 4);
        var options = Small(5);
        var generator = new OCTGANGenerator<double>(Architecture(), options);
        generator.Fit(data, columns, epochs: 4);
        // Algorithm 1, line 7 trains the interior t_i, which stay ordered strictly inside (0, 1); line 9 decays the
        // learning rates by ψ every ξ = 2 epochs: twice in four.
        var t = generator.TimePointValues;
        Assert.NotEqual(new[] { 1.0 / 3, 2.0 / 3 }, t);
        Assert.True(0 < t[0] && t[0] < t[1] && t[1] < 1, $"t = {string.Join(", ", t)}");
        Assert.Equal(options.LearningRate * 0.97 * 0.97, generator.GeneratorLearningRate, 12);
        // p(0) = z ⊕ c: the generator reads the noise and a one-hot block per discrete column.
        Assert.Equal(options.EmbeddingDimension + 3, generator.GeneratorInputWidth);
    }

    [Fact(Timeout = 600000)]
    public async Task ConditionalGenerator_HonoursTheRequestedCategory()
    {
        await Task.Yield();
        // Section 3.3: L_matching = H(c_s, d̂_s) makes the generated discrete column follow the condition.
        var (data, columns) = Table(150, 6);
        var generator = new OCTGANGenerator<double>(Architecture(), Small(7));
        generator.Fit(data, columns, epochs: 40);
        for (int category = 0; category < 3; category++)
        {
            var conditionColumn = new Vector<double>(Enumerable.Repeat(0.0, 100).ToArray());
            var conditionValue = new Vector<double>(Enumerable.Repeat((double)category, 100).ToArray());
            var rows = generator.Generate(100, conditionColumn, conditionValue);
            double hit = Enumerable.Range(0, rows.Rows).Average(i => Math.Abs(rows[i, 1] - category) < 0.5 ? 1.0 : 0.0);
            Assert.True(hit > 0.8, $"category {category}: {hit:P0} of the rows carry it");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_IsTheDeterministicGeneratorOverZAndC()
    {
        await Task.Yield();
        var (data, columns) = Table(100, 8);
        var generator = new OCTGANGenerator<double>(Architecture(), Small(9));
        generator.Fit(data, columns, epochs: 1);
        var p = new Tensor<double>(new[] { generator.GeneratorInputWidth });
        for (int i = 0; i < p.Length; i++) p[i] = 0.1 * (i + 1);
        var first = generator.Predict(p);
        var second = generator.Predict(p);
        Assert.Equal(first.ToArray(), second.ToArray());
        // Mode and category blocks are distributions; the mode-normalized value is a tanh.
        Assert.InRange(first[0], -1.0, 1.0);
        double categories = first[first.Length - 1] + first[first.Length - 2] + first[first.Length - 3];
        Assert.Equal(1.0, categories, 9);
        // Similar initial inputs give similar rows (the ODE is a homeomorphism, section 3.3).
        var nearby = p.Clone();
        nearby[0] += 1e-6;
        var moved = generator.Predict(nearby);
        Assert.True(Enumerable.Range(0, first.Length).Max(i => Math.Abs(moved[i] - first[i])) < 1e-3);
    }

    private sealed class DoubleTolerance : IEqualityComparer<double>
    {
        private readonly double _tolerance;
        public DoubleTolerance(double tolerance) => _tolerance = tolerance;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _tolerance;
        public int GetHashCode(double obj) => 0;
    }
}
