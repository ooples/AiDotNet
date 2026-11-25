using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Pins the whole-tensor (engine) vector, tensor and Jacobian paths of the activation functions to the
/// element-by-element definitions they replaced: each expected value below is computed with plain scalar
/// arithmetic from the documented formula.
/// </summary>
public class ActivationEngineParityTests
{
    private const double Tolerance = 1e-12;

    private static readonly double[] Probe =
    {
        -7.25, -3.5, -3.0, -1.2, -0.3, 0.0, 0.4, 0.8, 1.0, 1.2, 3.0, 5.9, 6.0, 7.3,
    };

    private static void AssertMatrix(double[,] expected, Matrix<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.GetLength(0), actual.Rows);
        Assert.Equal(expected.GetLength(1), actual.Columns);
        for (int i = 0; i < actual.Rows; i++)
        {
            for (int j = 0; j < actual.Columns; j++)
            {
                Assert.True(Math.Abs(expected[i, j] - actual[i, j]) <= tolerance,
                    $"{what}: [{i}, {j}] expected {expected[i, j]:R} but was {actual[i, j]:R}");
            }
        }
    }

    private static void AssertValues(IReadOnlyList<double> expected, IReadOnlyList<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{what}: [{i}] expected {expected[i]:R} but was {actual[i]:R}");
        }
    }

    private static double[,] SoftmaxForm(IReadOnlyList<double> p)
    {
        int n = p.Count;
        var jacobian = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                jacobian[i, j] = i == j ? p[i] * (1.0 - p[i]) : -p[i] * p[j];
            }
        }

        return jacobian;
    }

    public static IEnumerable<object[]> ElementwiseActivations()
    {
        yield return new object[] { new ReLUActivation<double>() };
        yield return new object[] { new ReLU6Activation<double>() };
        yield return new object[] { new LeakyReLUActivation<double>(0.1) };
        yield return new object[] { new ELUActivation<double>(1.3) };
        yield return new object[] { new HardSwishActivation<double>() };
        yield return new object[] { new SwishActivation<double>() };
        yield return new object[] { new GELUActivation<double>() };
        yield return new object[] { new BinarySpikingActivation<double>(1.0, 2.0, 0.5) };
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(ElementwiseActivations))]
    public async Task ElementwiseActivation_VectorJacobian_IsDiagonalOfScalarDerivative(ActivationFunctionBase<double> activation)
    {
        var jacobian = activation.Derivative(new Vector<double>(Probe));

        var expected = new double[Probe.Length, Probe.Length];
        for (int i = 0; i < Probe.Length; i++)
        {
            expected[i, i] = activation.Derivative(Probe[i]);
        }

        AssertMatrix(expected, jacobian, Tolerance, activation.GetType().Name);
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(ElementwiseActivations))]
    public async Task ElementwiseActivation_TensorDerivative_MatchesScalarDerivative(ActivationFunctionBase<double> activation)
    {
        var input = new Tensor<double>(new[] { 2, 7 }, new Vector<double>(Probe));

        var derivative = activation.Derivative(input);

        Assert.Equal(input.Shape.ToArray(), derivative.Shape.ToArray());
        AssertValues(Probe.Select(v => activation.Derivative(v)).ToArray(), derivative.ToArray(), Tolerance, activation.GetType().Name);
    }

    [Fact(Timeout = 120000)]
    public async Task SignActivation_VectorAndTensor_MatchScalarBranches()
    {
        var sign = new SignActivation<double>();
        var values = new[] { -2.5, -0.0, 0.0, 1e-300, 4.0, double.NaN, double.NegativeInfinity, double.PositiveInfinity };
        var expected = new[] { -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0 };

        AssertValues(expected, sign.Activate(new Vector<double>(values)).ToArray(), 0.0, "vector");
        AssertValues(expected, sign.Activate(new Tensor<double>(new[] { 2, 4 }, new Vector<double>(values))).ToArray(), 0.0, "tensor");
    }

    [Fact(Timeout = 120000)]
    public async Task SignActivation_Derivatives_AreZero()
    {
        var sign = new SignActivation<double>();

        var jacobian = sign.Derivative(new Vector<double>(new[] { -1.0, 0.0, 2.0 }));
        AssertMatrix(new double[3, 3], jacobian, 0.0, "vector jacobian");

        var stacked = sign.Derivative(new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { -1.0, 0.0, 2.0, 3.0, -4.0, 5.0 })));
        Assert.Equal(new[] { 2, 3, 3 }, stacked.Shape.ToArray());
        Assert.All(stacked.ToArray(), value => Assert.Equal(0.0, value));
    }

    [Fact(Timeout = 120000)]
    public async Task BinarySpikingActivation_Vector_FiresAtAndAboveThreshold()
    {
        var spiking = new BinarySpikingActivation<double>(1.0, 2.0, 0.5);
        var values = new[] { -1.0, 0.999999, 1.0, 1.000001, 3.0, double.NaN };

        var output = spiking.Activate(new Vector<double>(values));

        AssertValues(new[] { 0.0, 0.0, 1.0, 1.0, 1.0, 0.0 }, output.ToArray(), 0.0, "spikes");
    }

    [Fact(Timeout = 120000)]
    public async Task BinarySpikingActivation_Tensor_ForwardIsBinaryAndGradientIsSigmoidSurrogate()
    {
        const double threshold = 1.0;
        const double slope = 2.0;
        var spiking = new BinarySpikingActivation<double>(threshold, slope, 0.5);
        var values = new[] { -1.0, 0.6, 1.0, 1.3, 2.5, 0.9 };
        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(values));
        var engine = AiDotNetEngine.Current;

        using var tape = new GradientTape<double>();
        var output = spiking.Activate(input);
        var loss = engine.ReduceSum(output, new[] { 0, 1 }, keepDims: false);
        var gradients = tape.ComputeGradients(loss, new[] { input });

        AssertValues(values.Select(v => v >= threshold ? 1.0 : 0.0).ToArray(), output.ToArray(), 1e-15, "forward");

        Assert.True(gradients.TryGetValue(input, out var gradient));
        Assert.NotNull(gradient);
        var expectedGradient = values.Select(v =>
        {
            double s = 1.0 / (1.0 + Math.Exp(-slope * (v - threshold)));
            return slope * s * (1.0 - s);
        }).ToArray();
        AssertValues(expectedGradient, gradient.ToArray(), 1e-10, "surrogate gradient");
    }

    [Fact(Timeout = 120000)]
    public async Task SoftmaxFamily_Jacobians_MatchSoftmaxForm()
    {
        var input = new Vector<double>(new[] { 0.3, -1.1, 2.0, 0.0, 0.7 });

        var softmax = new SoftmaxActivation<double>();
        AssertMatrix(SoftmaxForm(softmax.Activate(input).ToArray()), softmax.Derivative(input), Tolerance, "softmax");

        var softmin = new SoftminActivation<double>();
        AssertMatrix(SoftmaxForm(softmin.Activate(input).ToArray()), softmin.Derivative(input), Tolerance, "softmin");

        var taylor = new TaylorSoftmaxActivation<double>(3);
        AssertMatrix(SoftmaxForm(taylor.Activate(input).ToArray()), taylor.Derivative(input), Tolerance, "taylor softmax");
    }

    [Fact(Timeout = 120000)]
    public async Task LogSoftmaxAndLogSoftmin_Jacobians_MatchDefinition()
    {
        var values = new[] { 0.3, -1.1, 2.0, 0.0, 0.7 };
        var input = new Vector<double>(values);
        int n = values.Length;

        double sum = values.Sum(Math.Exp);
        var softmax = values.Select(v => Math.Exp(v) / sum).ToArray();
        var expectedLogSoftmax = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                expectedLogSoftmax[i, j] = i == j ? 1.0 - softmax[i] : -softmax[j];
            }
        }

        AssertMatrix(expectedLogSoftmax, new LogSoftmaxActivation<double>().Derivative(input), Tolerance, "log softmax");

        var logSoftmin = new LogSoftminActivation<double>();
        var softmin = logSoftmin.Activate(input).ToArray().Select(Math.Exp).ToArray();
        var expectedLogSoftmin = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                expectedLogSoftmin[i, j] = i == j ? softmin[i] - 1.0 : softmin[j];
            }
        }

        AssertMatrix(expectedLogSoftmin, logSoftmin.Derivative(input), Tolerance, "log softmin");
    }

    [Theory(Timeout = 120000)]
    [InlineData(new[] { 0.3, -1.1, 2.0, 0.0, 0.7 })]
    [InlineData(new[] { 5.0, 1.0, 1.0, 1.0 })]
    [InlineData(new[] { 0.25, 0.25, 0.25, 0.25 })]
    [InlineData(new[] { -3.0, -2.5, -2.75 })]
    [InlineData(new[] { 1.5 })]
    public async Task SparsemaxActivation_Vector_MatchesSortedThresholdAlgorithm(double[] values)
    {
        var sparsemax = new SparsemaxActivation<double>();

        // Martins & Astudillo (2016), Algorithm 1
        var sorted = values.OrderByDescending(v => v).ToArray();
        double cumulative = 0.0;
        int k = 1;
        for (int i = 0; i < sorted.Length; i++)
        {
            cumulative += sorted[i];
            if (1.0 + (i + 1) * sorted[i] > cumulative)
            {
                k = i + 1;
            }
        }

        double tau = (sorted.Take(k).Sum() - 1.0) / k;
        var expected = values.Select(v => Math.Max(0.0, v - tau)).ToArray();

        var output = sparsemax.Activate(new Vector<double>(values)).ToArray();

        AssertValues(expected, output, Tolerance, "sparsemax");
        Assert.Equal(1.0, output.Sum(), 10);
    }

    [Fact(Timeout = 120000)]
    public async Task SparsemaxActivation_Jacobian_MatchesElementDefinition()
    {
        var sparsemax = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { 0.9, -1.1, 0.6, 0.0, 0.7 });
        var output = sparsemax.Activate(input).ToArray();
        int n = output.Length;
        Assert.Contains(output, v => v == 0.0);
        Assert.True(output.Count(v => v > 0.0) >= 2);

        var expected = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                expected[i, j] = output[i] > 0.0 ? (i == j ? 1.0 : -(output[j] / output[i])) : 0.0;
            }
        }

        AssertMatrix(expected, sparsemax.Derivative(input), Tolerance, "sparsemax jacobian");
    }

    [Fact(Timeout = 120000)]
    public async Task SphericalSoftmaxActivation_Jacobian_MatchesElementDefinition()
    {
        var spherical = new SphericalSoftmaxActivation<double>();
        var values = new[] { 0.3, -1.1, 2.0, 0.5 };
        var input = new Vector<double>(values);
        var output = spherical.Activate(input).ToArray();
        int n = values.Length;
        double norm = Math.Sqrt(values.Sum(v => v * v));

        var expected = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double term1 = i == j ? output[i] * (1.0 - output[i]) : -output[i] * output[j];
                double term2 = values[i] * values[j] / (norm * norm);
                expected[i, j] = (term1 - term2) / norm;
            }
        }

        AssertMatrix(expected, spherical.Derivative(input), Tolerance, "spherical softmax jacobian");
    }

    private static double[,] SquashJacobian(IReadOnlyList<double> x)
    {
        int n = x.Count;
        double normSquared = x.Sum(v => v * v);
        double norm = Math.Sqrt(normSquared);
        double scale = normSquared / (1.0 + normSquared);
        double normInv = norm > 1e-10 ? 1.0 / norm : 0.0;
        double onePlusSquared = (1.0 + normSquared) * (1.0 + normSquared);

        var jacobian = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double delta = i == j ? 1.0 : 0.0;
                double projection = normSquared > 1e-20 ? x[i] * x[j] / normSquared : 0.0;
                jacobian[i, j] = scale * ((delta - projection) * normInv) + 2.0 * (x[i] * x[j] / (onePlusSquared * norm));
            }
        }

        return jacobian;
    }

    [Fact(Timeout = 120000)]
    public async Task SquashActivation_VectorJacobian_MatchesElementDefinition()
    {
        var squash = new SquashActivation<double>();
        var values = new[] { 0.3, -1.1, 2.0 };

        AssertMatrix(SquashJacobian(values), squash.Derivative(new Vector<double>(values)), Tolerance, "squash jacobian");
    }

    [Fact(Timeout = 120000)]
    public async Task SquashActivation_TensorDerivative_StacksPerVectorJacobians()
    {
        var squash = new SquashActivation<double>();
        var rows = new[]
        {
            new[] { 0.3, -1.1, 2.0 },
            new[] { 1e-6, 2e-6, -3e-6 },
            new[] { 4.0, 0.0, -0.5 },
            new[] { -0.2, 0.2, 0.1 },
        };
        var input = new Tensor<double>(new[] { 2, 2, 3 }, new Vector<double>(rows.SelectMany(r => r).ToArray()));

        var derivative = squash.Derivative(input);

        Assert.Equal(new[] { 2, 2, 3, 3 }, derivative.Shape.ToArray());
        var actual = derivative.ToArray();
        for (int r = 0; r < rows.Length; r++)
        {
            var expected = SquashJacobian(rows[r]);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    double value = actual[(r * 3 + i) * 3 + j];
                    Assert.True(Math.Abs(expected[i, j] - value) <= Tolerance,
                        $"row {r} [{i}, {j}] expected {expected[i, j]:R} but was {value:R}");
                }
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MaxoutActivation_VectorAndJacobian_PickFirstLargestPiece()
    {
        var maxout = new MaxoutActivation<double>(3);
        // group 0: unique maximum in the middle; group 1: tie between pieces 0 and 2; group 2: all equal
        var input = new Vector<double>(new[] { 0.1, 0.9, -0.4, 2.0, -1.0, 2.0, 0.5, 0.5, 0.5 });

        AssertValues(new[] { 0.9, 2.0, 0.5 }, maxout.Activate(input).ToArray(), 0.0, "maxout");

        var expected = new double[3, 9];
        expected[0, 1] = 1.0;
        expected[1, 3] = 1.0;
        expected[2, 6] = 1.0;
        AssertMatrix(expected, maxout.Derivative(input), 0.0, "maxout jacobian");
    }

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    public async Task TaylorSoftmaxActivation_Vector_MatchesTruncatedSeries(int order)
    {
        var taylor = new TaylorSoftmaxActivation<double>(order);
        var values = new[] { 0.3, -0.4, 1.2, 0.0 };

        var series = values.Select(x =>
        {
            double result = 1.0;
            double term = 1.0;
            for (int n = 1; n <= order; n++)
            {
                term = term * x / n;
                result += term;
            }

            return result;
        }).ToArray();
        double sum = series.Sum();
        var expected = series.Select(v => v / sum).ToArray();

        AssertValues(expected, taylor.Activate(new Vector<double>(values)).ToArray(), Tolerance, "vector");
        AssertValues(expected, taylor.Activate(new Tensor<double>(new[] { 4 }, new Vector<double>(values))).ToArray(), Tolerance, "tensor");
    }

    private static double StandardGumbel(Random random)
    {
        const double floor = 1e-7;
        double inner = -Math.Log(Math.Max(random.NextDouble(), floor));
        return -Math.Log(Math.Max(inner, floor));
    }

    [Fact(Timeout = 120000)]
    public async Task GumbelSoftmaxActivation_Vector_ConsumesTheGeneratorInElementOrder()
    {
        const double temperature = 0.7;
        const int seed = 1234;
        var values = new[] { 0.3, -1.1, 2.0, 0.0, 0.7 };
        var reference = RandomHelper.CreateSeededRandom(seed);

        // logits = x + temperature * gumbel, then softmax(logits / temperature)
        var exponentials = values.Select(x => Math.Exp((x + temperature * StandardGumbel(reference)) / temperature)).ToArray();
        double sum = exponentials.Sum();
        var expected = exponentials.Select(v => v / sum).ToArray();

        var output = new GumbelSoftmaxActivation<double>(temperature, seed).Activate(new Vector<double>(values));

        AssertValues(expected, output.ToArray(), 1e-10, "gumbel softmax vector");
    }

    [Fact(Timeout = 120000)]
    public async Task GumbelSoftmaxActivation_Tensor_ConsumesTheGeneratorInRowMajorOrder()
    {
        const double temperature = 0.7;
        const int seed = 99;
        var rows = new[] { new[] { 0.3, -1.1, 2.0 }, new[] { 0.0, 0.7, -0.2 } };
        var reference = RandomHelper.CreateSeededRandom(seed);

        // softmax((x + gumbel) / temperature) along the last axis
        var expected = new List<double>();
        foreach (var row in rows)
        {
            var exponentials = row.Select(x => Math.Exp((x + StandardGumbel(reference)) / temperature)).ToArray();
            double sum = exponentials.Sum();
            expected.AddRange(exponentials.Select(v => v / sum));
        }

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(rows.SelectMany(r => r).ToArray()));
        var output = new GumbelSoftmaxActivation<double>(temperature, seed).Activate(input);

        AssertValues(expected, output.ToArray(), 1e-10, "gumbel softmax tensor");
    }

    [Fact(Timeout = 120000)]
    public async Task GumbelSoftmaxActivation_Jacobian_IsSoftmaxFormScaledByInverseTemperature()
    {
        const double temperature = 0.5;
        const int seed = 7;
        var input = new Vector<double>(new[] { 0.3, -1.1, 2.0, 0.0 });

        // Derivative draws one sample; a twin with the same seed reproduces it
        var sample = new GumbelSoftmaxActivation<double>(temperature, seed).Activate(input).ToArray();
        var expected = SoftmaxForm(sample);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                expected[i, j] *= 1.0 / temperature;
            }
        }

        AssertMatrix(expected, new GumbelSoftmaxActivation<double>(temperature, seed).Derivative(input), Tolerance, "gumbel jacobian");
    }

    [Theory(Timeout = 120000)]
    [InlineData(5)]
    [InlineData(8)]
    [InlineData(3)]
    [InlineData(2)]
    public async Task HierarchicalSoftmaxActivation_Vector_WalksEachClassPath(int numClasses)
    {
        var hierarchical = new HierarchicalSoftmaxActivation<double>(numClasses);
        var values = Enumerable.Range(0, numClasses).Select(i => 0.4 * i - 0.7).ToArray();
        var input = new Vector<double>(values);
        var weights = hierarchical.NodeWeightsTensor;
        int depth = weights.Shape[0];
        Assert.Equal(numClasses, weights.Shape[1]);
        var flatWeights = weights.ToArray();
        Assert.All(flatWeights, w => Assert.InRange(w, -0.5, 0.5));
        Assert.Contains(flatWeights, w => w != 0.0);

        var sigmoids = new double[depth];
        for (int d = 0; d < depth; d++)
        {
            double dot = 0.0;
            for (int j = 0; j < numClasses; j++)
            {
                dot += values[j] * flatWeights[d * numClasses + j];
            }

            sigmoids[d] = 1.0 / (1.0 + Math.Exp(-dot));
        }

        var expectedProbabilities = new double[numClasses];
        var expectedJacobian = new double[numClasses, numClasses];
        for (int c = 0; c < numClasses; c++)
        {
            double probability = 1.0;
            int node = 1;
            for (int d = 0; d < depth; d++)
            {
                bool goRight = (c & (1 << (depth - d - 1))) != 0;
                probability *= goRight ? sigmoids[d] : 1.0 - sigmoids[d];
                double factor = sigmoids[d] * (1.0 - sigmoids[d]);
                for (int j = 0; j < numClasses; j++)
                {
                    expectedJacobian[c, j] += (goRight ? factor : -factor) * flatWeights[d * numClasses + j];
                }

                node = node * 2 + (goRight ? 1 : 0);
                if (node >= numClasses)
                {
                    break;
                }
            }

            expectedProbabilities[c] = probability;
        }

        AssertValues(expectedProbabilities, hierarchical.Activate(input).ToArray(), Tolerance, "path probabilities");
        AssertMatrix(expectedJacobian, hierarchical.Derivative(input), Tolerance, "path derivatives");
    }
}
