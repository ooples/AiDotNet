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

    /// <summary>
    /// Central-difference Jacobian of <paramref name="function"/> at <paramref name="point"/>: column j is
    /// (f(x + h e_j) - f(x - h e_j)) / 2h. Independent of any closed form, so it catches a wrong formula.
    /// </summary>
    private static double[,] NumericalJacobian(Func<double[], double[]> function, double[] point, double step = 1e-6)
    {
        int outputs = function(point).Length;
        var jacobian = new double[outputs, point.Length];
        for (int j = 0; j < point.Length; j++)
        {
            var forward = (double[])point.Clone();
            var backward = (double[])point.Clone();
            forward[j] += step;
            backward[j] -= step;
            var high = function(forward);
            var low = function(backward);
            for (int i = 0; i < outputs; i++)
            {
                jacobian[i, j] = (high[i] - low[i]) / (2.0 * step);
            }
        }

        return jacobian;
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
    public async Task SparsemaxActivation_Jacobian_IsDiagOfSupportMinusSharedThreshold()
    {
        // Martins and Astudillo (2016), eq. 11: J = diag(s) - s s^T / |S|, s the indicator of the support.
        // The Jacobian used to put -(output[j] / output[i]) off the diagonal and 1 on it, which is not the
        // derivative of max(0, z - tau(z)) (it is not even symmetric, and its rows do not sum to zero).
        var sparsemax = new SparsemaxActivation<double>();
        var values = new[] { 0.9, -1.1, 0.6, 0.0, 0.7 };
        var input = new Vector<double>(values);
        var output = sparsemax.Activate(input).ToArray();
        int n = output.Length;
        int supportSize = output.Count(v => v > 0.0);
        Assert.Equal(3, supportSize);
        Assert.Equal(2, output.Count(v => v == 0.0));

        var expected = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                bool both = output[i] > 0.0 && output[j] > 0.0;
                expected[i, j] = (i == j && both ? 1.0 : 0.0) - (both ? 1.0 / supportSize : 0.0);
            }
        }

        // Hand values: 2/3 on the support's diagonal, -1/3 between two members of the support
        Assert.Equal(2.0 / 3.0, expected[0, 0], 15);
        Assert.Equal(-1.0 / 3.0, expected[0, 2], 15);
        Assert.Equal(0.0, expected[0, 1], 15);

        var jacobian = sparsemax.Derivative(input);
        AssertMatrix(expected, jacobian, Tolerance, "sparsemax jacobian");

        // Sparsemax is piecewise linear, so the central difference is exact away from a kink
        var numerical = NumericalJacobian(x => sparsemax.Activate(new Vector<double>(x)).ToArray(), values);
        AssertMatrix(numerical, jacobian, 1e-8, "sparsemax jacobian vs finite differences");
    }

    [Fact(Timeout = 120000)]
    public async Task SparsemaxActivation_Jacobian_OfASingleWinner_IsZero()
    {
        // One coordinate far above the rest takes all the mass: the output is the constant one-hot vector
        // in a neighbourhood, so every derivative is 0 (the support has one member and 1 - 1/1 = 0)
        var sparsemax = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { 5.0, 0.1, -2.0 });
        Assert.Equal(new[] { 1.0, 0.0, 0.0 }, sparsemax.Activate(input).ToArray());

        AssertMatrix(new double[3, 3], sparsemax.Derivative(input), Tolerance, "single-winner jacobian");
    }

    [Fact(Timeout = 120000)]
    public async Task SphericalSoftmaxActivation_Jacobian_IsTheChainRuleThroughTheNormalization()
    {
        var spherical = new SphericalSoftmaxActivation<double>();
        var values = new[] { 0.3, -1.1, 2.0, 0.5 };
        var input = new Vector<double>(values);
        var output = spherical.Activate(input).ToArray();
        int n = values.Length;
        double norm = Math.Sqrt(values.Sum(v => v * v));

        // y = softmax(u), u = x / ||x||: J = S (I - u u^T) / ||x|| with S = diag(y) - y y^T.
        // The Jacobian used to be (S - u u^T) / ||x||, which subtracts the projector from the softmax
        // Jacobian instead of multiplying by its complement.
        var softmaxForm = SoftmaxForm(output);
        var unit = values.Select(v => v / norm).ToArray();
        var expected = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            double alongUnit = 0.0;
            for (int k = 0; k < n; k++)
            {
                alongUnit += softmaxForm[i, k] * unit[k];
            }

            for (int j = 0; j < n; j++)
            {
                expected[i, j] = (softmaxForm[i, j] - alongUnit * unit[j]) / norm;
            }
        }

        var jacobian = spherical.Derivative(input);
        AssertMatrix(expected, jacobian, Tolerance, "spherical softmax jacobian");

        var numerical = NumericalJacobian(x => spherical.Activate(new Vector<double>(x)).ToArray(), values);
        AssertMatrix(numerical, jacobian, 1e-8, "spherical softmax jacobian vs finite differences");

        // The output does not change when x is scaled, so x is in the null space of the Jacobian
        for (int i = 0; i < n; i++)
        {
            double alongInput = 0.0;
            for (int j = 0; j < n; j++)
            {
                alongInput += jacobian[i, j] * values[j];
            }

            Assert.True(Math.Abs(alongInput) <= 1e-12, $"J x should be 0 but row {i} gave {alongInput:R}");
        }
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

        // y = softmax((x + gumbel) / temperature) (Jang et al., 2017, eq. 1; torch.nn.functional.gumbel_softmax
        // computes "(logits + gumbels) / tau"). The vector overload used to scale the noise by the temperature
        // first, which cancelled the division and left softmax(x / temperature + gumbel).
        var exponentials = values.Select(x => Math.Exp((x + StandardGumbel(reference)) / temperature)).ToArray();
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

    [Theory(Timeout = 120000)]
    [InlineData(0.25)]
    [InlineData(1.0)]
    [InlineData(3.0)]
    public async Task GumbelSoftmaxActivation_VectorAndTensorOverloads_DrawTheSameSample(double temperature)
    {
        const int seed = 23;
        var values = new[] { 0.3, -1.1, 2.0, 0.0, 0.9 };

        var fromVector = new GumbelSoftmaxActivation<double>(temperature, seed).Activate(new Vector<double>(values));
        var fromTensor = new GumbelSoftmaxActivation<double>(temperature, seed)
            .Activate(new Tensor<double>(new[] { values.Length }, new Vector<double>(values)));

        AssertValues(fromTensor.ToArray(), fromVector.ToArray(), 1e-12, "gumbel softmax vector vs tensor");
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
            // The factors class c picks up along its path, and whether each one is a right turn
            var factors = new List<double>();
            var rightTurns = new List<bool>();
            int node = 1;
            for (int d = 0; d < depth; d++)
            {
                bool goRight = (c & (1 << (depth - d - 1))) != 0;
                factors.Add(goRight ? sigmoids[d] : 1.0 - sigmoids[d]);
                rightTurns.Add(goRight);

                node = node * 2 + (goRight ? 1 : 0);
                if (node >= numClasses)
                {
                    break;
                }
            }

            double probability = 1.0;
            foreach (double factor in factors)
            {
                probability *= factor;
            }

            expectedProbabilities[c] = probability;

            // Product rule: differentiate one factor, keep the others. The Jacobian used to drop "the
            // others", which is only right for a path of length 1.
            for (int d = 0; d < factors.Count; d++)
            {
                double others = 1.0;
                for (int e = 0; e < factors.Count; e++)
                {
                    if (e != d)
                    {
                        others *= factors[e];
                    }
                }

                double slope = sigmoids[d] * (1.0 - sigmoids[d]);
                double coefficient = (rightTurns[d] ? slope : -slope) * others;
                for (int j = 0; j < numClasses; j++)
                {
                    expectedJacobian[c, j] += coefficient * flatWeights[d * numClasses + j];
                }
            }
        }

        var jacobian = hierarchical.Derivative(input);
        AssertValues(expectedProbabilities, hierarchical.Activate(input).ToArray(), Tolerance, "path probabilities");
        AssertMatrix(expectedJacobian, jacobian, Tolerance, "path derivatives");

        var numerical = NumericalJacobian(x => hierarchical.Activate(new Vector<double>(x)).ToArray(), values);
        AssertMatrix(numerical, jacobian, 1e-8, "path derivatives vs finite differences");
    }
}
