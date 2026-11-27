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
        await Task.Yield();

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
        await Task.Yield();

        var input = new Tensor<double>(new[] { 2, 7 }, new Vector<double>(Probe));

        var derivative = activation.Derivative(input);

        Assert.Equal(input.Shape.ToArray(), derivative.Shape.ToArray());
        AssertValues(Probe.Select(v => activation.Derivative(v)).ToArray(), derivative.ToArray(), Tolerance, activation.GetType().Name);
    }

    [Fact(Timeout = 120000)]
    public async Task SignActivation_VectorAndTensor_MatchScalarBranches()
    {
        await Task.Yield();

        var sign = new SignActivation<double>();
        var values = new[] { -2.5, -0.0, 0.0, 1e-300, 4.0, double.NaN, double.NegativeInfinity, double.PositiveInfinity };
        var expected = new[] { -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0 };

        AssertValues(expected, sign.Activate(new Vector<double>(values)).ToArray(), 0.0, "vector");
        AssertValues(expected, sign.Activate(new Tensor<double>(new[] { 2, 4 }, new Vector<double>(values))).ToArray(), 0.0, "tensor");
    }

    [Fact(Timeout = 120000)]
    public async Task SignActivation_Derivatives_AreZero()
    {
        await Task.Yield();

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
        await Task.Yield();

        var spiking = new BinarySpikingActivation<double>(1.0, 2.0, 0.5);
        var values = new[] { -1.0, 0.999999, 1.0, 1.000001, 3.0, double.NaN };

        var output = spiking.Activate(new Vector<double>(values));

        AssertValues(new[] { 0.0, 0.0, 1.0, 1.0, 1.0, 0.0 }, output.ToArray(), 0.0, "spikes");
    }

    [Fact(Timeout = 120000)]
    public async Task BinarySpikingActivation_Tensor_ForwardIsBinaryAndGradientIsSigmoidSurrogate()
    {
        await Task.Yield();

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

    private static readonly double[] SmoothPoint = { 0.3, -1.1, 2.0, 0.5 };

    /// <summary>A fresh activation for a key; Gumbel twins share a seed, so each draws the same first sample.</summary>
    private static ActivationFunctionBase<double> Create(string key) => key switch
    {
        "relu" => new ReLUActivation<double>(),
        "relu6" => new ReLU6Activation<double>(),
        "leaky-relu" => new LeakyReLUActivation<double>(0.1),
        "elu" => new ELUActivation<double>(1.3),
        "hard-swish" => new HardSwishActivation<double>(),
        "swish" => new SwishActivation<double>(),
        "gelu" => new GELUActivation<double>(),
        "softmax" => new SoftmaxActivation<double>(),
        "softmin" => new SoftminActivation<double>(),
        "log-softmax" => new LogSoftmaxActivation<double>(),
        "log-softmin" => new LogSoftminActivation<double>(),
        "taylor-0" => new TaylorSoftmaxActivation<double>(0),
        "taylor-1" => new TaylorSoftmaxActivation<double>(1),
        "taylor-2" => new TaylorSoftmaxActivation<double>(2),
        "taylor-3" => new TaylorSoftmaxActivation<double>(3),
        "taylor-6" => new TaylorSoftmaxActivation<double>(6),
        "sparsemax" => new SparsemaxActivation<double>(),
        "spherical" => new SphericalSoftmaxActivation<double>(),
        "squash" => new SquashActivation<double>(),
        "maxout" => new MaxoutActivation<double>(2),
        "gumbel-0.5" => new GumbelSoftmaxActivation<double>(0.5, 11),
        "gumbel-1" => new GumbelSoftmaxActivation<double>(1.0, 11),
        "gumbel-2" => new GumbelSoftmaxActivation<double>(2.0, 11),
        _ => throw new ArgumentOutOfRangeException(nameof(key), key, "unknown activation"),
    };

    // Every vector Jacobian in this file against central differences of its own forward pass: the check does not
    // know the formula, so a wrong derivative cannot pass by agreeing with itself.
    // Mutation: drop the negation in SoftminActivation.Derivative, or use the softmax form in TaylorSoftmax.
    [Theory(Timeout = 120000)]
    [InlineData("relu")]
    [InlineData("relu6")]
    [InlineData("leaky-relu")]
    [InlineData("elu")]
    [InlineData("hard-swish")]
    [InlineData("swish")]
    [InlineData("gelu")]
    [InlineData("softmax")]
    [InlineData("softmin")]
    [InlineData("log-softmax")]
    [InlineData("log-softmin")]
    [InlineData("taylor-0")]
    [InlineData("taylor-1")]
    [InlineData("taylor-2")]
    [InlineData("taylor-3")]
    [InlineData("taylor-6")]
    [InlineData("sparsemax")]
    [InlineData("spherical")]
    [InlineData("squash")]
    [InlineData("maxout")]
    [InlineData("gumbel-0.5")]
    [InlineData("gumbel-1")]
    [InlineData("gumbel-2")]
    public async Task VectorJacobian_MatchesCentralFiniteDifferences(string key)
    {
        await Task.Yield();

        var numerical = NumericalJacobian(x => Create(key).Activate(new Vector<double>(x)).ToArray(), SmoothPoint);

        var jacobian = Create(key).Derivative(new Vector<double>(SmoothPoint));

        // GELU's derivative is written with six-digit constants (0.797885 for sqrt(2/pi), as on master), so it is
        // 2.1e-7 away from the derivative of its own forward pass; every other Jacobian agrees to 1e-7.
        AssertMatrix(numerical, jacobian, key == "gelu" ? 1e-6 : 1e-7, key + " vs finite differences");
    }

    // 40 elements: longer than ScalarJacobianMaxLength, so this is the engine formulation of each Jacobian, and
    // VectorJacobian_MatchesCentralFiniteDifferences (4 elements) is the scalar one.
    private static readonly double[] LongSmoothPoint =
        Enumerable.Range(0, 40).Select(i => 1.3 * Math.Sin(1.7 * i + 0.3) + 0.011 * i + 0.05).ToArray();

    // Mutation: swap the factors of the outer product in OuterWithDiagonal.
    [Theory(Timeout = 120000)]
    [InlineData("relu")]
    [InlineData("swish")]
    [InlineData("softmax")]
    [InlineData("softmin")]
    [InlineData("log-softmax")]
    [InlineData("log-softmin")]
    [InlineData("taylor-0")]
    [InlineData("taylor-1")]
    [InlineData("taylor-2")]
    [InlineData("taylor-3")]
    [InlineData("taylor-6")]
    [InlineData("sparsemax")]
    [InlineData("spherical")]
    [InlineData("squash")]
    [InlineData("maxout")]
    [InlineData("gumbel-0.5")]
    [InlineData("gumbel-2")]
    public async Task VectorJacobian_AboveTheScalarLimit_MatchesCentralFiniteDifferences(string key)
    {
        await Task.Yield();

        Assert.True(LongSmoothPoint.Min(Math.Abs) > 1e-3, "the point must stay clear of the kinks at zero");
        var numerical = NumericalJacobian(x => Create(key).Activate(new Vector<double>(x)).ToArray(), LongSmoothPoint);

        var jacobian = Create(key).Derivative(new Vector<double>(LongSmoothPoint));

        AssertMatrix(numerical, jacobian, 1e-7, key + " (40 elements) vs finite differences");
    }

    [Fact(Timeout = 120000)]
    public async Task SoftminActivation_Jacobian_HasTheSignOfSoftmaxOfTheNegatedInput()
    {
        await Task.Yield();

        // softmin(x) = softmax(-x): raising x[i] lowers its own share and raises every other one
        var input = new Vector<double>(new[] { 0.3, -1.1, 2.0, 0.0, 0.7 });
        var softmin = new SoftminActivation<double>();
        var s = softmin.Activate(input).ToArray();

        var expected = SoftmaxForm(s);
        for (int i = 0; i < s.Length; i++)
        {
            for (int j = 0; j < s.Length; j++)
            {
                expected[i, j] = -expected[i, j];
            }
        }

        var jacobian = softmin.Derivative(input);
        AssertMatrix(expected, jacobian, Tolerance, "softmin");
        Assert.True(jacobian[1, 1] < 0, "diagonal must be negative");
        Assert.True(jacobian[1, 2] > 0, "off-diagonal must be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task SoftmaxActivation_Jacobian_MatchesSoftmaxForm()
    {
        await Task.Yield();

        var input = new Vector<double>(new[] { 0.3, -1.1, 2.0, 0.0, 0.7 });
        var softmax = new SoftmaxActivation<double>();

        AssertMatrix(SoftmaxForm(softmax.Activate(input).ToArray()), softmax.Derivative(input), Tolerance, "softmax");
    }

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(6)]
    public async Task TaylorSoftmaxActivation_Jacobian_DifferentiatesThePolynomial(int order)
    {
        await Task.Yield();

        // y[i] = f(x[i]) / Z, f = sum_{k<=order} x^k / k!, so dy[i]/dx[j] = (delta_ij - y[i]) f'(x[j]) / Z with
        // f' = sum_{k<order} x^k / k!
        double[] x = { 0.3, -1.1, 2.0, 0.5 };
        double Polynomial(double v, int terms)
        {
            double sum = 0, term = 1;
            for (int k = 0; k <= terms; k++)
            {
                sum += term;
                term *= v / (k + 1);
            }

            return terms < 0 ? 0 : sum;
        }

        double z = x.Sum(v => Polynomial(v, order));
        var expected = new double[x.Length, x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            for (int j = 0; j < x.Length; j++)
            {
                double y = Polynomial(x[i], order) / z;
                expected[i, j] = ((i == j ? 1.0 : 0.0) - y) * Polynomial(x[j], order - 1) / z;
            }
        }

        AssertMatrix(expected, new TaylorSoftmaxActivation<double>(order).Derivative(new Vector<double>(x)), 1e-12,
            "taylor softmax order " + order);
    }

    [Fact(Timeout = 120000)]
    public async Task LogSoftmaxAndLogSoftmin_Jacobians_MatchDefinition()
    {
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

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

    // The tensor overload runs Engine.TensorSquash, y = x r^2 / ((1 + r^2)(r + eps)) with eps = 1e-8, so its
    // Jacobian is A I + B x x^T with A = r^2 / ((1 + r^2)(r + eps)) and
    // B = (r + 2 eps - r^3) / ((1 + r^2)^2 (r + eps)^2), the derivative of that forward element by element.
    private static double[,] StabilizedSquashJacobian(IReadOnlyList<double> x)
    {
        const double eps = 1e-8;
        int n = x.Count;
        double normSquared = x.Sum(v => v * v);
        double norm = Math.Sqrt(normSquared);
        double onePlus = 1.0 + normSquared;
        double a = normSquared / (onePlus * (norm + eps));
        double b = (norm + 2 * eps - normSquared * norm) / (onePlus * onePlus * (norm + eps) * (norm + eps));

        var jacobian = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                jacobian[i, j] = (i == j ? a : 0.0) + b * x[i] * x[j];
            }
        }

        return jacobian;
    }

    [Fact(Timeout = 120000)]
    public async Task SquashActivation_VectorJacobian_MatchesElementDefinition()
    {
        await Task.Yield();

        var squash = new SquashActivation<double>();
        var values = new[] { 0.3, -1.1, 2.0 };

        AssertMatrix(SquashJacobian(values), squash.Derivative(new Vector<double>(values)), Tolerance, "squash jacobian");
    }

    [Fact(Timeout = 120000)]
    public async Task SquashActivation_TensorDerivative_StacksPerVectorJacobians()
    {
        await Task.Yield();

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
            var expected = StabilizedSquashJacobian(rows[r]);
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
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

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
        await Task.Yield();

        const double temperature = 0.5;
        const int seed = 7;
        var input = new Vector<double>(new[] { 0.3, -1.1, 2.0, 0.0 });

        // With no forward pass before it, Derivative draws one sample; a twin with the same seed reproduces it
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

    [Fact(Timeout = 120000)]
    public async Task SphericalSoftmaxActivation_Jacobian_AtTheZeroVector_IsFiniteAndUsesTheStabilisedNorm()
    {
        await Task.Yield();

        // The norm is sqrt(||x||^2 + 1e-12), as in the forward pass, so at x = 0 it is 1e-6 and u = 0:
        // J = S (I - u u^T) / norm = S / 1e-6, with S the softmax Jacobian of the uniform output
        const int n = 4;
        var jacobian = new SphericalSoftmaxActivation<double>().Derivative(new Vector<double>(new double[n]));

        var uniform = Enumerable.Repeat(1.0 / n, n).ToArray();
        var expected = SoftmaxForm(uniform);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                expected[i, j] /= 1e-6;
            }
        }

        AssertMatrix(expected, jacobian, 1e-3, "spherical jacobian at zero");
    }

    [Fact(Timeout = 120000)]
    public async Task GumbelSoftmaxActivation_Jacobian_BelongsToTheForwardPassBeforeIt()
    {
        await Task.Yield();

        // Mutation: call Activate(input) inside Derivative again: it draws a second sample and the Jacobian no
        // longer matches the output that was returned.
        const double temperature = 0.5;
        var input = new Vector<double>(new[] { 0.3, -1.1, 2.0, 0.0 });
        var gumbel = new GumbelSoftmaxActivation<double>(temperature, 31);

        var sample = gumbel.Activate(input).ToArray();
        var expected = SoftmaxForm(sample);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                expected[i, j] *= 1.0 / temperature;
            }
        }

        AssertMatrix(expected, gumbel.Derivative(input), Tolerance, "gumbel jacobian of the sample just drawn");
        AssertMatrix(expected, gumbel.Derivative(input), Tolerance, "a second call does not draw either");
    }

    [Fact(Timeout = 120000)]
    public async Task GumbelSoftmaxActivation_RankZeroTensor_IsOneDrawAndDoesNotThrow()
    {
        await Task.Yield();

        var scalar = new Tensor<double>(new[] { 0.7 }, Array.Empty<int>());

        var output = new GumbelSoftmaxActivation<double>(1.0, 5).Activate(scalar);

        Assert.Equal(1, output.Length);
        Assert.Equal(1.0, output.ToArray()[0], 12);
    }

    // ---- edges: large magnitudes, NaN, empty inputs

    [Fact(Timeout = 120000)]
    public async Task SparsemaxActivation_Vector_AtLargeMagnitudes_StaysOnTheSimplex()
    {
        await Task.Yield();

        // Mutation: drop the shift by the maximum: 1 + k z_(k) loses its 1 and the raw inputs come back.
        Assert.Equal(new[] { 1f, 0f, 0f },
            new SparsemaxActivation<float>().Activate(new Vector<float>(new[] { 2e7f, 1f, 0f })).ToArray());
        Assert.Equal(new[] { 0f, 1f, 0f },
            new SparsemaxActivation<float>().Activate(new Vector<float>(new[] { -1e8f, -1e8f + 16f, -2e8f })).ToArray());
        Assert.Equal(new[] { 1.0, 0.0, 0.0 },
            new SparsemaxActivation<double>().Activate(new Vector<double>(new[] { 1e17, 1.0, 0.0 })).ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task SparsemaxActivation_Vector_OrdinaryInput_IsUnchangedByTheShift()
    {
        await Task.Yield();

        // [0.5, 1.5, 0.1]: sorted 1.5, 0.5, 0.1; support {1.5, 0.5}, tau = (2.0 - 1) / 2 = 0.5 -> [0, 1, 0]
        AssertValues(new[] { 0.0, 1.0, 0.0 },
            new SparsemaxActivation<double>().Activate(new Vector<double>(new[] { 0.5, 1.5, 0.1 })).ToArray(), Tolerance, "sparsemax");
        // [0.2, 0.3, 0.1]: all in the support, tau = (0.6 - 1) / 3 -> x - tau
        double tau = (0.6 - 1.0) / 3.0;
        AssertValues(new[] { 0.2 - tau, 0.3 - tau, 0.1 - tau },
            new SparsemaxActivation<double>().Activate(new Vector<double>(new[] { 0.2, 0.3, 0.1 })).ToArray(), Tolerance, "sparsemax");
    }

    [Fact(Timeout = 120000)]
    public async Task SparsemaxActivation_Vector_NaNInput_StaysNaN()
    {
        await Task.Yield();

        // Mutation: drop NaNOrZero: the engine projection turns the NaN into 0.
        Assert.True(double.IsNaN(
            new SparsemaxActivation<double>().Activate(new Vector<double>(new[] { double.NaN, 1.0, 0.0 }))[0]));
        Assert.True(float.IsNaN(
            new SparsemaxActivation<float>().Activate(new Vector<float>(new[] { float.NaN, 1f, 0f }))[0]));
    }

    [Fact(Timeout = 120000)]
    public async Task MaxoutActivation_Vector_PropagatesANaNInAnyPieceOfAGroup()
    {
        await Task.Yield();

        // Mutation: return Activate(tensor).ToVector(): ReduceMax skips a NaN that is not the first piece.
        var later = new MaxoutActivation<double>(2).Activate(new Vector<double>(new[] { 1.0, double.NaN, 3.0, 2.0 }));
        Assert.True(double.IsNaN(later[0]));
        Assert.Equal(3.0, later[1]);

        var first = new MaxoutActivation<double>(2).Activate(new Vector<double>(new[] { double.NaN, 1.0, 3.0, 2.0 }));
        Assert.True(double.IsNaN(first[0]));
        Assert.Equal(3.0, first[1]);

        var single = new MaxoutActivation<float>(2).Activate(new Vector<float>(new[] { 1f, float.NaN, 3f, 2f }));
        Assert.True(float.IsNaN(single[0]));
        Assert.Equal(3f, single[1]);

        // An infinity is a value, not a NaN
        var infinite = new MaxoutActivation<double>(2).Activate(
            new Vector<double>(new[] { double.PositiveInfinity, 1.0, double.NegativeInfinity, 2.0 }));
        Assert.Equal(new[] { double.PositiveInfinity, 2.0 }, infinite.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task SphericalSoftmaxActivation_VectorForward_IsTheTensorForward_AndUniformAtZero()
    {
        await Task.Yield();

        // Mutation: divide by the bare norm in Activate(Vector): the zero vector gives NaN while the Jacobian
        // above is the finite Jacobian of the stabilised function.
        var spherical = new SphericalSoftmaxActivation<double>();

        Assert.Equal(new[] { 0.25, 0.25, 0.25, 0.25 }, spherical.Activate(new Vector<double>(new double[4])).ToArray());
        foreach (var values in new[] { new double[4], new[] { 1e-7, -2e-7, 0.0, 3e-7 }, new[] { 0.3, -1.1, 2.0, 0.5 } })
        {
            Assert.Equal(
                spherical.Activate(new Tensor<double>(values, new[] { values.Length })).ToArray(),
                spherical.Activate(new Vector<double>(values)).ToArray());
        }
    }

    public static IEnumerable<object[]> EveryVectorJacobianKey()
        => new[]
        {
            "relu", "relu6", "leaky-relu", "elu", "hard-swish", "swish", "gelu", "softmax", "softmin", "log-softmax",
            "log-softmin", "taylor-0", "taylor-3", "sparsemax", "spherical", "squash", "maxout", "gumbel-1",
        }.Select(k => new object[] { k });

    [Theory(Timeout = 120000)]
    [MemberData(nameof(EveryVectorJacobianKey))]
    public async Task VectorJacobian_OfAnEmptyVector_IsAnEmptyMatrix(string key)
    {
        await Task.Yield();

        // Mutation: remove an empty guard: TensorEye(0) / TensorTile throw.
        var jacobian = Create(key).Derivative(new Vector<double>(0));

        Assert.Equal(0, jacobian.Rows);
        Assert.Equal(0, jacobian.Columns);
        Assert.Equal(0, Create(key).Activate(new Vector<double>(0)).Length);
    }

    // ---- float, and the non-finite values the double probe did not hold

    private static readonly double[] EdgeProbe =
    {
        double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.0, double.Epsilon, -double.Epsilon,
        1e-310, 1e300, -1e300,
    };

    private static readonly float[] FloatProbe =
    {
        -7.25f, -3.5f, -3.0f, -1.2f, -0.3f, 0.0f, 0.4f, 0.8f, 1.0f, 1.2f, 1.25f, 3.0f, 5.9f, 6.0f, 7.3f,
        float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0.0f, float.Epsilon, 1e-40f, 1e30f, -1e30f,
    };

    /// <summary>NaN matches NaN, an infinity matches the same infinity, anything else within the tolerance.</summary>
    private static void AssertClose(double expected, double actual, double absolute, double relative, string what)
    {
        if (double.IsNaN(expected) || double.IsNaN(actual))
        {
            Assert.True(double.IsNaN(expected) && double.IsNaN(actual), $"{what}: expected {expected}, got {actual}");
            return;
        }

        if (double.IsInfinity(expected) || double.IsInfinity(actual))
        {
            Assert.True(expected.Equals(actual), $"{what}: expected {expected}, got {actual}");
            return;
        }

        double bound = Math.Max(absolute, relative * Math.Abs(expected));
        Assert.True(Math.Abs(expected - actual) <= bound, $"{what}: expected {expected:R}, got {actual:R} (bound {bound:R})");
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(ElementwiseActivations))]
    public async Task ElementwiseActivation_TensorDerivative_AtNonFiniteAndTinyValues_MatchesScalarDerivative(
        ActivationFunctionBase<double> activation)
    {
        await Task.Yield();

        // The same NaN / infinity pattern as the scalar definition, and the finite values within 1e-14.
        var derivative = activation.Derivative(new Tensor<double>(EdgeProbe, new[] { EdgeProbe.Length })).ToArray();

        for (int i = 0; i < EdgeProbe.Length; i++)
        {
            AssertClose(activation.Derivative(EdgeProbe[i]), derivative[i], 1e-14, 1e-14,
                $"{activation.GetType().Name} at {EdgeProbe[i]:R}");
        }
    }

    public static IEnumerable<object[]> FloatElementwiseActivations()
    {
        yield return new object[] { new ReLUActivation<float>() };
        yield return new object[] { new ReLU6Activation<float>() };
        yield return new object[] { new LeakyReLUActivation<float>(0.1) };
        yield return new object[] { new ELUActivation<float>(1.3) };
        yield return new object[] { new HardSwishActivation<float>() };
        yield return new object[] { new SwishActivation<float>() };
        yield return new object[] { new GELUActivation<float>() };
        yield return new object[] { new BinarySpikingActivation<float>(1f, 2f, 0.5f) };
    }

    // The engine paths are not bit-identical to the scalar loop in float (vectorized exp / tanh): the measured gap is
    // at most 3.9e-6 relative (Swish) and 16 ulps (GELU at 1.25). The bound asserted here is 1e-5 relative, 1e-6
    // absolute, with the NaN / infinity pattern exact.
    [Theory(Timeout = 120000)]
    [MemberData(nameof(FloatElementwiseActivations))]
    public async Task ElementwiseActivation_Float_TensorDerivativeAndJacobian_MatchScalarDerivativeWithinTheStatedBound(
        ActivationFunctionBase<float> activation)
    {
        await Task.Yield();

        string name = activation.GetType().Name;
        var derivative = activation.Derivative(new Tensor<float>(FloatProbe, new[] { FloatProbe.Length })).ToArray();
        var jacobian = activation.Derivative(new Vector<float>(FloatProbe));

        Assert.Equal(FloatProbe.Length, jacobian.Rows);
        for (int i = 0; i < FloatProbe.Length; i++)
        {
            float expected = activation.Derivative(FloatProbe[i]);
            AssertClose(expected, derivative[i], 1e-6, 1e-5, $"{name} tensor derivative at {FloatProbe[i]:R}");
            AssertClose(expected, jacobian[i, i], 1e-6, 1e-5, $"{name} jacobian diagonal at {FloatProbe[i]:R}");
            Assert.Equal(0f, jacobian[i, (i + 1) % FloatProbe.Length]);
        }
    }

    // Short vectors take the scalar route (ScalarJacobianMaxLength = 32); these are long enough for the engine
    // evaluation, so both routes are pinned to the same scalar derivative.
    // Mutation: write the tensor derivative to the wrong stride in WriteDiagonal.
    [Theory(Timeout = 120000)]
    [MemberData(nameof(ElementwiseActivations))]
    public async Task ElementwiseActivation_VectorJacobian_AboveTheScalarLimit_IsDiagonalOfScalarDerivative(
        ActivationFunctionBase<double> activation)
    {
        await Task.Yield();

        var values = Enumerable.Range(0, 5).SelectMany(_ => Probe).ToArray();
        Assert.True(values.Length > 64);

        var jacobian = activation.Derivative(new Vector<double>(values));

        var expected = new double[values.Length, values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            expected[i, i] = activation.Derivative(values[i]);
        }

        AssertMatrix(expected, jacobian, Tolerance, activation.GetType().Name);
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(FloatElementwiseActivations))]
    public async Task ElementwiseActivation_Float_VectorJacobian_AboveTheScalarLimit_MatchesScalarDerivativeWithinTheStatedBound(
        ActivationFunctionBase<float> activation)
    {
        await Task.Yield();

        var values = Enumerable.Range(0, 3).SelectMany(_ => FloatProbe).ToArray();
        Assert.True(values.Length > 64);

        var jacobian = activation.Derivative(new Vector<float>(values));

        for (int i = 0; i < values.Length; i++)
        {
            AssertClose(activation.Derivative(values[i]), jacobian[i, i], 1e-6, 1e-5,
                $"{activation.GetType().Name} jacobian diagonal at {values[i]:R}");
            Assert.Equal(0f, jacobian[i, (i + 1) % values.Length]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SignActivation_Float_VectorAndTensor_MatchScalarBranches()
    {
        await Task.Yield();

        var sign = new SignActivation<float>();
        float[] values = { -2f, -0.0f, 0f, 3f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.Epsilon };

        var fromVector = sign.Activate(new Vector<float>(values)).ToArray();
        var fromTensor = sign.Activate(new Tensor<float>(values, new[] { values.Length })).ToArray();

        for (int i = 0; i < values.Length; i++)
        {
            AssertClose(sign.Activate(values[i]), fromVector[i], 0, 0, $"sign vector at {values[i]:R}");
            AssertClose(sign.Activate(values[i]), fromTensor[i], 0, 0, $"sign tensor at {values[i]:R}");
        }
    }
}
