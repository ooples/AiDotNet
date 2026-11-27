using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Inputs at the edges of the numeric range, and subclass overrides, that the whole-tensor activation paths must
/// handle the way the element-by-element definitions do.
/// </summary>
public class ActivationNumericalStabilityTests
{
    /// <summary>An ELU whose scalar formulas a subclass replaced: f(x) = 3x, f'(x) = 3.</summary>
    private sealed class TripledElu : ELUActivation<double>
    {
        public override double Activate(double x) => 3.0 * x;

        public override double Derivative(double x) => 3.0;
    }

    [Fact(Timeout = 60000)]
    public async Task InheritedVectorJacobian_HonorsAnOverriddenScalarDerivative_AtEngineLengths()
    {
        await Task.Yield();

        // 33 elements: past the scalar break-even length, where the Jacobian came from ELU's tensor derivative
        var values = Enumerable.Range(0, 33).Select(i => i - 16.5).ToArray();
        var activation = new TripledElu();

        var jacobian = activation.Derivative(new Vector<double>(values));
        var forward = activation.Activate(new Vector<double>(values));

        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(3.0 * values[i], forward[i]);
            for (int j = 0; j < values.Length; j++)
            {
                Assert.Equal(i == j ? 3.0 : 0.0, jacobian[i, j]);
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task GumbelSoftmax_AtALowTemperature_StaysAFiniteDistribution()
    {
        await Task.Yield();

        // Noisy logits of order 1 divided by 0.001 are in the thousands: exp of them overflowed the sum
        var gumbel = new GumbelSoftmaxActivation<double>(temperature: 0.001, seed: 7);
        var input = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        for (int draw = 0; draw < 20; draw++)
        {
            var output = gumbel.Activate(input);
            Assert.All(output.ToArray(), p => Assert.True(p >= 0 && p <= 1, $"probability {p:R}"));
            Assert.Equal(1.0, output.Sum(), 12);

            var jacobian = gumbel.Derivative(input);
            for (int i = 0; i < jacobian.Rows; i++)
            {
                for (int j = 0; j < jacobian.Columns; j++)
                {
                    Assert.False(double.IsNaN(jacobian[i, j]) || double.IsInfinity(jacobian[i, j]));
                }
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LogSoftmin_Derivative_AtLargeEqualInputs_UsesTheUniformSoftmin()
    {
        await Task.Yield();

        var jacobian = new LogSoftminActivation<double>().Derivative(new Vector<double>(new[] { 1e17, 1e17 }));

        Assert.Equal(-0.5, jacobian[0, 0], 15);
        Assert.Equal(0.5, jacobian[0, 1], 15);
        Assert.Equal(0.5, jacobian[1, 0], 15);
        Assert.Equal(-0.5, jacobian[1, 1], 15);
    }

    [Fact(Timeout = 60000)]
    public async Task Sparsemax_TensorOverload_AtLargeFloatLogits_ReturnsADistribution()
    {
        await Task.Yield();

        var input = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(new[] { 2e7f, 2e7f, 3e7f, 1e7f }));

        var output = new SparsemaxActivation<float>().Activate(input).ToArray();

        Assert.Equal(new[] { 0.5f, 0.5f, 1f, 0f }, output);
    }

    [Fact(Timeout = 60000)]
    public async Task Squash_VectorJacobian_KeepsTheRadialTerm()
    {
        await Task.Yield();

        var squash = new SquashActivation<double>();

        Assert.Equal(0.5, squash.Derivative(new Vector<double>(new[] { 1.0 }))[0, 0], 15);
        // g'(r) = 2r / (1 + r^2)^2 at r = 2^30 is 2^-89 to double precision; the folded coefficient returned 0
        double large = squash.Derivative(new Vector<double>(new[] { Math.Pow(2, 30) }))[0, 0];
        Assert.Equal(Math.Pow(2, -89), large, Math.Pow(2, -89) * 1e-12);
        // The zero vector: y ~ |x| x, so the derivative is zero, not NaN
        var atZero = squash.Derivative(new Vector<double>(new[] { 0.0, 0.0 }));
        Assert.All(new[] { atZero[0, 0], atZero[0, 1], atZero[1, 0], atZero[1, 1] }, v => Assert.Equal(0.0, v));
    }

    [Fact(Timeout = 60000)]
    public async Task Squash_TensorDerivative_DifferentiatesTheStabilisedForward_AndHandlesEdges()
    {
        await Task.Yield();

        var squash = new SquashActivation<double>();
        var rows = new[]
        {
            new[] { 1e-8 },
            new[] { 0.0 },
            new[] { 0.7 },
        };
        var input = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(rows.SelectMany(r => r).ToArray()));

        var derivative = squash.Derivative(input).ToArray();

        // Central differences of the tensor forward, y = x r^2 / ((1 + r^2)(r + 1e-8)), away from 0; at 0 the
        // forward is x |x| / 1e-8 to leading order, whose derivative there is exactly 0 (asserted below)
        foreach (int r in new[] { 0, 2 })
        {
            double x = rows[r][0];
            double h = Math.Abs(x) * 1e-4;
            double slope = (Forward(x + h) - Forward(x - h)) / (2 * h);
            Assert.Equal(slope, derivative[r], Math.Max(Math.Abs(slope) * 1e-6, 1e-18));
        }

        Assert.Equal(0.0, derivative[1]);
        Assert.Equal(new[] { 0, 3, 3 }, squash.Derivative(new Tensor<double>(new[] { 0, 3 })).Shape.ToArray());

        static double Forward(double v)
        {
            double r = Math.Abs(v);
            return v * r * r / ((1 + r * r) * (r + 1e-8));
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TaylorSoftmax_ATermWhoseRawPowerOverflows_StaysFinite()
    {
        await Task.Yield();

        // order 4 at 7e9 in float: x^4 = 2.4e39 is past float's range, x^4 / 4! = 1e38 is not
        var output = new TaylorSoftmaxActivation<float>(order: 4)
            .Activate(new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 7e9f, 0f })))
            .ToArray();

        Assert.Equal(1f, output[0]);
        Assert.True(output[1] >= 0f && output[1] < 1e-30f, $"second probability {output[1]:R}");
    }
}
