using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regularization;

/// <summary>
/// What every regularizer must do to a coefficient, whatever the regularizer and whatever the
/// strength.
/// </summary>
/// <remarks>
/// <para>
/// The regularization suites in this repository are example-based: each one picks a handful of
/// strengths and checks the numbers that come back. Between them they use 0, 0.05, 0.1, 0.15, 0.2,
/// 0.25, 0.3, 0.5, 0.9 and 1.0 - every value at or below one.
/// </para>
/// <para>
/// <c>L2Regularization.Regularize</c> multiplied by <c>1 - strength</c>. At a strength above one
/// that factor is negative, so it FLIPPED THE SIGN of every coefficient instead of shrinking it.
/// The example suites could not see it: the bug begins immediately above the largest value any of
/// them tries, and 1.0 itself merely zeroes everything, which still reads as shrinkage.
/// </para>
/// <para>
/// It reached a user through ProximalGradientDescentOptimizer, whose proximal operator this is.
/// Minimising 0.5(w-1)^2 + 1.5|w| from w = 2 settled at -0.0714285671710968 on a problem whose
/// answer is 0 and whose iterates should never have left the positive half-line - the factor being
/// 1 - 1.5 = -0.5, with an exact fixed point at -1/14.
/// </para>
/// <para>
/// So this suite is written the other way round. It asserts PROPERTIES rather than numbers, sweeps
/// strengths that go well past one, and takes the regularizers from a single list so that a new one
/// is covered by every property here on the day it is added rather than on the day somebody
/// remembers to write its tests.
/// </para>
/// </remarks>
public class RegularizationContractTests
{
    /// <summary>
    /// Strengths that deliberately cross one and keep going. The interesting region for this class
    /// of bug is entirely above the range the example suites sample.
    /// </summary>
    private static readonly double[] Strengths =
        { 0.0, 0.01, 0.25, 0.5, 0.9, 1.0, 1.0001, 1.5, 2.0, 5.0, 100.0 };

    /// <summary>Coefficients spanning both signs, zero, and several magnitudes.</summary>
    private static readonly double[] Coefficients =
        { 0.0, 1e-9, -1e-9, 0.25, -0.25, 1.0, -1.0, 7.5, -7.5, 1234.5, -1234.5 };

    /// <summary>
    /// Every regularizer, built at a given strength.
    /// </summary>
    /// <remarks>
    /// One list, so a regularizer added to the library is covered by every property below without
    /// anyone writing a new test. Named alongside the instance so a failure says which one.
    /// </remarks>
    public static IEnumerable<object[]> AllRegularizers()
    {
        foreach (double strength in Strengths)
        {
            yield return new object[] { "None", strength };
            yield return new object[] { "L1", strength };
            yield return new object[] { "L2", strength };
            yield return new object[] { "Elastic", strength };
        }
    }

    private static IRegularization<double, Matrix<double>, Vector<double>> Build(string name, double strength)
    {
        var options = new RegularizationOptions { Strength = strength, L1Ratio = 0.5 };

        return name switch
        {
            "None" => new NoRegularization<double, Matrix<double>, Vector<double>>(),
            "L1" => new L1Regularization<double, Matrix<double>, Vector<double>>(
                        new RegularizationOptions { Type = RegularizationType.L1, Strength = strength }),
            "L2" => new L2Regularization<double, Matrix<double>, Vector<double>>(
                        new RegularizationOptions { Type = RegularizationType.L2, Strength = strength }),
            "Elastic" => new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(
                        new RegularizationOptions
                        { Type = RegularizationType.ElasticNet, Strength = strength, L1Ratio = 0.5 }),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "unknown regularizer"),
        };
    }

    /// <summary>
    /// A regularizer pulls coefficients TOWARD zero. It may take one all the way there. It may not
    /// take one past it.
    /// </summary>
    /// <remarks>
    /// This is the property the L2 defect broke, and the cheapest one to state: whatever else
    /// shrinkage means, a positive coefficient must not come back negative.
    /// </remarks>
    [Theory]
    [MemberData(nameof(AllRegularizers))]
    public void RegularizingNeverFlipsASign(string name, double strength)
    {
        var reg = Build(name, strength);
        var input = Vector<double>.FromArray(Coefficients);
        var output = reg.Regularize(input);

        for (int i = 0; i < Coefficients.Length; i++)
        {
            double before = Coefficients[i];
            double after = output[i];

            if (after == 0.0) continue;              // taken to zero is always allowed

            Assert.True(Math.Sign(before) == Math.Sign(after),
                $"{name} at strength {strength} turned {before} into {after}, " +
                "which is on the other side of zero");
        }
    }

    /// <summary>A regularizer may not move a coefficient further from zero than it started.</summary>
    [Theory]
    [MemberData(nameof(AllRegularizers))]
    public void RegularizingNeverGrowsACoefficient(string name, double strength)
    {
        var reg = Build(name, strength);
        var output = reg.Regularize(Vector<double>.FromArray(Coefficients));

        for (int i = 0; i < Coefficients.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) <= Math.Abs(Coefficients[i]) + 1e-12,
                $"{name} at strength {strength} moved {Coefficients[i]} to {output[i]}, " +
                "which is further from zero than it started");
        }
    }

    /// <summary>Zero has nowhere to be pulled to, so it stays.</summary>
    [Theory]
    [MemberData(nameof(AllRegularizers))]
    public void ZeroStaysZero(string name, double strength)
    {
        var reg = Build(name, strength);
        var output = reg.Regularize(Vector<double>.FromArray(new[] { 0.0, 0.0, 0.0 }));

        foreach (double v in output)
        {
            Assert.True(v == 0.0, $"{name} at strength {strength} moved zero to {v}");
        }
    }

    /// <summary>More strength never leaves a coefficient further from zero than less strength did.</summary>
    /// <remarks>
    /// The knob has to mean something, and it has to mean the same thing all the way along. A
    /// regularizer that shrinks harder up to some point and then less hard - or backwards - has a
    /// strength that is not a strength.
    /// </remarks>
    [Theory]
    [InlineData("None")]
    [InlineData("L1")]
    [InlineData("L2")]
    [InlineData("Elastic")]
    public void MoreStrengthNeverShrinksLess(string name)
    {
        var input = Vector<double>.FromArray(Coefficients);

        for (int s = 1; s < Strengths.Length; s++)
        {
            var weaker = Build(name, Strengths[s - 1]).Regularize(input);
            var stronger = Build(name, Strengths[s]).Regularize(input);

            for (int i = 0; i < Coefficients.Length; i++)
            {
                Assert.True(Math.Abs(stronger[i]) <= Math.Abs(weaker[i]) + 1e-12,
                    $"{name}: raising the strength from {Strengths[s - 1]} to {Strengths[s]} moved " +
                    $"{Coefficients[i]} from {weaker[i]} to {stronger[i]}, which is further from zero");
            }
        }
    }

    /// <summary>
    /// A proximal operator is non-expansive: it never pushes two points further apart than they
    /// began.
    /// </summary>
    /// <remarks>
    /// This is what makes proximal gradient descent converge at all, and it is exactly what a
    /// negative shrinkage factor destroys - multiplying by -0.5 maps 1 and -1, two units apart, to
    /// -0.5 and 0.5, still two apart, but maps 2 and 1 to -1 and -0.5, and reflects the whole line
    /// through the origin on the way. Stated over pairs so the failure names the pair.
    /// </remarks>
    [Theory]
    [MemberData(nameof(AllRegularizers))]
    public void RegularizingIsNonExpansive(string name, double strength)
    {
        var reg = Build(name, strength);

        for (int i = 0; i < Coefficients.Length; i++)
        {
            for (int j = i + 1; j < Coefficients.Length; j++)
            {
                var a = Vector<double>.FromArray(new[] { Coefficients[i] });
                var b = Vector<double>.FromArray(new[] { Coefficients[j] });

                double before = Math.Abs(Coefficients[i] - Coefficients[j]);
                double after = Math.Abs(reg.Regularize(a)[0] - reg.Regularize(b)[0]);

                Assert.True(after <= before + 1e-9,
                    $"{name} at strength {strength} moved {Coefficients[i]} and {Coefficients[j]} " +
                    $"from {before} apart to {after} apart");
            }
        }
    }
}
