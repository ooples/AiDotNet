using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Checks every loss function's tape gradient against numerical finite differences.
/// </summary>
/// <remarks>
/// <para>
/// The rest of the suite asserts gradient VALUES that were originally written against the
/// hand-written analytic derivatives this change deleted. That proves the tape agrees with what
/// was there before -- but if a hand-written derivative was itself wrong, the test enshrined the
/// error and still passes. Finite differences are computed from the loss's own forward pass and
/// nothing else, so they are an independent check: they can disagree with both.
/// </para>
/// <para>
/// This also covers the losses that never had an analytic derivative at all, whose
/// CalculateDerivative threw. They had no gradient test that could run; now they do.
/// </para>
/// <para>
/// The check is the central-difference approximation
/// <c>dL/dx_i ~= (L(x + h e_i) - L(x - h e_i)) / 2h</c>, whose error is O(h^2) rather than the
/// O(h) of a forward difference, so a loose tolerance is not hiding a real disagreement.
/// </para>
/// </remarks>
public class TapeGradientFiniteDifferenceTests
{
    // Large enough that (L(x+h) - L(x-h)) keeps enough significant digits in double precision,
    // small enough that the O(h^2) truncation term stays well under the tolerance.
    private const double Step = 1e-5;
    private const double Tolerance = 1e-4;

    /// <summary>
    /// Losses paired with an input that is valid for them, since several are only defined on a
    /// restricted domain -- probabilities in (0, 1), targets in {0, 1}, or {-1, +1} margins.
    /// </summary>
    public static IEnumerable<object[]> Losses()
    {
        // Regression: unrestricted real predictions and targets.
        var reg = new double[] { 0.7, -0.4, 1.3, 0.2 };
        var regTarget = new double[] { 0.5, -0.1, 1.0, 0.35 };

        yield return Case(new MeanSquaredErrorLoss<double>(), reg, regTarget);
        yield return Case(new MeanAbsoluteErrorLoss<double>(), reg, regTarget);
        yield return Case(new RootMeanSquaredErrorLoss<double>(), reg, regTarget);
        yield return Case(new HuberLoss<double>(), reg, regTarget);
        yield return Case(new LogCoshLoss<double>(), reg, regTarget);
        yield return Case(new MeanBiasErrorLoss<double>(), reg, regTarget);
        yield return Case(new CharbonnierLoss<double>(), reg, regTarget);
        yield return Case(new QuantileLoss<double>(), reg, regTarget);

        // Probabilities: predictions strictly inside (0, 1); targets are 0/1 labels.
        var prob = new double[] { 0.7, 0.2, 0.6, 0.35 };
        var binary = new double[] { 1.0, 0.0, 1.0, 0.0 };

        yield return Case(new BinaryCrossEntropyLoss<double>(), prob, binary);
        yield return Case(new CrossEntropyLoss<double>(), prob, binary);
        yield return Case(new DiceLoss<double>(), prob, binary);
        yield return Case(new JaccardLoss<double>(), prob, binary);
        yield return Case(new FocalLoss<double>(), prob, binary);

        // Margin-based: targets in {-1, +1}, predictions unrestricted scores.
        var scores = new double[] { 0.8, -0.6, 0.3, -1.2 };
        var signs = new double[] { 1.0, -1.0, 1.0, -1.0 };

        yield return Case(new HingeLoss<double>(), scores, signs);
        yield return Case(new SquaredHingeLoss<double>(), scores, signs);
        yield return Case(new ExponentialLoss<double>(), scores, signs);
        yield return Case(new ModifiedHuberLoss<double>(), scores, signs);

        // Logits: unrestricted, with the sigmoid/softmax applied inside the loss.
        yield return Case(new BinaryCrossEntropyWithLogitsLoss<double>(), scores, binary);

        // Strictly positive predictions and targets.
        var positive = new double[] { 1.4, 0.6, 2.1, 0.9 };
        var positiveTarget = new double[] { 1.1, 0.8, 1.7, 1.2 };

        yield return Case(new PoissonLoss<double>(), positive, positiveTarget);
        yield return Case(new KullbackLeiblerDivergence<double>(), prob, new double[] { 0.6, 0.25, 0.55, 0.4 });
    }

    private static object[] Case(ILossFunction<double> loss, double[] predicted, double[] actual)
        => new object[] { loss.GetType().Name, loss, predicted, actual };

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Losses))]
    public async Task TapeGradient_MatchesCentralFiniteDifference(
        string name, ILossFunction<double> loss, double[] predicted, double[] actual)
    {
        await Task.Yield();

        var target = Tensor<double>.FromVector(new Vector<double>(actual));
        var analytic = loss.ComputeGradient(
            Tensor<double>.FromVector(new Vector<double>(predicted)), target);

        Assert.Equal(predicted.Length, analytic.Length);

        for (int i = 0; i < predicted.Length; i++)
        {
            double numerical = (Evaluate(loss, predicted, actual, i, Step)
                                - Evaluate(loss, predicted, actual, i, -Step)) / (2 * Step);

            double got = analytic[i];

            Assert.False(double.IsNaN(got), $"{name}: tape gradient[{i}] is NaN.");
            Assert.False(double.IsInfinity(got), $"{name}: tape gradient[{i}] is Infinity.");

            // Relative where the gradient is large, absolute where it is near zero -- a fixed
            // absolute bound would be vacuous for one and unreachable for the other.
            double allowed = Tolerance * Math.Max(1.0, Math.Abs(numerical));
            Assert.True(
                Math.Abs(got - numerical) <= allowed,
                $"{name}: gradient[{i}] = {got:G17} but central differences give {numerical:G17} "
                + $"(difference {Math.Abs(got - numerical):G6} exceeds {allowed:G6}).");
        }
    }

    /// <summary>
    /// Evaluates the loss with a single component of the prediction displaced, using the loss's
    /// own tape forward so the comparison is against the same expression being differentiated.
    /// </summary>
    private static double Evaluate(
        ILossFunction<double> loss, double[] predicted, double[] actual, int index, double delta)
    {
        var shifted = (double[])predicted.Clone();
        shifted[index] += delta;

        var value = loss.ComputeTapeLoss(
            Tensor<double>.FromVector(new Vector<double>(shifted)),
            Tensor<double>.FromVector(new Vector<double>(actual)));

        return value[0];
    }
}
