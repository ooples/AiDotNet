using System;
using System.Threading.Tasks;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Cross-checks the multi-input tape forwards against central finite differences.
/// </summary>
/// <remarks>
/// <para>
/// Triplet, contrastive and noise-contrastive losses are not pointwise: they are defined over
/// three embedding sets, an embedding pair, or a target vector plus a noise matrix. Each used to
/// ship a hand-written derivative returning one gradient per input, and those derivatives were
/// the only way to obtain gradients for them. They are deleted, so what needs proving is that the
/// tape reaches EVERY input correctly, not just that it produces a number.
/// </para>
/// <para>
/// Finite differences are taken from each loss's own forward, so this is independent of both the
/// tape and the deleted math. A gradient that reached only one of three inputs -- the failure mode
/// that matters here, and the one a "gradients are finite" assertion cannot see -- shows up
/// immediately as a mismatch against a non-zero numerical slope.
/// </para>
/// </remarks>
public class MultiInputTapeGradientTests
{
    private const double Step = 1e-5;
    private const double Tolerance = 1e-4;

    [Fact(Timeout = 120000)]
    public async Task TripletLoss_TapeGradients_MatchFiniteDifferences_ForAllThreeInputs()
    {
        await Task.Yield();

        var loss = new TripletLoss<double>(margin: 1.0);

        // Chosen so the hinge is ACTIVE: the positive is not yet closer than the negative by the
        // margin, so the loss is non-zero and every input has a real gradient. On the flat side of
        // the hinge all three gradients are legitimately zero and the test would prove nothing.
        var anchor = Matrix(new[,] { { 0.10, 0.20, 0.30 }, { -0.40, 0.50, 0.10 } });
        var positive = Matrix(new[,] { { 0.35, 0.05, 0.60 }, { -0.10, 0.20, 0.45 } });
        var negative = Matrix(new[,] { { 0.20, 0.25, 0.40 }, { -0.35, 0.40, 0.20 } });

        var (a, p, n) = TripletGradients(loss, anchor, positive, negative);

        AssertMatchesNumeric("anchor", a, anchor,
            m => LossOf(loss, m, positive, negative));
        AssertMatchesNumeric("positive", p, positive,
            m => LossOf(loss, anchor, m, negative));
        AssertMatchesNumeric("negative", n, negative,
            m => LossOf(loss, anchor, positive, m));
    }

    [Theory(Timeout = 120000)]
    [InlineData(1.0)]
    [InlineData(0.0)]
    public async Task ContrastiveLoss_TapeGradients_MatchFiniteDifferences_ForBothEmbeddings(
        double similarityLabel)
    {
        await Task.Yield();

        var loss = new ContrastiveLoss<double>(margin: 1.0);

        // Both labels are exercised because they select different branches of the loss: a similar
        // pair is driven by d^2, a dissimilar one by max(0, margin - d)^2, and a derivative correct
        // for one can be wrong for the other.
        var v1 = new Vector<double>(new[] { 0.50, 1.00, -0.30 });
        var v2 = new Vector<double>(new[] { 0.62, 0.85, -0.15 });

        using var tape = new GradientTape<double>();
        var t1 = Tensor<double>.FromVector(v1);
        var t2 = Tensor<double>.FromVector(v2);
        var scalar = loss.ComputeTapeLoss(t1, t2, similarityLabel);
        var gradients = tape.ComputeGradients(scalar, new[] { t1, t2 });

        AssertVectorMatchesNumeric("output1", gradients[t1].ToVector(), v1,
            v => ScalarOf(loss.ComputeTapeLoss(
                Tensor<double>.FromVector(v), Tensor<double>.FromVector(v2), similarityLabel)));
        AssertVectorMatchesNumeric("output2", gradients[t2].ToVector(), v2,
            v => ScalarOf(loss.ComputeTapeLoss(
                Tensor<double>.FromVector(v1), Tensor<double>.FromVector(v), similarityLabel)));
    }

    [Fact(Timeout = 120000)]
    public async Task NceLoss_TapeGradients_MatchFiniteDifferences_ForTargetAndNoise()
    {
        await Task.Yield();

        var loss = new NoiseContrastiveEstimationLoss<double>(numNoiseSamples: 2);

        var targetLogits = new Vector<double>(new[] { 0.80, -0.40 });
        var noiseLogits = Matrix(new[,] { { -0.30, 0.25 }, { 0.60, -0.75 } });

        using var tape = new GradientTape<double>();
        var targetT = Tensor<double>.FromVector(targetLogits);
        var noiseT = Tensor<double>.FromMatrix(noiseLogits);
        var scalar = loss.ComputeTapeLoss(targetT, noiseT);
        var gradients = tape.ComputeGradients(scalar, new[] { targetT, noiseT });

        AssertVectorMatchesNumeric("targetLogits", gradients[targetT].ToVector(), targetLogits,
            v => ScalarOf(loss.ComputeTapeLoss(
                Tensor<double>.FromVector(v), Tensor<double>.FromMatrix(noiseLogits))));
        AssertMatchesNumeric("noiseLogits", gradients[noiseT].ToMatrix(), noiseLogits,
            m => ScalarOf(loss.ComputeTapeLoss(
                Tensor<double>.FromVector(targetLogits), Tensor<double>.FromMatrix(m))));
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private static (Matrix<double> Anchor, Matrix<double> Positive, Matrix<double> Negative)
        TripletGradients(TripletLoss<double> loss,
                         Matrix<double> anchor, Matrix<double> positive, Matrix<double> negative)
    {
        using var tape = new GradientTape<double>();

        var a = Tensor<double>.FromMatrix(anchor);
        var p = Tensor<double>.FromMatrix(positive);
        var n = Tensor<double>.FromMatrix(negative);

        var scalar = loss.ComputeTapeLoss(a, p, n);
        var gradients = tape.ComputeGradients(scalar, new[] { a, p, n });

        return (gradients[a].ToMatrix(), gradients[p].ToMatrix(), gradients[n].ToMatrix());
    }

    private static double LossOf(TripletLoss<double> loss,
                                 Matrix<double> anchor, Matrix<double> positive, Matrix<double> negative)
        => ScalarOf(loss.ComputeTapeLoss(
            Tensor<double>.FromMatrix(anchor),
            Tensor<double>.FromMatrix(positive),
            Tensor<double>.FromMatrix(negative)));

    private static double ScalarOf(Tensor<double> t) => t[0];

    private static void AssertMatchesNumeric(
        string name, Matrix<double> gradient, Matrix<double> point, Func<Matrix<double>, double> evaluate)
    {
        for (int r = 0; r < point.Rows; r++)
        {
            for (int c = 0; c < point.Columns; c++)
            {
                double numerical = (Shifted(evaluate, point, r, c, Step)
                                    - Shifted(evaluate, point, r, c, -Step)) / (2 * Step);
                Compare($"{name}[{r},{c}]", gradient[r, c], numerical);
            }
        }
    }

    private static void AssertVectorMatchesNumeric(
        string name, Vector<double> gradient, Vector<double> point, Func<Vector<double>, double> evaluate)
    {
        for (int i = 0; i < point.Length; i++)
        {
            double numerical = (ShiftedVector(evaluate, point, i, Step)
                                - ShiftedVector(evaluate, point, i, -Step)) / (2 * Step);
            Compare($"{name}[{i}]", gradient[i], numerical);
        }
    }

    private static void Compare(string label, double got, double numerical)
    {
        Assert.False(double.IsNaN(got), $"{label}: tape gradient is NaN.");
        Assert.False(double.IsInfinity(got), $"{label}: tape gradient is Infinity.");

        double allowed = Tolerance * Math.Max(1.0, Math.Abs(numerical));
        Assert.True(
            Math.Abs(got - numerical) <= allowed,
            $"{label}: tape gives {got:G17} but central differences give {numerical:G17} "
            + $"(difference {Math.Abs(got - numerical):G6} exceeds {allowed:G6}).");
    }

    private static double Shifted(
        Func<Matrix<double>, double> evaluate, Matrix<double> point, int row, int col, double delta)
    {
        var copy = new Matrix<double>(point.Rows, point.Columns);
        for (int r = 0; r < point.Rows; r++)
            for (int c = 0; c < point.Columns; c++)
                copy[r, c] = point[r, c];

        copy[row, col] += delta;
        return evaluate(copy);
    }

    private static double ShiftedVector(
        Func<Vector<double>, double> evaluate, Vector<double> point, int index, double delta)
    {
        var copy = new Vector<double>(point.Length);
        for (int i = 0; i < point.Length; i++) copy[i] = point[i];

        copy[index] += delta;
        return evaluate(copy);
    }

    private static Matrix<double> Matrix(double[,] values)
    {
        var m = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
            for (int c = 0; c < values.GetLength(1); c++)
                m[r, c] = values[r, c];

        return m;
    }
}
