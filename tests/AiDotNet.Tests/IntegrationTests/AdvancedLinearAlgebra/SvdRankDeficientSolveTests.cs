#nullable disable
using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for <c>SvdDecomposition&lt;T&gt;</c>'s singular-value cutoff, and for the
/// minimum-norm least-squares solution it makes possible.
/// </summary>
/// <remarks>
/// CRITICAL: These check answers against solutions that are known in closed form. If one fails,
/// FIX THE DECOMPOSITION — do not relax the assertion.
///
/// Before the cutoff existed, <c>Solve</c> compared each singular value against EXACT zero. A
/// rank-deficient matrix's mathematically-zero singular value comes out of a floating-point
/// computation as something like 4.9e-32 rather than 0, so it was inverted: on the 4-by-3 matrix
/// below that produced a solution of magnitude 9.0e15 with a residual of 642, where the correct
/// answer has magnitude 1.095 and a residual of zero.
/// </remarks>
public class SvdRankDeficientSolveTests
{
    /// <summary>
    /// A rank-deficient design: the third column is exactly twice the second, so the columns span
    /// a two-dimensional space and the least-squares fit has infinitely many exact solutions.
    /// </summary>
    private static Matrix<double> RankDeficient() => M(new[,]
    {
        { 1.0, 1.0, 2.0 },
        { 1.0, 2.0, 4.0 },
        { 1.0, 3.0, 6.0 },
        { 1.0, 4.0, 8.0 },
    });

    /// <summary>Targets lying exactly on <c>y = 1 + x</c>, so a residual of zero is achievable.</summary>
    private static Vector<double> Targets() => V(2.0, 3.0, 4.0, 5.0);

    /// <summary>
    /// Every solution of the form <c>(1, t, (1 - t)/2)</c> fits the data exactly. Minimizing
    /// <c>t² + ((1-t)/2)²</c> gives <c>t = 1/5</c>, so the minimum-norm answer is
    /// <c>(1, 0.2, 0.4)</c> with a length of sqrt(1.2).
    /// </summary>
    [Fact]
    public void Solve_OnARankDeficientMatrix_ReturnsTheMinimumNormSolution()
    {
        var solution = new SvdDecomposition<double>(RankDeficient()).Solve(Targets());

        Assert.Equal(1.0, solution[0], 8);
        Assert.Equal(0.2, solution[1], 8);
        Assert.Equal(0.4, solution[2], 8);
    }

    /// <summary>The answer must actually fit — a small norm is worthless if the residual is not zero.</summary>
    [Fact]
    public void Solve_OnARankDeficientMatrix_LeavesNoResidual()
    {
        var design = RankDeficient();
        var targets = Targets();
        var solution = new SvdDecomposition<double>(design).Solve(targets);

        double total = 0.0;
        for (int row = 0; row < design.Rows; row++)
        {
            double predicted = 0.0;
            for (int column = 0; column < design.Columns; column++)
            {
                predicted += design[row, column] * solution[column];
            }

            total += (predicted - targets[row]) * (predicted - targets[row]);
        }

        Assert.True(total < 1e-20, $"residual was {total}");
    }

    /// <summary>
    /// Minimum-norm means exactly that: no other exact fit is shorter. Every member of the family
    /// is checked against the returned one.
    /// </summary>
    [Fact]
    public void Solve_OnARankDeficientMatrix_IsShorterThanEveryOtherExactFit()
    {
        var solution = new SvdDecomposition<double>(RankDeficient()).Solve(Targets());

        double chosen = 0.0;
        for (int i = 0; i < solution.Length; i++) chosen += solution[i] * solution[i];

        for (double t = -2.0; t <= 2.0; t += 0.01)
        {
            // (1, t, (1 - t)/2) fits exactly for every t.
            double other = 1.0 + t * t + (1.0 - t) * (1.0 - t) / 4.0;
            Assert.True(other >= chosen - 1e-12, $"t = {t} gave a shorter exact fit");
        }
    }

    /// <summary>The numerical rank is reported, and it is 2 rather than the 3 columns present.</summary>
    [Fact]
    public void Rank_CountsOnlySingularValuesAboveTheCutoff()
    {
        Assert.Equal(2, new SvdDecomposition<double>(RankDeficient()).Rank);
    }

    /// <summary>A full-rank matrix reports its full rank, so the cutoff is not over-eager.</summary>
    [Fact]
    public void Rank_IsFullOnAWellConditionedMatrix()
    {
        var design = M(new[,]
        {
            { 1.0, 0.0, 0.0 },
            { 0.0, 2.0, 0.0 },
            { 0.0, 0.0, 3.0 },
            { 1.0, 1.0, 1.0 },
        });

        Assert.Equal(3, new SvdDecomposition<double>(design).Rank);
    }

    /// <summary>
    /// The default is the LAPACK and SciPy convention, <c>max(m, n) * eps</c>: for a 4-by-3 matrix
    /// that is 4 * 2.22e-16.
    /// </summary>
    [Fact]
    public void RelativeTolerance_DefaultsToTheLapackConvention()
    {
        var svd = new SvdDecomposition<double>(RankDeficient());

        Assert.Equal(4.0 * 2.220446049250313e-16, svd.RelativeTolerance, 15);
    }

    /// <summary>
    /// A caller who genuinely wants every non-zero singular value inverted can ask for it, and
    /// gets the enormous answer that follows. Keeping that reachable matters: the cutoff is a
    /// numerical judgement, and judgements should be overridable.
    /// </summary>
    [Fact]
    public void RelativeTolerance_OfZero_RestoresTheExactArithmeticPseudoinverse()
    {
        var svd = new SvdDecomposition<double>(
            RankDeficient(), SvdAlgorithmType.GolubReinsch, relativeTolerance: 0.0);

        Assert.Equal(3, svd.Rank);

        var solution = svd.Solve(Targets());

        double length = 0.0;
        for (int i = 0; i < solution.Length; i++) length += solution[i] * solution[i];

        Assert.True(Math.Sqrt(length) > 1e10, $"expected the blow-up, got {Math.Sqrt(length)}");
    }

    [Fact]
    public void RelativeTolerance_RejectsANegativeValue()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new SvdDecomposition<double>(
                RankDeficient(), SvdAlgorithmType.GolubReinsch, relativeTolerance: -1e-12));
    }

    /// <summary>
    /// A full-rank system must be unaffected by the change: the cutoff drops nothing, and the
    /// answer still agrees with QR, which never forms the pseudoinverse at all.
    /// </summary>
    [Fact]
    public void Solve_OnAFullRankSystem_StillAgreesWithQr()
    {
        var design = M(new[,]
        {
            { 1.0, 0.0 },
            { 1.0, 1.0 },
            { 1.0, 2.0 },
            { 1.0, 3.0 },
            { 1.0, 4.0 },
        });
        var targets = V(1.1, 2.9, 5.2, 6.8, 9.1);

        var bySvd = new SvdDecomposition<double>(design).Solve(targets);
        var byQr = new QrDecomposition<double>(design).Solve(targets);

        Assert.Equal(byQr[0], bySvd[0], 8);
        Assert.Equal(byQr[1], bySvd[1], 8);
    }

    /// <summary>
    /// A square non-singular system: the SVD answer must match the exact solution, so the cutoff
    /// has not quietly truncated a legitimate direction.
    /// </summary>
    [Fact]
    public void Solve_OnASquareSystem_MatchesTheExactAnswer()
    {
        //  x +  y +  z = 6
        // 2x -  y + 3z = 9        ->  x = 1, y = 2, z = 3
        //  x + 4y -  z = 6
        var design = M(new[,]
        {
            { 1.0, 1.0, 1.0 },
            { 2.0, -1.0, 3.0 },
            { 1.0, 4.0, -1.0 },
        });

        var solution = new SvdDecomposition<double>(design).Solve(V(6.0, 9.0, 6.0));

        Assert.Equal(1.0, solution[0], 8);
        Assert.Equal(2.0, solution[1], 8);
        Assert.Equal(3.0, solution[2], 8);
    }

    // ---------------------------------------------------------------- helpers

    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }
}
