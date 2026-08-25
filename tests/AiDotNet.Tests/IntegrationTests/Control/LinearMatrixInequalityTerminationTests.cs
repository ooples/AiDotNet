using AiDotNet.Control;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Pins that the LMI search never reports success on an estimate its own grader disagrees with.
/// </summary>
/// <remarks>
/// <para>
/// The search direction comes from a power iteration on a Gershgorin-SHIFTED matrix. That shift is
/// what makes the resulting eigenvalue unreliable as a STOPPING test: adding max-row-sum + 1 to
/// every eigenvalue leaves their relative separation tiny, so a fixed iteration budget returns a
/// dominant eigenvalue that is close but not close enough.
/// </para>
/// <para>
/// Measured before the fix, on a quadratic-stability LMI for a stabilised inverted pendulum: the
/// search stopped at iteration 100 of an allowed 5000 believing it had succeeded, and the matrix it
/// returned had a smallest eigenvalue of -1.46e-02. The final factorization then disagreed and the
/// status came back IterationLimit -- so the solver contradicted itself, and its reported iteration
/// count was misleading about how much of the budget had been used.
/// </para>
/// </remarks>
public class LinearMatrixInequalityTerminationTests
{
    /// <summary>
    /// A trivially feasible LMI: <c>F0 = -I</c>, one basis matrix <c>I</c>, so any coefficient
    /// above 1 satisfies it.
    /// </summary>
    private static (Matrix<double> Constant, List<Matrix<double>> Basis) TrivialProblem(int size)
    {
        var constant = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++) constant[i, i] = -1.0;

        var element = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++) element[i, i] = 1.0;

        return (constant, new List<Matrix<double>> { element });
    }

    [Fact]
    public void AFeasibleResultIsAlwaysConfirmedByFactorization()
    {
        var (constant, basis) = TrivialProblem(4);

        var result = new LinearMatrixInequalitySolver<double>().Solve(constant, basis);

        // When the solver says Feasible, the matrix it returns must actually be positive
        // semidefinite -- the two claims are now the same test rather than two that can disagree.
        if (result.Status == LinearMatrixInequalityStatus.Feasible)
        {
            Assert.True(LinearMatrixInequalitySolver<double>.IsPositiveSemidefinite(result.Matrix),
                "the solver reported Feasible but the returned matrix is not positive semidefinite");
        }
    }

    [Fact]
    public void TheSearchDoesNotStopEarlyOnAnUnverifiedEstimate()
    {
        // A problem whose entries are large enough that the Gershgorin shift swamps the spectrum,
        // which is exactly the regime where the power-iteration estimate drifts from the truth.
        int size = 6;
        var constant = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++) constant[i, i] = -1e-4;

        var basis = new List<Matrix<double>>();
        for (int k = 0; k < size; k++)
        {
            var element = new Matrix<double>(size, size);
            element[k, k] = 1.0;

            // Off-diagonal coupling with a large scale, so the shift dominates.
            if (k + 1 < size)
            {
                element[k, k + 1] = 40.0;
                element[k + 1, k] = 40.0;
            }

            basis.Add(element);
        }

        var options = new LinearMatrixInequalityOptions { MaxIterations = 500 };
        var result = new LinearMatrixInequalitySolver<double>(options).Solve(constant, basis);

        // Whatever the outcome, the two must agree: a Feasible status and a matrix that fails the
        // factorization is the self-contradiction this test exists to prevent.
        bool actuallyPsd = LinearMatrixInequalitySolver<double>.IsPositiveSemidefinite(result.Matrix);

        Assert.True(result.Status != LinearMatrixInequalityStatus.Feasible || actuallyPsd,
            $"status {result.Status} with IsPositiveSemidefinite = {actuallyPsd}");
    }

    [Fact]
    public void AnInitialGuessLetsTheSearchStartAwayFromADegenerateOrigin()
    {
        var (constant, basis) = TrivialProblem(4);

        // From the origin the assembled matrix is F0 itself -- here a multiple of the identity,
        // where every direction is a maximal eigendirection and the power iteration has no
        // distinguished eigenvector to return. Seeding sidesteps that entirely, and on the
        // quadratic-stability problem of course Lesson 9.7 it is the difference between a solve
        // that fails after 5000 iterations and one declared Feasible on iteration 1.
        var seed = Vector<double>.FromArray(new[] { 2.0 });

        var seeded = new LinearMatrixInequalitySolver<double>().Solve(constant, basis, seed);

        Assert.Equal(LinearMatrixInequalityStatus.Feasible, seeded.Status);
        Assert.True(LinearMatrixInequalitySolver<double>.IsPositiveSemidefinite(seeded.Matrix),
            "the seeded solve reported Feasible but its matrix is not positive semidefinite");
    }

    [Fact]
    public void AnInfeasibleProblemIsNotReportedAsFeasible()
    {
        // F0 = -I with a basis that can only ever add a NEGATIVE semidefinite term, so no
        // coefficient makes the sum positive semidefinite.
        var constant = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++) constant[i, i] = -1.0;

        var element = new Matrix<double>(3, 3);
        element[0, 0] = 1.0;

        var result = new LinearMatrixInequalitySolver<double>()
            .Solve(constant, new List<Matrix<double>> { element });

        // Only the first diagonal entry can be raised; the other two stay at -1 forever.
        Assert.NotEqual(LinearMatrixInequalityStatus.Feasible, result.Status);
    }
}
