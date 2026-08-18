using AiDotNet.Models.Options;
using AiDotNet.Solvers.InteriorPoint;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// The interior-point solver must handle a quadratic program whose variables have no lower bounds.
/// </summary>
/// <remarks>
/// <para>
/// The standard-form rewrite splits every free variable into <c>z⁺ − z⁻</c>, which gives the
/// projected Hessian the rank-deficient <c>[[Q, −Q], [−Q, Q]]</c> structure. The Newton system's
/// kernel <c>K = Q + Z⁻¹S</c> then goes numerically singular along the split directions as the
/// iterates approach the boundary and the smallest <c>s/x</c> ratios fall towards zero. The
/// regularization the solver already applied to the m×m normal matrix was never applied to that
/// n×n kernel, so its LU factorization threw, <c>NewtonSystem.Build</c> returned null, and the
/// solve was abandoned wherever it happened to be.
/// </para>
/// <para>
/// Measured before the fix on the problem below: status <c>IterationLimit</c> after 20 iterations at
/// (16.876, 13.124) costing 741.82 against a true optimum of (15, 15) costing 675 — and raising
/// <c>MaxIterations</c> from 100 to 2000 changed nothing, because the exit was the failed
/// factorization rather than the iteration cap. That fixed iteration count under two very different
/// caps is what identified the real exit path.
/// </para>
/// </remarks>
public class InteriorPointFreeVariableQpTests
{
    /// <summary>
    /// Two generators meeting a demand of 30, the cheaper one capped at 15.
    /// </summary>
    /// <remarks>
    /// Minimize <c>2x₁² + x₂²</c> (so Q = diag(4, 2) under the solver's ½xᵀQx convention) subject to
    /// <c>x₁ + x₂ = 30</c> and <c>x₂ ≤ 15</c>. The cap binds, giving (15, 15) at a cost of 675 with a
    /// capacity multiplier of 30 — verifiable by hand and reproduced independently by the active-set
    /// solver.
    /// </remarks>
    private static QuadraticProgram<double> Program(bool declareLowerBounds)
    {
        var quadratic = new Matrix<double>(2, 2);
        quadratic[0, 0] = 4.0;
        quadratic[1, 1] = 2.0;

        var equality = new Matrix<double>(1, 2);
        equality[0, 0] = 1.0;
        equality[0, 1] = 1.0;

        var inequality = new Matrix<double>(1, 2);
        inequality[0, 0] = 0.0;
        inequality[0, 1] = 1.0;

        return new QuadraticProgram<double>(
            quadratic: quadratic,
            linear: new Vector<double>(2),
            inequalityMatrix: inequality,
            inequalityBounds: Vector<double>.FromArray(new[] { 15.0 }),
            equalityMatrix: equality,
            equalityBounds: Vector<double>.FromArray(new[] { 30.0 }),
            lowerBounds: declareLowerBounds
                ? Vector<double>.FromArray(new[] { 0.0, 0.0 })
                : null);
    }

    private static void AssertSolvesTheGeneratorProblem(QuadraticProgramSolution<double> solution)
    {
        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.NotNull(solution.Solution);

        Assert.True(Math.Abs(solution.Solution![0] - 15.0) < 1e-4, $"x1 = {solution.Solution[0]}");
        Assert.True(Math.Abs(solution.Solution[1] - 15.0) < 1e-4, $"x2 = {solution.Solution[1]}");
        Assert.True(Math.Abs(solution.ObjectiveValue - 675.0) < 1e-3,
            $"objective = {solution.ObjectiveValue}");
    }

    [Fact]
    public void FreeVariables_ReachTheOptimum()
    {
        AssertSolvesTheGeneratorProblem(new InteriorPointSolver<double>().Solve(Program(false)));
    }

    [Fact]
    public void DeclaredLowerBounds_ReachTheSameOptimum()
    {
        // The formulation that avoided the split and therefore always worked. Kept so that a future
        // change which fixes the free case by breaking this one cannot pass.
        AssertSolvesTheGeneratorProblem(new InteriorPointSolver<double>().Solve(Program(true)));
    }

    [Fact]
    public void FreeVariables_DoNotDependOnTheIterationCap()
    {
        // The diagnostic that located the defect, kept as an assertion. If the solve ever exits
        // through the failed-factorization path again, the answer will stop improving with the cap
        // and this comparison will show it.
        var tight = new InteriorPointSolver<double>(
            new InteriorPointSolverOptions { MaxIterations = 100 }).Solve(Program(false));
        var loose = new InteriorPointSolver<double>(
            new InteriorPointSolverOptions { MaxIterations = 2000 }).Solve(Program(false));

        AssertSolvesTheGeneratorProblem(tight);
        AssertSolvesTheGeneratorProblem(loose);

        Assert.True(tight.Iterations < 100,
            $"the solve should converge well inside the cap; took {tight.Iterations}");
    }

    [Fact]
    public void FreeVariables_AgreeWithTheActiveSetSolver()
    {
        // An independent method on the same program. These two have almost nothing in common -- one
        // guesses the active set and lands on the boundary, the other refuses to approach it -- so
        // agreement is real evidence rather than self-consistency.
        var interior = new InteriorPointSolver<double>().Solve(Program(false));
        var activeSet = new ActiveSetQuadraticProgramSolver<double>().Solve(Program(false));

        Assert.Equal(LinearProgramStatus.Optimal, interior.Status);
        Assert.Equal(LinearProgramStatus.Optimal, activeSet.Status);

        Assert.True(Math.Abs(interior.Solution![0] - activeSet.Solution![0]) < 1e-4,
            $"x1: interior {interior.Solution[0]}, active set {activeSet.Solution[0]}");
        Assert.True(Math.Abs(interior.Solution[1] - activeSet.Solution[1]) < 1e-4,
            $"x2: interior {interior.Solution[1]}, active set {activeSet.Solution[1]}");
        Assert.True(Math.Abs(interior.ObjectiveValue - activeSet.ObjectiveValue) < 1e-3,
            $"objective: interior {interior.ObjectiveValue}, active set {activeSet.ObjectiveValue}");
    }

    [Fact]
    public void FreeVariables_RecoverTheCapacityMultiplier()
    {
        // The multiplier is the point of solving this problem at all -- it is the price of one more
        // unit of capacity, and hand calculation puts it at 30.
        var solution = new InteriorPointSolver<double>().Solve(Program(false));

        Assert.NotNull(solution.InequalityMultipliers);
        Assert.True(Math.Abs(Math.Abs(solution.InequalityMultipliers![0]) - 30.0) < 1e-2,
            $"capacity multiplier = {solution.InequalityMultipliers[0]}, expected magnitude 30");
    }
}
