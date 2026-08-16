#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Solvers.InteriorPoint;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Solvers;

/// <summary>
/// Integration tests for the Mehrotra predictor-corrector interior-point solver.
/// </summary>
/// <remarks>
/// CRITICAL: Every expected value here is either hand-solvable, fixed by optimization theory
/// (strong duality, the KKT conditions), or cross-checked against an independent algorithm — the
/// simplex method for linear programs and the active-set method for quadratic ones. Two unrelated
/// algorithms agreeing on an answer is real evidence; a solver agreeing with itself is not. If a
/// test fails, FIX THE SOLVER — do not relax the assertion to match the output.
///
/// The tolerance is looser than the simplex suite's on purpose. An interior-point method approaches
/// the optimum from strictly inside the feasible region and never lands exactly on the boundary, so
/// its answer is correct to the convergence tolerance rather than exact — which is the defining
/// trade-off of the method, not a defect in this implementation.
/// </remarks>
public class InteriorPointSolverIntegrationTests
{
    private const double Tolerance = 1e-5;

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

    private static InteriorPointSolver<double> Solver() => new();

    #region Linear programs with known optima

    /// <summary>
    /// The textbook furniture problem:
    ///   maximize 30·t + 20·c  subject to  4t + 3c ≤ 240,  2t + c ≤ 100,  t, c ≥ 0.
    /// Written as a minimization of the negated objective. The optimum sits where both constraints
    /// bind: 4t + 3c = 240 and 2t + c = 100 give t = 30, c = 40, for a profit of 1700.
    /// </summary>
    [Fact]
    public void Solve_FurnitureProblem_FindsTheKnownVertex()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(30.0, solution.Solution[0], 4);
        Assert.Equal(40.0, solution.Solution[1], 4);
        Assert.Equal(-1700.0, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// minimize -x - y  subject to  x + 2y ≤ 4,  3x + 2y ≤ 6.
    /// Both constraints bind at the optimum: subtracting gives 2x = 2, so x = 1 and y = 1.5.
    /// </summary>
    [Fact]
    public void Solve_TwoBindingConstraints_FindsTheirIntersection()
    {
        var program = new LinearProgram<double>(
            objective: V(-1, -1),
            inequalityMatrix: M(new[,] { { 1.0, 2.0 }, { 3.0, 2.0 } }),
            inequalityBounds: V(4, 6));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 4);
        Assert.Equal(1.5, solution.Solution[1], 4);
        Assert.Equal(-2.5, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// An equality constraint alone: minimize x + 2y subject to x + y = 3, x, y ≥ 0. Since y costs
    /// twice what x does, the cheapest way to reach a total of 3 is all x.
    /// </summary>
    [Fact]
    public void Solve_EqualityConstraint_SpendsOnTheCheaperVariable()
    {
        var program = new LinearProgram<double>(
            objective: V(1, 2),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(3));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0], 4);
        Assert.Equal(0.0, solution.Solution[1], 4);
        Assert.Equal(3.0, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// A negative right-hand side, which the standard-form rewrite negates:
    ///   minimize x  subject to  -x ≤ -2, x ≥ 0.  That is x ≥ 2, so the optimum is x = 2.
    /// </summary>
    [Fact]
    public void Solve_NegativeRightHandSide_HandlesTheNegatedRow()
    {
        var program = new LinearProgram<double>(
            objective: V(1),
            inequalityMatrix: M(new[,] { { -1.0 } }),
            inequalityBounds: V(-2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(2.0, solution.Solution[0], 4);
    }

    /// <summary>
    /// A variable free to go negative, which the rewrite splits into a difference of two
    /// non-negative parts: minimize x subject to x ≥ -5, with no lower bound of zero.
    /// </summary>
    [Fact]
    public void Solve_FreeVariable_ReachesTheNegativeOptimum()
    {
        var program = new LinearProgram<double>(
            objective: V(1),
            inequalityMatrix: M(new[,] { { -1.0 } }),
            inequalityBounds: V(5),
            lowerBounds: V(double.NegativeInfinity),
            upperBounds: V(double.PositiveInfinity));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-5.0, solution.Solution[0], 4);
    }

    /// <summary>
    /// Finite upper bounds become extra rows in the rewrite:
    ///   minimize -x - y subject to 0 ≤ x ≤ 3, 0 ≤ y ≤ 2. Both variables go to their ceilings.
    /// </summary>
    [Fact]
    public void Solve_BoxBounds_PushesEachVariableToItsCeiling()
    {
        var program = new LinearProgram<double>(
            objective: V(-1, -1),
            lowerBounds: V(0, 0),
            upperBounds: V(3, 2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(3.0, solution.Solution[0], 4);
        Assert.Equal(2.0, solution.Solution[1], 4);
        Assert.Equal(-5.0, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// A non-zero lower bound shifts the variable in the rewrite:
    ///   minimize x subject to 2 ≤ x ≤ 7. The optimum is the floor.
    /// </summary>
    [Fact]
    public void Solve_ShiftedLowerBound_FindsTheFloor()
    {
        var program = new LinearProgram<double>(
            objective: V(1),
            lowerBounds: V(2),
            upperBounds: V(7));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(2.0, solution.Solution[0], 4);
    }

    /// <summary>
    /// A blend problem with a mix of senses, hand-solvable by inspection:
    ///   minimize 2x + 3y  subject to  x + y = 10,  x ≤ 6,  x, y ≥ 0.
    /// x is the cheaper input, so it is pushed to its ceiling of 6 and y covers the rest.
    /// </summary>
    [Fact]
    public void Solve_BlendProblem_UsesTheCheapInputToItsLimit()
    {
        var program = new LinearProgram<double>(
            objective: V(2, 3),
            inequalityMatrix: M(new[,] { { 1.0, 0.0 } }),
            inequalityBounds: V(6),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(10));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(6.0, solution.Solution[0], 4);
        Assert.Equal(4.0, solution.Solution[1], 4);
        Assert.Equal(24.0, solution.ObjectiveValue, 4);
    }

    #endregion

    #region Agreement with the simplex method

    /// <summary>
    /// The two algorithms share nothing but the problem statement — simplex walks the vertices,
    /// interior point cuts through the middle — so agreeing on the optimal objective across a range
    /// of shapes is strong evidence both are right.
    /// </summary>
    [Theory]
    [MemberData(nameof(CrossCheckPrograms))]
    public void Solve_AgreesWithSimplex_OnTheOptimalObjective(string name, LinearProgram<double> program)
    {
        var simplexSolution = new SimplexSolver<double>().Solve(program);
        var interiorSolution = Solver().Solve(program);

        Assert.True(
            simplexSolution.Status == interiorSolution.Status,
            $"{name}: simplex reported {simplexSolution.Status} but interior point reported " +
            $"{interiorSolution.Status}.");

        if (simplexSolution.Status != LinearProgramStatus.Optimal) return;

        Assert.True(
            Math.Abs(simplexSolution.ObjectiveValue - interiorSolution.ObjectiveValue) < Tolerance,
            $"{name}: simplex found {simplexSolution.ObjectiveValue} but interior point found " +
            $"{interiorSolution.ObjectiveValue}.");
    }

    public static TheoryData<string, LinearProgram<double>> CrossCheckPrograms()
    {
        var data = new TheoryData<string, LinearProgram<double>>();

        data.Add("furniture", new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100)));

        data.Add("three-variable transport", new LinearProgram<double>(
            objective: V(4, 6, 3),
            inequalityMatrix: M(new[,] { { -1.0, -1.0, 0.0 }, { 0.0, -1.0, -1.0 } }),
            inequalityBounds: V(-5, -7)));

        data.Add("mixed senses", new LinearProgram<double>(
            objective: V(2, 3),
            inequalityMatrix: M(new[,] { { 1.0, 0.0 } }),
            inequalityBounds: V(6),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(10)));

        data.Add("box bounded", new LinearProgram<double>(
            objective: V(-5, -4, -3),
            inequalityMatrix: M(new[,] { { 2.0, 3.0, 1.0 }, { 4.0, 1.0, 2.0 }, { 3.0, 4.0, 2.0 } }),
            inequalityBounds: V(5, 11, 8),
            lowerBounds: V(0, 0, 0),
            upperBounds: V(4, 4, 4)));

        data.Add("degenerate vertex", new LinearProgram<double>(
            objective: V(-1, -1),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 0.0 } }),
            inequalityBounds: V(4, 4, 3)));

        data.Add("free variable", new LinearProgram<double>(
            objective: V(1, 1),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(2),
            lowerBounds: V(double.NegativeInfinity, 0),
            upperBounds: V(double.PositiveInfinity, double.PositiveInfinity)));

        return data;
    }

    #endregion

    #region Infeasible and unbounded

    /// <summary>
    /// x ≥ 3 and x ≤ 1 cannot both hold. The solver must prove this with a Farkas certificate
    /// rather than run out of iterations.
    /// </summary>
    [Fact]
    public void Solve_ContradictoryConstraints_ReportsInfeasible()
    {
        var program = new LinearProgram<double>(
            objective: V(1),
            inequalityMatrix: M(new[,] { { -1.0 }, { 1.0 } }),
            inequalityBounds: V(-3, 1));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Infeasible, solution.Status);
        Assert.Null(solution.Solution);
    }

    /// <summary>
    /// Two equalities demanding different totals of the same sum are contradictory.
    /// </summary>
    [Fact]
    public void Solve_ContradictoryEqualities_ReportsInfeasible()
    {
        var program = new LinearProgram<double>(
            objective: V(1, 1),
            equalityMatrix: M(new[,] { { 1.0, 1.0 }, { 1.0, 1.0 } }),
            equalityBounds: V(2, 5));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Infeasible, solution.Status);
    }

    /// <summary>
    /// minimize -x with no upper limit on x: the objective falls forever along the ray x → ∞.
    /// </summary>
    [Fact]
    public void Solve_ObjectiveWithNoCeiling_ReportsUnbounded()
    {
        var program = new LinearProgram<double>(
            objective: V(-1, 0),
            inequalityMatrix: M(new[,] { { 0.0, 1.0 } }),
            inequalityBounds: V(5));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Unbounded, solution.Status);
        Assert.Null(solution.Solution);
    }

    #endregion

    #region Duality

    /// <summary>
    /// Strong duality: at the optimum the primal objective equals the dual objective bᵀy. This also
    /// pins the sign convention of the reported duals, which must match the simplex solver's so the
    /// two are interchangeable.
    /// </summary>
    [Fact]
    public void Solve_DualValues_SatisfyStrongDuality()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.NotNull(solution.InequalityDualValues);

        double dualObjective =
            240 * solution.InequalityDualValues[0] + 100 * solution.InequalityDualValues[1];

        Assert.Equal(solution.ObjectiveValue, dualObjective, 4);
    }

    /// <summary>
    /// The duals must agree with the simplex method's, not merely satisfy duality on their own — a
    /// caller swapping one solver for the other must get the same shadow prices.
    /// </summary>
    [Fact]
    public void Solve_DualValues_AgreeWithSimplex()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var simplexSolution = new SimplexSolver<double>().Solve(program);
        var interiorSolution = Solver().Solve(program);

        Assert.NotNull(simplexSolution.InequalityDualValues);
        Assert.NotNull(interiorSolution.InequalityDualValues);

        for (int r = 0; r < 2; r++)
        {
            Assert.Equal(
                simplexSolution.InequalityDualValues[r],
                interiorSolution.InequalityDualValues[r],
                4);
        }
    }

    /// <summary>
    /// Complementary slackness: a constraint that is not tight at the optimum has a dual value of
    /// zero — capacity you are not using is worth nothing.
    /// </summary>
    [Fact]
    public void Solve_SlackConstraint_HasZeroDualValue()
    {
        var program = new LinearProgram<double>(
            objective: V(-1),
            inequalityMatrix: M(new[,] { { 1.0 }, { 1.0 } }),
            inequalityBounds: V(2, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.0, solution.InequalityDualValues[1], 4);
        Assert.True(
            Math.Abs(solution.InequalityDualValues[0]) > Tolerance,
            "The binding constraint should carry a non-zero shadow price.");
    }

    #endregion

    #region Quadratic programs

    /// <summary>
    /// An unconstrained-in-practice quadratic: minimize ½(x² + y²) − x − 2y over x, y ≥ 0. The
    /// gradient vanishes at (1, 2), which is already non-negative, so the bounds never bind.
    /// </summary>
    [Fact]
    public void Solve_QuadraticWithInactiveBounds_FindsTheStationaryPoint()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-1, -2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 4);
        Assert.Equal(2.0, solution.Solution[1], 4);
        Assert.Equal(-2.5, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// A binding equality: minimize ½(x² + y²) subject to x + y = 2. Symmetry and the KKT
    /// conditions put the answer at (1, 1) — the closest point on the line to the origin.
    /// </summary>
    [Fact]
    public void Solve_QuadraticWithEquality_FindsTheProjection()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(0, 0),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 4);
        Assert.Equal(1.0, solution.Solution[1], 4);
        Assert.Equal(1.0, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// A binding inequality: minimize ½(x² + y²) − 3x − 3y subject to x + y ≤ 2. The unconstrained
    /// minimum is (3, 3), which violates the constraint, so the answer is its projection onto the
    /// boundary line — (1, 1) by symmetry.
    /// </summary>
    [Fact]
    public void Solve_QuadraticWithBindingInequality_LandsOnTheBoundary()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-3, -3),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(2));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(1.0, solution.Solution[0], 4);
        Assert.Equal(1.0, solution.Solution[1], 4);
        Assert.Equal(-5.0, solution.ObjectiveValue, 4);
    }

    /// <summary>
    /// A non-diagonal Hessian, so the two variables genuinely interact:
    ///   minimize ½(2x² + 2xy + 2y²) − 4x − 6y  subject to  x + y ≤ 3.
    /// The unconstrained stationary point solves 2x + y = 4, x + 2y = 6, giving (2/3, 8/3) with a
    /// sum of 10/3 — above the limit, so the constraint binds. On x + y = 3 the objective reduces to
    /// a one-dimensional quadratic minimized at x = 1/2, y = 5/2.
    /// </summary>
    [Fact]
    public void Solve_CoupledQuadratic_MatchesTheHandDerivedOptimum()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 2.0, 1.0 }, { 1.0, 2.0 } }),
            linear: V(-4, -6),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(3));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.5, solution.Solution[0], 4);
        Assert.Equal(2.5, solution.Solution[1], 4);
    }

    /// <summary>
    /// The interior-point and active-set methods identify the optimum by completely different means
    /// — one approaches from inside, the other guesses which constraints bind and corrects — so
    /// agreement across a range of quadratic programs is real evidence rather than self-consistency.
    /// </summary>
    [Theory]
    [MemberData(nameof(CrossCheckQuadraticPrograms))]
    public void Solve_AgreesWithActiveSet_OnQuadraticPrograms(
        string name, QuadraticProgram<double> program)
    {
        var activeSetSolution = new ActiveSetQuadraticProgramSolver<double>().Solve(program);
        var interiorSolution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, activeSetSolution.Status);
        Assert.Equal(LinearProgramStatus.Optimal, interiorSolution.Status);

        Assert.True(
            Math.Abs(activeSetSolution.ObjectiveValue - interiorSolution.ObjectiveValue) < 1e-4,
            $"{name}: active set found {activeSetSolution.ObjectiveValue} but interior point found " +
            $"{interiorSolution.ObjectiveValue}.");

        for (int i = 0; i < program.VariableCount; i++)
        {
            Assert.True(
                Math.Abs(activeSetSolution.Solution[i] - interiorSolution.Solution[i]) < 1e-4,
                $"{name}: the two solvers disagree on variable {i} — " +
                $"{activeSetSolution.Solution[i]} against {interiorSolution.Solution[i]}.");
        }
    }

    public static TheoryData<string, QuadraticProgram<double>> CrossCheckQuadraticPrograms()
    {
        var data = new TheoryData<string, QuadraticProgram<double>>();

        data.Add("inactive bounds", new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-1, -2)));

        data.Add("binding inequality", new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(-3, -3),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(2)));

        data.Add("coupled Hessian", new QuadraticProgram<double>(
            quadratic: M(new[,] { { 2.0, 1.0 }, { 1.0, 2.0 } }),
            linear: V(-4, -6),
            inequalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            inequalityBounds: V(3)));

        data.Add("equality constrained", new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(0, 0),
            equalityMatrix: M(new[,] { { 1.0, 1.0 } }),
            equalityBounds: V(2)));

        data.Add("three variables", new QuadraticProgram<double>(
            quadratic: M(new[,] { { 2.0, 0.0, 0.0 }, { 0.0, 2.0, 0.0 }, { 0.0, 0.0, 2.0 } }),
            linear: V(-2, -4, -6),
            inequalityMatrix: M(new[,] { { 1.0, 1.0, 1.0 } }),
            inequalityBounds: V(4)));

        return data;
    }

    /// <summary>
    /// Non-negativity is the default lower bound, so a quadratic whose unconstrained minimum is
    /// negative must be clipped at zero — the classic non-negative least-squares shape.
    /// </summary>
    [Fact]
    public void Solve_QuadraticWantingNegativeValues_ClipsAtZero()
    {
        var program = new QuadraticProgram<double>(
            quadratic: M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 } }),
            linear: V(2, -3));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0.0, solution.Solution[0], 4);
        Assert.Equal(3.0, solution.Solution[1], 4);
    }

    #endregion

    #region Configuration and validation

    [Fact]
    public void Constructor_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new InteriorPointSolver<double>(null));
    }

    [Fact]
    public void Constructor_NonPositiveIterationLimit_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new InteriorPointSolver<double>(new InteriorPointSolverOptions { MaxIterations = 0 }));
    }

    [Fact]
    public void Constructor_FullStepToBoundary_Throws()
    {
        // A step of the full distance lands exactly on the boundary, where the next iteration
        // divides by zero — so this has to be rejected rather than quietly clamped.
        Assert.Throws<ArgumentException>(() =>
            new InteriorPointSolver<double>(
                new InteriorPointSolverOptions { FractionToBoundary = 1.0 }));
    }

    [Fact]
    public void Solve_NullLinearProgram_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve((LinearProgram<double>)null));
    }

    [Fact]
    public void Solve_NullQuadraticProgram_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Solver().Solve((QuadraticProgram<double>)null));
    }

    /// <summary>
    /// A tighter tolerance must produce an answer at least as accurate, never a worse one.
    /// </summary>
    [Fact]
    public void Solve_TighterTolerance_DoesNotDegradeAccuracy()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var loose = new InteriorPointSolver<double>(
            new InteriorPointSolverOptions { Tolerance = 1e-4 }).Solve(program);
        var tight = new InteriorPointSolver<double>(
            new InteriorPointSolverOptions { Tolerance = 1e-10 }).Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, loose.Status);
        Assert.Equal(LinearProgramStatus.Optimal, tight.Status);

        double looseError = Math.Abs(loose.ObjectiveValue + 1700.0);
        double tightError = Math.Abs(tight.ObjectiveValue + 1700.0);

        Assert.True(
            tightError <= looseError + 1e-12,
            $"Tightening the tolerance made the answer worse: {tightError} against {looseError}.");
    }

    /// <summary>
    /// The iteration count is reported, and a well-scaled problem converges in far fewer steps than
    /// the limit — the property that makes interior-point methods scale.
    /// </summary>
    [Fact]
    public void Solve_WellScaledProblem_ConvergesInFewIterations()
    {
        var program = new LinearProgram<double>(
            objective: V(-30, -20),
            inequalityMatrix: M(new[,] { { 4.0, 3.0 }, { 2.0, 1.0 } }),
            inequalityBounds: V(240, 100));

        var solution = Solver().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.InRange(solution.Iterations, 1, 40);
    }

    #endregion
}
