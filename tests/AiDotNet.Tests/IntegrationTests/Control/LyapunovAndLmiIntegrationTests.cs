#nullable disable
using AiDotNet.Control;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Integration tests for the Lyapunov equation solver and the linear matrix inequality solver.
/// </summary>
/// <remarks>
/// CRITICAL: Lyapunov solutions are checked against closed forms where one exists and against the
/// equation's own residual otherwise — substituting the answer back in is a check the solver cannot
/// satisfy by converging to the wrong place. Positive definiteness is verified in this file by an
/// independent Cholesky factorization written here, not by calling back into the solver's own test.
/// If a test fails, FIX THE SOLVER — do not relax the assertion.
/// </remarks>
public class LyapunovAndLmiIntegrationTests
{
    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    /// <summary>
    /// An independent positive-definiteness test, written here so the assertions do not depend on
    /// the same code path the solver uses internally.
    /// </summary>
    private static bool IsPositiveDefinite(Matrix<double> matrix, double tolerance = 1e-9)
    {
        int n = matrix.Rows;
        var factor = new double[n, n];

        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c <= r; c++)
            {
                double total = matrix[r, c];
                for (int k = 0; k < c; k++) total -= factor[r, k] * factor[c, k];

                if (r == c)
                {
                    if (total <= tolerance) return false;
                    factor[r, c] = Math.Sqrt(total);
                }
                else
                {
                    factor[r, c] = total / factor[c, c];
                }
            }
        }

        return true;
    }

    #region Lyapunov: closed forms

    /// <summary>
    /// The scalar continuous equation 2ap + q = 0 gives p = -q/(2a). With a = -1 and q = 2, p = 1.
    /// </summary>
    [Fact]
    public void Lyapunov_ScalarContinuous_MatchesTheClosedForm()
    {
        var solution = new LyapunovSolver<double>().SolveContinuous(
            M(new[,] { { -1.0 } }), M(new[,] { { 2.0 } }));

        Assert.Equal(1.0, solution[0, 0], 10);
    }

    /// <summary>
    /// The scalar discrete equation a²p - p + q = 0 gives p = q/(1 - a²). With a = 0.5 and q = 1,
    /// p = 4/3.
    /// </summary>
    [Fact]
    public void Lyapunov_ScalarDiscrete_MatchesTheClosedForm()
    {
        var solution = new LyapunovSolver<double>().SolveDiscrete(
            M(new[,] { { 0.5 } }), M(new[,] { { 1.0 } }));

        Assert.Equal(4.0 / 3.0, solution[0, 0], 10);
    }

    /// <summary>
    /// A diagonal continuous system decouples completely: each entry solves its own scalar equation,
    /// so P is diagonal with entries -q_ii / (2 a_ii).
    /// </summary>
    [Fact]
    public void Lyapunov_DiagonalSystem_DecouplesIntoScalarSolutions()
    {
        var solution = new LyapunovSolver<double>().SolveContinuous(
            M(new[,] { { -2.0, 0.0 }, { 0.0, -4.0 } }),
            M(new[,] { { 4.0, 0.0 }, { 0.0, 8.0 } }));

        Assert.Equal(1.0, solution[0, 0], 10);
        Assert.Equal(1.0, solution[1, 1], 10);
        Assert.Equal(0.0, solution[0, 1], 10);
    }

    #endregion

    #region Lyapunov: residuals and structure

    /// <summary>
    /// A coupled continuous system, verified by substituting the answer back into the equation.
    /// </summary>
    [Fact]
    public void Lyapunov_CoupledContinuous_SatisfiesTheEquation()
    {
        var a = M(new[,] { { -1.0, 2.0, 0.0 }, { 0.0, -3.0, 1.0 }, { 0.5, 0.0, -2.0 } });
        var q = M(new[,] { { 2.0, 0.3, 0.0 }, { 0.3, 1.0, 0.1 }, { 0.0, 0.1, 3.0 } });

        var p = new LyapunovSolver<double>().SolveContinuous(a, q);

        // Residual of A'P + PA + Q.
        double worst = 0.0;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                double value = q[r, c];
                for (int k = 0; k < 3; k++) value += a[k, r] * p[k, c] + p[r, k] * a[k, c];
                worst = Math.Max(worst, Math.Abs(value));
            }
        }

        Assert.True(worst < 1e-10, $"The Lyapunov residual was {worst}.");
    }

    /// <summary>
    /// The same for the discrete equation A'PA - P + Q = 0.
    /// </summary>
    [Fact]
    public void Lyapunov_CoupledDiscrete_SatisfiesTheEquation()
    {
        var a = M(new[,] { { 0.5, 0.2, 0.0 }, { 0.0, 0.4, 0.1 }, { 0.1, 0.0, 0.3 } });
        var q = M(new[,] { { 1.0, 0.2, 0.0 }, { 0.2, 2.0, 0.1 }, { 0.0, 0.1, 1.5 } });

        var p = new LyapunovSolver<double>().SolveDiscrete(a, q);

        double worst = 0.0;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                double value = q[r, c] - p[r, c];
                for (int k = 0; k < 3; k++)
                {
                    for (int l = 0; l < 3; l++) value += a[k, r] * p[k, l] * a[l, c];
                }

                worst = Math.Max(worst, Math.Abs(value));
            }
        }

        Assert.True(worst < 1e-10, $"The Stein residual was {worst}.");
    }

    /// <summary>
    /// Lyapunov's theorem: for a stable system and a positive-definite Q, the solution is positive
    /// definite — which is a stability certificate, not merely a verdict.
    /// </summary>
    [Fact]
    public void Lyapunov_StableSystem_YieldsAPositiveDefiniteCertificate()
    {
        var a = M(new[,] { { 0.5, 0.4 }, { -0.2, 0.3 } });

        var p = new LyapunovSolver<double>().SolveDiscrete(a, Matrix<double>.CreateIdentity(2));

        Assert.True(
            IsPositiveDefinite(p),
            "A stable system must admit a positive-definite Lyapunov certificate.");
    }

    /// <summary>
    /// And the converse: an unstable system's solution is not positive definite, so the test
    /// distinguishes rather than always answering yes.
    /// </summary>
    [Fact]
    public void Lyapunov_UnstableSystem_YieldsNoCertificate()
    {
        // Eigenvalues 1.5 and 0.5: one outside the unit circle.
        var a = M(new[,] { { 1.5, 0.0 }, { 0.0, 0.5 } });

        var p = new LyapunovSolver<double>().SolveDiscrete(a, Matrix<double>.CreateIdentity(2));

        Assert.False(
            IsPositiveDefinite(p),
            "An unstable system must not produce a positive-definite certificate — if it did, the " +
            "certificate would prove something false.");
    }

    /// <summary>
    /// The solution is symmetric as a matter of theory, and must come back exactly symmetric.
    /// </summary>
    [Fact]
    public void Lyapunov_Solution_IsExactlySymmetric()
    {
        var p = new LyapunovSolver<double>().SolveContinuous(
            M(new[,] { { -1.0, 2.0, 0.5 }, { 0.0, -3.0, 1.0 }, { 0.5, 0.0, -2.0 } }),
            Matrix<double>.CreateIdentity(3));

        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++) Assert.Equal(p[r, c], p[c, r]);
        }
    }

    #endregion

    #region Gramians

    /// <summary>
    /// The scalar discrete controllability Gramian is the sum of a^(2k) b², a geometric series
    /// equal to b² / (1 - a²).
    /// </summary>
    [Fact]
    public void Gramian_ScalarDiscrete_MatchesTheGeometricSeries()
    {
        const double A = 0.6;
        const double B = 2.0;

        var gramian = new LyapunovSolver<double>().ControllabilityGramian(
            M(new[,] { { A } }), M(new[,] { { B } }), ControlTimeDomain.Discrete);

        Assert.Equal(B * B / (1 - A * A), gramian[0, 0], 10);
    }

    /// <summary>
    /// A controllable system has a positive-definite controllability Gramian: every direction of the
    /// state space can be reached.
    /// </summary>
    [Fact]
    public void Gramian_ControllableSystem_IsPositiveDefinite()
    {
        var gramian = new LyapunovSolver<double>().ControllabilityGramian(
            M(new[,] { { 0.5, 1.0 }, { 0.0, 0.4 } }),
            M(new[,] { { 0.0 }, { 1.0 } }),
            ControlTimeDomain.Discrete);

        Assert.True(IsPositiveDefinite(gramian));
    }

    /// <summary>
    /// An uncontrollable system's Gramian is singular: the input cannot reach one direction at all,
    /// which is exactly what the Gramian is for detecting.
    /// </summary>
    [Fact]
    public void Gramian_UncontrollableSystem_IsSingular()
    {
        // The second state is untouched by the input and decoupled from the first.
        var gramian = new LyapunovSolver<double>().ControllabilityGramian(
            M(new[,] { { 0.5, 0.0 }, { 0.0, 0.4 } }),
            M(new[,] { { 1.0 }, { 0.0 } }),
            ControlTimeDomain.Discrete);

        Assert.False(
            IsPositiveDefinite(gramian),
            "A direction the input cannot reach must show up as a singular Gramian.");
    }

    #endregion

    #region Lyapunov validation

    [Fact]
    public void Lyapunov_ContinuousWithZeroEigenvalue_Throws()
    {
        // A = 0 makes two eigenvalues sum to zero, so the equation has no unique solution.
        Assert.Throws<InvalidOperationException>(() =>
            new LyapunovSolver<double>().SolveContinuous(
                new Matrix<double>(2, 2), Matrix<double>.CreateIdentity(2)));
    }

    [Fact]
    public void Lyapunov_DiscreteWithUnitEigenvalue_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            new LyapunovSolver<double>().SolveDiscrete(
                Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(2)));
    }

    [Fact]
    public void Lyapunov_NonSquareStateMatrix_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LyapunovSolver<double>().SolveContinuous(
                new Matrix<double>(2, 3), Matrix<double>.CreateIdentity(2)));
    }

    [Fact]
    public void Lyapunov_MismatchedConstantTerm_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LyapunovSolver<double>().SolveContinuous(
                Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(3)));
    }

    #endregion

    #region Linear matrix inequalities

    /// <summary>
    /// The simplest possible inequality: find coefficients making a diagonal matrix positive
    /// semidefinite. F(x) = -I + x0·E00 + x1·E11 needs both coefficients at least one.
    /// </summary>
    [Fact]
    public void Lmi_DiagonalFeasibility_FindsAVerifiedPoint()
    {
        var constantTerm = M(new[,] { { -1.0, 0.0 }, { 0.0, -1.0 } });
        var basis = new List<Matrix<double>>
        {
            M(new[,] { { 1.0, 0.0 }, { 0.0, 0.0 } }),
            M(new[,] { { 0.0, 0.0 }, { 0.0, 1.0 } }),
        };

        var result = new LinearMatrixInequalitySolver<double>()
            .Solve(constantTerm, basis, Vector<double>.FromArray(new[] { 5.0, 5.0 }));

        Assert.Equal(LinearMatrixInequalityStatus.Feasible, result.Status);

        // Verified here rather than trusted: the returned matrix really is positive semidefinite.
        Assert.True(result.Matrix[0, 0] >= -1e-9 && result.Matrix[1, 1] >= -1e-9);
        Assert.True(result.SmallestEigenvalue >= -1e-9);
    }

    /// <summary>
    /// Starting from an infeasible point, the search must move to a feasible one.
    /// </summary>
    [Fact]
    public void Lmi_FromAnInfeasibleStart_ReachesFeasibility()
    {
        var constantTerm = M(new[,] { { -1.0, 0.0 }, { 0.0, -1.0 } });
        var basis = new List<Matrix<double>>
        {
            M(new[,] { { 1.0, 0.0 }, { 0.0, 0.0 } }),
            M(new[,] { { 0.0, 0.0 }, { 0.0, 1.0 } }),
        };

        var result = new LinearMatrixInequalitySolver<double>(
                new LinearMatrixInequalityOptions { InitialStepSize = 2.0, MaxIterations = 20000 })
            .Solve(constantTerm, basis);

        Assert.Equal(LinearMatrixInequalityStatus.Feasible, result.Status);
        Assert.True(IsPositiveDefinite(result.Matrix, tolerance: -1e-9));
    }

    /// <summary>
    /// A genuine control problem: find a Lyapunov certificate by solving an inequality rather than an
    /// equation. The unknowns are the entries of a symmetric P, and the block-diagonal inequality
    /// asks for P ⪰ εI and -(AᵀP + PA) ⪰ εI at once — the two conditions of Lyapunov's theorem.
    /// The result is cross-checked for the property it claims.
    /// </summary>
    [Fact]
    public void Lmi_LyapunovStability_FindsACertificateForAStableSystem()
    {
        var a = M(new[,] { { -1.0, 0.5 }, { 0.0, -2.0 } });
        const double Epsilon = 0.05;

        // Variables are (p11, p12, p22); each contributes a symmetric basis matrix.
        var parameterizations = new List<Matrix<double>>
        {
            M(new[,] { { 1.0, 0.0 }, { 0.0, 0.0 } }),
            M(new[,] { { 0.0, 1.0 }, { 1.0, 0.0 } }),
            M(new[,] { { 0.0, 0.0 }, { 0.0, 1.0 } }),
        };

        var constantTerm = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++) constantTerm[i, i] = -Epsilon;

        var basis = new List<Matrix<double>>();
        foreach (var p in parameterizations)
        {
            var block = new Matrix<double>(4, 4);

            // Top-left block: P itself.
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++) block[r, c] = p[r, c];
            }

            // Bottom-right block: -(A'P + PA).
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    double value = 0.0;
                    for (int k = 0; k < 2; k++) value += a[k, r] * p[k, c] + p[r, k] * a[k, c];
                    block[2 + r, 2 + c] = -value;
                }
            }

            basis.Add(block);
        }

        var result = new LinearMatrixInequalitySolver<double>(
                new LinearMatrixInequalityOptions { InitialStepSize = 2.0, MaxIterations = 20000 })
            .Solve(constantTerm, basis, Vector<double>.FromArray(new[] { 1.0, 0.0, 1.0 }));

        Assert.Equal(LinearMatrixInequalityStatus.Feasible, result.Status);

        // Rebuild P from the coefficients and check it really certifies stability.
        var certificate = M(new[,]
        {
            { result.Variables[0], result.Variables[1] },
            { result.Variables[1], result.Variables[2] },
        });

        Assert.True(IsPositiveDefinite(certificate), "The certificate P must be positive definite.");

        var decrease = new Matrix<double>(2, 2);
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 2; c++)
            {
                double value = 0.0;
                for (int k = 0; k < 2; k++)
                {
                    value += a[k, r] * certificate[k, c] + certificate[r, k] * a[k, c];
                }

                decrease[r, c] = -value;
            }
        }

        Assert.True(
            IsPositiveDefinite(decrease),
            "-(A'P + PA) must be positive definite for P to certify stability.");
    }

    /// <summary>
    /// When no coefficient can help, the search must report that it ran out rather than claim the
    /// problem is infeasible — a search that failed has not proved anything.
    /// </summary>
    [Fact]
    public void Lmi_WhenNoDirectionHelps_ReportsIterationLimitNotInfeasible()
    {
        var constantTerm = M(new[,] { { -1.0, 0.0 }, { 0.0, -1.0 } });

        // A basis matrix of zeros cannot change F(x) at all.
        var basis = new List<Matrix<double>> { new Matrix<double>(2, 2) };

        var result = new LinearMatrixInequalitySolver<double>(
                new LinearMatrixInequalityOptions { MaxIterations = 50 })
            .Solve(constantTerm, basis);

        Assert.Equal(LinearMatrixInequalityStatus.IterationLimit, result.Status);
    }

    /// <summary>
    /// A constant term that is already positive semidefinite must be recognized immediately from the
    /// origin.
    /// </summary>
    [Fact]
    public void Lmi_AlreadyFeasibleAtTheOrigin_StopsImmediately()
    {
        var result = new LinearMatrixInequalitySolver<double>().Solve(
            Matrix<double>.CreateIdentity(2),
            new List<Matrix<double>> { Matrix<double>.CreateIdentity(2) });

        Assert.Equal(LinearMatrixInequalityStatus.Feasible, result.Status);
        Assert.Equal(1, result.Iterations);
    }

    #endregion

    #region LMI validation

    [Fact]
    public void Lmi_EmptyBasis_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearMatrixInequalitySolver<double>().Solve(
                Matrix<double>.CreateIdentity(2), new List<Matrix<double>>()));
    }

    [Fact]
    public void Lmi_MismatchedBasisMatrix_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearMatrixInequalitySolver<double>().Solve(
                Matrix<double>.CreateIdentity(2),
                new List<Matrix<double>> { Matrix<double>.CreateIdentity(3) }));
    }

    [Fact]
    public void Lmi_NonSquareConstantTerm_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearMatrixInequalitySolver<double>().Solve(
                new Matrix<double>(2, 3),
                new List<Matrix<double>> { Matrix<double>.CreateIdentity(2) }));
    }

    [Fact]
    public void Lmi_InitialGuessOfWrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearMatrixInequalitySolver<double>().Solve(
                Matrix<double>.CreateIdentity(2),
                new List<Matrix<double>> { Matrix<double>.CreateIdentity(2) },
                Vector<double>.FromArray(new[] { 1.0, 2.0 })));
    }

    [Fact]
    public void Lmi_NonPositiveIterationLimit_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearMatrixInequalitySolver<double>(
                new LinearMatrixInequalityOptions { MaxIterations = 0 }));
    }

    #endregion
}
