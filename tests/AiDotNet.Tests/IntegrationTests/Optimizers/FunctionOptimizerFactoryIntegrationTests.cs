#nullable disable
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests for the <c>CreateForFunction</c> factories, which are the only way to build a
/// gradient optimizer with no model attached, and for the strong Wolfe line search that makes
/// <c>LBFGSOptimizer.Minimize</c> converge at the rate the algorithm is published with.
/// </summary>
/// <remarks>
/// CRITICAL: These tests check answers against KNOWN optima and against evaluation budgets derived
/// from what the algorithms actually promise. If one fails, FIX THE OPTIMIZER, do not relax the
/// assertion.
///
/// Before the factories existed, <c>Minimize</c> was reachable on exactly one optimizer
/// (<c>LBFGSOptimizer</c>) even though it was implemented on the shared base class: every other
/// constructor demanded an <c>IFullModel</c>, which a plain function does not have.
/// </remarks>
public class FunctionOptimizerFactoryIntegrationTests
{
    private const int MaxIterations = 5000;
    private const double GradientTolerance = 1e-8;

    /// <summary>f(x) = sum((x_i - 3)^2); gradient 2(x - 3). Minimum 0 at every coordinate 3.</summary>
    private static (double objective, Vector<double> gradient) ShiftedSphere(Vector<double> x)
    {
        double sum = 0;
        var gradient = new Vector<double>(x.Length);
        for (int i = 0; i < x.Length; i++)
        {
            double delta = x[i] - 3.0;
            sum += delta * delta;
            gradient[i] = 2.0 * delta;
        }

        return (sum, gradient);
    }

    /// <summary>
    /// The chained Rosenbrock function in n variables, minimum 0 at every coordinate 1.
    /// </summary>
    private static (double objective, Vector<double> gradient) ChainedRosenbrock(Vector<double> x)
    {
        double sum = 0;
        var gradient = new Vector<double>(x.Length);

        for (int i = 0; i + 1 < x.Length; i++)
        {
            double flat = 1.0 - x[i];
            double curve = x[i + 1] - x[i] * x[i];

            sum += flat * flat + 100.0 * curve * curve;

            gradient[i] += -2.0 * flat - 400.0 * x[i] * curve;
            gradient[i + 1] += 200.0 * curve;
        }

        return (sum, gradient);
    }

    /// <summary>The usual hard starting point, alternating -1.2 and 1.</summary>
    private static Vector<double> RosenbrockStart(int dimension)
    {
        var start = new Vector<double>(dimension);
        for (int i = 0; i < dimension; i++) start[i] = i % 2 == 0 ? -1.2 : 1.0;
        return start;
    }

    /// <summary>Counts how many times an objective is asked for a value.</summary>
    private sealed class CountingObjective
    {
        private readonly Func<Vector<double>, (double objective, Vector<double> gradient)> _inner;

        public CountingObjective(
            Func<Vector<double>, (double objective, Vector<double> gradient)> inner)
            => _inner = inner;

        public int Count { get; private set; }

        public (double objective, Vector<double> gradient) Evaluate(Vector<double> x)
        {
            Count++;
            return _inner(x);
        }
    }

    /// <summary>
    /// Every factory in the gradient family, named so a failure says which one broke.
    /// </summary>
    public static IEnumerable<object[]> GradientOptimizerFactories()
    {
        yield return Factory("Adagrad", () => AdagradOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("AdaDelta", () => AdaDeltaOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("AdaMax", () => AdaMaxOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("Adam", () => AdamOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("AdamW", () => AdamWOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("AMSGrad", () => AMSGradOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("BFGS", () => BFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("ConjugateGradient", () => ConjugateGradientOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("CoordinateDescent", () => CoordinateDescentOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("DFP", () => DFPOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("GradientDescent", () => GradientDescentOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("LAMB", () => LAMBOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("LARS", () => LARSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("LBFGS", () => LBFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("Lion", () => LionOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("MiniBatchGradientDescent", () => MiniBatchGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("Momentum", () => MomentumOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("Nadam", () => NadamOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("Nesterov", () => NesterovAcceleratedGradientOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("RMSProp", () => RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("StochasticGradientDescent", () => StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
        yield return Factory("TrustRegion", () => TrustRegionOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction());
    }

    private static object[] Factory(
        string name, Func<IFunctionOptimizer<double>> build) => new object[] { name, build };

    /// <summary>
    /// Every factory produces an optimizer that can actually minimize. The objective is the easiest
    /// possible one — a round bowl — so a failure means the construction path is broken rather than
    /// the method being unsuited to the problem.
    /// </summary>
    [Theory]
    [MemberData(nameof(GradientOptimizerFactories))]
    public void CreateForFunction_ProducesAWorkingMinimizer(
        string name, Func<IFunctionOptimizer<double>> build)
    {
        var optimizer = build();
        Assert.NotNull(optimizer);

        var start = new Vector<double>(new[] { 10.0, -8.0, 4.0 });
        var (startObjective, _) = ShiftedSphere(start);

        Vector<double> answer = optimizer.Minimize(start, ShiftedSphere, MaxIterations, 1e-6);
        var (endObjective, _) = ShiftedSphere(answer);

        Assert.True(
            endObjective < startObjective,
            $"{name} did not reduce the objective: {startObjective} -> {endObjective}.");

        for (int i = 0; i < answer.Length; i++)
        {
            Assert.False(
                double.IsNaN(answer[i]) || double.IsInfinity(answer[i]),
                $"{name} returned a non-finite coordinate at index {i}.");
        }
    }

    /// <summary>
    /// The factory must not leak state between instances: two independently created optimizers run
    /// on the same problem have to agree exactly.
    /// </summary>
    [Fact]
    public void CreateForFunction_GivesIndependentInstances()
    {
        var start = new Vector<double>(new[] { -1.2, 1.0 });

        Vector<double> first = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(start, ChainedRosenbrock, MaxIterations, GradientTolerance);

        var reused = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction();
        reused.Minimize(
            new Vector<double>(new[] { 5.0, 5.0 }), ChainedRosenbrock, MaxIterations, GradientTolerance);
        Vector<double> afterOtherWork = reused.Minimize(
            start, ChainedRosenbrock, MaxIterations, GradientTolerance);

        Assert.Equal(first[0], afterOtherWork[0], 9);
        Assert.Equal(first[1], afterOtherWork[1], 9);
    }

    /// <summary>
    /// The options object reaches the optimizer. A memory of one leaves L-BFGS with almost no
    /// curvature information, so it must take strictly more evaluations than the default ten.
    /// </summary>
    [Fact]
    public void CreateForFunction_AppliesTheOptionsItIsGiven()
    {
        var start = RosenbrockStart(10);

        var starved = new CountingObjective(ChainedRosenbrock);
        LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction(new LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                MemorySize = 1,
            })
            .Minimize(start, starved.Evaluate, MaxIterations, GradientTolerance);

        var normal = new CountingObjective(ChainedRosenbrock);
        LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(start, normal.Evaluate, MaxIterations, GradientTolerance);

        Assert.True(
            starved.Count > normal.Count,
            $"A memory of one should cost more than the default ten; got {starved.Count} against " +
            $"{normal.Count}.");
    }

    /// <summary>
    /// L-BFGS reaches Rosenbrock's optimum, and does it within the evaluation budget the algorithm
    /// is published with.
    /// </summary>
    /// <remarks>
    /// The budgets here are the point of the test. Nocedal and Wright's L-BFGS assumes a line
    /// search satisfying the Wolfe conditions; accepting steps on sufficient decrease alone leaves
    /// the correction pairs needing Powell damping, and damping costs the method its superlinear
    /// convergence. Measured on this exact problem, sufficient-decrease-only took 253, 7997 and
    /// 8379 evaluations at 2, 5 and 10 variables against 64, 75 and 101 with the Wolfe search. A
    /// budget of 400 fails loudly if that regresses while staying far above the observed counts.
    /// </remarks>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(20)]
    public void Lbfgs_SolvesRosenbrockWithinASuperlinearBudget(int dimension)
    {
        var counter = new CountingObjective(ChainedRosenbrock);

        Vector<double> answer = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(RosenbrockStart(dimension), counter.Evaluate, MaxIterations, 1e-6);

        var (objective, _) = ChainedRosenbrock(answer);

        Assert.True(objective < 1e-8, $"Expected the optimum at n = {dimension}; got f = {objective}.");

        Assert.True(
            counter.Count <= 400,
            $"L-BFGS needed {counter.Count} evaluations at n = {dimension}. Above 400 means the " +
            "line search stopped satisfying the Wolfe conditions and convergence has gone linear.");
    }

    /// <summary>
    /// The sufficient-decrease search is still available and still works, so turning the Wolfe
    /// search off is a supported choice rather than a broken path.
    /// </summary>
    [Fact]
    public void Lbfgs_WithoutTheWolfeSearch_StillReachesTheOptimum()
    {
        var options = new LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            UseStrongWolfeLineSearch = false,
        };

        Vector<double> answer = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction(options)
            .Minimize(RosenbrockStart(5), ChainedRosenbrock, MaxIterations, 1e-6);

        var (objective, _) = ChainedRosenbrock(answer);
        Assert.True(objective < 1e-6, $"Expected the optimum; got f = {objective}.");
    }

    /// <summary>
    /// A tighter curvature constant makes the line search fussier without breaking it. Both settings
    /// must find the same answer, because the conditions bound the step rather than choosing it.
    /// </summary>
    [Fact]
    public void Lbfgs_CurvatureConstant_ChangesTheSearchNotTheAnswer()
    {
        var start = RosenbrockStart(5);

        Vector<double> permissive = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(start, ChainedRosenbrock, MaxIterations, 1e-8);

        Vector<double> fussy = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction(new LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                WolfeCurvatureConstant = 0.1,
            })
            .Minimize(start, ChainedRosenbrock, MaxIterations, 1e-8);

        for (int i = 0; i < permissive.Length; i++)
        {
            Assert.Equal(1.0, permissive[i], 5);
            Assert.Equal(1.0, fussy[i], 5);
        }
    }

    /// <summary>
    /// Powell damping exists to repair pairs a sufficient-decrease step can leave unusable. A Wolfe
    /// step never produces such a pair, so the damping factor must stop mattering — which is the
    /// direct evidence that the curvature condition is genuinely being enforced.
    /// </summary>
    [Fact]
    public void Lbfgs_WithTheWolfeSearch_IsUnaffectedByTheDampingFactor()
    {
        var start = RosenbrockStart(5);

        var damped = new CountingObjective(ChainedRosenbrock);
        LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(start, damped.Evaluate, MaxIterations, 1e-6);

        var undamped = new CountingObjective(ChainedRosenbrock);
        LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction(new LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                PowellDampingFactor = 0.0,
            })
            .Minimize(start, undamped.Evaluate, MaxIterations, 1e-6);

        Assert.Equal(damped.Count, undamped.Count);
    }

    /// <summary>
    /// The strong Wolfe conditions are only jointly satisfiable when c1 &lt; c2. A configuration that
    /// violates it must fall back to the sufficient-decrease search rather than looping or throwing.
    /// </summary>
    [Fact]
    public void Lbfgs_WithAnImpossibleWolfePair_FallsBackRatherThanFailing()
    {
        var options = new LBFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            ArmijoConstant = 0.5,
            WolfeCurvatureConstant = 0.4,
        };

        Vector<double> answer = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction(options)
            .Minimize(new Vector<double>(new[] { 10.0, -8.0, 4.0 }), ShiftedSphere, MaxIterations, 1e-6);

        for (int i = 0; i < answer.Length; i++)
        {
            Assert.Equal(3.0, answer[i], 4);
        }
    }

    /// <summary>
    /// BFGS reaches Rosenbrock's optimum within the budget a line-searched quasi-Newton method is
    /// entitled to.
    /// </summary>
    /// <remarks>
    /// Before BFGS had a Minimize override it inherited the base loop, which has no line search at
    /// all, and needed 51/1911/1166/673 evaluations at 2/5/10/20 variables. With the shared strong
    /// Wolfe search it needs 65/65/100/145 — and now beats L-BFGS at every size above two, which
    /// is the ordering the theory predicts, since it keeps the whole curvature history rather
    /// than ten pairs.
    /// </remarks>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(20)]
    [InlineData(50)]
    public void Bfgs_SolvesRosenbrockWithinALineSearchedBudget(int dimension)
    {
        var counter = new CountingObjective(ChainedRosenbrock);

        Vector<double> answer = BFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(RosenbrockStart(dimension), counter.Evaluate, MaxIterations, 1e-8);

        var (objective, _) = ChainedRosenbrock(answer);

        Assert.True(objective < 1e-10, $"Expected the optimum at n = {dimension}; got f = {objective}.");

        Assert.True(
            counter.Count <= 600,
            $"BFGS needed {counter.Count} evaluations at n = {dimension}. Above 600 means the " +
            "line search stopped being applied on the plain-function path.");
    }

    /// <summary>
    /// Nonlinear conjugate gradient converges too, and its storage is three vectors regardless of
    /// the problem size.
    /// </summary>
    /// <remarks>
    /// Two things had to be right for this. The recursion has to remember the DIRECTION rather
    /// than the step, and the previous GRADIENT has to be the one at the previous point — an
    /// earlier draft recorded the gradient at the new point, which made every beta zero and turned
    /// the method into steepest descent: 165066 evaluations at two variables, against 379 now.
    /// </remarks>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(20)]
    public void ConjugateGradient_SolvesRosenbrock(int dimension)
    {
        var counter = new CountingObjective(ChainedRosenbrock);

        Vector<double> answer = ConjugateGradientOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(RosenbrockStart(dimension), counter.Evaluate, MaxIterations, 1e-8);

        var (objective, _) = ChainedRosenbrock(answer);

        Assert.True(objective < 1e-10, $"Expected the optimum at n = {dimension}; got f = {objective}.");

        Assert.True(
            counter.Count <= 8000,
            $"Conjugate gradient needed {counter.Count} evaluations at n = {dimension}. Above " +
            "8000 means the recursion has degenerated into steepest descent.");
    }

    /// <summary>
    /// Conjugate gradient defaults to a much stricter curvature constant than the quasi-Newton
    /// methods, because its direction is only a descent direction when the search is accurate.
    /// </summary>
    [Fact]
    public void ConjugateGradient_DefaultsToTheStricterCurvatureConstant()
    {
        var options = new ConjugateGradientOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        Assert.Equal(0.1, options.WolfeCurvatureConstant, 12);

        var quasiNewton = new BFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        Assert.Equal(0.9, quasiNewton.WolfeCurvatureConstant, 12);
    }

    /// <summary>
    /// DFP is the weakest of the three and is included as the historical comparison it is: it
    /// solves the small cases and loses ground as the problem grows, which is the empirical result
    /// that made BFGS the default.
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    public void Dfp_SolvesTheSmallCases(int dimension)
    {
        var counter = new CountingObjective(ChainedRosenbrock);

        Vector<double> answer = DFPOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction()
            .Minimize(RosenbrockStart(dimension), counter.Evaluate, MaxIterations, 1e-8);

        var (objective, _) = ChainedRosenbrock(answer);

        Assert.True(objective < 1e-10, $"Expected the optimum at n = {dimension}; got f = {objective}.");
    }

    /// <summary>
    /// The line-search options now live on the shared base, so every gradient optimizer has them
    /// and a caller can tune any of them.
    /// </summary>
    [Fact]
    public void LineSearchOptions_AreSharedAcrossTheGradientFamily()
    {
        var bfgs = new BFGSOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            UseStrongWolfeLineSearch = false,
            ArmijoConstant = 1e-3,
            LineSearchMaxZoomSteps = 5,
        };

        Assert.False(bfgs.UseStrongWolfeLineSearch);
        Assert.Equal(1e-3, bfgs.ArmijoConstant, 12);
        Assert.Equal(5, bfgs.LineSearchMaxZoomSteps);

        // And turning the Wolfe search off still reaches the answer, on the backtracking fallback.
        Vector<double> answer = BFGSOptimizer<double, Tensor<double>, Tensor<double>>
            .CreateForFunction(bfgs)
            .Minimize(RosenbrockStart(5), ChainedRosenbrock, MaxIterations, 1e-6);

        Assert.True(ChainedRosenbrock(answer).objective < 1e-6);
    }

}
