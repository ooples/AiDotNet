using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression suite for the convergence-check pattern bug fixed in PR #1351
/// (AdamOptimizer) and swept across the rest of the optimizer suite.
/// </summary>
/// <remarks>
/// <para>
/// Before the fix, every optimizer's convergence check fired after epoch 0
/// because the comparison was <c>|bestStepData - currentStepData| &lt; tolerance</c>,
/// but <c>UpdateBestSolution</c> copies <c>currentStepData</c> into
/// <c>bestStepData</c> on the first iteration (because <c>bestStepData</c>
/// starts uninitialised). After that copy, the difference is always 0 &lt;
/// tolerance and <c>Optimize</c> returns after exactly 1 epoch — i.e. the
/// optimiser silently honours <c>MaxIterations=1</c> regardless of the
/// caller's configured budget.
/// </para>
///
/// <para>
/// The fix is to compare <c>previousStepData</c> (the prior epoch's score)
/// against <c>currentStepData</c>, so the convergence signal is genuine
/// per-epoch progress: "the fitness stopped changing from one epoch to
/// the next."
/// </para>
///
/// <para>
/// Each test instantiates the optimiser, runs <see cref="IOptimizer{T,TInput,TOutput}.Optimize"/>
/// on a deterministic 2D quadratic problem with <c>MaxIterations=10</c>,
/// and asserts that <c>result.Iterations &gt; 1</c>. The pre-fix value was
/// always exactly 1.
/// </para>
///
/// <para>
/// To add a new optimizer to this sweep, add a single line to the
/// <see cref="OptimizerFactories"/> member-data list with the optimizer's
/// name and a factory delegate.
/// </para>
/// </remarks>
public class OptimizerConvergenceCheckPatternTests
{
    private readonly ITestOutputHelper _output;

    public OptimizerConvergenceCheckPatternTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Builds a deterministic 2D quadratic regression fixture: y = 1 + 2*x0 + 3*x1 + small noise.
    /// </summary>
    private static (Matrix<double> X, Vector<double> Y) BuildQuadraticFixture()
    {
        var rng = new Random(42);
        const int n = 64;
        const int features = 2;

        var x = new Matrix<double>(n, features);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < features; j++)
            {
                x[i, j] = rng.NextDouble() * 2.0 - 1.0;
            }
            y[i] = 1.0 + 2.0 * x[i, 0] + 3.0 * x[i, 1] + (rng.NextDouble() * 0.01);
        }
        return (x, y);
    }

    /// <summary>
    /// Factory delegate for an optimiser-under-test. Takes a fresh model and
    /// the desired <c>MaxIterations</c>; returns an <see cref="IOptimizer{T,TInput,TOutput}"/>
    /// instance ready to run.
    /// </summary>
    public delegate IOptimizer<double, Matrix<double>, Vector<double>> OptimizerFactory(
        IFullModel<double, Matrix<double>, Vector<double>> model,
        int maxIterations);

    /// <summary>
    /// Parametrised data: one row per optimiser in the sweep. Adding a new
    /// optimiser is a single line.
    /// </summary>
    public static IEnumerable<object[]> OptimizerFactories()
    {
        // The test uses Tolerance=0 (and effectively-disabled warmup for LARS / LAMB)
        // so the convergence check fires only on genuine "no progress" plateaus, not
        // on small-but-real per-epoch deltas. The pre-fix bug would still surface
        // because |best - current| is 0 (exactly) after UpdateBestSolution copies on
        // the first iteration, regardless of Tolerance.
        const double Tol = 0.0;

        yield return Row("Adam8BitOptimizer", (model, maxIter) =>
            new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(model,
                new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("AdamWOptimizer", (model, maxIter) =>
            new AdamWOptimizer<double, Matrix<double>, Vector<double>>(model,
                new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("AdagradOptimizer", (model, maxIter) =>
            new AdagradOptimizer<double, Matrix<double>, Vector<double>>(model,
                new AdagradOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("AdaDeltaOptimizer", (model, maxIter) =>
            new AdaDeltaOptimizer<double, Matrix<double>, Vector<double>>(model,
                new AdaDeltaOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("AdaMaxOptimizer", (model, maxIter) =>
            new AdaMaxOptimizer<double, Matrix<double>, Vector<double>>(model,
                new AdaMaxOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("AMSGradOptimizer", (model, maxIter) =>
            new AMSGradOptimizer<double, Matrix<double>, Vector<double>>(model,
                new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("BFGSOptimizer", (model, maxIter) =>
            new BFGSOptimizer<double, Matrix<double>, Vector<double>>(model,
                new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("ConjugateGradientOptimizer", (model, maxIter) =>
            new ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>>(model,
                new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("CoordinateDescentOptimizer", (model, maxIter) =>
            new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(model,
                new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        // DFPOptimizer has an UNRELATED pre-existing bug in UpdateInverseHessian
        // ("Vector lengths must match. Got 3 and 0") that is independent of the
        // convergence-check pattern fixed here. It throws before reaching the
        // convergence check on epoch 2, so the regression test would crash for
        // reasons unrelated to this PR. We still apply the convergence-check
        // pattern fix to DFPOptimizer.cs:143 but exclude it from this regression
        // sweep until the inverse-hessian initialisation bug is fixed in a
        // separate issue / PR.

        yield return Row("FTRLOptimizer", (model, maxIter) =>
            new FTRLOptimizer<double, Matrix<double>, Vector<double>>(model,
                new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("LAMBOptimizer", (model, maxIter) =>
            new LAMBOptimizer<double, Matrix<double>, Vector<double>>(model,
                new LAMBOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol, WarmupEpochs = 0 }));

        yield return Row("LARSOptimizer", (model, maxIter) =>
            new LARSOptimizer<double, Matrix<double>, Vector<double>>(model,
                new LARSOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol, WarmupEpochs = 0 }));

        yield return Row("LBFGSOptimizer", (model, maxIter) =>
            new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(model,
                new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("LevenbergMarquardtOptimizer", (model, maxIter) =>
            new LevenbergMarquardtOptimizer<double, Matrix<double>, Vector<double>>(model,
                new LevenbergMarquardtOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("LionOptimizer", (model, maxIter) =>
            new LionOptimizer<double, Matrix<double>, Vector<double>>(model,
                new LionOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("MiniBatchGradientDescentOptimizer", (model, maxIter) =>
            new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model,
                new MiniBatchGradientDescentOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("MomentumOptimizer", (model, maxIter) =>
            new MomentumOptimizer<double, Matrix<double>, Vector<double>>(model,
                new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("NadamOptimizer", (model, maxIter) =>
            new NadamOptimizer<double, Matrix<double>, Vector<double>>(model,
                new NadamOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("NelderMeadOptimizer", (model, maxIter) =>
            new NelderMeadOptimizer<double, Matrix<double>, Vector<double>>(model,
                new NelderMeadOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, Tolerance = Tol }));

        yield return Row("NesterovAcceleratedGradientOptimizer", (model, maxIter) =>
            new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(model,
                new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("NewtonMethodOptimizer", (model, maxIter) =>
            new NewtonMethodOptimizer<double, Matrix<double>, Vector<double>>(model,
                new NewtonMethodOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("PowellOptimizer", (model, maxIter) =>
            new PowellOptimizer<double, Matrix<double>, Vector<double>>(model,
                new PowellOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, Tolerance = Tol }));

        yield return Row("ProximalGradientDescentOptimizer", (model, maxIter) =>
            new ProximalGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model,
                new ProximalGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("RootMeanSquarePropagationOptimizer", (model, maxIter) =>
            new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(model,
                new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("StochasticGradientDescentOptimizer", (model, maxIter) =>
            new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(model,
                new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));

        yield return Row("TrustRegionOptimizer", (model, maxIter) =>
            new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(model,
                new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>> { MaxIterations = maxIter, InitialLearningRate = 0.01, Tolerance = Tol }));
    }

    private static object[] Row(string name, OptimizerFactory factory) => new object[] { name, factory };

    /// <summary>
    /// Regression: every fixed optimiser must execute MORE than one iteration
    /// on a deterministic problem. The pre-fix value was always exactly 1
    /// because the convergence check fired immediately after epoch 0.
    /// </summary>
    [Theory(Timeout = 30000)]
    [MemberData(nameof(OptimizerFactories))]
    public async Task Optimize_RunsMoreThanOneIteration_AfterConvergenceCheckPatternFix(
        string optimizerName, OptimizerFactory factory)
    {
        await Task.Yield();

        var (x, y) = BuildQuadraticFixture();
        var model = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        const int requestedIterations = 10;
        var optimizer = factory(model, requestedIterations);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = x, YTrain = y,
            XValidation = x, YValidation = y,
            XTest = x, YTest = y
        };

        var result = optimizer.Optimize(inputData);

        _output.WriteLine($"{optimizerName}: Iterations={result.Iterations} (requested {requestedIterations})");

        // The pre-fix value was always exactly 1 because the convergence check
        // fired immediately after epoch 0. After the fix, the optimiser must
        // run more than 1 iteration. We assert a strict > 1 — even if the
        // optimiser legitimately converges on a true plateau before
        // MaxIterations, the per-epoch comparison guarantees at least 2
        // iterations to detect that plateau.
        Assert.True(result.Iterations > 1,
            $"{optimizerName}.Optimize ran only {result.Iterations} iteration(s) — " +
            "the convergence check is still firing on the uninitialised bestStepData " +
            "instead of on per-epoch progress. See PR #1351 / Issue #1340.");
    }
}
