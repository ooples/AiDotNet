#nullable disable
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests for the general-function minimization API — <c>IFunctionOptimizer&lt;T&gt;</c>
/// and <c>IDerivativeFreeFunctionOptimizer&lt;T&gt;</c> — which lets any optimizer minimize a plain
/// scalar objective with no model or dataset, the way <c>scipy.optimize.minimize</c> does.
/// </summary>
/// <remarks>
/// CRITICAL: These tests check answers against KNOWN optima, not against whatever the code
/// currently produces. If one fails, FIX THE OPTIMIZER, do not relax the assertion.
///
/// Benchmark functions and their optima:
/// - Sphere:     f(x) = sum(x_i^2),                      minimum 0 at the origin.
/// - Quadratic:  f(x) = (x-a)^T A (x-a) with A positive definite, minimum 0 at x = a.
/// - Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2,        minimum 0 at (1, 1).
/// - Himmelblau: f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2,    four minima with value 0, one at (3, 2).
/// </remarks>
public class FunctionMinimizationIntegrationTests
{
    private const int MaxIterations = 5000;
    private const double GradientTolerance = 1e-8;

    #region Benchmark objectives

    /// <summary>f(x) = sum(x_i^2); gradient 2x. Minimum 0 at the origin.</summary>
    private static (double objective, Vector<double> gradient) Sphere(Vector<double> x)
    {
        double sum = 0;
        var gradient = new Vector<double>(x.Length);
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * x[i];
            gradient[i] = 2.0 * x[i];
        }

        return (sum, gradient);
    }

    /// <summary>
    /// f(x) = sum(scale_i * (x_i - centre_i)^2); gradient 2*scale_i*(x_i - centre_i).
    /// An axis-aligned positive-definite quadratic with minimum 0 at <paramref name="centre"/>.
    /// Differing scales make it ill-conditioned, which separates adaptive methods from plain
    /// gradient descent.
    /// </summary>
    private static (double objective, Vector<double> gradient) Quadratic(
        Vector<double> x, double[] centre, double[] scale)
    {
        double sum = 0;
        var gradient = new Vector<double>(x.Length);
        for (int i = 0; i < x.Length; i++)
        {
            double delta = x[i] - centre[i];
            sum += scale[i] * delta * delta;
            gradient[i] = 2.0 * scale[i] * delta;
        }

        return (sum, gradient);
    }

    /// <summary>Rosenbrock's banana function; minimum 0 at (1, 1).</summary>
    private static (double objective, Vector<double> gradient) Rosenbrock(Vector<double> v)
    {
        double x = v[0], y = v[1];
        double objective = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);

        var gradient = new Vector<double>(2);
        gradient[0] = -2 * (1 - x) - 400 * x * (y - x * x);
        gradient[1] = 200 * (y - x * x);

        return (objective, gradient);
    }

    /// <summary>Himmelblau's function; four separate minima, all with value 0.</summary>
    private static double Himmelblau(Vector<double> v)
    {
        double x = v[0], y = v[1];
        return (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7);
    }

    private static Vector<double> Point(params double[] values) => Vector<double>.FromArray(values);

    #endregion

    #region Gradient-based optimizers: IFunctionOptimizer

    /// <summary>
    /// Every gradient optimizer must drive the sphere function to its known minimum at the origin.
    /// This is the headline claim of the rollout: the whole family gained a working
    /// <c>Minimize</c> by reusing each optimizer's own <c>Step</c>, so if any one of them is wired
    /// up incorrectly it shows here.
    /// </summary>
    [Theory]
    [InlineData("GradientDescent")]
    [InlineData("Momentum")]
    [InlineData("Nesterov")]
    [InlineData("Adam")]
    [InlineData("AdamW")]
    [InlineData("AdaMax")]
    [InlineData("Nadam")]
    [InlineData("AMSGrad")]
    [InlineData("Adagrad")]
    [InlineData("RMSProp")]
    [InlineData("AdaDelta")]
    [InlineData("Lion")]
    [InlineData("LBFGS")]
    public void Minimize_SphereFunction_ReachesKnownMinimumAtOrigin(string optimizerName)
    {
        var optimizer = CreateOptimizer(optimizerName);

        var result = optimizer.Minimize(Point(3.0, -4.0, 2.5), Sphere, MaxIterations, GradientTolerance);

        // The minimum is exactly the origin; allow a small residual for finite iteration counts.
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(
                Math.Abs(result[i]) < 1e-3,
                $"{optimizerName}: component {i} is {result[i]}, expected within 1e-3 of 0.");
        }

        var (finalObjective, _) = Sphere(result);
        Assert.True(
            finalObjective < 1e-6,
            $"{optimizerName}: final objective {finalObjective}, expected below 1e-6.");
    }

    /// <summary>
    /// The minimum of an axis-aligned quadratic is its centre, which is not the origin — this
    /// catches an implementation that merely shrinks parameters toward zero.
    /// </summary>
    [Theory]
    [InlineData("Adam")]
    [InlineData("AdamW")]
    [InlineData("Nadam")]
    [InlineData("AMSGrad")]
    [InlineData("RMSProp")]
    [InlineData("LBFGS")]
    public void Minimize_IllConditionedQuadratic_ReachesKnownCentre(string optimizerName)
    {
        var centre = new[] { 2.0, -3.0, 0.5 };
        var scale = new[] { 1.0, 20.0, 0.1 };
        var optimizer = CreateOptimizer(optimizerName);

        var result = optimizer.Minimize(
            Point(0.0, 0.0, 0.0),
            x => Quadratic(x, centre, scale),
            MaxIterations,
            GradientTolerance);

        for (int i = 0; i < centre.Length; i++)
        {
            Assert.True(
                Math.Abs(result[i] - centre[i]) < 1e-2,
                $"{optimizerName}: component {i} is {result[i]}, expected within 1e-2 of {centre[i]}.");
        }
    }

    /// <summary>
    /// Rosenbrock's curved valley is the standard stress test for a quasi-Newton method. L-BFGS
    /// should locate (1, 1) closely; this specifically exercises the two-loop recursion and the
    /// Armijo line search that <c>Minimize</c> now runs against the caller's objective.
    /// </summary>
    [Fact]
    public void Minimize_Rosenbrock_LBFGSReachesKnownMinimum()
    {
        var optimizer = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction();

        var result = optimizer.Minimize(Point(-1.2, 1.0), Rosenbrock, MaxIterations, 1e-10);

        Assert.True(Math.Abs(result[0] - 1.0) < 1e-3, $"x = {result[0]}, expected within 1e-3 of 1.");
        Assert.True(Math.Abs(result[1] - 1.0) < 1e-3, $"y = {result[1]}, expected within 1e-3 of 1.");
    }

    /// <summary>
    /// The optimizer must not modify the caller's starting vector — it is an input, not a buffer.
    /// </summary>
    [Fact]
    public void Minimize_DoesNotMutateCallerStartingPoint()
    {
        var start = Point(3.0, -4.0);
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);

        optimizer.Minimize(start, Sphere, 100, GradientTolerance);

        Assert.Equal(3.0, start[0]);
        Assert.Equal(-4.0, start[1]);
    }

    /// <summary>
    /// Each call resets accumulated state, so running the same problem twice on the same instance
    /// must give the same answer. Without the reset, Adam's step counter would carry over and its
    /// bias correction would differ on the second run.
    /// </summary>
    [Fact]
    public void Minimize_CalledTwiceOnSameInstance_ProducesIdenticalResults()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);

        var first = optimizer.Minimize(Point(3.0, -4.0), Sphere, 200, GradientTolerance);
        var second = optimizer.Minimize(Point(3.0, -4.0), Sphere, 200, GradientTolerance);

        Assert.Equal(first[0], second[0], 12);
        Assert.Equal(first[1], second[1], 12);
    }

    /// <summary>
    /// Already at the minimum, the gradient is zero, so the very first convergence check must fire
    /// and the point must come back unchanged.
    /// </summary>
    [Fact]
    public void Minimize_StartingAtMinimum_ReturnsImmediatelyWithoutMoving()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);

        var result = optimizer.Minimize(Point(0.0, 0.0), Sphere, MaxIterations, 1e-10);

        Assert.Equal(0.0, result[0], 12);
        Assert.Equal(0.0, result[1], 12);
    }

    #endregion

    #region Projected minimization (bound constraints)

    /// <summary>
    /// With the unconstrained minimum at the origin but every variable confined to [1, 5], the
    /// constrained solution is the corner nearest the origin: all ones. Every iterate must also be
    /// feasible, which is what distinguishes a projection from clamping the answer at the end.
    /// </summary>
    [Fact]
    public void Minimize_WithBoxProjection_StopsAtConstrainedOptimum()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);

        Vector<double> ProjectOntoBox(Vector<double> point)
        {
            var projected = new Vector<double>(point.Length);
            for (int i = 0; i < point.Length; i++)
            {
                projected[i] = Math.Max(1.0, Math.Min(5.0, point[i]));
            }

            return projected;
        }

        bool everyEvaluationWasFeasible = true;

        (double, Vector<double>) FeasibilityCheckingSphere(Vector<double> x)
        {
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] < 1.0 - 1e-12 || x[i] > 5.0 + 1e-12) everyEvaluationWasFeasible = false;
            }

            return Sphere(x);
        }

        var result = optimizer.Minimize(
            Point(4.0, 3.0), FeasibilityCheckingSphere, MaxIterations, GradientTolerance, ProjectOntoBox);

        Assert.True(everyEvaluationWasFeasible, "An iterate left the feasible box [1, 5].");
        Assert.True(Math.Abs(result[0] - 1.0) < 1e-3, $"x = {result[0]}, expected within 1e-3 of 1.");
        Assert.True(Math.Abs(result[1] - 1.0) < 1e-3, $"y = {result[1]}, expected within 1e-3 of 1.");
    }

    /// <summary>
    /// An infeasible starting point must be projected before the objective ever sees it.
    /// </summary>
    [Fact]
    public void Minimize_WithProjection_ProjectsInfeasibleStartingPoint()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);
        double firstSeenValue = double.NaN;

        var result = optimizer.Minimize(
            Point(-10.0),
            x =>
            {
                if (double.IsNaN(firstSeenValue)) firstSeenValue = x[0];
                return Sphere(x);
            },
            50,
            GradientTolerance,
            point => Point(Math.Max(1.0, point[0])));

        Assert.Equal(1.0, firstSeenValue, 12);
        Assert.True(result[0] >= 1.0 - 1e-12);
    }

    /// <summary>A projection that changes the vector length is a programming error, not a silent
    /// reshape.</summary>
    [Fact]
    public void Minimize_ProjectionReturningWrongLength_Throws()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);

        Assert.Throws<ArgumentException>(() => optimizer.Minimize(
            Point(1.0, 2.0), Sphere, 10, GradientTolerance, _ => Point(1.0)));
    }

    #endregion

    #region Derivative-free optimizers

    /// <summary>
    /// Nelder-Mead needs no gradient at all and must still find the sphere minimum.
    /// </summary>
    [Fact]
    public void Minimize_DerivativeFree_NelderMeadReachesSphereMinimum()
    {
        var optimizer = new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>();

        var result = optimizer.Minimize(
            Point(3.0, -4.0, 2.5), x => Sphere(x).objective, MaxIterations, 1e-12);

        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(
                Math.Abs(result[i]) < 1e-4,
                $"Component {i} is {result[i]}, expected within 1e-4 of 0.");
        }
    }

    /// <summary>
    /// Nelder-Mead on Rosenbrock is the classic <c>fminsearch</c> demonstration; it converges to
    /// (1, 1) without any derivative information.
    /// </summary>
    [Fact]
    public void Minimize_DerivativeFree_NelderMeadSolvesRosenbrock()
    {
        var optimizer = new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>();

        var result = optimizer.Minimize(
            Point(-1.2, 1.0), x => Rosenbrock(x).objective, MaxIterations, 1e-12);

        Assert.True(Math.Abs(result[0] - 1.0) < 1e-3, $"x = {result[0]}, expected within 1e-3 of 1.");
        Assert.True(Math.Abs(result[1] - 1.0) < 1e-3, $"y = {result[1]}, expected within 1e-3 of 1.");
    }

    /// <summary>
    /// Himmelblau has four distinct minima, all with value 0. Starting near (3, 2) the search must
    /// land on that one — a test that the simplex moves downhill locally rather than wandering.
    /// </summary>
    [Fact]
    public void Minimize_DerivativeFree_NelderMeadFindsNearestHimmelblauMinimum()
    {
        var optimizer = new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>();

        var result = optimizer.Minimize(Point(2.5, 2.5), Himmelblau, MaxIterations, 1e-12);

        Assert.True(Math.Abs(result[0] - 3.0) < 1e-3, $"x = {result[0]}, expected within 1e-3 of 3.");
        Assert.True(Math.Abs(result[1] - 2.0) < 1e-3, $"y = {result[1]}, expected within 1e-3 of 2.");
        Assert.True(Himmelblau(result) < 1e-6, $"Objective {Himmelblau(result)}, expected below 1e-6.");
    }

    /// <summary>
    /// A starting coordinate of exactly zero cannot be perturbed proportionally. If the initial
    /// simplex construction did not fall back to an absolute step, the simplex would be degenerate
    /// along that axis and the search could never move in it.
    /// </summary>
    [Fact]
    public void Minimize_DerivativeFree_ZeroStartingCoordinateStillSearchesThatAxis()
    {
        var optimizer = new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>();
        var centre = new[] { 0.0, 1.5 };
        var scale = new[] { 1.0, 1.0 };

        // Starts at the origin, where the first coordinate is exactly zero, and the answer
        // requires moving the SECOND coordinate to 1.5 while keeping the first at 0.
        var result = optimizer.Minimize(
            Point(0.0, 0.0), x => Quadratic(x, centre, scale).objective, MaxIterations, 1e-12);

        Assert.True(Math.Abs(result[0]) < 1e-4, $"x = {result[0]}, expected within 1e-4 of 0.");
        Assert.True(Math.Abs(result[1] - 1.5) < 1e-4, $"y = {result[1]}, expected within 1e-4 of 1.5.");
    }

    #endregion

    #region Argument validation

    [Fact]
    public void Minimize_NullStartingPoint_Throws()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);
        Assert.ThrowsAny<ArgumentException>(() =>
            optimizer.Minimize(null, Sphere, 10, GradientTolerance));
    }

    [Fact]
    public void Minimize_EmptyStartingPoint_Throws()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);
        Assert.Throws<ArgumentException>(() =>
            optimizer.Minimize(new Vector<double>(0), Sphere, 10, GradientTolerance));
    }

    [Fact]
    public void Minimize_NonPositiveIterationCount_Throws()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);
        Assert.Throws<ArgumentException>(() =>
            optimizer.Minimize(Point(1.0), Sphere, 0, GradientTolerance));
    }

    [Fact]
    public void Minimize_GradientOfWrongLength_Throws()
    {
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null);
        Assert.Throws<ArgumentException>(() => optimizer.Minimize(
            Point(1.0, 2.0), _ => (1.0, new Vector<double>(3)), 10, GradientTolerance));
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Builds a model-free optimizer instance by name, with a learning rate large enough that the
    /// fixed iteration budget is enough for the benchmark problems.
    /// </summary>
    private static GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> CreateOptimizer(
        string name)
    {
        return name switch
        {
            "GradientDescent" => new GradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new GradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.1 }),
            "Momentum" => new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "Nesterov" => new NesterovAcceleratedGradientOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new NesterovAcceleratedGradientOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "Adam" => new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "AdamW" => new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "AdaMax" => new AdaMaxOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AdaMaxOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "Nadam" => new NadamOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new NadamOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "AMSGrad" => new AMSGradOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AMSGradOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "Adagrad" => new AdagradOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AdagradOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.5 }),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05 }),
            "AdaDelta" => new AdaDeltaOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AdaDeltaOptimizerOptions<double, Tensor<double>, Tensor<double>>()),
            "Lion" => new LionOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new LionOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.01 }),
            "LBFGS" => LBFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(),
            _ => throw new ArgumentException($"Unknown optimizer '{name}'.", nameof(name)),
        };
    }

    #endregion
}
