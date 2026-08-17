using AiDotNet.Classification.SVM;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.Enums;
using AiDotNet.GaussianProcesses;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Solvers.Constrained;
using AiDotNet.Solvers.InteriorPoint;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Review;

[Trait("Category", "Integration")]
public class Pr2010ReviewFollowUpTests
{
    [Fact(Timeout = 120000)]
    public async Task NuSvc_RejectsNuThatClassDistributionCannotSupport()
    {
        await Task.Yield();

        var x = Column(Enumerable.Range(0, 10).Select(i => (double)i).ToArray());
        var y = Vector<double>.FromArray(new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
        var model = new NuSupportVectorClassifier<double>(
            new SVMOptions<double> { Kernel = KernelType.Linear, Seed = 42 }, null, nu: 0.5);

        var exception = Assert.Throws<ArgumentException>(() => model.Train(x, y));

        Assert.Contains("must be at most 0.2", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task NuSvc_AsymmetricBoundary_UsesTheSolverBias()
    {
        await Task.Yield();

        var x = Column(new[] { 1.0, 2.0, 3.0, 8.0, 9.0, 10.0 });
        var y = Vector<double>.FromArray(new[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 });
        var model = new NuSupportVectorClassifier<double>(
            new SVMOptions<double> { Kernel = KernelType.Linear, MaxIterations = 1000, Seed = 42 },
            null,
            nu: 0.5);

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(y.ToArray(), predictions.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task DetrTapeLoss_RejectsIncompatibleStructuredTargetShape()
    {
        await Task.Yield();

        var loss = new DETRSetLoss<double>(numClasses: 2);
        var predicted = new Tensor<double>([1, 3, 6]);
        var target = new Tensor<double>([2, 3, 5]);

        var exception = Assert.Throws<ArgumentException>(() => loss.ComputeTapeLoss(predicted, target));

        Assert.Contains("Received predicted", exception.Message, StringComparison.Ordinal);
        Assert.Contains("and target", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task HyperparameterProjection_RejectsContradictoryPositiveBounds()
    {
        await Task.Yield();

        var optimizer = new HyperparameterOptimizer<double>(maxIterations: 2);

        var exception = Assert.Throws<ArgumentException>(() => optimizer.GradientDescent(
            new Dictionary<string, double> { ["lengthScale"] = 1e-8 },
            values => (0.0, new Dictionary<string, double> { ["lengthScale"] = 0.0 }),
            parameterBounds: new Dictionary<string, (double min, double max)>
            {
                ["lengthScale"] = (0.0, 1e-9),
            }));

        Assert.Contains("upper bound", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task HyperparameterProjection_RejectsNonFiniteAndInvertedBounds()
    {
        await Task.Yield();

        var optimizer = new HyperparameterOptimizer<double>(maxIterations: 2);
        Func<Dictionary<string, double>,
            (double Loss, Dictionary<string, double> Gradient)> objective =
            _ => (0.0, new Dictionary<string, double> { ["offset"] = 0.0 });

        Assert.Throws<ArgumentException>(() => optimizer.GradientDescent(
            new Dictionary<string, double> { ["offset"] = 0.0 },
            objective,
            parameterBounds: new Dictionary<string, (double min, double max)>
            {
                ["offset"] = (double.NaN, 1.0),
            }));

        Assert.Throws<ArgumentException>(() => optimizer.GradientDescent(
            new Dictionary<string, double> { ["offset"] = 0.0 },
            objective,
            parameterBounds: new Dictionary<string, (double min, double max)>
            {
                ["offset"] = (2.0, 1.0),
            }));
    }

    [Fact(Timeout = 120000)]
    public async Task ControlOptions_CopyConstructorsPreserveSeedAndOwnMutableValues()
    {
        await Task.Yield();

        var lmi = new LinearMatrixInequalityOptions
        {
            Seed = 17,
            MaxIterations = 123,
            Margin = 1e-6,
            InitialStepSize = 0.25,
            PowerIterations = 44,
        };
        var lmiCopy = new LinearMatrixInequalityOptions(lmi);

        Assert.Equal(lmi.Seed, lmiCopy.Seed);
        Assert.Equal(lmi.MaxIterations, lmiCopy.MaxIterations);
        Assert.Equal(lmi.Margin, lmiCopy.Margin);
        Assert.Equal(lmi.InitialStepSize, lmiCopy.InitialStepSize);
        Assert.Equal(lmi.PowerIterations, lmiCopy.PowerIterations);

        var lower = V(-2.0);
        var terminal = Matrix<double>.CreateIdentity(1);
        var nonlinear = new NonlinearModelPredictiveControllerOptions<double>
        {
            Seed = 23,
            Horizon = 7,
            SqpIterations = 3,
            StepSize = 0.5,
            Tolerance = 1e-7,
            InputLowerBounds = lower,
            InputUpperBounds = V(2.0),
            TerminalCost = terminal,
            WarmStart = false,
        };
        var nonlinearCopy = new NonlinearModelPredictiveControllerOptions<double>(nonlinear);

        lower[0] = -99.0;
        terminal[0, 0] = 99.0;

        Assert.Equal(23, nonlinearCopy.Seed);
        Assert.Equal(7, nonlinearCopy.Horizon);
        Assert.Equal(3, nonlinearCopy.SqpIterations);
        Assert.Equal(0.5, nonlinearCopy.StepSize);
        Assert.Equal(1e-7, nonlinearCopy.Tolerance);
        Assert.Equal(-2.0, nonlinearCopy.InputLowerBounds![0]);
        Assert.Equal(2.0, nonlinearCopy.InputUpperBounds![0]);
        Assert.Equal(1.0, nonlinearCopy.TerminalCost![0, 0]);
        Assert.False(nonlinearCopy.WarmStart);
    }

    [Fact(Timeout = 120000)]
    public async Task PrincipalComponentRegression_HandlesConstantColumnsAndClonesCompleteState()
    {
        await Task.Yield();

        var x = new Matrix<double>(8, 3);
        var y = new Vector<double>(8);
        for (int i = 0; i < x.Rows; i++)
        {
            x[i, 0] = i;
            x[i, 1] = 5.0;
            x[i, 2] = i % 2 == 0 ? -1.0 : 1.0;
            y[i] = 7.5;
        }

        var model = new PrincipalComponentRegression<double>(
            new PrincipalComponentRegressionOptions<double> { NumComponents = 2 });
        model.Train(x, y);
        var expected = model.Predict(x);
        var clone = (PrincipalComponentRegression<double>)model.Clone();
        var actual = clone.Predict(x);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(double.IsFinite(expected[i]));
            Assert.Equal(7.5, expected[i], 8);
            Assert.Equal(expected[i], actual[i], 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task StepwiseForwardSelection_PreservesOriginalFeatureIndicesAfterRemoval()
    {
        await Task.Yield();

        var x = new Matrix<double>(60, 3);
        var y = new Vector<double>(60);
        for (int i = 0; i < x.Rows; i++)
        {
            x[i, 0] = (i * 17 % 13) - 6;
            x[i, 1] = i % 2 == 0 ? -1.0 : 1.0;
            x[i, 2] = (i % 5) - 2.0;
            y[i] = 100.0 * x[i, 1] + 10.0 * x[i, 2];
        }

        var model = new StepwiseRegression<double>(new StepwiseRegressionOptions<double>
        {
            Method = StepwiseMethod.Forward,
            MaxFeatures = 2,
            MinImprovement = 0.0,
        });
        model.Train(x, y);

        var active = model.GetActiveFeatureIndices().ToArray();
        Assert.Contains(1, active);
        Assert.Contains(2, active);
        Assert.Equal(active.Length, active.Distinct().Count());
    }

    [Fact(Timeout = 120000)]
    public async Task TimeSeriesRoundTrip_PreservesTrendSeasonalClockAndLagSeed()
    {
        await Task.Yield();

        var options = new TimeSeriesRegressionOptions<double>
        {
            LagOrder = 2,
            IncludeTrend = true,
            SeasonalPeriod = 4,
            AutocorrelationCorrection = false,
            ModelType = TimeSeriesModelType.AutoRegressive,
        };
        var x = new Matrix<double>(32, 1);
        var y = new Vector<double>(32);
        for (int i = 0; i < x.Rows; i++)
        {
            x[i, 0] = i;
            y[i] = 20.0 + 1.5 * i + (i % 4 == 1 ? 5.0 : 0.0);
        }

        var future = new Matrix<double>(5, 1);
        for (int i = 0; i < future.Rows; i++) future[i, 0] = x.Rows + i;

        var model = new TimeSeriesRegression<double>(options);
        model.Train(x, y);
        var expected = model.Predict(future);

        var restored = new TimeSeriesRegression<double>(new TimeSeriesRegressionOptions<double>());
        restored.Deserialize(model.Serialize());
        var actual = restored.Predict(future);

        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    [Fact(Timeout = 120000)]
    public async Task TimeSeriesDeserialize_AcceptsPayloadBeforeRecursiveStateTrailer()
    {
        await Task.Yield();

        var options = new TimeSeriesRegressionOptions<double>
        {
            LagOrder = 2,
            ModelType = TimeSeriesModelType.AutoRegressive,
        };
        var model = new TimeSeriesRegression<double>(options);
        var x = Column(Enumerable.Range(0, 12).Select(i => (double)i).ToArray());
        var y = Vector<double>.FromArray(Enumerable.Range(0, 12).Select(i => 3.0 * i + 2.0).ToArray());
        model.Train(x, y);

        byte[] current = model.Serialize();
        int appendedBytes = sizeof(int) + options.LagOrder * sizeof(double) + sizeof(int);
        byte[] legacy = current.Take(current.Length - appendedBytes).ToArray();
        var restored = new TimeSeriesRegression<double>();

        restored.Deserialize(legacy);
        Vector<double> prediction = restored.Predict(Column(new[] { 12.0 }));

        Assert.True(double.IsFinite(prediction[0]));
    }

    [Fact(Timeout = 120000)]
    public async Task LinearProgram_OwnsInputsAndRejectsWrongDirectionInfinity()
    {
        await Task.Yield();

        var objective = V(1.0, 2.0);
        var constraints = new Matrix<double>(1, 2);
        constraints[0, 0] = 3.0;
        var bounds = V(4.0);
        var program = new LinearProgram<double>(objective, constraints, bounds);

        objective[0] = 99.0;
        constraints[0, 0] = 99.0;
        bounds[0] = 99.0;

        Assert.Equal(1.0, program.Objective[0]);
        Assert.Equal(3.0, program.InequalityMatrix![0, 0]);
        Assert.Equal(4.0, program.InequalityBounds![0]);
        Assert.Throws<ArgumentException>(() => new LinearProgram<double>(
            V(1.0), lowerBounds: V(double.PositiveInfinity)));
        Assert.Throws<ArgumentException>(() => new LinearProgram<double>(
            V(1.0), upperBounds: V(double.NegativeInfinity)));
    }

    [Fact(Timeout = 120000)]
    public async Task QuadraticProgram_ValidatesEntireObjectiveAndOwnsInputs()
    {
        await Task.Yield();

        var quadratic = Matrix<double>.CreateIdentity(2);
        var linear = V(1.0, 2.0);
        var program = new QuadraticProgram<double>(quadratic, linear);
        quadratic[0, 0] = 99.0;
        linear[0] = 99.0;

        Assert.Equal(1.0, program.Quadratic[0, 0]);
        Assert.Equal(1.0, program.Linear[0]);

        var nonFiniteDiagonal = Matrix<double>.CreateIdentity(2);
        nonFiniteDiagonal[1, 1] = double.NaN;
        Assert.Throws<ArgumentException>(() => new QuadraticProgram<double>(nonFiniteDiagonal, V(0.0, 0.0)));
        Assert.Throws<ArgumentException>(() => new QuadraticProgram<double>(
            Matrix<double>.CreateIdentity(2), V(0.0, double.PositiveInfinity)));
    }

    [Fact(Timeout = 120000)]
    public async Task OptimizerOptionCopies_PreserveInheritedAndAlgorithmSpecificSettings()
    {
        await Task.Yield();

        var lbfgs = new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            Seed = 71,
            MaxIterations = 29,
            Tolerance = 3e-7,
            MemorySize = 13,
            LineSearchMaxSteps = 17,
            PowellDampingFactor = 0.35,
        };
        var lbfgsCopy = new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(lbfgs);

        Assert.Equal(lbfgs.Seed, lbfgsCopy.Seed);
        Assert.Equal(lbfgs.MaxIterations, lbfgsCopy.MaxIterations);
        Assert.Equal(lbfgs.Tolerance, lbfgsCopy.Tolerance);
        Assert.Equal(lbfgs.MemorySize, lbfgsCopy.MemorySize);
        Assert.Equal(lbfgs.LineSearchMaxSteps, lbfgsCopy.LineSearchMaxSteps);
        Assert.Equal(lbfgs.PowellDampingFactor, lbfgsCopy.PowellDampingFactor);

        var nelderMead = new NelderMeadOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            Seed = 83,
            MaxIterations = 31,
            Tolerance = 7e-8,
            InitialSimplexStep = 0.12,
            UseAdaptiveParameters = true,
            AdaptationRate = 0.07,
        };
        var nelderMeadCopy = new NelderMeadOptimizerOptions<double, Matrix<double>, Vector<double>>(nelderMead);

        Assert.Equal(nelderMead.Seed, nelderMeadCopy.Seed);
        Assert.Equal(nelderMead.MaxIterations, nelderMeadCopy.MaxIterations);
        Assert.Equal(nelderMead.Tolerance, nelderMeadCopy.Tolerance);
        Assert.Equal(nelderMead.InitialSimplexStep, nelderMeadCopy.InitialSimplexStep);
        Assert.Equal(nelderMead.UseAdaptiveParameters, nelderMeadCopy.UseAdaptiveParameters);
        Assert.Equal(nelderMead.AdaptationRate, nelderMeadCopy.AdaptationRate);
    }

    [Fact(Timeout = 120000)]
    public async Task AugmentedLagrangian_UnconstrainedIterationLimitIsNotReportedOptimal()
    {
        await Task.Yield();

        var solver = new AugmentedLagrangianSolver<double>(
            new AugmentedLagrangianSolverOptions { StationarityTolerance = 1e-8 },
            new FrozenOptimizer());
        var problem = new ConstrainedProblem<double>(
            point => (point[0] * point[0], V(2.0 * point[0])));

        var solution = solver.Solve(problem, V(3.0));

        Assert.Equal(LinearProgramStatus.IterationLimit, solution.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task AugmentedLagrangian_RejectsConstraintJacobianWidthMismatch()
    {
        await Task.Yield();

        var solver = new AugmentedLagrangianSolver<double>();
        var problem = new ConstrainedProblem<double>(
            point => (point[0] * point[0], V(2.0 * point[0])),
            equalityConstraints: point => (V(point[0]), new Matrix<double>(1, 2)));

        var exception = Assert.Throws<ArgumentException>(() => solver.Solve(problem, V(1.0)));

        Assert.Contains("1x1 Jacobian", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task InteriorPoint_LeavesQuadraticVariableUnboundedBelowByDefault()
    {
        await Task.Yield();

        var quadratic = new Matrix<double>(1, 1);
        quadratic[0, 0] = 1.0;
        var program = new QuadraticProgram<double>(quadratic, V(1.0));

        var solution = new InteriorPointSolver<double>().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(-1.0, solution.Solution![0], 5);
    }

    [Fact(Timeout = 120000)]
    public async Task Simplex_UnboundedIntegerTypedProgram_DoesNotMaterializeInfinity()
    {
        await Task.Yield();

        var program = new LinearProgram<int>(Vector<int>.FromArray([-1]));

        var solution = new SimplexSolver<int>().Solve(program);

        Assert.Equal(LinearProgramStatus.Unbounded, solution.Status);
    }

    [Fact(Timeout = 120000)]
    public async Task BranchAndBound_DecimalProgramWithoutUpperBounds_DoesNotConvertInfinity()
    {
        await Task.Yield();

        var relaxation = new LinearProgram<decimal>(Vector<decimal>.FromArray([1m]));
        var program = new IntegerProgram<decimal>(relaxation);

        var solution = new BranchAndBoundSolver<decimal>().Solve(program);

        Assert.Equal(LinearProgramStatus.Optimal, solution.Status);
        Assert.Equal(0m, solution.Solution![0]);
    }

    [Fact(Timeout = 120000)]
    public async Task LbfgsRoundTrip_PreservesSubsequentOptimizationBehavior()
    {
        await Task.Yield();

        Func<Vector<double>, (double objective, Vector<double> gradient)> objective = point =>
            (0.5 * point[0] * point[0] + 25.0 * point[1] * point[1],
                V(point[0], 50.0 * point[1]));

        var optimizer = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction();
        optimizer.Minimize(
            V(4.0, 2.0),
            objective,
            maxIterations: 5,
            tolerance: 1e-12);

        var restored = LBFGSOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction();
        restored.Deserialize(optimizer.Serialize());

        var expected = optimizer.Minimize(V(-3.0, 1.5), objective, 3, 1e-12);
        var actual = restored.Minimize(V(-3.0, 1.5), objective, 3, 1e-12);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 15);
        }
    }

    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static Matrix<double> Column(double[] values)
    {
        var matrix = new Matrix<double>(values.Length, 1);
        for (int i = 0; i < values.Length; i++) matrix[i, 0] = values[i];
        return matrix;
    }

    private sealed class FrozenOptimizer : IFunctionOptimizer<double>
    {
        public Vector<double> Minimize(
            Vector<double> initialParameters,
            Func<Vector<double>, (double objective, Vector<double> gradient)> objectiveAndGradient,
            int maxIterations,
            double tolerance)
            => initialParameters.Clone();
    }
}
