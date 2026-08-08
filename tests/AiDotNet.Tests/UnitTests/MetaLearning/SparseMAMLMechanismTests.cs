using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Verifies that <see cref="SparseMAMLAlgorithm{T, TInput, TOutput}"/> implements von Oswald et al.,
/// "Learning where to learn: Gradient sparsity in meta and continual learning" (arXiv:2110.14402).
/// </summary>
/// <remarks>
/// The method is defined by WHERE the gate comes from. This class previously derived a mask from
/// per-parameter gradient-magnitude z-scores against an EMA — a fixed rule with nothing meta-learned,
/// under which sparsity cannot emerge because it is imposed. These tests pin the three properties
/// that distinguish a learned gate from a rule: it starts open, it is judged by the query loss, and
/// it multiplies the update rather than the parameter.
/// </remarks>
public class SparseMAMLMechanismTests
{
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task(int seed)
    {
        var rng = new Random(seed);
        var sx = new Matrix<double>(4, 3);
        var qx = new Matrix<double>(4, 3);
        var sy = new Vector<double>(4);
        var qy = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++) { sx[i, j] = rng.NextDouble() - 0.5; qx[i, j] = rng.NextDouble() - 0.5; }
            sy[i] = i % 2; qy[i] = i % 2;
        }
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = sx, SupportSetY = sy, QuerySetX = qx, QuerySetY = qy,
            NumWays = 2, NumShots = 2, NumQueryPerClass = 2,
        };
    }

    private static SparseMAMLAlgorithm<double, Matrix<double>, Vector<double>> Algorithm(
        Action<SparseMAMLOptions<double, Matrix<double>, Vector<double>>>? tweak = null)
    {
        var options = new SparseMAMLOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
        {
            InnerLearningRate = 0.05,
            OuterLearningRate = 0.01,
            AdaptationSteps = 2,
            Seed = 23,
        };
        tweak?.Invoke(options);
        return new SparseMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    [Fact]
    public void EveryGateStartsOpen_SoSparsityMustBeDiscovered()
    {
        // "Patterned sparsity EMERGES from this process." Starting from an already-sparse mask would
        // beg the question the paper is answering.
        var algorithm = Algorithm();

        for (int d = 0; d < algorithm.GateLogits.Length; d++)
        {
            Assert.True(algorithm.Gate(d) > 0.5, $"Gate {d} did not start open.");
        }
        Assert.Equal(0.0, algorithm.Sparsity, 12);
    }

    [Fact]
    public void GateIsASigmoidOfItsLogit_AndStaysInTheUnitInterval()
    {
        // A gate outside (0, 1) would either invert or amplify the update rather than gate it.
        foreach (double logit in new[] { -20.0, -1.0, 0.0, 1.0, 20.0 })
        {
            var algorithm = Algorithm(o => o.InitialGateLogit = logit);
            double expected = 1.0 / (1.0 + Math.Exp(-logit));
            Assert.Equal(expected, algorithm.Gate(0), 9);
            Assert.InRange(algorithm.Gate(0), 0.0, 1.0);
        }
    }

    [Fact]
    public void GateIsLearned_NotDerivedFromGradientMagnitudes()
    {
        // The decisive difference from the z-score rule this replaced. Under a rule the mask is a
        // fixed function of the current gradients and carries no state across meta-iterations; a
        // learned gate moves because the META-objective moved it.
        var algorithm = Algorithm();
        var before = algorithm.GateLogits.Clone();

        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            new[] { Task(1), Task(2), Task(3) });
        algorithm.MetaTrain(batch);

        bool moved = Enumerable.Range(0, before.Length)
            .Any(d => Math.Abs(before[d] - algorithm.GateLogits[d]) > 1e-12);
        Assert.True(moved, "The gate logits never moved, so nothing was meta-learned.");
    }

    [Fact]
    public void ClosedGateFreezesItsParameter()
    {
        // The gate multiplies the UPDATE. A parameter whose gate is shut must come out of adaptation
        // exactly as it went in, whatever its gradient was.
        var algorithm = Algorithm(o => o.InitialGateLogit = -40.0);   // sigmoid ~ 0
        var model = new LinearVectorModel(3);
        var before = model.GetParameters();

        var adapted = algorithm.Adapt(Task(7));
        var after = AdaptedParams(adapted);

        for (int d = 0; d < Math.Min(before.Length, after.Length); d++)
        {
            Assert.Equal(before[d], after[d], 9);
        }
    }

    [Fact]
    public void OpenGateLetsAdaptationThrough()
    {
        // The complement of the previous test: with gates open the same task must move parameters,
        // so the freeze above is attributable to the gate and not to a dead inner loop.
        var algorithm = Algorithm(o => o.InitialGateLogit = 40.0);
        var task = Task(7);

        var before = AdaptedParams(algorithm.Adapt(task));

        var closed = Algorithm(o => o.InitialGateLogit = -40.0);
        var frozen = AdaptedParams(closed.Adapt(task));

        bool differ = Enumerable.Range(0, Math.Min(before.Length, frozen.Length))
            .Any(d => Math.Abs(before[d] - frozen[d]) > 1e-9);
        Assert.True(differ, "Open and closed gates produced the same adaptation.");
    }

    [Fact]
    public void SparsityReportsTheFractionGatedOff()
    {
        Assert.Equal(0.0, Algorithm(o => o.InitialGateLogit = 5.0).Sparsity, 12);
        Assert.Equal(1.0, Algorithm(o => o.InitialGateLogit = -5.0).Sparsity, 12);
    }

    [Fact]
    public void GateMultipliesTheUpdate_NotTheParameter()
    {
        // Scaling the parameter would change the function the model computes. Scaling the update
        // changes only where learning may happen — which is what "learning where to learn" means.
        // A half-open gate must therefore leave the parameters between the frozen and fully-open
        // outcomes, not shrink them toward zero.
        var task = Task(4);
        var open = AdaptedParams(Algorithm(o => o.InitialGateLogit = 40.0).Adapt(task));
        var half = AdaptedParams(Algorithm(o => o.InitialGateLogit = 0.0).Adapt(task));
        var shut = AdaptedParams(Algorithm(o => o.InitialGateLogit = -40.0).Adapt(task));

        for (int d = 0; d < open.Length; d++)
        {
            double lo = Math.Min(open[d], shut[d]);
            double hi = Math.Max(open[d], shut[d]);
            Assert.InRange(half[d], lo - 1e-9, hi + 1e-9);
        }
    }

    [Fact]
    public void PerParameterRateVariantIsOptional()
    {
        // The paper's "more expressive model where learning rates are meta-learned"; off by default,
        // which is its sparse-MAML variant.
        Assert.False(new SparseMAMLOptions<double, Matrix<double>, Vector<double>>(
            new LinearVectorModel(3)).MetaLearnPerParameterRates);

        var expressive = Algorithm(o => o.MetaLearnPerParameterRates = true);
        var loss = expressive.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(
            new[] { Task(5), Task(6) }));
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
    }

    [Fact]
    public void MetaTrainProducesFiniteLossAndAdaptsAfterwards()
    {
        var algorithm = Algorithm();
        var task = Task(11);
        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(
            new[] { task, Task(12) }));

        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
        Assert.NotNull(algorithm.Adapt(task).Predict(task.QuerySetX));
    }

    [Fact]
    public void AlgorithmTypeIsSparseMAML()
    {
        Assert.Equal(AiDotNet.MetaLearning.MetaLearningAlgorithmType.SparseMAML, Algorithm().AlgorithmType);
    }

    /// <summary>
    /// Reads the adapted parameter vector. Adapt returns IModel, which carries no parameter
    /// contract, so the concrete adapted model is what exposes them.
    /// </summary>
    private static Vector<double> AdaptedParams(
        AiDotNet.Interfaces.IModel<Matrix<double>, Vector<double>, AiDotNet.Models.ModelMetadata<double>> adapted)
        => ((AdaptedMetaModel<double, Matrix<double>, Vector<double>>)adapted).GetParameters();
}
