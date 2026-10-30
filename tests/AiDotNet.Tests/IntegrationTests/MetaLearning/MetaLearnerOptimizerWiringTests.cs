using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// The MetaOptimizer and InnerOptimizer a meta-learner is given take its outer and inner steps; without them the
/// steps stay plain SGD. An optimizer for a loop the algorithm has no gradient step in is rejected (#2397).
/// </summary>
public class MetaLearnerOptimizerWiringTests
{
    private sealed class Probe : MetaLearnerBase<double, Matrix<double>, Vector<double>>
    {
        public Probe(
            IGradientBasedOptimizer<double, Matrix<double>, Vector<double>>? metaOptimizer,
            IGradientBasedOptimizer<double, Matrix<double>, Vector<double>>? innerOptimizer,
            MetaLearnerOptimizerUse use = MetaLearnerOptimizerUse.InnerAndOuter)
            : base(new LinearVectorModel(3), new MeanSquaredErrorLoss<double>(), new MetaLearnerOptionsBase<double>(),
                null, metaOptimizer, innerOptimizer, use)
        {
        }

        public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MAML;

        public override double MetaTrain(TaskBatch<double, Matrix<double>, Vector<double>> taskBatch) =>
            throw new NotSupportedException();

        public override IModel<Matrix<double>, Vector<double>, ModelMetadata<double>> Adapt(
            IMetaLearningTask<double, Matrix<double>, Vector<double>> task) => throw new NotSupportedException();

        public Vector<double> Outer(Vector<double> p, Vector<double> g, double lr) => ApplyOuterGradients(p, g, lr);

        public Vector<double>[] Outer(double lr, params (Vector<double>, Vector<double>)[] groups) =>
            ApplyOuterGradients(lr, groups);

        public Vector<double> Inner(Vector<double> p, Vector<double> g, double lr) => ApplyInnerGradients(p, g, lr);
    }

    private static AdamOptimizer<double, Matrix<double>, Vector<double>> Adam(double lr) =>
        new(null, new AdamOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = lr });

    private static Vector<double> V(params double[] values) => new(values);

    [Fact]
    public void WithoutOptimizers_StepsArePlainSgd()
    {
        var probe = new Probe(null, null);

        var outer = probe.Outer(V(1, 1), V(2, -4), 0.5);
        var inner = probe.Inner(V(1, 1), V(2, -4), 0.25);

        Assert.Equal(new[] { 0.0, 3.0 }, outer.ToArray());
        Assert.Equal(new[] { 0.5, 2.0 }, inner.ToArray());
    }

    [Fact]
    public void MetaOptimizer_TakesTheOuterStep_AtItsOwnRate()
    {
        var probe = new Probe(Adam(0.1), null);

        // Adam's first step is lr * g / (|g| + eps): 0.1 per coordinate whatever the gradient's scale. SGD at the
        // passed rate 0.5 would move these by 1.0, -2.0 and 0.0005.
        var updated = probe.Outer(V(1, 1, 1), V(2, -4, 0.001), 0.5);

        Assert.Equal(0.9, updated[0], 6);
        Assert.Equal(1.1, updated[1], 6);
        Assert.Equal(0.9, updated[2], 4);
    }

    [Fact]
    public void GroupedOuterStep_IsOneOptimizerStepOverTheConcatenation()
    {
        var grouped = new Probe(Adam(0.1), null);
        var whole = new Probe(Adam(0.1), null);
        var body = V(1, 2);
        var head = V(3);
        var empty = new Vector<double>(0);

        // Two steps, so Adam's moments matter: a grouped step must carry the same state as one step over [body, head].
        var g1 = grouped.Outer(0.5, (body, V(1, -2)), (empty, empty), (head, V(0.5)));
        var g2 = grouped.Outer(0.5, (g1[0], V(-3, 1)), (g1[1], empty), (g1[2], V(2)));
        var w1 = whole.Outer(V(1, 2, 3), V(1, -2, 0.5), 0.5);
        var w2 = whole.Outer(w1, V(-3, 1, 2), 0.5);

        Assert.Empty(g2[1].ToArray());
        Assert.Equal(new[] { w2[0], w2[1] }, g2[0].ToArray());
        Assert.Equal(new[] { w2[2] }, g2[2].ToArray());
    }

    [Fact]
    public void InnerOptimizer_StateBelongsToOneAdaptation()
    {
        var probe = new Probe(null, Adam(0.1));
        var start = V(0.0);

        // Adaptation 1 steps on a large gradient; adaptation 2 starts again from the meta-parameters with a small
        // one. A reset makes its first step Adam's first step again, 0.1; carried moments would shrink it to ~0.068.
        var a1 = probe.Inner(start, V(1.0), 0.5);
        probe.Inner(a1, V(1.0), 0.5);
        var b1 = probe.Inner(start, V(0.01), 0.5);

        Assert.Equal(-0.1, b1[0], 6);
    }

    [Fact]
    public void InnerOptimizer_KeepsItsStateWithinAnAdaptation()
    {
        var probe = new Probe(null, Adam(0.1));

        var a1 = probe.Inner(V(0.0), V(1.0), 0.5);
        var a2 = probe.Inner(a1, V(0.01), 0.5);

        // Continuing from the vector it returned keeps the moments, so this step is not a fresh 0.1.
        Assert.InRange(a1[0] - a2[0], 0.05, 0.09);
    }

    [Fact]
    public void AnOptimizerForALoopTheAlgorithmDoesNotRun_IsRejected()
    {
        Assert.Throws<ArgumentException>(() => new Probe(null, Adam(0.1), MetaLearnerOptimizerUse.Outer));
        Assert.Throws<ArgumentException>(() => new Probe(Adam(0.1), null, MetaLearnerOptimizerUse.Inner));
        Assert.Throws<ArgumentException>(() => new Probe(Adam(0.1), null, MetaLearnerOptimizerUse.None));
        _ = new Probe(Adam(0.1), null, MetaLearnerOptimizerUse.Outer);
    }
}
