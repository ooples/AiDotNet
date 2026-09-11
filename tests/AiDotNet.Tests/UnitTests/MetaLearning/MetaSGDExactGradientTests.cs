using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Meta-SGD as Li et al. 2017 define it: theta and the per-parameter coefficients are trained together on the
/// exact gradient of the query loss (#2155).
/// </summary>
/// <remarks>
/// Meta-SGD used to leave theta untouched, estimate alpha's gradient with a one-sided finite difference, and train
/// momentum, direction and the Adam betas on scaled copies of alpha's gradient. The gradient checks here compare
/// every reported derivative against a central difference of the episode's query loss.
/// </remarks>
public class MetaSGDExactGradientTests
{
    private const int Features = 3;
    private const int SupportRows = 5;
    private const int QueryRows = 4;

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var weights = Enumerable.Range(0, Features).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray();
        double bias = rng.NextDouble() - 0.5;

        (Matrix<double> X, Vector<double> Y) Rows(int count)
        {
            var x = new Matrix<double>(count, Features);
            var y = new Vector<double>(count);
            for (int r = 0; r < count; r++)
            {
                double target = bias;
                for (int c = 0; c < Features; c++)
                {
                    x[r, c] = rng.NextDouble() * 2.0 - 1.0;
                    target += weights[c] * x[r, c];
                }

                y[r] = target;
            }

            return (x, y);
        }

        var support = Rows(SupportRows);
        var query = Rows(QueryRows);
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = support.X,
            SupportSetY = support.Y,
            QuerySetX = query.X,
            QuerySetY = query.Y,
            NumWays = 1,
            NumShots = SupportRows,
            NumQueryPerClass = QueryRows,
            Name = $"regression-{seed}",
        };
    }

    private static Vector<double> Fill(int length, Func<int, double> value)
    {
        var vector = new Vector<double>(length);
        for (int i = 0; i < length; i++) vector[i] = value(i);
        return vector;
    }

    private static MetaSGDAlgorithm<double, Matrix<double>, Vector<double>> CreateLearner(
        Action<MetaSGDOptions<double, Matrix<double>, Vector<double>>>? configure = null)
    {
        var options = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(Features));
        configure?.Invoke(options);
        var learner = new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(options);

        // Distinct coefficients per parameter, so a gradient routed to the wrong entry cannot pass.
        var optimizer = learner.LearnedOptimizer;
        int n = optimizer.NumParameters;
        optimizer.LearningRates = Fill(n, i => 0.05 + 0.02 * i);
        optimizer.Momentums = Fill(n, i => 0.3 + 0.05 * i);
        optimizer.Directions = Fill(n, i => 1.0 - 0.1 * i);
        return learner;
    }

    /// <summary>The coefficient entries in the order the episode gradient reports them.</summary>
    private static List<(string Name, Func<Vector<double>> Get, Action<Vector<double>> Set, int Index)> CoefficientEntries(
        MetaSGDAlgorithm<double, Matrix<double>, Vector<double>> learner,
        MetaSGDOptions<double, Matrix<double>, Vector<double>> options)
    {
        var o = learner.LearnedOptimizer;
        int n = o.NumParameters;
        var entries = new List<(string, Func<Vector<double>>, Action<Vector<double>>, int)>();
        if (options.LearnLearningRate)
            for (int i = 0; i < n; i++) entries.Add(("alpha", () => o.LearningRates, v => o.LearningRates = v, i));
        if (options.LearnMomentum)
            for (int i = 0; i < n; i++) entries.Add(("momentum", () => o.Momentums, v => o.Momentums = v, i));
        if (options.LearnDirection)
            for (int i = 0; i < n; i++) entries.Add(("direction", () => o.Directions, v => o.Directions = v, i));
        if (options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && options.LearnAdamBetas)
        {
            for (int i = 0; i < n; i++)
            {
                entries.Add(("beta1", () => o.AdamBeta1, v => o.AdamBeta1 = v, i));
                entries.Add(("beta2", () => o.AdamBeta2, v => o.AdamBeta2 = v, i));
                entries.Add(("epsilon", () => o.AdamEpsilon, v => o.AdamEpsilon = v, i));
            }
        }

        return entries;
    }

    private static double CentralDifference(Func<double> loss, Action<double> shift, double h)
    {
        shift(h);
        double plus = loss();
        shift(-2 * h);
        double minus = loss();
        shift(h);
        return (plus - minus) / (2 * h);
    }

    private static void AssertClose(double analytic, double numeric, string what)
        => Assert.True(Math.Abs(analytic - numeric) <= 1e-6 + 1e-4 * Math.Abs(numeric),
            $"{what}: analytic {analytic:G10} vs central difference {numeric:G10}.");

    [Theory]
    [InlineData(MetaSGDUpdateRuleType.SGD, false, false)]
    [InlineData(MetaSGDUpdateRuleType.SGD, true, false)]
    [InlineData(MetaSGDUpdateRuleType.SGD, false, true)]
    [InlineData(MetaSGDUpdateRuleType.SGDWithMomentum, false, false)]
    [InlineData(MetaSGDUpdateRuleType.Adam, false, false)]
    [InlineData(MetaSGDUpdateRuleType.RMSprop, false, false)]
    [InlineData(MetaSGDUpdateRuleType.AdaGrad, false, false)]
    [InlineData(MetaSGDUpdateRuleType.AdaDelta, false, false)]
    public void EpisodeGradient_MatchesCentralDifferences(MetaSGDUpdateRuleType rule, bool learnDirection, bool clip)
    {
        MetaSGDOptions<double, Matrix<double>, Vector<double>>? options = null;
        var learner = CreateLearner(o =>
        {
            o.UpdateRuleType = rule;
            o.InnerSteps = 2; // two steps, so optimizer state carries from one step into the next
            o.LearnDirection = learnDirection;
            o.LearnMomentum = rule == MetaSGDUpdateRuleType.SGDWithMomentum;
            o.LearnAdamBetas = rule == MetaSGDUpdateRuleType.Adam;
            o.UseFirstOrder = false;
            o.GradientClipThreshold = clip ? 0.05 : null;
            options = o;
        });
        Assert.NotNull(options);
        var task = CreateTask(seed: 11);

        var (_, thetaGradient, coefficientGradient) = learner.EpisodeGradientForTesting(task);

        var model = learner.GetMetaModel();
        double Loss() => learner.EpisodeLossForTesting(task);
        var theta = model.GetParameters();
        Assert.Equal(theta.Length, thetaGradient.Length);
        for (int i = 0; i < theta.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(Loss, h =>
            {
                var shifted = model.GetParameters();
                shifted[index] += h;
                model.SetParameters(shifted);
            }, 1e-6);
            AssertClose(thetaGradient[i], numeric, $"{rule}: dL/dtheta[{i}]");
        }

        var entries = CoefficientEntries(learner, options ?? throw new InvalidOperationException());
        Assert.Equal(entries.Count, coefficientGradient.Length);
        for (int k = 0; k < entries.Count; k++)
        {
            var (name, get, set, index) = entries[k];
            double numeric = CentralDifference(Loss, h =>
            {
                var shifted = get();
                shifted[index] += h;
                set(shifted);
            }, 1e-6);
            AssertClose(coefficientGradient[k], numeric, $"{rule}: dL/d{name}[{index}]");
        }
    }

    [Fact]
    public void FirstOrder_DropsOnlyTheHessianTerm()
    {
        // First-order: dL/dtheta is the query gradient at the adapted parameters. The coefficient gradients stay
        // exact, since the Hessian enters them only through theta's path.
        var exact = CreateLearner(o => o.UseFirstOrder = false);
        var firstOrder = CreateLearner(o => o.UseFirstOrder = true);
        var task = CreateTask(seed: 12);

        var (_, exactTheta, exactCoefficients) = exact.EpisodeGradientForTesting(task);
        var (_, firstTheta, firstCoefficients) = firstOrder.EpisodeGradientForTesting(task);

        Assert.Equal(exactCoefficients, firstCoefficients);
        Assert.True(exactTheta.Zip(firstTheta, (a, b) => Math.Abs(a - b)).Max() > 1e-8,
            "The exact theta gradient carries the support Hessian; the first-order one does not.");
    }

    [Fact]
    public void MetaTrain_UpdatesThetaAndAlphaTogether()
    {
        var learner = CreateLearner();
        var thetaBefore = learner.GetMetaModel().GetParameters().ToArray();
        var alphaBefore = learner.LearnedOptimizer.LearningRates.ToArray();

        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { CreateTask(1), CreateTask(2) }));

        Assert.NotEqual(thetaBefore, learner.GetMetaModel().GetParameters().ToArray());
        Assert.NotEqual(alphaBefore, learner.LearnedOptimizer.LearningRates.ToArray());
    }

    [Fact]
    public void MetaTrain_DescendsTheMetaObjective()
    {
        var learner = CreateLearner(o => o.OuterLearningRate = 0.05);
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            Enumerable.Range(0, 4).Select(i => CreateTask(100 + i)).ToArray());

        double first = learner.MetaTrain(batch);
        double last = first;
        for (int step = 0; step < 40; step++) last = learner.MetaTrain(batch);

        Assert.True(last < first, $"Meta-training on a fixed batch went from {first} to {last}.");
    }

    [Fact]
    public void Adapt_IsOneSgdStepWithTheLearnedRates()
    {
        // Li et al. 2017, eq. 2: theta' = theta - alpha o grad L_train(theta).
        var learner = CreateLearner(o => o.UpdateRuleType = MetaSGDUpdateRuleType.SGD);
        var task = CreateTask(seed: 21);
        var theta = learner.GetMetaModel().GetParameters();
        var alpha = learner.LearnedOptimizer.LearningRates;

        var reference = new LinearVectorModel(Features);
        reference.SetParameters(theta);
        var gradient = reference.ComputeGradients(task.SupportSetX, task.SupportSetY, new MeanSquaredErrorLoss<double>());
        reference.SetParameters(Fill(theta.Length, i => theta[i] - alpha[i] * gradient[i]));
        var expected = reference.Predict(task.QuerySetX);

        var actual = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    [Fact]
    public void Alpha_IsUnboundedByDefault_SoItsSignCanFlip()
    {
        // Alpha "decides both the update direction and learning rate" (Li et al. 2017), so a negative entry is
        // legitimate. The old default floor of 1e-6 made it impossible.
        var learner = CreateLearner(o => o.OuterLearningRate = 0.01);
        var optimizer = learner.LearnedOptimizer;
        optimizer.LearningRates = Fill(optimizer.NumParameters, _ => 0.001);

        optimizer.UpdateMetaParameters(Fill(optimizer.GetMetaParameterCount(), _ => 1.0));

        Assert.All(optimizer.LearningRates.ToArray(), a => Assert.Equal(-0.009, a, 12));
    }

    [Fact]
    public void Alpha_IsClampedOnlyToBoundsTheCallerSets()
    {
        var learner = CreateLearner(o =>
        {
            o.OuterLearningRate = 0.01;
            o.MinLearningRate = 1e-6;
        });
        var optimizer = learner.LearnedOptimizer;
        optimizer.LearningRates = Fill(optimizer.NumParameters, _ => 0.001);

        optimizer.UpdateMetaParameters(Fill(optimizer.GetMetaParameterCount(), _ => 1.0));

        Assert.All(optimizer.LearningRates.ToArray(), a => Assert.Equal(1e-6, a, 15));
    }

    [Fact]
    public void Defaults_AreThePapersSetting()
    {
        var options = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(Features));

        Assert.Equal(1, options.InnerSteps);
        Assert.Equal(1, options.AdaptationSteps);
        Assert.False(options.UseFirstOrder);
        Assert.False(options.LearnDirection);
        Assert.Null(options.MinLearningRate);
        Assert.Null(options.MaxLearningRate);
        Assert.True(options.IsValid());
    }

    [Fact]
    public void Serialize_CarriesTheLearnedCoefficients()
    {
        var trained = CreateLearner();
        trained.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { CreateTask(31), CreateTask(32) }));
        var restored = new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(
            new MetaSGDOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(Features)));

        using (ModelPersistenceGuard.InternalOperation())
        {
            restored.Deserialize(trained.Serialize());
        }

        Assert.Equal(trained.LearnedOptimizer.LearningRates.ToArray(), restored.LearnedOptimizer.LearningRates.ToArray());
        Assert.Equal(trained.GetMetaModel().GetParameters().ToArray(), restored.GetMetaModel().GetParameters().ToArray());
    }
}
