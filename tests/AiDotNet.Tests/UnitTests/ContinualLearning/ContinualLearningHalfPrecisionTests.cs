using System;
using System.Reflection;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Half-precision references for the continual-learning reductions that master computed in double.
/// Half rounds 1e-10 to zero and overflows past 65504, so a floor or a square taken directly in T
/// turns these finite master results into infinities.
/// </summary>
public class ContinualLearningHalfPrecisionTests
{
    private static T InvokeDistillation<T>(string method, double[] teacherLogits, double[] studentLogits, Func<double, T> from)
    {
        var lwf = new LearningWithoutForgetting<T, Tensor<T>, Tensor<T>>(new MeanSquaredErrorLoss<T>(), distillationTemperature: 1.0);
        var info = typeof(LearningWithoutForgetting<T, Tensor<T>, Tensor<T>>)
            .GetMethod(method, BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(info);
        var teacher = new Vector<T>(Array.ConvertAll(teacherLogits, v => from(v)));
        var student = new Vector<T>(Array.ConvertAll(studentLogits, v => from(v)));
        object? result = info?.Invoke(lwf, new object[] { teacher, student });
        Assert.IsType<T>(result);
        return result is T value ? value : from(double.NaN);
    }

    [Fact]
    public void KLDivergence_Half_StudentProbabilityUnderflow_UsesTheFloor()
    {
        // teacher = [0.5, 0.5]; student = [1, e^-100] and e^-100 is 0 in Half.
        // KL = 0.5 (ln 0.5 - ln 1) + 0.5 (ln 0.5 - ln 1e-10) = 10.8198 (master, in double).
        double expected = 0.5 * Math.Log(0.5) + 0.5 * (Math.Log(0.5) - Math.Log(1e-10));
        Half kl = InvokeDistillation<Half>("ComputeKLDivergence", new[] { 0.0, 0.0 }, new[] { 0.0, -100.0 }, v => (Half)v);

        Assert.InRange((double)kl, expected - 0.05, expected + 0.05);
    }

    [Fact]
    public void SoftCrossEntropy_Half_StudentProbabilityUnderflow_UsesTheFloor()
    {
        // -(0.5 ln 1 + 0.5 ln 1e-10) = 11.5129
        double expected = -0.5 * Math.Log(1e-10);
        Half ce = InvokeDistillation<Half>("ComputeSoftCrossEntropy", new[] { 0.0, 0.0 }, new[] { 0.0, -100.0 }, v => (Half)v);

        Assert.InRange((double)ce, expected - 0.05, expected + 0.05);
    }

    [Fact]
    public void KLDivergence_Double_MatchesReference()
    {
        double[] teacherLogits = { 1.0, -0.5, 2.0 };
        double[] studentLogits = { 0.25, 0.75, -1.0 };
        double[] t = Softmax(teacherLogits);
        double[] s = Softmax(studentLogits);
        double expected = 0;
        for (int i = 0; i < t.Length; i++)
        {
            expected += t[i] * Math.Log(t[i] / Math.Max(s[i], 1e-10));
        }

        double kl = InvokeDistillation<double>("ComputeKLDivergence", teacherLogits, studentLogits, v => v);

        Assert.Equal(expected, kl, 10);
    }

    [Fact]
    public void StableL2Norm_Half_DoesNotOverflowOnLargeComponents()
    {
        // 300^2 + 400^2 = 250000 overflows Half; the norm itself, 500, does not.
        var v = new Vector<Half>(new[] { (Half)300.0f, (Half)(-400.0f) });

        Half norm = MemoryAwareSynapses<Half, Tensor<Half>, Tensor<Half>>.StableL2Norm(v);

        Assert.Equal(500.0, (double)norm, 1);
    }

    [Fact]
    public void StableL2Norm_Double_MatchesEuclideanNorm()
    {
        var v = new Vector<double>(new[] { 3.0, -4.0, 12.0 });

        Assert.Equal(13.0, MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>.StableL2Norm(v), 12);
        Assert.Equal(0.0, MemoryAwareSynapses<double, Tensor<double>, Tensor<double>>.StableL2Norm(new Vector<double>(3)), 12);
    }

    [Fact]
    public void ImportanceMeanMetrics_Half_AccumulateBeyondHalfRange()
    {
        // 70000 ones: master's double mean is exactly 1; a Half sum stops growing at 2048.
        var ones = new Vector<Half>(70000);
        ones.Fill((Half)1.0f);
        var loss = new MeanSquaredErrorLoss<Half>();
        object[] strategies =
        {
            new ElasticWeightConsolidation<Half, Tensor<Half>, Tensor<Half>>(loss),
            new ExpectedGradientLength<Half, Tensor<Half>, Tensor<Half>>(loss),
            new MemoryAwareSynapses<Half, Tensor<Half>, Tensor<Half>>(loss),
            new SynapticIntelligence<Half, Tensor<Half>, Tensor<Half>>(loss),
        };

        foreach (var strategy in strategies)
        {
            var method = strategy.GetType().GetMethod("ComputeMean", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(method);
            Assert.Equal(1.0, (double)(method?.Invoke(strategy, new object[] { ones }) ?? double.NaN), 12);
            Assert.True(double.IsNaN((double)(method?.Invoke(strategy, new object[] { new Vector<Half>(0) }) ?? 0.0)),
                $"{strategy.GetType().Name}: the mean of no values is 0 / 0, as in master");
        }
    }

    [Fact]
    public void SynapticIntelligenceLayerStatistics_Half_ChunkMeansBeyondHalfRange()
    {
        // 50000 importances: 10 chunks of 5000. Master's double chunk means are 1, 2, ..., 10; a Half
        // sum of a 5000-element chunk stops growing at 2048 (or loses its low bits).
        var importance = new Vector<Half>(50000);
        for (int chunk = 0; chunk < 10; chunk++)
        {
            for (int i = 0; i < 5000; i++)
            {
                importance[chunk * 5000 + i] = (Half)(chunk + 1.0f);
            }
        }

        var si = new SynapticIntelligence<Half, Tensor<Half>, Tensor<Half>>(new MeanSquaredErrorLoss<Half>());
        var method = si.GetType().GetMethod("ComputeLayerStatistics", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);
        method?.Invoke(si, new object[] { importance });

        Assert.Equal(10, si.LayerImportance.Count);
        for (int chunk = 0; chunk < 10; chunk++)
        {
            Assert.Equal(chunk + 1.0, si.LayerImportance[chunk], 12);
        }
    }

    [Fact]
    public void GemState_TaskGradients_AreWrittenAsDoubleArrays()
    {
        // Master's state format: LoadStateFromSerialization reads each gradient element with GetDouble.
        var gem = new GradientEpisodicMemory<Half, Tensor<Half>, Tensor<Half>>(new MeanSquaredErrorLoss<Half>());
        var type = gem.GetType();
        var gradientsField = type.GetField("_taskGradients", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(gradientsField);
        var gradients = gradientsField?.GetValue(gem) as System.Collections.Generic.List<Vector<Half>>;
        Assert.NotNull(gradients);
        gradients?.Add(new Vector<Half>(new[] { (Half)1.5f, (Half)(-0.25f), (Half)300.0f }));

        var getState = type.GetMethod("GetStateForSerialization", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(getState);
        var state = getState?.Invoke(gem, Array.Empty<object>()) as System.Collections.Generic.Dictionary<string, object>;
        Assert.NotNull(state);

        var written = Assert.IsType<System.Collections.Generic.List<double[]>>(state?["TaskGradients"]);
        Assert.Single(written);
        Assert.Equal(new[] { 1.5, -0.25, 300.0 }, written[0]);
    }

    [Fact]
    public void VariationalUpdateVariance_Half_ZeroGradientKeepsFiniteLogVariance()
    {
        // logVar <- 0.9 logVar + 0.1 log(max(g^2 + 1e-8, 1e-10)). With g = 0, 1e-8 rounds to 0 in Half, so
        // master's floor gives 0.9 (-3) + 0.1 ln 1e-10 = -5.0026 rather than -infinity.
        var vcl = new AiDotNet.ContinualLearning.VariationalContinualLearning<Half>();
        var field = vcl.GetType().GetField("_posteriorLogVar", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(field);
        field?.SetValue(vcl, new Vector<Half>(new[] { (Half)(-3.0f), (Half)(-3.0f) }));

        vcl.UpdateVariance(new Vector<Half>(2), 0.1);

        var logVar = field?.GetValue(vcl) as Vector<Half>;
        Assert.NotNull(logVar);
        double expected = 0.9 * -3.0 + 0.1 * Math.Log(1e-10);
        for (int i = 0; i < 2; i++)
        {
            Assert.InRange((double)(logVar ?? new Vector<Half>(2))[i], expected - 0.02, expected + 0.02);
        }
    }

    [Fact]
    public void LearnerAccuracies_Half_SaveAsJsonNumbersAndRoundTrip()
    {
        // Master wrote accuracies as JSON doubles. Serializing Half directly through Newtonsoft gives an
        // empty object, which then cannot be read back.
        var accuracies = new System.Collections.Generic.Dictionary<int, Half>
        {
            [0] = (Half)0.75f,
            [2] = (Half)0.5f,
        };

        var json = AiDotNet.ContinualLearning.Trainers.ContinualLearnerBase<Half, Tensor<Half>, Tensor<Half>>.AccuraciesToJson(accuracies);

        Assert.Equal(Newtonsoft.Json.Linq.JTokenType.Float, json["0"]?.Type);
        Assert.Equal(0.75, (double?)json["0"] ?? double.NaN, 12);
        Assert.Equal(0.5, (double?)json["2"] ?? double.NaN, 12);

        var restored = new System.Collections.Generic.Dictionary<int, Half>();
        AiDotNet.ContinualLearning.Trainers.ContinualLearnerBase<Half, Tensor<Half>, Tensor<Half>>.AccuraciesFromJson(
            Newtonsoft.Json.Linq.JObject.Parse(json.ToString()), restored);

        Assert.Equal(2, restored.Count);
        Assert.Equal(0.75, (double)restored[0], 12);
        Assert.Equal(0.5, (double)restored[2], 12);
    }

    private static double[] Softmax(double[] logits)
    {
        double max = double.NegativeInfinity;
        foreach (double v in logits)
        {
            max = Math.Max(max, v);
        }

        var result = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = Math.Exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    [Fact]
    public void GemTrainerViolationRate_CountsBeyondHalfRange()
    {
        // 2500 of 3000 batches violate. A mean of 0/1 flags in Half stops counting at 2048.
        var flags = new System.Collections.Generic.List<bool>();
        for (int i = 0; i < 3000; i++)
        {
            flags.Add(i < 2500);
        }

        double rate = AiDotNet.ContinualLearning.Trainers.GEMTrainer<Half, Tensor<Half>, Tensor<Half>>.ViolationRate(flags);

        Assert.Equal(2500.0 / 3000.0, rate, 12);
    }

    [Fact]
    public void GemProjection_Half_LargeTaskGradientGramStaysFinite()
    {
        // Stored task gradient [300, 0], proposed gradient [-1, 0] (margin 0, no normalization).
        // Dual: lambda = (0 - g . g_k) / (g_k . g_k) = 300 / 90000, so g' = g + lambda g_k = [0, 0].
        // In Half the Gram entry 90000 overflows past 65504 and the projection was lost.
        var gem = new GradientEpisodicMemory<Half, Tensor<Half>, Tensor<Half>>(
            new MeanSquaredErrorLoss<Half>(),
            new GEMOptions<Half> { NormalizeGradients = false, Margin = 0.0 });
        gem.StoreTaskGradient(new Vector<Half>(new[] { (Half)300.0f, (Half)0.0f }));

        var projected = gem.ProjectGradient(new Vector<Half>(new[] { (Half)(-1.0f), (Half)0.0f }));

        Assert.Equal(0.0, (double)projected[0], 3);
        Assert.Equal(0.0, (double)projected[1], 3);
    }
}
