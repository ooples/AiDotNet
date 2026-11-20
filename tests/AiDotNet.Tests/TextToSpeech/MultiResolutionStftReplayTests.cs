using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MultiResolutionStftReplayTests
{
    private static readonly int[] Ffts = { 4, 8 }, Hops = { 2, 3 }, Windows = { 4, 6 };

    private static double[] Magnitudes(double[] audio, int fft, int hop, int window)
    {
        int frames = 1 + audio.Length / hop, bins = fft / 2 + 1, offset = (fft - window) / 2;
        var result = new double[frames * bins];
        for (int frame = 0; frame < frames; frame++)
            for (int bin = 0; bin < bins; bin++)
            {
                double real = 0, imaginary = 0;
                for (int n = 0; n < window; n++)
                {
                    int index = frame * hop + offset + n - fft / 2;
                    while (index < 0 || index >= audio.Length)
                        index = index < 0 ? -index : 2 * audio.Length - 2 - index;
                    double weight = .5 - .5 * Math.Cos(2 * Math.PI * n / window);
                    double angle = 2 * Math.PI * (offset + n) * bin / fft;
                    real += audio[index] * (weight * Math.Cos(angle));
                    imaginary += audio[index] * (-weight * Math.Sin(angle));
                }
                result[frame * bins + bin] = Math.Sqrt(Math.Max(real * real + imaginary * imaginary, 1e-7));
            }
        return result;
    }

    private static double Reference(double[][] prediction, double[][] target, bool joint)
    {
        double spectral = 0, logarithmic = 0;
        for (int resolution = 0; resolution < Ffts.Length; resolution++)
        {
            double batchSquaredError = 0, batchEnergy = 0, batchLog = 0;
            int elements = 0;
            for (int sample = 0; sample < prediction.Length; sample++)
            {
                var p = Magnitudes(prediction[sample], Ffts[resolution], Hops[resolution], Windows[resolution]);
                var t = Magnitudes(target[sample], Ffts[resolution], Hops[resolution], Windows[resolution]);
                double squaredError = 0, energy = 0, log = 0;
                for (int i = 0; i < p.Length; i++)
                {
                    squaredError += (t[i] - p[i]) * (t[i] - p[i]);
                    energy += t[i] * t[i];
                    log += Math.Abs(Math.Log(t[i]) - Math.Log(p[i]));
                }
                if (joint)
                {
                    batchSquaredError += squaredError;
                    batchEnergy += energy;
                    batchLog += log;
                    elements += p.Length;
                }
                else
                {
                    spectral += Math.Sqrt(squaredError) / Math.Max(Math.Sqrt(energy), 1e-12) / prediction.Length;
                    logarithmic += log / p.Length / prediction.Length;
                }
            }
            if (joint)
            {
                spectral += Math.Sqrt(batchSquaredError) / Math.Max(Math.Sqrt(batchEnergy), 1e-12);
                logarithmic += batchLog / elements;
            }
        }
        return (spectral + .37 * logarithmic) / Ffts.Length;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Replayed_loss_matches_independent_DFT_and_generated_gradients_while_target_gradients_stop(bool joint)
    {
        var engine = new CpuEngine();
        var loss = new MultiResolutionStftLoss<double>(engine, Ffts, Hops, Windows);
        var prediction = Enumerable.Range(0, 2).Select(_ => new Tensor<double>(new[] { 12 })).ToArray();
        var target = Enumerable.Range(0, 2).Select(_ => new Tensor<double>(new[] { 12 })).ToArray();
        var parameters = prediction.Concat(target).ToArray();
        void SetData(int step)
        {
            for (int sample = 0; sample < 2; sample++)
                for (int i = 0; i < 12; i++)
                {
                    prediction[sample].SetFlat(i, .1 + .4 * Math.Sin(i * .41 + step * .29 + sample));
                    target[sample].SetFlat(i, -.05 + .6 * Math.Cos(i * .37 - step * .43 + sample * .6));
                }
            foreach (var parameter in parameters) parameter.IncrementVersion();
        }
        Tensor<double> Objective()
        {
            var terms = joint ? loss.ForwardJointBatch(prediction, target) : loss.Forward(prediction, target);
            return engine.TensorAdd(terms.SpectralConvergence, engine.TensorMultiplyScalar(terms.LogMagnitude, .37));
        }
        SetData(0);
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 3; step++)
            {
                SetData(step + 1);
                var p = prediction.Select(value => value.ToArray()).ToArray();
                var t = target.Select(value => value.ToArray()).ToArray();
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                double actualLoss = plan.Step().ToArray()[0];
                Assert.True(Math.Abs(actualLoss - Reference(p, t, joint)) < 1e-10);
                for (int sample = 0; sample < 2; sample++)
                {
                    var gradient = plan.Gradients[sample].ToArray();
                    for (int i = 0; i < p[sample].Length; i++)
                    {
                        const double delta = 1e-6;
                        double value = p[sample][i];
                        p[sample][i] = value + delta;
                        double plus = Reference(p, t, joint);
                        p[sample][i] = value - delta;
                        double minus = Reference(p, t, joint);
                        p[sample][i] = value;
                        double expected = (plus - minus) / (2 * delta);
                        Assert.True(Math.Abs(gradient[i] - expected) < 2e-5 * Math.Max(1, Math.Abs(expected)),
                            $"joint={joint}, step={step}, sample={sample}, index={i}: {gradient[i]:R} vs {expected:R}");
                    }
                    Assert.All(plan.Gradients[2 + sample].ToArray(), value => Assert.Equal(0d, value));
                    Assert.Equal(p[sample], prediction[sample].ToArray());
                    Assert.Equal(t[sample], target[sample].ToArray());
                }
            }
        }
    }

    [Fact]
    public void Target_inverse_norm_rebinds_fresh_float_leaves_and_widens_before_squaring_and_summing()
    {
        var engine = new CpuEngine();
        var target = new Tensor<float>(Enumerable.Repeat(8e18f, 32).ToArray(), new[] { 32 });
        var weight = new Tensor<float>(new[] { .7f }, new[] { 1 });
        var parameters = new[] { weight };
        Tensor<float> Objective() => engine.TensorMultiply(weight,
            MultiResolutionStftLoss<float>.InverseTargetNorm(engine, new[] { target }));
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 3; step++)
            {
                target = new Tensor<float>(Enumerable.Repeat((8 + step) * 1e18f, 32).ToArray(), new[] { 32 });
                double value = target.ToArray()[0];
                float expected = (float)(1 / Math.Sqrt(32 * value * value));
                Assert.True(expected > 0 && !float.IsInfinity(expected));
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                Assert.Equal(.7f * expected, plan.Step().ToArray()[0]);
                Assert.Equal(expected, plan.Gradients[0].ToArray()[0]);
            }
        }
#if NET5_0_OR_GREATER
        var halfTargets = new Tensor<Half>(Enumerable.Repeat((Half)200, 32).ToArray(), new[] { 32 });
        var halfInverse = MultiResolutionStftLoss<Half>.InverseTargetNorm(engine, new[] { halfTargets });
        Assert.Equal((Half)(1 / Math.Sqrt(32 * 200d * 200d)), halfInverse.ToArray()[0]);
#endif
    }
}
