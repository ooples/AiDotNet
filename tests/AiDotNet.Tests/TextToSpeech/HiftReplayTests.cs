using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class HiftReplayTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private sealed class CountingRandom(int seed) : Random(seed)
    {
        internal int Calls { get; private set; }
        public override double NextDouble() { Calls++; return base.NextDouble(); }
    }

    private static HiftGenerator<T> Build<T>(IEngine engine, bool byFrames)
        => new(engine, new List<LayerBase<T>>(), new HiftConfiguration(
            InChannels: 2, BaseChannels: 8, Harmonics: 1, SampleRate: byFrames ? 24000 : 22050,
            NsfAlpha: 0.1, NsfSigma: 0.003, VoicedThreshold: 0,
            UpsampleRates: new[] { 2 }, UpsampleKernels: new[] { 4 }, FftSize: 4, HopSize: 1,
            ResblockKernels: new[] { 3 }, ResblockDilations: new[] { new[] { 1 } },
            SourceKernels: new[] { 3 }, SourceDilations: new[] { new[] { 1 } },
            LeakySlope: 0.1, AudioLimit: 0.99, F0Channels: 2), new Random(7));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Deterministic_harmonics_replay_current_derived_F0_without_a_derivative(bool byFrames)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var model = Build<double>(engine, byFrames);
        var p = new Tensor<double>(new[] { 2.0 }, new[] { 1 });
        var f0 = new Tensor<double>(new[] { 0.0, 110.0, 220.0 }, new[] { 3 });
        var phases = new[] { 0.123, 0.37 };
        var noise = new double[2, 6];
        for (int h = 0; h < 2; h++) for (int s = 0; s < 6; s++) noise[h, s] = (h * 6 + s - 5) / 7.0;
        var parameters = new[] { p, f0 };
        Tensor<double> Generate(Tensor<double> input) => byFrames
            ? model.HarmonicsByFrames(input, phases, noise) : model.Harmonics(input, phases, noise);
        Tensor<double> harmonics;
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            harmonics = Generate(engine.TensorAddScalar(f0, 0.0));
            plan = scope.CompileTraining(parameters, engine.TensorMultiply(p, engine.ReduceSum(harmonics, null, false)));
            Assert.Null(scope.HostReadOp);
        }
        using (plan)
        {
            Assert.True(plan.HasForwardOnlyHeterogeneousReplay);
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0, weightDecay: 0);
            double[][] contours = { new[] { 0.0, 110.0, 220.0 }, new[] { 220.0, 0.0, 55.0 }, new[] { 0.0, 440.0, 330.0 } };
            for (int step = 0; step < contours.Length; step++)
            {
                for (int i = 0; i < 3; i++) f0[i] = contours[step][i];
                var expected = byFrames ? OriginalHarmonicsByFrames(model, f0, phases, noise)
                    : OriginalHarmonics(model, f0, phases, noise);
                Assert.Equal(expected.ToArray(), Generate(f0).ToArray());
                double sum = engine.ReduceSum(expected, null, false).ToArray()[0];
                Assert.Equal(2 * sum, plan.Step().ToArray()[0]);
                Assert.Equal(expected.ToArray(), harmonics.ToArray());
                Assert.Equal(sum, plan.Gradients[0].ToArray()[0]);
                Assert.Equal(new long[] { step + 1, 0 }, plan.ParameterAcceptedUpdates);
            }
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Float_and_Half_sources_replay_exact_double_draws_without_probe_draws(bool half, bool byFrames)
    {
        if (half) CheckSource<Half>(byFrames);
        else CheckSource<float>(byFrames);
    }

    private static void CheckSource<T>(bool byFrames)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var model = Build<T>(engine, byFrames);
        var eager = Build<T>(engine, byFrames);
        eager.SourceMerge.SetParameters(model.SourceMerge.GetParameters());
        var ops = MathHelper.GetNumericOperations<T>();
        var p = new Tensor<T>(new[] { ops.FromDouble(2) }, new[] { 1 });
        var f0 = new Tensor<T>(new[] { ops.Zero, ops.FromDouble(110), ops.FromDouble(220) }, new[] { 3 });
        var mergeParameters = TapeTrainingStep<T>.CollectParameters(new[] { model.SourceMerge }, -1);
        Assert.Equal(2, mergeParameters.Count);
        var parameters = new[] { p, f0 }.Concat(mergeParameters).ToArray();
        var random = new CountingRandom(17);
        var referenceRandom = new CountingRandom(17);
        Tensor<T>? currentSource = null;
        Tensor<T> Objective()
        {
            var source = model.Source(engine.TensorAddScalar(f0, ops.Zero), random);
            currentSource = source;
            return engine.TensorMultiply(p, engine.ReduceSum(source, null, false));
        }
        using var recording = ClosureObjectiveGuard<T>.BeginRecording();
        using var draws = CompiledStepRandom<T>.BeginRecording();
        CompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            plan = scope.CompileTraining(parameters, Objective());
            Assert.Null(scope.HostReadOp);
        }
        using (plan)
        {
            var tracedSource = Assert.IsType<Tensor<T>>(currentSource);
            var guard = ClosureObjectiveGuard<T>.TakeRecorded(out var reason);
            if (guard is null) throw new InvalidOperationException(reason);
            guard.AttachTo(plan);
            CompiledStepRandom<T>.Attach(plan, draws);
            Assert.Equal(1, CompiledStepRandom<T>.DrawCount(plan));
            Assert.True(plan.HasForwardOnlyHeterogeneousReplay);
            Assert.False(plan.IsGraphStepEligible);
            if (typeof(T) == typeof(float)) plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0, weightDecay: 0);
            double[][] contours = { new[] { 0.0, 110.0, 220.0 }, new[] { 220.0, 0.0, 55.0 }, new[] { 0.0, 440.0, 330.0 } };
            for (int step = 0; step < contours.Length; step++)
            {
                for (int i = 0; i < 3; i++) f0[i] = ops.FromDouble(contours[step][i]);
                if (step > 0)
                {
                    int beforeProbe = random.Calls;
                    using (CompiledStepRandom<T>.SkipDraws())
                        Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                    Assert.Equal(beforeProbe, random.Calls);
                    CompiledStepRandom<T>.RedrawFor(plan);
                }
                var expected = eager.Merge(ReferenceExcitation(eager, f0, referenceRandom));
                var sum = engine.ReduceSum(expected, null, false);
                var expectedLoss = engine.TensorMultiply(p, sum).ToArray()[0];
                Assert.Equal(expectedLoss, plan.Step().ToArray()[0]);
                Assert.Equal(expected.ToArray(), tracedSource.ToArray());
                Assert.Equal(sum.ToArray()[0], plan.Gradients[0].ToArray()[0]);
                // One phase draw and 2 draws per Gaussian for each of 2 harmonics and 6 samples.
                Assert.Equal(25 * (step + 1), random.Calls);
                Assert.Equal(referenceRandom.Calls, random.Calls);
                if (typeof(T) == typeof(float))
                    Assert.Equal(new long[] { step + 1, 0, step + 1, step + 1 }, plan.ParameterAcceptedUpdates);
            }
        }
    }

    private static Tensor<T> ReferenceExcitation<T>(HiftGenerator<T> model, Tensor<T> f0, Random random)
    {
        // The pre-repair eager draw order is an independent oracle: phase first, then two fresh uniforms per Gaussian.
        int harmonics = model.Configuration.Harmonics + 1, samples = f0.Length * model.Configuration.UpsampleScale;
        var phases = new double[harmonics];
        for (int h = 1; h < harmonics; h++) phases[h] = -Math.PI + 2 * Math.PI * random.NextDouble();
        var noise = new double[harmonics, samples];
        for (int h = 0; h < harmonics; h++) for (int s = 0; s < samples; s++)
            noise[h, s] = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        if (model.Configuration.SampleRate == 22050 || model.Configuration.OriginalSineGen)
            return OriginalHarmonics(model, f0, phases, noise);
        var fractions = new double[harmonics];
        for (int h = 1; h < harmonics; h++) fractions[h] = phases[h] / (2 * Math.PI) + 0.5;
        return OriginalHarmonicsByFrames(model, f0, fractions, noise);
    }

    private static Tensor<T> OriginalHarmonicsByFrames<T>(HiftGenerator<T> model, Tensor<T> f0, double[] initialPhases, double[,] noise)
    {
        var c = model.Configuration;
        var NumOps = MathHelper.GetNumericOperations<T>();
        int frames = f0.Length, scale = c.UpsampleScale, samples = frames * scale, harmonics = c.Harmonics + 1;
        var sines = new Tensor<T>(new[] { 1, harmonics, samples });
        // F.interpolate(mode="linear", align_corners=False) with a scale factor: source = (i + 0.5) / factor − 0.5,
        // clamped at 0, the output length ⌊length · factor⌋.
        static double[] Interpolate(double[] values, double factor)
        {
            int length = values.Length, size = (int)Math.Floor(length * factor);
            double step = 1.0 / factor;
            var result = new double[size];
            for (int i = 0; i < size; i++)
            {
                double source = Math.Max(0.0, (i + 0.5) * step - 0.5);
                int low = Math.Min((int)source, length - 1);
                int high = low < length - 1 ? low + 1 : low;
                double weight = source - low;
                result[i] = (1 - weight) * values[low] + weight * values[high];
            }
            return result;
        }
        for (int h = 0; h < harmonics; h++)
        {
            var increments = new double[samples];
            for (int s = 0; s < samples; s++)
            {
                double value = NumOps.ToDouble(f0[s / scale]) * (h + 1) / c.SampleRate;
                increments[s] = value - Math.Floor(value);
            }
            increments[0] += h == 0 ? 0.0 : initialPhases[h];
            var perFrame = Interpolate(increments, 1.0 / scale);
            double cumulative = 0;
            for (int i = 0; i < perFrame.Length; i++)
            {
                cumulative += perFrame[i];
                perFrame[i] = cumulative * 2 * Math.PI * scale;
            }
            var phase = Interpolate(perFrame, scale);
            for (int s = 0; s < samples; s++)
            {
                double f = NumOps.ToDouble(f0[s / scale]);
                bool voiced = f > c.VoicedThreshold;
                double noiseAmplitude = voiced ? c.NsfSigma : c.NsfAlpha / 3;
                double sine = s < phase.Length ? Math.Sin(phase[s]) : 0.0;
                sines[0, h, s] = NumOps.FromDouble((voiced ? c.NsfAlpha * sine : 0.0) + noiseAmplitude * noise[h, s]);
            }
        }
        return sines;
    }

    private static Tensor<T> OriginalHarmonics<T>(HiftGenerator<T> model, Tensor<T> f0, double[] phases, double[,] noise)
    {
        var c = model.Configuration;
        var NumOps = MathHelper.GetNumericOperations<T>();
        int frames = f0.Length, scale = c.UpsampleScale, samples = frames * scale, harmonics = c.Harmonics + 1;
        var sines = new Tensor<T>(new[] { 1, harmonics, samples });
        for (int h = 0; h < harmonics; h++)
        {
            double cumulative = 0, phase = h == 0 ? 0.0 : phases[h];
            for (int s = 0; s < samples; s++)
            {
                double f = NumOps.ToDouble(f0[s / scale]);
                cumulative += f * (h + 1) / c.SampleRate;
                double theta = 2 * Math.PI * (cumulative - Math.Floor(cumulative));
                bool voiced = f > c.VoicedThreshold;
                double noiseAmplitude = voiced ? c.NsfSigma : c.NsfAlpha / 3;
                sines[0, h, s] = NumOps.FromDouble((voiced ? c.NsfAlpha * Math.Sin(theta + phase) : 0.0) + noiseAmplitude * noise[h, s]);
            }
        }
        return sines;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Source_merge_gradients_match_the_tanh_linear_reference_and_F0_is_absent(bool byFrames)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var model = Build<double>(engine, byFrames);
        var f0 = new Tensor<double>(new[] { 0.0, 110.0, 220.0 }, new[] { 3 });
        var parameters = TapeTrainingStep<double>.CollectParameters(new[] { model.SourceMerge }, -1);
        var excitation = ReferenceExcitation(model, f0, new Random(17)).ToArray();
        using var tape = new GradientTape<double>();
        var source = model.Source(f0, new Random(17));
        var gradients = tape.ComputeGradients(engine.ReduceSum(source, null, false), new[] { f0 }.Concat(parameters).ToArray());
        Assert.False(gradients.ContainsKey(f0));
        var output = source.ToArray();
        var expectedWeights = new double[2];
        double expectedBias = 0;
        for (int s = 0; s < 6; s++)
        {
            double slope = 1 - output[s] * output[s];
            expectedBias += slope;
            for (int h = 0; h < 2; h++) expectedWeights[h] += slope * excitation[h * 6 + s];
        }
        var actualWeights = gradients[parameters[0]].ToArray();
        for (int h = 0; h < 2; h++) Assert.Equal(expectedWeights[h], actualWeights[h], 12);
        Assert.Equal(expectedBias, gradients[parameters[1]].ToArray()[0], 12);
    }

    [Fact]
    public void Forward_records_the_merge_but_keeps_the_pitch_predictor_frozen()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var model = Build<double>(engine, byFrames: true);
        var mel = new Tensor<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1 }, new[] { 1, 2, 4 });
        var merge = TapeTrainingStep<double>.CollectParameters(new[] { model.SourceMerge }, -1);
        var pitch = TapeTrainingStep<double>.CollectParameters(model.F0Convs.Concat(new[] { model.F0Classifier }), -1);
        using var tape = new GradientTape<double>();
        var audio = model.Forward(mel, new Random(17));
        var gradients = tape.ComputeGradients(engine.ReduceSum(audio, null, false), merge.Concat(pitch).ToArray());
        Assert.All(merge, parameter => Assert.True(gradients.ContainsKey(parameter)));
        Assert.All(pitch, parameter => Assert.False(gradients.ContainsKey(parameter)));
    }
}
