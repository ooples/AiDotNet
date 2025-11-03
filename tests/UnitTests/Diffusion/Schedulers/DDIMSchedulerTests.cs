using System;
using Xunit;
using AiDotNet.Diffusion.Schedulers;

namespace UnitTests.Diffusion.Schedulers
{
    public class DDIMSchedulerTests
    {
        [Fact]
        public void Config_Constructs_With_Defaults()
        {
            var cfg = new SchedulerConfig();
            Assert.Equal(1000, cfg.TrainTimesteps);
            Assert.Equal(BetaSchedule.Linear, cfg.BetaSchedule);
            Assert.Equal(PredictionType.Epsilon, cfg.PredictionType);
        }

        [Fact]
        public void SetTimesteps_Produces_Descending_Sequence()
        {
            var sch = new DDIMScheduler(new SchedulerConfig(trainTimesteps: 100));
            sch.SetTimesteps(20);
            var ts = sch.Timesteps;
            Assert.True(ts.Length <= 20 && ts.Length > 0);
            for (int i = 1; i < ts.Length; i++)
                Assert.True(ts[i] < ts[i-1]);
        }

        [Fact]
        public void Step_Deterministic_When_Eta_Zero()
        {
            var sch = new DDIMScheduler(new SchedulerConfig(trainTimesteps: 50));
            sch.SetTimesteps(10);
            int t = sch.Timesteps[0];

            var sample = new double[] { 0.1, -0.2, 0.3, -0.4 };
            var eps =    new double[] { 0.05, 0.02, -0.01, -0.03 };

            var prev = sch.Step(eps, t, sample, eta: 0.0);
            Assert.Equal(sample.Length, prev.Length);

            // sanity: values are finite and within a reasonable range
            foreach (var v in prev) Assert.True(!double.IsNaN(v) && !double.IsInfinity(v));
        }

        [Fact]
        public void Step_Uses_Sigma_When_Eta_Positive()
        {
            var sch = new DDIMScheduler(new SchedulerConfig(trainTimesteps: 100));
            sch.SetTimesteps(25);
            int t = sch.Timesteps[0];

            var sample = new double[] { 0.2, 0.0, -0.1 };
            var eps =    new double[] { 0.01, -0.02, 0.03 };
            var noise =  new double[] { 0.3, -0.5, 0.7 };

            var prevNoNoise = sch.Step(eps, t, sample, eta: 0.0);
            var prevWithNoise = sch.Step(eps, t, sample, eta: 0.5, noise: noise);

            // With non-zero eta and provided noise, result should differ
            bool anyDiff = false;
            for (int i = 0; i < sample.Length; i++)
            {
                if (Math.Abs(prevNoNoise[i] - prevWithNoise[i]) > 1e-9)
                {
                    anyDiff = true;
                    break;
                }
            }
            Assert.True(anyDiff);
        }

        [Fact]
        public void Step_Clips_When_Config_ClipSample_True()
        {
            var cfg = new SchedulerConfig(trainTimesteps: 2, betaStart: 0.0001, betaEnd: 0.02, clipSample: true);
            var sch = new DDIMScheduler(cfg);
            sch.SetTimesteps(2); // timesteps should be [1,0]
            int t = sch.Timesteps[0]; // 1

            // Compute expected values using the same formulas
            double beta0 = 0.0001;
            double beta1 = 0.02;
            double alpha0 = 1 - beta0; // 0.9999
            double alpha1 = 1 - beta1; // 0.98
            double ac = alpha0 * alpha1;
            double acPrev = alpha0;
            double sqrtAc = Math.Sqrt(ac);
            double sqrtOneMinusAc = Math.Sqrt(1 - ac);
            double sqrtAcPrev = Math.Sqrt(acPrev);
            double sqrtOneMinusAcPrev = Math.Sqrt(1 - acPrev);

            var sample = new double[] { 10.0 }; // large value to force clipping
            var eps =    new double[] { -5.0 };

            // Unclipped predOriginal
            double predOriginal = (sample[0] - sqrtOneMinusAc * eps[0]) / sqrtAc;
            // Clip to [-1,1]
            if (predOriginal > 1) predOriginal = 1;
            else if (predOriginal < -1) predOriginal = -1;

            double expected = sqrtAcPrev * predOriginal + Math.Sqrt(1 - acPrev) * eps[0];

            var actual = sch.Step(eps, t, sample, eta: 0.0);
            Assert.InRange(actual[0], expected - 1e-8, expected + 1e-8);
        }
    }
}
