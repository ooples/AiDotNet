using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio;
using AiDotNet.Audio.Classification;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Audio.MusicAnalysis;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio
{
    public class AudioEngineOpsTests
    {
        [Fact]
        public void HalfRowAndEventIndices_RemainExactBeyond2048()
        {
            var engine = AiDotNetEngine.Current;
            var scores = new Half[2050];
            scores[2049] = (Half)1;
            Assert.Equal(new[] { 2049 }, AudioEngineOps.RowFirstArgMax(engine, new Tensor<Half>(scores, [1, 2050])));

            var ties = Enumerable.Repeat((Half)1, 2049).ToArray();
            var segments = new int[ties.Length];
            Assert.Equal(new[] { 2048 }, AudioEventDetectionOps.ArgMaxPerSegment(engine, ties, segments, 1, last: true));
            Assert.Equal(new[] { 0 }, AudioEventDetectionOps.ArgMaxPerSegment(engine, ties, segments, 1, last: false));
        }

        [Fact]
        public void HalfSpectralStatistics_FullScaleValues_RemainFinite()
        {
            var engine = AiDotNetEngine.Current;
            var spectrum = Tensor<Half>.CreateDefault([1, 2], Half.MaxValue);
            Assert.Equal(0.5, AudioEngineOps.SpectralCentroidBins(engine, spectrum), 12);
            Assert.Equal(1.0, AudioEngineOps.SpectralFlatness(engine, spectrum), 12);
            var moments = AudioEngineOps.ColumnMeanStd(engine,
                new Tensor<Half>(new[] { Half.MaxValue, -Half.MaxValue }, [2, 1]));
            Assert.Equal((Half)0, moments.Mean[0]);
            Assert.Equal(Half.MaxValue, moments.Std[0]);
        }

        [Fact]
        public void HalfEventStatistics_LargeCount_PreservesMeanConfidence()
        {
            var events = Enumerable.Range(0, 70000).Select(i => new AiDotNet.Interfaces.AudioEvent<Half>
            {
                EventType = "tone", Confidence = (Half)0.5, StartTime = i, EndTime = i + 1
            }).ToArray();
            var stats = AudioEventDetectionOps.ComputeEventStatistics(AiDotNetEngine.Current, events)["tone"];
            Assert.Equal(70000, stats.Count);
            Assert.Equal((Half)0.5, stats.AverageConfidence);
        }

        [Fact]
        public void FloatLargeEmbedding_NormalizesWithoutOverflow()
        {
            var normalized = AudioEngineOps.L2NormalizedData(AiDotNetEngine.Current,
                new Tensor<float>(new[] { 1e20f }, [1]));
            Assert.Equal(new[] { 1f }, normalized);
            var fingerprinter = new ChromaprintFingerprinter<float>();
            var fingerprint = new AiDotNet.Interfaces.AudioFingerprint<float> { Data = new[] { 1e20f }, Hash = null };
            Assert.Equal(1.0, fingerprinter.ComputeSimilarity(fingerprint, fingerprint), 12);
        }

        [Fact]
        public void SymmetricHann_HalfSingleton_EqualsOne()
        {
            Assert.Equal(new[] { (Half)1 }, AudioEngineOps.SymmetricHannWindow<Half>(AiDotNetEngine.Current, 1).ToArray());
            Assert.Equal(new[] { (Half)0, (Half)1, (Half)0 },
                AudioEngineOps.SymmetricHannWindow<Half>(AiDotNetEngine.Current, 3).ToArray());
        }

        [Fact]
        public void BeatTracker_RevalidatesMutatedTempoBounds()
        {
            var options = new BeatTrackerOptions();
            var tracker = new BeatTracker<double>(options);
            options.MaxTempo = options.MinTempo / 2;
            Assert.Throws<ArgumentOutOfRangeException>(() => tracker.Track(new Tensor<double>([0])));
        }

        /// <summary>
        /// A silent onset envelope has no peaks to align with, so confidence is 0 (the double result).
        /// The 1e-10 floor rounds to 0 in Half; compared in Half the guard let 0 / 0 through as NaN.
        /// </summary>
        [Fact]
        public void BeatTracker_HalfSilentEnvelope_ConfidenceIsZero()
        {
            var tracker = new BeatTracker<Half>();
            var method = typeof(BeatTracker<Half>).GetMethod("ComputeConfidence",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("ComputeConfidence");
            var confidence = (Half)(method.Invoke(tracker, new object[]
            {
                new Tensor<Half>([64]), new System.Collections.Generic.List<double> { 0.0, 0.5 }
            }) ?? throw new InvalidOperationException("null confidence"));
            Assert.Equal((Half)0, confidence);
        }

        /// <summary>
        /// The tempo penalty for distances d = maxWindow .. 1 is
        /// exp(-0.5 ((d - p) / p)^2 / flexibility^2). Past 2048 a Half arange no longer holds every
        /// distance, so the weights must equal the double formula rounded once to Half.
        /// </summary>
        [Fact]
        public void BeatTracker_HalfTempoPenalty_MatchesDoubleBeyond2048Frames()
        {
            const int maxWindow = 4001;
            const double period = 2000, flexibility = 0.2;
            var penalty = BeatTracker<Half>.TempoPenaltyByDistance(AiDotNetEngine.Current, maxWindow, period, flexibility);
            Assert.Equal(maxWindow, penalty.Length);
            for (int i = 0; i < maxWindow; i++)
            {
                double d = maxWindow - i;
                double deviation = Math.Abs(d - period) / period;
                Assert.Equal((Half)Math.Exp(-0.5 * deviation * deviation / (flexibility * flexibility)), penalty[i]);
            }
        }

        [Fact]
        public void ClapHalfResampling_PreservesImpulseBeyondExactIntegerRange()
        {
            const int inputs = 5000, outputs = 4096, impulse = 4097;
            var samples = new Half[inputs];
            samples[impulse] = (Half)1;
            var actual = ClapAudioFrontEnd<Half>.AntialiasedBilinearShrink(new Tensor<Half>(samples, [inputs, 1]), outputs);
            double scale = inputs / (double)outputs;
            for (int i = 0; i < outputs; i++)
            {
                double center = scale * (i + 0.5);
                int first = Math.Max((int)(center - scale + 0.5), 0);
                int last = Math.Min((int)(center + scale + 0.5), inputs);
                double total = 0, weighted = 0;
                for (int source = first; source < last; source++)
                {
                    double weight = Math.Max(0, 1 - Math.Abs((source - center + 0.5) / scale));
                    total += weight;
                    if (source == impulse) weighted += weight;
                }
                Assert.Equal((Half)(weighted / total), actual[i, 0]);
            }
        }

        /// <summary>
        /// Wav2Vec2 normalization of a Half clip longer than Half.MaxValue (65,504) samples: the count
        /// and the raw sums do not fit in Half, so the mean and std must be formed without them. The
        /// result must match a double reference computed from the same Half-rounded samples.
        /// </summary>
        [Fact(Timeout = 120000)]
        public async Task Standardize_HalfClipLongerThanHalfMax_MatchesDoubleReference()
        {
            await Task.Yield();
            const int length = 70000;
            var half = MathHelper.GetNumericOperations<Half>();
            var samples = new Half[length];
            var reference = new double[length];
            for (int i = 0; i < length; i++)
            {
                samples[i] = half.FromDouble(0.5 * Math.Sin(i * 0.01) + 0.25);
                reference[i] = half.ToDouble(samples[i]);
            }

            double mean = 0;
            for (int i = 0; i < length; i++) mean += reference[i];
            mean /= length;
            double sumSq = 0;
            for (int i = 0; i < length; i++) sumSq += (reference[i] - mean) * (reference[i] - mean);
            double std = Math.Max(Math.Sqrt(sumSq / length), 1e-7);

            var result = AudioEngineOps.Standardize(
                AiDotNetEngine.Current, new Tensor<Half>(samples, [length]), 1e-7).ToArray();

            Assert.Equal(length, result.Length);
            for (int i = 0; i < length; i++)
            {
                double expected = (reference[i] - mean) / std;
                double actual = half.ToDouble(result[i]);
                Assert.False(double.IsNaN(actual) || double.IsInfinity(actual), $"sample {i} is {actual}");
                // Half has a 2^-11 unit roundoff; the mean, the deviation, the std and the division each
                // round in Half, so outputs of magnitude up to ~2.2 carry a few 1e-3 of error.
                Assert.True(Math.Abs(actual - expected) <= 2e-2, $"sample {i}: expected {expected}, got {actual}");
            }
        }

        /// <summary>
        /// Full-scale Half samples: subtracting the mean (about 21,835) from -65,504 would overflow Half
        /// to -inf, so the samples must be scaled before the mean is removed. Double reference
        /// (population std): [65504, -65504, 65504] standardizes to [1/sqrt(2), -sqrt(2), 1/sqrt(2)].
        /// </summary>
        [Fact(Timeout = 60000)]
        public async Task Standardize_FullScaleHalf_DoesNotOverflow()
        {
            await Task.Yield();
            var half = MathHelper.GetNumericOperations<Half>();
            var samples = new[] { half.FromDouble(65504), half.FromDouble(-65504), half.FromDouble(65504) };
            var result = AudioEngineOps.Standardize(
                AiDotNetEngine.Current, new Tensor<Half>(samples, [3]), 1e-7).ToArray();
            double[] expected = { 1 / Math.Sqrt(2), -Math.Sqrt(2), 1 / Math.Sqrt(2) };
            for (int i = 0; i < expected.Length; i++)
            {
                double actual = half.ToDouble(result[i]);
                Assert.True(Math.Abs(actual - expected[i]) <= 1e-2, $"sample {i}: expected {expected[i]}, got {actual}");
            }
        }
    }
}
