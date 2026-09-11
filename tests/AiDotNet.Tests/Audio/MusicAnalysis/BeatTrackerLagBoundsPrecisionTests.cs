using AiDotNet.Audio.MusicAnalysis;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.MusicAnalysis
{
    /// <summary>
    /// Regression test for the autocorrelation lag bounds in <see cref="BeatTracker{T}"/>'s tempo
    /// estimate. The bounds used to compute the frame rate as SampleRate / HopLength in integer
    /// arithmetic, truncating it before converting BPM to a lag, so the lag window could be too short
    /// to contain the true beat period.
    /// </summary>
    public class BeatTrackerLagBoundsPrecisionTests
    {
        [Fact]
        public void Track_NonIntegerFrameRate_SearchesTheFullTempoLagRange()
        {
            // Frame rate = 2900 / 1000 = 2.9 frames/s. Clicks every 4 hops -> beat period of 4 frames,
            // i.e. 60 * 2.9 / 4 = 43.5 BPM, which lies inside [MinTempo, MaxTempo] = [30, 60].
            // Correct lag window: [(int)(2.9 * 60 / 60), (int)(2.9 * 60 / 30)) = [2, 5) -> finds lag 4.
            // Truncated frame rate 2 gave [2, 4) -> lag 4 was never searched and the estimate was
            // lag 3 = 58 BPM.
            const int hop = 1000;
            const int periods = 60;
            var options = new BeatTrackerOptions
            {
                SampleRate = 2900,
                HopLength = hop,
                FftSize = 512,
                MinTempo = 30.0,
                MaxTempo = 60.0,
                SmoothingWindow = 0
            };
            var tracker = new BeatTracker<double>(options);

            // One unit click per beat, offset so it falls inside exactly one (centred) STFT frame.
            var audio = new Tensor<double>([periods * 4 * hop]);
            for (int beat = 0; beat < periods; beat++)
            {
                audio[beat * 4 * hop + 128] = 1.0;
            }

            var result = tracker.Track(audio);

            Assert.Equal(43.5, result.Tempo, 6);
        }
    }
}
