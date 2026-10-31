using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The Tacotron 2 mel spectrogram and FastSpeech 2 frame energy match the published preprocessing.
/// </summary>
/// <remarks>
/// <c>ReferenceData/librosa_tacotron_mel.json</c> was produced with librosa 1.0.0 from the 22050 Hz test signal of
/// <c>WorldPitchDetectorTests</c>: <c>|librosa.stft(n_fft=1024, hop_length=256, win_length=1024, window='hann',
/// center=True, pad_mode='reflect')|</c>, <c>librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0,
/// fmax=8000)</c>, <c>log(max(mel, 1e-5))</c> (Tacotron 2's dynamic-range compression), and the per-frame L2 norm
/// of the magnitude (FastSpeech 2's energy).
/// </remarks>
public class TacotronSpectrogramTests
{
    private static JObject Reference()
    {
        const string fileName = "librosa_tacotron_mel.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static double[] Signal()
    {
        const int fs = 22050;
        const double seconds = 1.2;
        int n = (int)(fs * seconds);
        var x = new double[n];
        double phase = 0.0;
        long state = 12345;
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / fs;
            double f0 = 120.0 + 100.0 * t / seconds;
            phase += 2 * Math.PI * f0 / fs;
            double v = 0.0;
            if (!(0.55 * seconds <= t && t < 0.70 * seconds))
            {
                for (int h = 1; h < 6; h++) v += Math.Sin(h * phase) / h;
                v *= 0.3;
            }
            state = (1103515245L * state + 12345L) % 2147483648L;
            v += 0.01 * ((state / 2147483648.0) - 0.5);
            x[i] = v;
        }
        return x;
    }

    [Fact(Timeout = 60000)]
    public async Task MelBasis_MatchesLibrosa()
    {
        await Task.Yield();
        var basis = new TacotronSpectrogram().MelBasis;
        var expected = Reference()["mel_basis_rows_0_5_40_79"]!.Select(r => r.Select(v => (double)v).ToArray()).ToArray();
        int[] rows = { 0, 5, 40, 79 };
        for (int r = 0; r < rows.Length; r++)
            for (int k = 0; k < expected[r].Length; k++)
                Assert.True(Math.Abs(expected[r][k] - basis[rows[r], k]) <= 1e-7,
                    $"mel basis [{rows[r]},{k}]: librosa {expected[r][k]:R}, port {basis[rows[r], k]:R}.");
    }

    [Fact(Timeout = 60000)]
    public async Task LogMelAndEnergy_MatchLibrosa()
    {
        await Task.Yield();
        var reference = Reference();
        var spec = new TacotronSpectrogram();
        var x = Signal();

        var magnitude = spec.Magnitude(x);
        Assert.Equal((int)reference["frames"]!, magnitude.GetLength(0));

        var energy = spec.Energy(magnitude);
        var expectedEnergy = reference["energy"]!.Select(v => (double)v).ToArray();
        for (int f = 0; f < expectedEnergy.Length; f++)
            Assert.True(Math.Abs(expectedEnergy[f] - energy[f]) <= 1e-7 * Math.Max(1, expectedEnergy[f]),
                $"energy[{f}]: librosa {expectedEnergy[f]:R}, port {energy[f]:R}.");

        var mel = spec.LogMel(magnitude);
        var expectedMel = reference["mel_first_frames"]!.Select(r => r.Select(v => (double)v).ToArray()).ToArray();
        for (int f = 0; f < expectedMel.Length; f++)
            for (int m = 0; m < expectedMel[f].Length; m++)
                Assert.True(Math.Abs(expectedMel[f][m] - mel[f, m]) <= 1e-6,
                    $"log-mel[{f},{m}]: librosa {expectedMel[f][m]:R}, port {mel[f, m]:R}.");
    }
}
