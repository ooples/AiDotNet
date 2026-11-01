using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Classification;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Tzanetakis and Cook 2002: the 30 musical surface features (timbral texture, beat histogram, pitch histograms) of
/// section III and the per-genre diagonal Gaussian mixtures fitted with EM of section IV (#2397).
/// </summary>
public class GenreClassifierPaperTests
{
    private const int Rate = 22050;

    private static int Index(string name) => Array.IndexOf(MusicalSurfaceFeatures.Names, name);

    private static double[] Features(GenreClassifier<double> model, Func<double, double> signal, double seconds = 3.0)
    {
        var audio = new Tensor<double>(new[] { (int)(Rate * seconds) });
        for (int i = 0; i < audio.Length; i++) audio[i] = signal(i / (double)Rate);
        var features = ((IGenreClassifier<double>)model).ExtractFeatures(audio);
        Assert.Equal(new[] { 1, MusicalSurfaceFeatures.Length }, features.Shape.ToArray());
        return Enumerable.Range(0, MusicalSurfaceFeatures.Length).Select(i => features[0, i]).ToArray();
    }

    // A harmonic tone: the fundamental and two overtones.
    private static double Harmonic(double t, double f) =>
        0.5 * Math.Sin(2 * Math.PI * f * t) + 0.25 * Math.Sin(4 * Math.PI * f * t) + 0.125 * Math.Sin(6 * Math.PI * f * t);

    // Decaying 2 kHz bursts, one per beat.
    private static double Clicks(double t, double bpm)
    {
        double sinceBeat = t % (60.0 / bpm);
        return Math.Exp(-60 * sinceBeat) * Math.Sin(2 * Math.PI * 2000 * t);
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new GenreClassifierOptions();
        Assert.Equal((22050, 512, 512, 43), (o.SampleRate, o.FftSize, o.HopLength, o.TextureWindowFrames));
        Assert.Equal((65536, 32768, 3), (o.BeatWindow, o.BeatHop, o.MixtureComponents));
        Assert.Equal(
            new[] { "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock" },
            GenreClassifier<double>.StandardGenres);
        using var model = new GenreClassifier<double>();
        var layer = Assert.IsType<GaussianMixtureClassifierLayer<double>>(Assert.Single(model.Layers));
        Assert.Equal(3, layer.Components);
    }

    [Fact(Timeout = 120000)]
    public async Task Timbral_CentroidAndZeroCrossingsOfAToneAreItsFrequency()
    {
        await Task.Yield();
        using var model = new GenreClassifier<double>();
        var f = Features(model, t => 0.5 * Math.Sin(2 * Math.PI * 1000 * t));
        // A 1 kHz tone: the magnitude centroid sits at 1 kHz (within the 43 Hz bins and the window's leakage), and a
        // 512-sample window holds 2 * 1000 * 512 / 22050 = 46.4 zero crossings.
        Assert.InRange(f[Index("CentroidMean")], 900, 1300);
        Assert.InRange(f[Index("ZeroCrossingsMean")], 45, 48);
        Assert.InRange(f[Index("CentroidVariance")], 0, 1000);
        // A steady tone: every window has the texture window's mean energy, so almost none falls below it.
        Assert.InRange(f[Index("LowEnergy")], 0, 0.6);
    }

    [Fact(Timeout = 120000)]
    public async Task BeatHistogram_FindsTheTempoOfAClickTrain()
    {
        await Task.Yield();
        using var model = new GenreClassifier<double>();
        foreach (double bpm in new[] { 90.0, 120.0 })
        {
            var f = Features(model, t => Clicks(t, bpm));
            Assert.InRange(f[Index("P1")], bpm - 3, bpm + 3);
            Assert.InRange(f[Index("A0")], 0.0, 1.0);
            Assert.True(f[Index("BeatSum")] > 0);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task PitchHistograms_PlaceAHarmonicToneOnItsNoteAndPitchClass()
    {
        await Task.Yield();
        using var model = new GenreClassifier<double>();
        var f = Features(model, t => Harmonic(t, 440));
        // A4 is MIDI 69 (equation 12); its pitch class 9 sits at (7 * 9) mod 12 = 3 on the circle of fifths (14).
        Assert.Equal(69, f[Index("UP0")]);
        Assert.Equal(3, f[Index("FP0")]);
        Assert.True(f[Index("FA0")] > 0);
    }

    [Fact(Timeout = 60000)]
    public async Task SimpleGaussian_LogLikelihoodIsTheClassGaussiansDensity()
    {
        await Task.Yield();
        // One component per class is the paper's simple Gaussian classifier: the EM fit is the sample mean and the
        // (population) variance of each feature, and the output is the diagonal Gaussian log density.
        var options = new GenreClassifierOptions { MixtureComponents = 1, CustomGenres = new[] { "a", "b" } };
        using var model = new GenreClassifier<double>(options);
        int n = 6, d = MusicalSurfaceFeatures.Length;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var inputs = new Tensor<double>(new[] { 2 * n, d });
        var targets = new Tensor<double>(new[] { 2 * n, 2 });
        for (int i = 0; i < 2 * n; i++)
        {
            for (int j = 0; j < d; j++) inputs[i, j] = (i < n ? 0 : 5) + rng.NextDouble() * (j + 1);
            targets[i, i < n ? 0 : 1] = 1;
        }
        model.Train(inputs, targets);

        var x = Enumerable.Range(0, d).Select(j => 1.0 + 0.1 * j).ToArray();
        double expected = 0;
        for (int j = 0; j < d; j++)
        {
            var column = Enumerable.Range(0, n).Select(i => inputs[i, j]).ToArray();
            double mean = column.Average(), variance = column.Select(v => (v - mean) * (v - mean)).Average();
            expected += -0.5 * (Math.Log(2 * Math.PI * variance) + (x[j] - mean) * (x[j] - mean) / variance);
        }
        var logLikelihood = model.Layers[0].Forward(new Tensor<double>(x, new[] { d }));
        Assert.Equal(expected, logLikelihood[0], 6);
    }

    [Fact(Timeout = 60000)]
    public async Task Mixture_FitsSeparatedModesAndClassifiesHeldOutSamples()
    {
        await Task.Yield();
        // Genre "a" has two modes and "b" one between them: a single Gaussian per genre cannot separate them, GMM(2)
        // can.
        var options = new GenreClassifierOptions { MixtureComponents = 2, CustomGenres = new[] { "a", "b" }, Seed = 1 };
        using var model = new GenreClassifier<double>(options);
        int d = MusicalSurfaceFeatures.Length;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4);
        double Center(int i) => i % 3 == 0 ? -4 : i % 3 == 1 ? 4 : 0;
        Tensor<double> Samples(int count, out int[] labels)
        {
            var x = new Tensor<double>(new[] { count, d });
            labels = new int[count];
            for (int i = 0; i < count; i++)
            {
                labels[i] = i % 3 == 2 ? 1 : 0;
                for (int j = 0; j < d; j++) x[i, j] = Center(i) + 0.5 * (rng.NextDouble() - 0.5);
            }
            return x;
        }
        var train = Samples(60, out var trainLabels);
        var targets = new Tensor<double>(new[] { 60, 2 });
        for (int i = 0; i < 60; i++) targets[i, trainLabels[i]] = 1;
        model.Train(train, targets);

        var test = Samples(30, out var testLabels);
        var probabilities = model.Predict(test);
        for (int i = 0; i < 30; i++)
        {
            int predicted = probabilities[i, 1] > probabilities[i, 0] ? 1 : 0;
            Assert.Equal(testLabels[i], predicted);
            Assert.Equal(1.0, probabilities[i, 0] + probabilities[i, 1], 9);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task EndToEnd_LearnsTwoGenresFromAudio()
    {
        await Task.Yield();
        var options = new GenreClassifierOptions { CustomGenres = new[] { "tonal", "percussive" }, MixtureComponents = 1, Seed = 2 };
        using var model = new GenreClassifier<double>(options);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(6);
        Func<double, double> Clip(int genre, int variant)
        {
            double pitch = 220 * Math.Pow(2, variant / 12.0), bpm = 100 + 10 * variant, phase = rng.NextDouble();
            return genre == 0 ? t => Harmonic(t + phase, pitch) : t => Clicks(t + phase, bpm);
        }
        int perGenre = 4, d = MusicalSurfaceFeatures.Length;
        var inputs = new Tensor<double>(new[] { 2 * perGenre, d });
        var targets = new Tensor<double>(new[] { 2 * perGenre, 2 });
        for (int g = 0; g < 2; g++)
            for (int v = 0; v < perGenre; v++)
            {
                var f = Features(model, Clip(g, v), 1.5);
                for (int j = 0; j < d; j++) inputs[g * perGenre + v, j] = f[j];
                targets[g * perGenre + v, g] = 1;
            }
        model.Train(inputs, targets);

        for (int g = 0; g < 2; g++)
        {
            var clip = Clip(g, perGenre);
            var audio = new Tensor<double>(new[] { (int)(Rate * 1.5) });
            for (int i = 0; i < audio.Length; i++) audio[i] = clip(i / (double)Rate);
            var result = model.Classify(audio);
            Assert.Equal(options.CustomGenres[g], result.PredictedGenre);
        }
    }
}
