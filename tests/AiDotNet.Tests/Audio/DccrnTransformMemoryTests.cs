#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Audio.Enhancement;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// DCCRN's STFT pair against the torch.stft / torch.istft definitions it implements: periodic Hann
/// window, center=True with zero padding of n_fft / 2, and an inverse that applies the one-sided
/// Hermitian bin weights and divides by the overlap-added squared window.
/// </summary>
public class DccrnTransformMemoryTests
{
    private static DCCRN<T> Model<T>(int size, int hop)
    {
        var model = (DCCRN<T>)FormatterServices.GetUninitializedObject(typeof(DCCRN<T>));
        (typeof(NeuralNetworkBase<T>).GetField("NumOps", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("NumOps")).SetValue(model, MathHelper.GetNumericOperations<T>());
        foreach (var (name, value) in new[] { ("_fftSize", size), ("_hopSize", hop) })
            (typeof(DCCRN<T>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException(name)).SetValue(model, value);
        return model;
    }

    private static MethodInfo Method<T>(string name)
        => typeof(DCCRN<T>).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(name);

    private static Tensor<T> Stft<T>(DCCRN<T> model, Tensor<T> audio)
        => Assert.IsType<Tensor<T>>(Method<T>("ComputeComplexSTFT").Invoke(model, new object[] { audio }));

    private static Tensor<T> Istft<T>(DCCRN<T> model, Tensor<T> stft, int? length)
        => Assert.IsType<Tensor<T>>(Method<T>("ComputeInverseSTFT").Invoke(model, new object?[] { stft, length }));

    /// <summary>
    /// An impulse at sample N / 4 of an N-sample clip, hop N: centre padding puts it at position 3N / 4
    /// of frame 0 (periodic Hann weight 0.5) and outside frame 1, so frame 0 holds
    /// 0.5 * exp(-2 pi i k (3N / 4) / N) and frame 1 is zero.
    /// </summary>
    [Theory]
    [InlineData(32)]
    [InlineData(16384)]
    public void Stft_Impulse_MatchesCenteredTorchStft(int size)
    {
        var model = Model<double>(size, size);
        var audio = new Tensor<double>(new[] { size });
        audio.SetFlat(size / 4, 1);
        var spectrum = Stft(model, audio);
        int bins = size / 2 + 1, position = 3 * size / 4;
        Assert.Equal(new[] { 1, 2, bins, 2 }, spectrum.Shape);
        for (int k = 0; k < bins; k++)
        {
            double phase = 2 * Math.PI * ((long)k * position % size) / size;
            Assert.Equal(0.5 * Math.Cos(phase), spectrum[0, 0, k, 0], 10);
            Assert.Equal(-0.5 * Math.Sin(phase), spectrum[0, 1, k, 0], 10);
            Assert.Equal(0.0, spectrum[0, 0, k, 1], 10);
            Assert.Equal(0.0, spectrum[0, 1, k, 1], 10);
        }
    }

    /// <summary>
    /// torch.istft(torch.stft(x)) reconstructs x: this needs the factor 2 on the bins between DC and
    /// Nyquist, the division by the overlap-added squared window, and the trim of the centre padding.
    /// Without a length the output has hop * (frames - 1) samples, torch's default.
    /// </summary>
    [Theory]
    [InlineData(32, 8, 301)]
    [InlineData(32, 16, 7)]
    [InlineData(16384, 8192, 20000)]
    public void StftThenInverse_ReconstructsSignal(int size, int hop, int length)
    {
        var model = Model<double>(size, hop);
        var random = RandomHelper.CreateSeededRandom(17);
        var samples = new double[length];
        for (int i = 0; i < length; i++) samples[i] = random.NextDouble() * 2 - 1;
        var spectrum = Stft(model, new Tensor<double>(samples, new[] { length }));
        int frames = length / hop + 1;
        Assert.Equal(frames, spectrum.Shape[3]);

        var reconstructed = Istft(model, spectrum, length);
        Assert.Equal(length, reconstructed.Length);
        for (int i = 0; i < length; i++)
            Assert.Equal(samples[i], reconstructed.GetFlat(i), 9);

        var defaultLength = Istft(model, spectrum, null);
        Assert.Equal(hop * (frames - 1), defaultLength.Length);
        for (int i = 0; i < defaultLength.Length; i++)
            Assert.Equal(samples[i], defaultLength.GetFlat(i), 9);
    }

    /// <summary>
    /// One frame holding only bin 3 (real 1, imaginary 0.25) inverts to the real irfft
    /// (2 / N)(cos(2 pi 3 n / N) - 0.25 sin(2 pi 3 n / N)) times the window, divided by the squared
    /// window where that exceeds 1e-11; the centre padding of N / 2 is trimmed first.
    /// </summary>
    [Fact]
    public void Inverse_SingleBin_MatchesIrfftWithWindowEnvelope()
    {
        const int size = 32, hop = 32;
        var model = Model<double>(size, hop);
        var spectrum = new Tensor<double>(new[] { 1, 2, size / 2 + 1, 1 });
        spectrum[0, 0, 3, 0] = 1;
        spectrum[0, 1, 3, 0] = 0.25;
        var reconstructed = Istft(model, spectrum, size / 2);
        Assert.Equal(size / 2, reconstructed.Length);
        for (int i = 0; i < size / 2; i++)
        {
            int n = i + size / 2;
            double phase = 2 * Math.PI * (3L * n % size) / size;
            double frame = 2.0 / size * (Math.Cos(phase) - 0.25 * Math.Sin(phase));
            double window = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / size);
            double expected = window * window > 1e-11 ? frame * window / (window * window) : 0;
            Assert.Equal(expected, reconstructed.GetFlat(i), 12);
        }
    }

    [Fact]
    public void HalfStft_PhaseProductsDoNotOverflowAtOrdinaryFftSize()
    {
        const int size = 512, bins = size / 2 + 1, position = 3 * size / 4;
        var model = Model<Half>(size, size);
        var audio = new Tensor<Half>(new[] { size });
        audio.SetFlat(size / 4, (Half)1);
        var spectrum = Stft(model, audio);
        for (int k = 0; k < bins; k++)
        {
            double phase = 2 * Math.PI * ((long)k * position % size) / size;
            Assert.InRange(Math.Abs((double)spectrum[0, 0, k, 0] - 0.5 * Math.Cos(phase)), 0, 0.001);
            Assert.InRange(Math.Abs((double)spectrum[0, 1, k, 0] + 0.5 * Math.Sin(phase)), 0, 0.001);
        }
    }
}
#endif
