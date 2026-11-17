using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanResidentMelTests
{
    private const int N = 16, Hop = 4, Window = 12, Length = 64;
    private const double Epsilon = .125;

    [Fact]
    public void Cpu_default_preserves_folded_double_window_dft()
    {
        var cpu = new CpuEngine();
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = cpu;
        try
        {
            var audio = Input();
            var mel = new DifferentiableMel<double>(cpu, 4000, N, Hop, Window, 4, 0, 2000, true);
            var expected = Oracle(audio.ToArray(), stagedFloatWindow: false);
            Close(expected.Magnitude, mel.Magnitude(audio, Epsilon).ToArray(), 2e-12);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [SkippableFact]
    public void Cuda_windowed_magnitude_and_waveform_vjp_match_independent_dft()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver unavailable.");
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "Native GPU unavailable.");
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            var samples = Input().ToArray();
            var input = new Tensor<float>(samples.Select(value => (float)value).ToArray(), new[] { Length });
            var expected = Oracle(input.ToArray().Select(value => (double)value).ToArray(), stagedFloatWindow: true);
            var mel = new DifferentiableMel<float>(gpu, 4000, N, Hop, Window, 4, 0, 2000, true);
            using var tape = new GradientTape<float>();
            var magnitude = mel.Magnitude(input, Epsilon);
            var upstream = new Tensor<float>(expected.Upstream.Select(value => (float)value).ToArray(), magnitude.Shape.ToArray());
            var gradient = tape.ComputeGradients(gpu.ReduceSum(gpu.TensorMultiply(magnitude, upstream)), new[] { input })[input];
            Close(expected.Magnitude, magnitude.ToArray().Select(value => (double)value).ToArray(), 3e-5);
            Close(expected.Gradient, gradient.ToArray().Select(value => (double)value).ToArray(), 3e-5);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [SkippableFact]
    public void Cuda_training_records_real_fft_semantic_without_materializing_the_windowed_transform()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver unavailable.");
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "Native GPU unavailable.");
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            var mel = new DifferentiableMel<float>(gpu, 4000, N, Hop, Window, 4, 0, 2000, true);
            var input = new Tensor<float>(Input().ToArray().Select(value => (float)value).ToArray(), new[] { Length });
            using var graph = GraphMode.EnableTraining(new[] { input });
            var magnitude = mel.Magnitude(input, Epsilon);
            Assert.NotNull(magnitude.LazySource);
            Assert.Equal("TensorRfftMagnitude", magnitude.LazySource!.OpName);
            Assert.Equal(new[] { Length / Hop, N / 2 + 1 }, magnitude.Shape.ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static Tensor<double> Input() => new(Enumerable.Range(0, Length)
        .Select(i => Math.Sin(i * .37) * .25 + (i % 7 - 3) * .03125).ToArray(), new[] { Length });

    // Analytic real DFT and frame/reflection adjoint. No engine FFT, matrix
    // product, autograd implementation, or production coefficient table is used.
    private static (double[] Magnitude, double[] Gradient, double[] Upstream) Oracle(double[] samples, bool stagedFloatWindow)
    {
        int pad = (N - Hop) / 2, bins = N / 2 + 1, frames = 1 + (Length + 2 * pad - N) / Hop;
        var magnitude = new double[frames * bins]; var gradient = new double[Length];
        var upstream = Enumerable.Range(0, magnitude.Length).Select(i => (i % 7 - 3) * .0625).ToArray();
        var window = new double[N]; int windowStart = (N - Window) / 2;
        for (int i = 0; i < Window; i++)
        {
            if (stagedFloatWindow)
            {
                float angle = i * (float)(2 * Math.PI / Window);
                float cosine = MathF.Cos(angle), weighted = -.5f * cosine;
                window[windowStart + i] = weighted + .5f;
            }
            else window[windowStart + i] = .5 - .5 * Math.Cos(2 * Math.PI * i / Window);
        }
        for (int frame = 0; frame < frames; frame++)
            for (int bin = 0; bin < bins; bin++)
            {
                double real = 0, imaginary = 0;
                for (int t = 0; t < N; t++)
                {
                    double angle = 2 * Math.PI * bin * t / N;
                    int index = Reflect(frame * Hop + t - pad);
                    double sample = samples[index] * window[t];
                    real += sample * Math.Cos(angle); imaginary -= sample * Math.Sin(angle);
                }
                int at = frame * bins + bin;
                double value = Math.Sqrt(real * real + imaginary * imaginary + Epsilon);
                magnitude[at] = value;
                for (int t = 0; t < N; t++)
                {
                    double angle = 2 * Math.PI * bin * t / N;
                    gradient[Reflect(frame * Hop + t - pad)] += upstream[at] / value
                        * (real * Math.Cos(angle) - imaginary * Math.Sin(angle)) * window[t];
                }
            }
        return (magnitude, gradient, upstream);
        static int Reflect(int index) => index < 0 ? -index : index >= Length ? 2 * Length - 2 - index : index;
    }
    private static void Close(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(double.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= tolerance * (1 + Math.Abs(expected[i])),
                $"[{i}] actual={actual[i]:G17} expected={expected[i]:G17}");
    }
}
