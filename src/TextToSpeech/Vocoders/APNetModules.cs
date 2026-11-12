using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The centred short-time Fourier transform on the gradient tape (<c>torch.stft(center=True)</c>: reflect padding of
/// n_fft / 2, a periodic Hann window of the window length centred in the FFT frame, or with no window a rectangular one
/// of the window length): the real and imaginary parts <c>[1, bins, frames]</c> of a waveform, frames = 1 + samples / hop.
/// </summary>
internal sealed class CenteredComplexStft<T>
{
    private readonly IEngine _engine;
    private readonly int _fft;
    private readonly int _hop;
    private readonly Tensor<T> _cos;
    private readonly Tensor<T> _sin;

    public CenteredComplexStft(IEngine engine, int fftSize, int hopSize, int windowSize, bool hannWindow = true)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        _engine = engine;
        _fft = fftSize;
        _hop = hopSize;
        int bins = fftSize / 2 + 1, offset = (fftSize - windowSize) / 2;
        _cos = new Tensor<T>(new[] { fftSize, bins });
        _sin = new Tensor<T>(new[] { fftSize, bins });
        for (int n = 0; n < windowSize; n++)
        {
            double w = hannWindow ? 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / windowSize) : 1.0;
            for (int k = 0; k < bins; k++)
            {
                double a = 2 * Math.PI * (offset + n) * k / fftSize;
                _cos[offset + n, k] = ops.FromDouble(w * Math.Cos(a));
                _sin[offset + n, k] = ops.FromDouble(-w * Math.Sin(a));
            }
        }
    }

    public int Bins => _fft / 2 + 1;

    /// <summary>The real and imaginary parts <c>[1, bins, 1 + samples / hop]</c> of <paramref name="audio"/>
    /// <c>[samples]</c>.</summary>
    public (Tensor<T> Re, Tensor<T> Im) Forward(Tensor<T> audio)
    {
        int length = audio.Length, pad = _fft / 2, frames = 1 + length / _hop;
        var index = new Tensor<int>(new[] { frames * _fft });
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < _fft; n++)
                index[f * _fft + n] = DifferentiableMel<T>.Reflect(f * _hop + n - pad, length);
        var framed = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(audio, new[] { length, 1 }), index, 0), new[] { frames, _fft });
        Tensor<T> Columns(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, Bins, frames });
        return (Columns(_engine.TensorMatMul(framed, _cos)), Columns(_engine.TensorMatMul(framed, _sin)));
    }
}

/// <summary>
/// APNet's frame-level spectrum predictor (Ai and Ling 2023, §III-A/B, Fig. 2–3; reference yangai520/APNet
/// <c>models.py</c>): a weight-normalized input convolution, a residual convolution network of P parallel blocks — each
/// Q subblocks of LReLU → dilated convolution → LReLU → convolution with a residual connection — whose outputs are
/// averaged and passed through LReLU, and one or more weight-normalized output convolutions to the FFT bins.
/// </summary>
/// <remarks>The ASP has one output (the log amplitude spectrum); the PSP two (the pseudo real and imaginary parts
/// whose two-argument arctangent is the phase, Eq. 5). The subblocks' slope is 0.1 and the slope after the average
/// PyTorch's default 0.01 (reference); the blocks and output convolutions start from N(0, 0.01)
/// (<c>init_weights</c>).</remarks>
internal sealed class ApNetPredictor<T>
{
    private const double SubblockSlope = 0.1;
    private const double OutputSlope = 0.01;
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly NormedConv1DLayer<T> _input;
    private readonly List<List<(NormedConv1DLayer<T> Dilated, NormedConv1DLayer<T> Plain)>> _blocks = new();
    private readonly List<NormedConv1DLayer<T>> _outputs = new();

    public ApNetPredictor(IEngine engine, Random initialization, int melChannels, int channels, int[] kernels, int[][] dilations,
        int inputKernel, int outputKernel, int bins, int outputCount, List<LayerBase<T>> owner)
    {
        _engine = engine;
        Func<double> normal = () =>
        {
            double u1 = 1.0 - initialization.NextDouble(), u2 = initialization.NextDouble();
            return 0.01 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        };
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int dilation, bool reinitialize)
        {
            var conv = new NormedConv1DLayer<T>(input, output, kernel, 1, dilation, 1, (kernel * dilation - dilation) / 2, false,
                ConvolutionNormalization.Weight);
            // init_weights touches only the weight; the bias keeps its default draw.
            if (reinitialize) conv.Reinitialize(normal, keepBias: true);
            owner.Add(conv);
            _layers.Add(conv);
            return conv;
        }
        _input = Conv(melChannels, channels, inputKernel, 1, false);
        for (int b = 0; b < kernels.Length; b++)
        {
            int k = kernels[b];
            var block = new List<(NormedConv1DLayer<T>, NormedConv1DLayer<T>)>();
            foreach (int d in dilations[b]) block.Add((Conv(channels, channels, k, d, true), Conv(channels, channels, k, 1, true)));
            _blocks.Add(block);
        }
        for (int i = 0; i < outputCount; i++) _outputs.Add(Conv(channels, bins, outputKernel, 1, true));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The output convolutions' maps <c>[1, bins, frames]</c> of a mel spectrogram <c>[1, mel, frames]</c>.</summary>
    public IReadOnlyList<Tensor<T>> Forward(Tensor<T> mel)
    {
        var x = _input.Forward(mel);
        Tensor<T>? sum = null;
        foreach (var block in _blocks)
        {
            var y = x;
            foreach (var (dilated, plain) in block)
            {
                var t = dilated.Forward(VocoderOps.LeakyRelu(_engine, y, SubblockSlope));
                t = plain.Forward(VocoderOps.LeakyRelu(_engine, t, SubblockSlope));
                y = _engine.TensorAdd(t, y);
            }
            sum = sum is null ? y : _engine.TensorAdd(sum, y);
        }
        var h = VocoderOps.LeakyRelu(_engine, _engine.TensorMultiplyScalar(sum!, MathHelper.GetNumericOperations<T>().FromDouble(1.0 / _blocks.Count)), OutputSlope);
        return _outputs.Select(o => o.Forward(h)).ToList();
    }
}

/// <summary>APNet's phase differences (§III-C, Eq. 13–21; reference <c>phase_loss</c>): along frequency (group delay)
/// and along time (phase time difference), as the reference's difference matrices compute them —
/// <c>D[0] = −P[0]</c>, <c>D[j] = P[j − 1] − P[j]</c>.</summary>
internal static class PhaseDifferences
{
    /// <summary>The differences of <c>[1, bins, frames]</c> along <paramref name="axis"/> (1: frequency, 2: time).</summary>
    public static Tensor<T> Along<T>(IEngine engine, Tensor<T> p, int axis)
    {
        int n = p.Shape[axis];
        var shape = (int[])p._shape.Clone();
        shape[axis] = 1;
        var zero = new Tensor<T>(shape);
        var start = new int[3];
        var length = (int[])p._shape.Clone();
        length[axis] = n - 1;
        var previous = engine.TensorConcatenate(new[] { zero, engine.TensorSlice(p, start, length) }, axis);
        return engine.TensorSubtract(previous, p);
    }
}
