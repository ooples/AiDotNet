using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.AudioGen;

public partial class AudioGenModel<T>
{
    /// <summary>EnCodec latent dimension D (Défossez et al. 2022, §4.1).</summary>
    private const int CodecLatentDim = 128;

    /// <summary>SEANet base channel count C (Défossez et al. 2022, §3.1).</summary>
    private const int CodecBaseChannels = 32;

    /// <summary>LSTM layers in the SEANet decoder (Défossez et al. 2022, §3.1).</summary>
    private const int CodecLstmLayers = 2;

    /// <summary>Kernel of the first and last decoder convolutions.</summary>
    private const int CodecOuterKernel = 7;

    /// <summary>Kernel of the first convolution in each residual unit.</summary>
    private const int CodecResidualKernel = 3;

    /// <summary>EnCodec token rate in frames per second (16 kHz / hop 320).</summary>
    private const int CodecTokensPerSecond = 50;

    /// <summary>The paper's upsampling ratios, applied in this order when they divide the hop.</summary>
    private static readonly int[] PaperUpsampleRatios = { 8, 5, 4, 2 };

    /// <summary>One embedding table per codebook: token id -> D-dimensional code vector.</summary>
    private List<EmbeddingLayer<T>> _codecCodebooks = new List<EmbeddingLayer<T>>();

    /// <summary>Conv1d(D -> C·2^B, kernel 7).</summary>
    private Conv1DLayer<T>? _codecInputConv;

    /// <summary>The decoder LSTM stack (applied with a skip connection).</summary>
    private List<LSTMLayer<T>> _codecLstm = new List<LSTMLayer<T>>();

    /// <summary>One transposed convolution per upsampling ratio.</summary>
    private List<Conv1DTransposeLayer<T>> _codecUpsample = new List<Conv1DTransposeLayer<T>>();

    /// <summary>Residual unit convolutions, two per upsampling block (kernel 3, then kernel 1).</summary>
    private List<Conv1DLayer<T>> _codecResidual = new List<Conv1DLayer<T>>();

    /// <summary>Final Conv1d(C -> channels, kernel 7).</summary>
    private Conv1DLayer<T>? _codecOutputConv;

    /// <summary>Upsampling ratio of each block; their product is the hop.</summary>
    private int[] _codecRatios = Array.Empty<int>();

    /// <summary>The ELU used before every decoder convolution.</summary>
    private readonly ELUActivation<T> _codecElu = new ELUActivation<T>();

    /// <summary>
    /// Splits <paramref name="hop"/> into upsampling ratios: each paper ratio that still divides it,
    /// in the paper's order, then the smallest prime factors of what remains.
    /// </summary>
    private static int[] CodecUpsampleRatios(int hop)
    {
        var ratios = new List<int>();
        int remaining = hop;
        foreach (int ratio in PaperUpsampleRatios)
        {
            if (remaining % ratio == 0)
            {
                ratios.Add(ratio);
                remaining /= ratio;
            }
        }

        for (int factor = 2; remaining > 1; factor++)
        {
            while (remaining % factor == 0)
            {
                ratios.Add(factor);
                remaining /= factor;
            }
        }

        return ratios.ToArray();
    }

    /// <summary>Builds the native EnCodec decoder for the configured sample rate and channel count.</summary>
    private void InitializeCodecDecoder()
    {
        int hop = Math.Max(1, (int)Math.Round((double)_sampleRate / CodecTokensPerSecond));
        _codecRatios = CodecUpsampleRatios(hop);

        _codecCodebooks = new List<EmbeddingLayer<T>>(_numCodebooks);
        for (int cb = 0; cb < _numCodebooks; cb++)
            _codecCodebooks.Add(new EmbeddingLayer<T>(_codebookSize, CodecLatentDim));

        int width = CodecBaseChannels << _codecRatios.Length;
        _codecInputConv = new Conv1DLayer<T>(CodecLatentDim, width, CodecOuterKernel);

        _codecLstm = new List<LSTMLayer<T>>(CodecLstmLayers);
        for (int l = 0; l < CodecLstmLayers; l++)
            _codecLstm.Add(new LSTMLayer<T>(width));

        _codecUpsample = new List<Conv1DTransposeLayer<T>>(_codecRatios.Length);
        _codecResidual = new List<Conv1DLayer<T>>(2 * _codecRatios.Length);
        foreach (int ratio in _codecRatios)
        {
            int half = width / 2;
            // Kernel 2r, stride r, no built-in trimming: the r extra samples are cut in DecodeAudioNative.
            _codecUpsample.Add(new Conv1DTransposeLayer<T>(width, half, 2 * ratio, ratio, 0));
            // Residual unit with compression 2: dim -> dim / 2 (kernel 3) -> dim (kernel 1).
            _codecResidual.Add(new Conv1DLayer<T>(half, Math.Max(1, half / 2), CodecResidualKernel));
            _codecResidual.Add(new Conv1DLayer<T>(Math.Max(1, half / 2), half, 1));
            width = half;
        }

        _codecOutputConv = new Conv1DLayer<T>(width, Math.Max(1, _channels), CodecOuterKernel);
    }

    /// <summary>
    /// Decodes generated EnCodec tokens <c>[1, numCodebooks, numTokens]</c> (or the same values flat,
    /// codebook-major) to a waveform of <c>durationSeconds * sampleRate</c> samples: <c>[samples]</c>
    /// for mono, <c>[channels, samples]</c> otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// AudioGen (Kreuk et al. 2022, §3.1) generates the discrete tokens of an EnCodec model and decodes
    /// them with that model's decoder. This is the EnCodec decoder (Défossez et al. 2022, §3.1 and
    /// Figure 1): residual vector quantization is undone by summing each codebook's learned embedding
    /// of its token, then a SEANet decoder maps the D-dimensional latent sequence to audio:
    /// </para>
    /// <list type="number">
    /// <item>Conv1d(D -> C·2^B, kernel 7).</item>
    /// <item>A two-layer LSTM over time with a skip connection (<c>y = LSTM(x) + x</c>).</item>
    /// <item>For each upsampling ratio r (B blocks): ELU, ConvTranspose1d(stride r, kernel 2r) halving
    /// the channels, then a residual unit: ELU, Conv1d(kernel 3) to half width, ELU, Conv1d(kernel 1)
    /// back, added to its input.</item>
    /// <item>ELU and a final Conv1d(C -> channels, kernel 7).</item>
    /// </list>
    /// <para>
    /// The paper's 16 kHz model uses D = 128, C = 32 and ratios (8, 5, 4, 2), i.e. a hop of 320 samples
    /// and 50 tokens per second. The hop here is the sample rate divided by the 50 Hz token rate, split
    /// into those ratios where they divide it. In native mode the decoder's weights are trainable model
    /// state like the language model's; in ONNX mode the supplied audio decoder is used instead.
    /// </para>
    /// </remarks>
    private Tensor<T> DecodeAudioNative(Tensor<T> audioCodes, double durationSeconds)
    {
        if (_codecInputConv is null || _codecOutputConv is null)
            throw new InvalidOperationException("The native EnCodec decoder is not initialized.");

        int channels = Math.Max(1, _channels);
        int targetSamples = (int)(durationSeconds * _sampleRate);
        int[] outputShape = channels == 1 ? new[] { Math.Max(0, targetSamples) } : new[] { channels, Math.Max(0, targetSamples) };
        int numTokens = audioCodes.Shape.Length >= 3 ? audioCodes.Shape[2] : audioCodes.Length / Math.Max(1, _numCodebooks);
        if (numTokens <= 0 || targetSamples <= 0 || _numCodebooks <= 0)
            return new Tensor<T>(outputShape);

        // Token ids as [numCodebooks, numTokens]
        Tensor<T> codes;
        if (audioCodes.Shape.Length >= 3)
        {
            if (audioCodes.Shape[1] < _numCodebooks)
                throw new ArgumentException(
                    $"Audio codes carry {audioCodes.Shape[1]} codebooks; the decoder needs {_numCodebooks}.", nameof(audioCodes));
            codes = Engine.Reshape(
                Engine.TensorSlice(audioCodes, new[] { 0, 0, 0 }, new[] { 1, _numCodebooks, numTokens }),
                new[] { _numCodebooks, numTokens });
        }
        else
        {
            var flat = Engine.Reshape(audioCodes, new[] { audioCodes.Length });
            codes = Engine.Reshape(
                Engine.TensorSlice(flat, new[] { 0 }, new[] { _numCodebooks * numTokens }),
                new[] { _numCodebooks, numTokens });
        }

        // Residual vector quantization: the latent is the sum of every codebook's code vector.
        Tensor<T>? latent = null;
        for (int cb = 0; cb < _numCodebooks; cb++)
        {
            var ids = Engine.TensorSlice(codes, new[] { cb, 0 }, new[] { 1, numTokens });
            var embedded = _codecCodebooks[cb].Forward(ids); // [1, numTokens, D]
            latent = latent is null ? embedded : Engine.TensorAdd(latent, embedded);
        }

        if (latent is null)
            return new Tensor<T>(outputShape);

        // [1, numTokens, D] -> [1, D, numTokens]
        var x = _codecInputConv.Forward(Engine.TensorPermute(latent, new[] { 0, 2, 1 }));

        // LSTM over time with a skip connection: [1, C, T] <-> [1, T, C]
        var sequence = Engine.TensorPermute(x, new[] { 0, 2, 1 });
        var recurrent = sequence;
        foreach (var lstm in _codecLstm)
            recurrent = lstm.Forward(recurrent);
        x = Engine.TensorPermute(Engine.TensorAdd(recurrent, sequence), new[] { 0, 2, 1 });

        int length = numTokens;
        for (int block = 0; block < _codecRatios.Length; block++)
        {
            int ratio = _codecRatios[block];

            // ELU -> transposed conv; keep the centre length * ratio samples (non-causal unpadding).
            var upsampled = _codecUpsample[block].Forward(_codecElu.Activate(x));
            int extra = upsampled.Shape[2] - length * ratio;
            int left = extra - extra / 2;
            length *= ratio;
            x = Engine.TensorSlice(upsampled, new[] { 0, 0, left }, new[] { upsampled.Shape[0], upsampled.Shape[1], length });

            // Residual unit: x + Conv1(ELU(Conv3(ELU(x))))
            var inner = _codecResidual[2 * block].Forward(_codecElu.Activate(x));
            inner = _codecResidual[2 * block + 1].Forward(_codecElu.Activate(inner));
            x = Engine.TensorAdd(x, inner);
        }

        var waveform = _codecOutputConv.Forward(_codecElu.Activate(x)); // [1, channels, numTokens * hop]

        // Fit to the requested duration: trim, or zero-pad a short decode.
        int produced = waveform.Shape[2];
        int kept = Math.Min(produced, targetSamples);
        var fitted = Engine.Reshape(
            Engine.TensorSlice(waveform, new[] { 0, 0, 0 }, new[] { 1, channels, kept }), new[] { channels, kept });
        if (kept < targetSamples)
            fitted = Engine.Concat(new[] { fitted, new Tensor<T>(new[] { channels, targetSamples - kept }) }, 1);

        return channels == 1 ? Engine.Reshape(fitted, new[] { targetSamples }) : fitted;
    }
}
