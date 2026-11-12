using AiDotNet.Attributes;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.AudioGen;

public partial class AudioGenModel<T>
{
    /// <summary>EnCodec latent dimension D (Défossez et al. 2022, §4.1).</summary>
    private int CodecLatentDim => _options.EncodecDimension;

    /// <summary>
    /// SEANet base channel count C. AudioGen's own 16 kHz compression model
    /// (facebook/audiogen-medium, <c>seanet.n_filters: 64</c>) uses 64, so its first decoder
    /// convolution is 128 -> 1024 channels; the EnCodec paper's general models use 32.
    /// </summary>
    private int CodecBaseChannels => _options.EncodecFilters;

    /// <summary>LSTM layers in the SEANet decoder (Défossez et al. 2022, §3.1).</summary>
    private int CodecLstmLayers => _options.EncodecLstmLayers;

    /// <summary>Kernel of the first and last decoder convolutions.</summary>
    private const int CodecOuterKernel = 7;

    /// <summary>Kernel of the first convolution in each residual unit.</summary>
    private const int CodecResidualKernel = 3;

    /// <summary>EnCodec token rate in frames per second (16 kHz / hop 320).</summary>
    private const int CodecTokensPerSecond = 50;

    /// <summary>The paper's upsampling ratios, applied in this order when they divide the hop.</summary>
    private static readonly int[] PaperUpsampleRatios = { 8, 5, 4, 2 };

    /// <summary>
    /// Every EnCodec decoder weight, in the order <see cref="InitializeCodecDecoder"/> builds them.
    /// </summary>
    /// <remarks>
    /// AudioGen decodes with a pretrained EnCodec that stays frozen while the language model trains
    /// (Kreuk et al. 2022, §3.1), so these are frozen parameters: saved, loaded and cloned with the
    /// model, never updated by an optimizer.
    /// </remarks>
    [FrozenParameter]
    private List<Tensor<T>> _codecWeights = new List<Tensor<T>>();

    /// <summary>The audiocraft state-dict name of each entry of <see cref="_codecWeights"/>.</summary>
    private List<string> _codecWeightNames = new List<string>();

    /// <summary>The frozen EnCodec decoder weights, in build order (for persistence checks).</summary>
    internal IReadOnlyList<Tensor<T>> CodecDecoderWeights => _codecWeights;

    /// <summary>Decodes token ids with the native EnCodec decoder (no sampling involved).</summary>
    internal Tensor<T> DecodeCodecTokens(Tensor<T> audioCodes, double durationSeconds)
        => DecodeAudioNative(audioCodes, durationSeconds);

    /// <summary>Upsampling ratio of each decoder block; their product is the hop.</summary>
    private int[] _codecRatios = Array.Empty<int>();

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

    /// <summary>
    /// Builds the EnCodec decoder weights for the configured sample rate, codebooks and channels,
    /// then loads pretrained values from <see cref="AudioGenOptions.EncodecWeightsPath"/> when given.
    /// </summary>
    /// <remarks>
    /// Without pretrained weights every tensor is drawn uniformly from
    /// <c>[-1/sqrt(fanIn), 1/sqrt(fanIn)]</c> (PyTorch's convolution and LSTM default), seeded from
    /// the model's layer-initialization scope so a seeded model is reproducible.
    /// </remarks>
    private void InitializeCodecDecoder()
    {
        int hop = Math.Max(1, (int)Math.Round((double)_sampleRate / CodecTokensPerSecond));
        _codecRatios = CodecUpsampleRatios(hop);
        _codecWeights = new List<Tensor<T>>();
        _codecWeightNames = new List<string>();
        var random = LayerInitializationSeedScope.NextRandom();

        void Add(string name, int[] shape, int fanIn)
        {
            T bound = NumOps.FromDouble(1.0 / Math.Sqrt(Math.Max(1, fanIn)));
            _codecWeights.Add(Engine.TensorRandomUniformRange(shape, NumOps.Negate(bound), bound, random.Next()));
            _codecWeightNames.Add(name);
        }

        for (int q = 0; q < _numCodebooks; q++)
            Add($"{CodebookPrefix}{q}{CodebookSuffix}", new[] { _codebookSize, CodecLatentDim }, CodecLatentDim);

        // SEANetDecoder.model: [0] conv, [1] LSTM, then per ratio ELU / ConvTranspose / residual block,
        // then ELU and the output conv.
        int width = CodecBaseChannels << _codecRatios.Length;
        Add("decoder.model.0.conv.conv.weight", new[] { width, CodecLatentDim, CodecOuterKernel }, CodecLatentDim * CodecOuterKernel);
        Add("decoder.model.0.conv.conv.bias", new[] { width }, CodecLatentDim * CodecOuterKernel);
        for (int l = 0; l < CodecLstmLayers; l++)
        {
            Add($"decoder.model.1.lstm.weight_ih_l{l}", new[] { 4 * width, width }, width);
            Add($"decoder.model.1.lstm.weight_hh_l{l}", new[] { 4 * width, width }, width);
            Add($"decoder.model.1.lstm.bias_ih_l{l}", new[] { 4 * width }, width);
            Add($"decoder.model.1.lstm.bias_hh_l{l}", new[] { 4 * width }, width);
        }

        int module = 2;
        foreach (int ratio in _codecRatios)
        {
            int half = width / 2;
            int hidden = Math.Max(1, half / 2);
            Add($"decoder.model.{module + 1}.convtr.convtr.weight", new[] { width, half, 2 * ratio }, half * 2 * ratio);
            Add($"decoder.model.{module + 1}.convtr.convtr.bias", new[] { half }, half * 2 * ratio);
            Add($"decoder.model.{module + 2}.block.1.conv.conv.weight", new[] { hidden, half, CodecResidualKernel }, half * CodecResidualKernel);
            Add($"decoder.model.{module + 2}.block.1.conv.conv.bias", new[] { hidden }, half * CodecResidualKernel);
            Add($"decoder.model.{module + 2}.block.3.conv.conv.weight", new[] { half, hidden, 1 }, hidden);
            Add($"decoder.model.{module + 2}.block.3.conv.conv.bias", new[] { half }, hidden);
            width = half;
            module += 3;
        }

        int channels = Math.Max(1, _channels);
        Add($"decoder.model.{module + 1}.conv.conv.weight", new[] { channels, width, CodecOuterKernel }, width * CodecOuterKernel);
        Add($"decoder.model.{module + 1}.conv.conv.bias", new[] { channels }, width * CodecOuterKernel);

        if (_options.EncodecWeightsPath is string path && path.Length > 0)
            LoadCodecWeights(path);
    }

    /// <summary>
    /// Replaces the decoder weights with a pretrained EnCodec checkpoint in safetensors format.
    /// </summary>
    /// <remarks>
    /// Names follow audiocraft's compression-model state dict, verified against the
    /// <c>best_state</c> of facebook/audiogen-medium <c>compression_state_dict.bin</c> (convert it to
    /// safetensors without the <c>best_state.</c> prefix): <c>quantizer.vq.layers.{q}._codebook.embed</c>
    /// [2048, 128], <c>decoder.model.0.conv.conv</c>, <c>decoder.model.1.lstm.{weight,bias}_{ih,hh}_l{0,1}</c>,
    /// <c>decoder.model.{3,6,9,12}.convtr.convtr</c>, <c>decoder.model.{4,7,10,13}.block.{1,3}.conv.conv</c>
    /// and <c>decoder.model.15.conv.conv</c>. That checkpoint stores convolutions weight-normalized
    /// (<c>weight_g</c> [d0, 1, 1] and <c>weight_v</c>); they are folded into the plain weight.
    /// Hugging Face transformers' EnCodec naming (<c>decoder.layers.*</c>) is not supported.
    /// </remarks>
    private void LoadCodecWeights(string path)
    {
        var checkpoint = new SafeTensorsLoader<T>().Load(path);
        for (int i = 0; i < _codecWeights.Count; i++)
        {
            string name = _codecWeightNames[i];
            Tensor<T> value;
            if (checkpoint.TryGetValue(name, out var direct))
            {
                value = direct;
            }
            else if (name.EndsWith(".weight", StringComparison.Ordinal)
                && checkpoint.TryGetValue(name + "_g", out var gain)
                && checkpoint.TryGetValue(name + "_v", out var direction))
            {
                value = WeightNormalized(gain, direction);
            }
            else
            {
                throw new InvalidOperationException($"EnCodec checkpoint '{path}' has no tensor '{name}'.");
            }

            var expected = _codecWeights[i].Shape.ToArray();
            if (!expected.SequenceEqual(value.Shape.ToArray()))
                throw new InvalidOperationException(
                    $"EnCodec tensor '{name}' has shape [{string.Join(", ", value.Shape.ToArray())}]; " +
                    $"this model needs [{string.Join(", ", expected)}].");
            _codecWeights[i] = value;
        }

        // A checkpoint with more codebooks than NumCodebooks would load, and the decoder would silently use only the
        // first ones, changing every decoded sample: require the configured count to match the checkpoint's.
        var extraCodebooks = checkpoint.Keys.Select(CodebookIndex).Where(q => q >= _numCodebooks).ToList();
        if (extraCodebooks.Count > 0)
            throw new InvalidOperationException(
                $"EnCodec checkpoint '{path}' has codebook {extraCodebooks.Min()} ('{CodebookPrefix}{extraCodebooks.Min()}" +
                $"{CodebookSuffix}'), but this model is configured with {_numCodebooks} codebooks; set NumCodebooks to " +
                "the checkpoint's codebook count.");
    }

    private const string CodebookPrefix = "quantizer.vq.layers.";
    private const string CodebookSuffix = "._codebook.embed";

    /// <summary>The codebook index of a <c>quantizer.vq.layers.{q}._codebook.embed</c> name, or -1 for any other name.</summary>
    private static int CodebookIndex(string name)
    {
        if (!name.StartsWith(CodebookPrefix, StringComparison.Ordinal) || !name.EndsWith(CodebookSuffix, StringComparison.Ordinal)
            || name.Length <= CodebookPrefix.Length + CodebookSuffix.Length)
            return -1;
        string index = name.Substring(CodebookPrefix.Length, name.Length - CodebookPrefix.Length - CodebookSuffix.Length);
        return int.TryParse(index, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture,
            out int q) ? q : -1;
    }

    /// <summary><c>g * v / ||v||</c>, the norm taken over every axis but the first (weight_norm, dim 0).</summary>
    private Tensor<T> WeightNormalized(Tensor<T> gain, Tensor<T> direction)
    {
        var shape = direction.Shape.ToArray();
        var axes = Enumerable.Range(1, shape.Length - 1).ToArray();
        var norm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(direction), axes, keepDims: true));
        var scale = Engine.TensorDivide(Engine.Reshape(gain, norm.Shape.ToArray()), norm);
        return Engine.TensorMultiply(direction, Engine.TensorBroadcastTo(scale, shape));
    }

    /// <summary>Conv1d with audiocraft's non-causal reflection padding, including its short-input rule.</summary>
    private Tensor<T> CodecConv(Tensor<T> x, Tensor<T> weight, Tensor<T> bias)
    {
        int padding = (weight.Shape[2] - 1) / 2;
        if (padding > 0)
        {
            int length = x.Shape[2];
            // audiocraft.modules.conv.pad1d temporarily appends zeros before reflecting short inputs.
            int extra = Math.Max(0, padding - length + 1);
            if (extra > 0)
                x = Engine.PadNd(x, new[] { 0, extra }, AiDotNet.Tensors.Engines.PadMode.Constant, NumOps.Zero);
            x = Engine.PadNd<T>(x, new[] { padding, padding }, AiDotNet.Tensors.Engines.PadMode.Reflect, NumOps.Zero);
            if (extra > 0)
                x = Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { x.Shape[0], x.Shape[1], length + 2 * padding });
        }
        var y = Engine.Conv1D(x, weight, 1, 0, 1);
        return Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, bias.Length, 1 }), y.Shape.ToArray()));
    }

    /// <summary>One PyTorch-convention LSTM layer (gates i, f, g, o) over a <c>[T, H]</c> sequence.</summary>
    private Tensor<T> CodecLstm(Tensor<T> sequence, Tensor<T> weightIh, Tensor<T> weightHh, Tensor<T> biasIh, Tensor<T> biasHh)
    {
        int steps = sequence.Shape[0];
        int hiddenSize = weightHh.Shape[1];
        var bias = Engine.Reshape(Engine.TensorAdd(biasIh, biasHh), new[] { 1, 4 * hiddenSize });
        var inputGates = Engine.TensorAdd(
            Engine.TensorMatMul(sequence, Engine.TensorTranspose(weightIh)),
            Engine.TensorBroadcastTo(bias, new[] { steps, 4 * hiddenSize }));
        var recurrent = Engine.TensorTranspose(weightHh);
        var h = new Tensor<T>(new[] { 1, hiddenSize });
        var c = new Tensor<T>(new[] { 1, hiddenSize });
        var outputs = new Tensor<T>[steps];
        for (int t = 0; t < steps; t++)
        {
            var gates = Engine.TensorAdd(
                Engine.TensorSlice(inputGates, new[] { t, 0 }, new[] { 1, 4 * hiddenSize }),
                Engine.TensorMatMul(h, recurrent));
            var input = Engine.TensorSigmoid(Engine.TensorSlice(gates, new[] { 0, 0 }, new[] { 1, hiddenSize }));
            var forget = Engine.TensorSigmoid(Engine.TensorSlice(gates, new[] { 0, hiddenSize }, new[] { 1, hiddenSize }));
            var cell = Engine.TensorTanh(Engine.TensorSlice(gates, new[] { 0, 2 * hiddenSize }, new[] { 1, hiddenSize }));
            var output = Engine.TensorSigmoid(Engine.TensorSlice(gates, new[] { 0, 3 * hiddenSize }, new[] { 1, hiddenSize }));
            c = Engine.TensorAdd(Engine.TensorMultiply(forget, c), Engine.TensorMultiply(input, cell));
            h = Engine.TensorMultiply(output, Engine.TensorTanh(c));
            outputs[t] = h;
        }

        return Engine.Concat(outputs, 0);
    }

    /// <summary>
    /// Decodes generated EnCodec tokens <c>[1, numCodebooks, numTokens]</c> (or the same values flat,
    /// codebook-major) to a waveform of <c>durationSeconds * sampleRate</c> samples: <c>[samples]</c>
    /// for mono, <c>[channels, samples]</c> otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// AudioGen (Kreuk et al. 2022, §3.1) generates the discrete tokens of a pretrained, frozen
    /// EnCodec model and decodes them with that model's decoder. This is the EnCodec decoder
    /// (Défossez et al. 2022, §3.1 and Figure 1): residual vector quantization is undone by summing
    /// each codebook's embedding of its token, then a SEANet decoder maps the D-dimensional latents
    /// to audio:
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
    /// AudioGen's 16 kHz compression model uses D = 128, C = 64, ratios (8, 5, 4, 2), a hop of 320
    /// samples, 50 tokens per second and 4 codebooks of 2048 entries. The hop here is the sample rate divided by the 50 Hz token rate, split
    /// into those ratios where they divide it. Pretrained weights load from
    /// <see cref="AudioGenOptions.EncodecWeightsPath"/>; in ONNX mode the supplied audio decoder is
    /// used instead.
    /// </para>
    /// </remarks>
    private Tensor<T> DecodeAudioNative(Tensor<T> audioCodes, double durationSeconds)
    {
        if (_codecWeights.Count == 0)
            throw new InvalidOperationException("The native EnCodec decoder is not initialized.");

        int channels = Math.Max(1, _channels);
        int targetSamples = (int)(durationSeconds * _sampleRate);
        int[] outputShape = channels == 1 ? new[] { Math.Max(0, targetSamples) } : new[] { channels, Math.Max(0, targetSamples) };
        int numTokens = audioCodes.Shape.Length >= 3 ? audioCodes.Shape[2] : audioCodes.Length / Math.Max(1, _numCodebooks);
        if (numTokens <= 0 || targetSamples <= 0 || _numCodebooks <= 0)
            return new Tensor<T>(outputShape);

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

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
        int w = 0;
        Tensor<T>? latent = null;
        for (int q = 0; q < _numCodebooks; q++)
        {
            var ids = Engine.Reshape(Engine.TensorSlice(codes, new[] { q, 0 }, new[] { 1, numTokens }), new[] { numTokens }).Cast<int>();
            var embedded = Engine.TensorIndexSelect(_codecWeights[w++], ids, 0); // [numTokens, D]
            latent = latent is null ? embedded : Engine.TensorAdd(latent, embedded);
        }

        if (latent is null)
            return new Tensor<T>(outputShape);

        // [numTokens, D] -> [1, D, numTokens]
        var x = CodecConv(Engine.Reshape(Engine.TensorTranspose(latent), new[] { 1, CodecLatentDim, numTokens }), _codecWeights[w], _codecWeights[w + 1]);
        w += 2;

        // LSTM over time with a skip connection: [1, C, T] <-> [T, C]
        int width = x.Shape[1];
        var sequence = Engine.TensorTranspose(Engine.Reshape(x, new[] { width, numTokens }));
        var recurrent = sequence;
        for (int l = 0; l < CodecLstmLayers; l++)
        {
            recurrent = CodecLstm(recurrent, _codecWeights[w], _codecWeights[w + 1], _codecWeights[w + 2], _codecWeights[w + 3]);
            w += 4;
        }
        x = Engine.Reshape(Engine.TensorTranspose(Engine.TensorAdd(recurrent, sequence)), new[] { 1, width, numTokens });

        int length = numTokens;
        foreach (int ratio in _codecRatios)
        {
            // ELU -> transposed conv (kernel 2r, stride r); keep the centre length * ratio samples.
            var upsampled = Engine.TensorConvTranspose1D(Engine.ELU(x), _codecWeights[w], ratio, 0, 0);
            var upBias = _codecWeights[w + 1];
            upsampled = Engine.TensorAdd(upsampled, Engine.TensorBroadcastTo(
                Engine.Reshape(upBias, new[] { 1, upBias.Length, 1 }), upsampled.Shape.ToArray()));
            int extra = upsampled.Shape[2] - length * ratio;
            int left = extra - extra / 2;
            length *= ratio;
            x = Engine.TensorSlice(upsampled, new[] { 0, 0, left }, new[] { 1, upsampled.Shape[1], length });

            // Residual unit: x + Conv1(ELU(Conv3(ELU(x))))
            var inner = CodecConv(Engine.ELU(x), _codecWeights[w + 2], _codecWeights[w + 3]);
            inner = CodecConv(Engine.ELU(inner), _codecWeights[w + 4], _codecWeights[w + 5]);
            x = Engine.TensorAdd(x, inner);
            w += 6;
        }

        var waveform = CodecConv(Engine.ELU(x), _codecWeights[w], _codecWeights[w + 1]); // [1, channels, numTokens * hop]

        // Fit to the requested duration: trim, or zero-pad a short decode.
        int kept = Math.Min(waveform.Shape[2], targetSamples);
        var fitted = Engine.Reshape(
            Engine.TensorSlice(waveform, new[] { 0, 0, 0 }, new[] { 1, channels, kept }), new[] { channels, kept });
        if (kept < targetSamples)
            fitted = Engine.Concat(new[] { fitted, new Tensor<T>(new[] { channels, targetSamples - kept }) }, 1);

        return channels == 1 ? Engine.Reshape(fitted, new[] { targetSamples }) : fitted;
    }
}
