using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Pitch;

/// <summary>
/// FCPE, the fast context-based pitch estimator (torchfcpe's <c>CFNaiveMelPE</c> with its bundled
/// <c>fcpe_c_v001.pt</c>), as SpiritLM Expressive extracts F0 for its pitch tokens. A 128-bin log-mel spectrogram (16 kHz,
/// 1024-point STFT, 160-sample hop) goes through two 3-tap convolutions (group norm and leaky ReLU between them), six
/// conformer convolution modules with residuals (the bundled model is <c>conv_only</c>: no attention), a layer norm and a
/// weight-normalized linear map to 360 sigmoid saliences over a cent grid from 32.70 Hz to 1975.5 Hz.
/// </summary>
/// <remarks>
/// <para>The F0 of a frame is the salience-weighted mean of the cents in a 9-bin window around the most salient bin
/// (<c>local_argmax</c>; the window clamped at the grid's ends, repeating the end bin), <c>10 · 2^(cents / 1200)</c> Hz;
/// a frame whose top salience is at most the threshold is unvoiced (0 Hz).</para>
/// <para>The spectrogram is torchfcpe's <c>MelModule</c>: the waveform padded by 432 samples on each side (reflected, or with
/// zeros when the right pad would exceed it), a periodic Hann window, magnitude <c>√(re² + im² + 10⁻⁹)</c>, librosa's
/// Slaney mel bank to 8 kHz and <c>log(max(·, 10⁻⁵))</c>; <c>samples / 160 + 1</c> frames, the last repeated.</para>
/// </remarks>
internal sealed class FcpePitchEstimator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly double[,] _melBasis;

    public const int SampleRate = 16000, FftSize = 1024, Hop = 160;

    internal sealed record ConvModule(LayerNormalizationLayer<T> Norm, NormedConv1DLayer<T> Pointwise1, NormedConv1DLayer<T> Depthwise, NormedConv1DLayer<T> Pointwise2);

    public FcpePitchEstimator(IEngine engine, List<LayerBase<T>> layers, int mels = 128, int hidden = 512, int layerCount = 6, int bins = 360,
        double f0Min = 32.70, double f0Max = 1975.5)
    {
        _engine = engine;
        Mels = mels;
        Hidden = hidden;
        Bins = bins;
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int groups = 1) =>
            Own(new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, 1, groups, kernel / 2, false, ConvolutionNormalization.None));
        InputConv1 = Conv(mels, hidden, 3);
        InputNorm = Own(new GroupNormalizationLayer<T>(4, hidden, 1e-5));
        InputConv2 = Conv(hidden, hidden, 3);
        int inner = hidden * 2;
        for (int i = 0; i < layerCount; i++)
            Modules.Add(new ConvModule(Own(new LayerNormalizationLayer<T>(hidden, 1e-5)), Conv(hidden, inner * 2, 1), Conv(inner, inner, 31, inner), Conv(inner, hidden, 1)));
        OutputNorm = Own(new LayerNormalizationLayer<T>(hidden, 1e-5));
        Output = Own(new DenseLayer<T>(bins, (IActivationFunction<T>)new IdentityActivation<T>()));
        // torch.linspace(cent(f0_min), cent(f0_max), bins); a checkpoint's float32 table replaces it.
        double low = Cents(f0Min), high = Cents(f0Max);
        CentTable = new double[bins];
        for (int i = 0; i < bins; i++) CentTable[i] = low + (high - low) * i / (bins - 1);
        _melBasis = MelFilterbanks.Slaney(SampleRate, FftSize, mels, 0, 8000);
    }

    public int Mels { get; }
    public int Hidden { get; }
    public int Bins { get; }
    public NormedConv1DLayer<T> InputConv1 { get; }
    public GroupNormalizationLayer<T> InputNorm { get; }
    public NormedConv1DLayer<T> InputConv2 { get; }
    public List<ConvModule> Modules { get; } = new();
    public LayerNormalizationLayer<T> OutputNorm { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>The cents of each salience bin (the checkpoint's <c>cent_table</c>).</summary>
    public double[] CentTable { get; private set; }

    public static double Cents(double hz) => 1200.0 * Math.Log(hz / 10.0, 2);

    /// <summary>Declares the cent grid as model state: a checkpoint's float32 table replaces the computed one, and no
    /// parameter vector carries it.</summary>
    public void DeclareCentTable(AiDotNet.Models.ModelStateRegistry<T> state, string name) =>
        state.DeclareDoubleVector(name, () => new Vector<double>(CentTable), table =>
        {
            if (table is null) return;
            if (table.Length != Bins) throw new InvalidDataException($"State '{name}' holds {table.Length} cents for {Bins} bins.");
            CentTable = table.ToArray();
        });

    /// <summary>The log-mel spectrogram <c>[frames, mels]</c> of 16 kHz audio (torchfcpe's <c>Wav2MelModule</c>).</summary>
    public double[,] LogMel(IReadOnlyList<double> audio)
    {
        int length = audio.Count;
        if (length == 0) throw new ArgumentException("The audio is empty.", nameof(audio));
        int padLeft = (FftSize - Hop) / 2, padRight = Math.Max((FftSize - Hop + 1) / 2, FftSize - length - padLeft);
        bool reflect = padRight < length;
        double Sample(int i)
        {
            int j = i - padLeft;
            if (j >= 0 && j < length) return audio[j];
            if (!reflect) return 0;
            if (j < 0) j = -j;
            if (j >= length) j = 2 * (length - 1) - j;
            return audio[j];
        }
        int padded = length + padLeft + padRight, stftFrames = (padded - FftSize) / Hop + 1, frames = length / Hop + 1, bins = FftSize / 2 + 1;
        var window = new double[FftSize];
        for (int n = 0; n < FftSize; n++) window[n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / FftSize);
        var fft = new FastFourierTransform<double>();
        var mel = new double[frames, Mels];
        var magnitude = new double[bins];
        int computed = Math.Min(stftFrames, frames);
        for (int f = 0; f < computed; f++)
        {
            var frame = new Vector<double>(FftSize);
            for (int n = 0; n < FftSize; n++) frame[n] = Sample(f * Hop + n) * window[n];
            var spectrum = fft.Forward(frame);
            for (int k = 0; k < bins; k++)
            {
                double re = spectrum[k].Real, im = spectrum[k].Imaginary;
                magnitude[k] = Math.Sqrt(re * re + im * im + 1e-9);
            }
            for (int m = 0; m < Mels; m++)
            {
                double sum = 0;
                for (int k = 0; k < bins; k++) sum += _melBasis[m, k] * magnitude[k];
                mel[f, m] = Math.Log(Math.Max(sum, 1e-5));
            }
        }
        for (int f = computed; f < frames; f++)
            for (int m = 0; m < Mels; m++) mel[f, m] = mel[computed - 1, m];
        return mel;
    }

    /// <summary>The sigmoid saliences <c>[frames, bins]</c> of a log-mel spectrogram <c>[frames, mels]</c>.</summary>
    public Tensor<T> Saliences(double[,] mel)
    {
        int frames = mel.GetLength(0);
        var rows = new Tensor<T>(new[] { frames, Mels });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < Mels; m++) rows[f, m] = NumOps.FromDouble(mel[f, m]);
        Tensor<T> Channels(Tensor<T> r) => _engine.Reshape(_engine.TensorTranspose(r), new[] { 1, r.Shape[1], frames });
        Tensor<T> Rows(Tensor<T> c) => _engine.TensorTranspose(_engine.Reshape(c, new[] { c.Shape[1], frames }));
        var h = InputConv1.Forward(Channels(rows));
        h = _engine.Reshape(InputNorm.Forward(_engine.Reshape(h, new[] { 1, Hidden, frames, 1 })), new[] { 1, Hidden, frames });
        h = InputConv2.Forward(_engine.LeakyReLU(h, NumOps.FromDouble(0.01)));
        var x = Rows(h);
        foreach (var module in Modules)
        {
            var c = module.Pointwise1.Forward(Channels(module.Norm.Forward(x)));
            int inner = c.Shape[1] / 2;
            c = _engine.TensorMultiply(_engine.TensorSlice(c, new[] { 0, 0, 0 }, new[] { 1, inner, frames }),
                _engine.Sigmoid(_engine.TensorSlice(c, new[] { 0, inner, 0 }, new[] { 1, inner, frames })));
            c = module.Depthwise.Forward(c);
            c = _engine.TensorMultiply(c, _engine.Sigmoid(c));
            x = _engine.TensorAdd(x, Rows(module.Pointwise2.Forward(c)));
        }
        return _engine.Sigmoid(Output.Forward(OutputNorm.Forward(x)));
    }

    /// <summary>The F0 in Hz of each spectrogram frame (0 when unvoiced), by the local-argmax decoder.</summary>
    public double[] F0(IReadOnlyList<double> audio, double threshold = 0.05)
    {
        Tensor<T> salience;
        using (new NoGradScope<T>()) salience = Saliences(LogMel(audio));
        int frames = salience.Shape[0];
        var f0 = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            int best = 0;
            double confident = NumOps.ToDouble(salience[f, 0]);
            for (int b = 1; b < Bins; b++)
            {
                double v = NumOps.ToDouble(salience[f, b]);
                if (v > confident) { confident = v; best = b; }
            }
            if (confident <= threshold) { f0[f] = 0; continue; }
            double weighted = 0, total = 0;
            for (int o = -4; o <= 4; o++)
            {
                int b = Math.Min(Bins - 1, Math.Max(0, best + o));
                double y = NumOps.ToDouble(salience[f, b]);
                weighted += CentTable[b] * y;
                total += y;
            }
            f0[f] = 10.0 * Math.Pow(2.0, weighted / total / 1200.0);
        }
        return f0;
    }

    /// <summary>
    /// SpiritLM's F0 and voicing (<c>FCPEF0Extractor.compute_f0_uv</c>): FCPE's F0, frames under <paramref name="f0Min"/> Hz
    /// unvoiced and filled by linear interpolation between voiced frames (the ends held), capped at
    /// <paramref name="f0Max"/>, both brought by nearest-neighbour interpolation to <c>samples / hop + 1</c> frames.
    /// </summary>
    /// <remarks>Audio with no voiced frame keeps 0 Hz throughout (the release's interpolation fails on it).</remarks>
    public (double[] F0, double[] Voiced) SpiritLmF0(IReadOnlyList<double> audio, int hop = 80, double threshold = 0.05, double f0Min = 50, double f0Max = 1100)
    {
        var f0 = F0(audio, threshold);
        int frames = f0.Length;
        var unvoiced = new bool[frames];
        var voicedIndex = new List<int>();
        for (int f = 0; f < frames; f++)
        {
            unvoiced[f] = f0[f] < f0Min;
            if (!unvoiced[f]) voicedIndex.Add(f);
        }
        var filled = (double[])f0.Clone();
        if (voicedIndex.Count > 0)
        {
            // torch_interp: searchsorted (left) into the voiced frames, the ends held.
            for (int f = 0; f < frames; f++)
            {
                if (!unvoiced[f]) continue;
                int right = voicedIndex.BinarySearch(f);
                if (right < 0) right = ~right;
                if (right >= voicedIndex.Count) { filled[f] = f0[voicedIndex[^1]]; continue; }
                if (right == 0) { filled[f] = f0[voicedIndex[0]]; continue; }
                int xl = voicedIndex[right - 1], xr = voicedIndex[right];
                filled[f] = f0[xl] + (f - xl) * (f0[xr] - f0[xl]) / (xr - xl);
            }
        }
        for (int f = 0; f < frames; f++) if (filled[f] > f0Max) filled[f] = f0Max;
        int target = audio.Count / hop + 1;
        var outF0 = new double[target];
        var voiced = new double[target];
        for (int t = 0; t < target; t++)
        {
            int source = NearestIndex(t, frames, target);
            outF0[t] = filled[source];
            voiced[t] = unvoiced[source] ? 0 : 1;
        }
        return (outF0, voiced);
    }

    /// <summary>torch's nearest-neighbour source index (<c>F.interpolate(mode="nearest")</c> with a size: a float32 scale).</summary>
    internal static int NearestIndex(int output, int inputSize, int outputSize)
    {
        if (outputSize == inputSize) return output;
        if (outputSize == 2 * inputSize) return output >> 1;
        float scale = (float)inputSize / outputSize;
        return Math.Min((int)Math.Floor(output * scale), inputSize - 1);
    }

    /// <summary>Loads torchfcpe's state dict (<c>input_stack</c>, <c>net.encoder_layers.{i}.conformer.net</c>, <c>norm</c>,
    /// the weight-normalized <c>output_proj</c> and the <c>cent_table</c> buffer).</summary>
    public void LoadTorchWeights(Func<string, double[]?> take)
    {
        double[] Read(string name) => take(name) ?? throw new InvalidDataException($"The FCPE checkpoint has no tensor '{name}'.");
        InputConv1.LoadTorchWeights(Read("input_stack.0.weight"), null, Read("input_stack.0.bias"));
        var gamma = InputNorm.GetGammaTensor();
        var beta = InputNorm.GetBetaTensor();
        var g = Read("input_stack.1.weight");
        var b = Read("input_stack.1.bias");
        for (int i = 0; i < Hidden; i++) { gamma[i] = NumOps.FromDouble(g[i]); beta[i] = NumOps.FromDouble(b[i]); }
        _engine.InvalidatePersistentTensor(gamma);
        _engine.InvalidatePersistentTensor(beta);
        InputConv2.LoadTorchWeights(Read("input_stack.3.weight"), null, Read("input_stack.3.bias"));
        for (int i = 0; i < Modules.Count; i++)
        {
            string p = $"net.encoder_layers.{i}.conformer.net";
            var m = Modules[i];
            TorchParameters.LayerNorm(_engine, m.Norm, Read($"{p}.0.weight"), Read($"{p}.0.bias"));
            m.Pointwise1.LoadTorchWeights(Read($"{p}.2.weight"), null, Read($"{p}.2.bias"));
            m.Depthwise.LoadTorchWeights(Read($"{p}.4.conv.weight"), null, Read($"{p}.4.conv.bias"));
            m.Pointwise2.LoadTorchWeights(Read($"{p}.6.weight"), null, Read($"{p}.6.bias"));
        }
        TorchParameters.LayerNorm(_engine, OutputNorm, Read("norm.weight"), Read("norm.bias"));
        // weight = g · v / ‖v‖ per output row.
        // The bundled checkpoint has weight_g/weight_v; newer torch saves the parametrization.
        var v = take("output_proj.weight_v") ?? Read("output_proj.parametrizations.weight.original1");
        var gain = take("output_proj.weight_g") ?? Read("output_proj.parametrizations.weight.original0");
        var weight = new double[v.Length];
        for (int o = 0; o < Bins; o++)
        {
            double norm = 0;
            for (int i = 0; i < Hidden; i++) norm += v[o * Hidden + i] * v[o * Hidden + i];
            norm = Math.Sqrt(norm);
            for (int i = 0; i < Hidden; i++) weight[o * Hidden + i] = gain[o] * v[o * Hidden + i] / norm;
        }
        TorchParameters.Linear(_engine, Output, Hidden, Bins, weight, Read("output_proj.bias"));
        CentTable = Read("cent_table");
    }
}
