using AiDotNet.Audio.Features;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Paraformer's streaming encoder as FunASR runs it on a recording (Gao et al. 2022; <c>ParaformerStreaming.infer_encoder</c>
/// with <c>WavFrontendOnline</c> and <c>SANMEncoderChunkOpt</c>), which Step-Audio's linguistic tokenizer reads: the audio in
/// strides of <c>chunk[1] · 960</c> samples; each stride's Kaldi filterbanks continued from the last (the unread samples
/// carried), stacked by LFR with a splice cache of the frames not yet stacked, normalized; scaled by √width, positions
/// continuing from the previous chunk, <c>chunk[0] + chunk[2]</c> frames of the previous input prepended; SAN-M blocks
/// attending also to up to <c>look-back · chunk[1]</c> earlier frames' keys and values (each chunk's minus its last
/// <c>chunk[2]</c>); the final norm; and the chunk's own frames kept.
/// </summary>
/// <remarks>The last stride is the final one (the remainder of the recording): LFR pads it with its last frame. Dither
/// is Kaldi's (Gaussian noise of amplitude <see cref="Dither"/> on each framed window, on the 16-bit scale), which the
/// release leaves at 1.0.</remarks>
internal sealed class StreamingParaformerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int SampleRate = 16000, FrameLength = 400, FrameShift = 160;
    private readonly IEngine _engine;
    private readonly KaldiFbank<T> _fbank;

    public StreamingParaformerEncoder(IEngine engine, List<LayerBase<T>> layers, SanmEncoderConfiguration configuration, Random initialization)
    {
        if (configuration.TpBlocks != 0) throw new ArgumentException("Paraformer's streaming encoder has no tp blocks.", nameof(configuration));
        _engine = engine;
        Encoder = new SanmEncoder<T>(engine, layers, configuration, initialization);
        _fbank = new KaldiFbank<T>(engine, SampleRate, configuration.Mels, WindowType.Hamming);
    }

    public SanmEncoder<T> Encoder { get; }

    /// <summary>The chunk sizes (lookback frames, chunk frames, lookahead frames) and attention look-back in chunks;
    /// Step-Audio's tokenizer uses [0, 4, 5] and 4.</summary>
    public int[] ChunkSize { get; set; } = { 0, 4, 5 };

    /// <inheritdoc cref="ChunkSize"/>
    public int LookBack { get; set; } = 4;

    /// <summary>Kaldi's dither amplitude on 16-bit samples (the release's 1.0); 0 is deterministic.</summary>
    public double Dither { get; set; } = 1.0;

    /// <summary>The generator the dither draws from.</summary>
    public Random DitherRandom { get; set; } = new Random(0);

    /// <summary>The encoder's frames <c>[n, width]</c> of a 16 kHz recording.</summary>
    public Tensor<T> Encode(IReadOnlyList<double> audio)
    {
        var c = Encoder.Configuration;
        int stride = ChunkSize[1] * 960, n = audio.Count / stride + 1, dim = c.InputSize, overlap = ChunkSize[0] + ChunkSize[2];
        var frontend = new FrontendState();
        var overlapRows = new Tensor<T>(new[] { overlap, dim });
        var caches = new (Tensor<T> Keys, Tensor<T> Values)?[Encoder.Blocks.Count];
        int positionStart = 0;
        var outputs = new List<Tensor<T>>();
        for (int k = 0; k < n; k++)
        {
            bool final = k == n - 1;
            int start = k * stride, count = Math.Max(0, Math.Min(stride, audio.Count - start));
            if (final && count == 0) break;
            var piece = new double[count];
            for (int i = 0; i < count; i++) piece[i] = audio[start + i];
            var speech = Frontend(piece, final, frontend);
            if (speech is null || speech.Shape[0] == 0)
            {
                if (final) break;
                throw new InvalidOperationException("A non-final stride gave no features.");
            }
            int length = speech.Shape[0];
            // forward_chunk: scaled, positioned, the previous input's last frames prepended.
            var x = Encoder.AddPositions(_engine.TensorMultiplyScalar(speech, NumOps.FromDouble(Math.Sqrt(c.Width))), positionStart);
            positionStart += length;
            x = _engine.TensorConcatenate(new[] { overlapRows, x }, 0);
            int total = x.Shape[0];
            overlapRows = _engine.TensorSlice(x, new[] { total - overlap, 0 }, new[] { overlap, dim });
            for (int b = 0; b < Encoder.Blocks.Count; b++)
            {
                var past = caches[b];
                var (output, keys, values) = Encoder.RunChunk(Encoder.Blocks[b], x, past?.Keys, past?.Values);
                // The stride's keys and values without its last chunk[2] frames, kept for look-back.
                int keep = keys.Shape[0] - ChunkSize[2];
                var strideKeys = _engine.TensorSlice(keys, new[] { 0, 0 }, new[] { Math.Max(0, keep), keys.Shape[1] });
                var strideValues = _engine.TensorSlice(values, new[] { 0, 0 }, new[] { Math.Max(0, keep), values.Shape[1] });
                if (past is null) caches[b] = (strideKeys, strideValues);
                else
                {
                    var allKeys = _engine.TensorConcatenate(new[] { past.Value.Keys, strideKeys }, 0);
                    var allValues = _engine.TensorConcatenate(new[] { past.Value.Values, strideValues }, 0);
                    int limit = LookBack * ChunkSize[1], rows = allKeys.Shape[0];
                    if (LookBack != -1 && rows > limit)
                    {
                        allKeys = _engine.TensorSlice(allKeys, new[] { rows - limit, 0 }, new[] { limit, allKeys.Shape[1] });
                        allValues = _engine.TensorSlice(allValues, new[] { rows - limit, 0 }, new[] { limit, allValues.Shape[1] });
                    }
                    caches[b] = (allKeys, allValues);
                }
                x = output;
            }
            x = Encoder.AfterNorm.Forward(x);
            outputs.Add(_engine.TensorSlice(x, new[] { total - length, 0 }, new[] { length, c.Width }));
            if (final) break;
        }
        return outputs.Count == 0 ? new Tensor<T>(new[] { 0, c.Width }) : _engine.TensorConcatenate(outputs.ToArray(), 0);
    }

    // ---------------------------------------------------------------- WavFrontendOnline

    private sealed class FrontendState
    {
        public double[] InputCache = Array.Empty<double>();
        public int ReserveWaveforms;
        public List<double[]>? SpliceCache;
    }

    // Python's a[start:stop] length for a sequence of `length`.
    private static int SliceLength(int length, int start, int stop)
    {
        int Clamp(int i) => i < 0 ? Math.Max(0, length + i) : Math.Min(length, i);
        return Math.Max(0, Clamp(stop) - Clamp(start));
    }

    private Tensor<T>? Frontend(double[] piece, bool final, FrontendState state)
    {
        var c = Encoder.Configuration;
        var input = state.InputCache.Concat(piece).ToArray();
        int frames = input.Length >= FrameLength ? (input.Length - FrameLength) / FrameShift + 1 : 0;
        int remainder = input.Length - frames * FrameShift;
        state.InputCache = input.Skip(remainder == 0 ? 0 : input.Length - remainder).ToArray();
        int lfrHalf = (c.LfrM - 1) / 2;
        if (frames > 0)
        {
            int waveformLength = (frames - 1) * FrameShift + FrameLength;
            var scaled = new double[input.Length];
            for (int i = 0; i < input.Length; i++) scaled[i] = input[i] * 32768.0;
            var fbank = _fbank.Forward(new Tensor<T>(new[] { scaled.Length }, new Vector<T>(scaled.Select(v => NumOps.FromDouble(v)).ToArray())),
                Dither, DitherRandom);
            var rows = new List<double[]>();
            for (int f = 0; f < frames; f++)
            {
                var row = new double[c.Mels];
                for (int m = 0; m < c.Mels; m++) row[m] = NumOps.ToDouble(fbank[f, m]);
                rows.Add(row);
            }
            int waveforms = state.ReserveWaveforms + waveformLength;
            state.SpliceCache ??= Enumerable.Repeat(rows[0], lfrHalf).Select(r => (double[])r.Clone()).ToList();
            if (frames + state.SpliceCache.Count >= c.LfrM)
            {
                var all = state.SpliceCache.Concat(rows).ToList();
                int fromWaveforms = (int)((waveforms - FrameLength) / (double)FrameShift + 1);
                int minus = state.ReserveWaveforms == 0 ? lfrHalf : 0;
                var (lfr, splice, spliceIndex) = Lfr(all, final);
                state.SpliceCache = splice;
                int reserveIndex = spliceIndex - minus;
                state.ReserveWaveforms = SliceLength(waveforms, reserveIndex * FrameShift, fromWaveforms * FrameShift);
                return Normalize(lfr);
            }
            state.ReserveWaveforms = SliceLength(waveforms, 0, -(FrameLength - FrameShift));
            state.SpliceCache.AddRange(rows);
            return null;
        }
        if (final && state.SpliceCache is not null)
        {
            var (lfr, splice, _) = Lfr(state.SpliceCache, true);
            state.SpliceCache = splice;
            return Normalize(lfr);
        }
        return null;
    }

    // WavFrontendOnline.apply_lfr: stacks of m frames every n, the right context not yet complete kept for the next call
    // (padded with the last frame when final).
    private (List<double[]> Rows, List<double[]> Splice, int SpliceIndex) Lfr(List<double[]> inputs, bool final)
    {
        var c = Encoder.Configuration;
        int m = c.LfrM, step = c.LfrN, total = inputs.Count;
        int count = (int)Math.Ceiling((total - (m - 1) / 2) / (double)step);
        int spliceIndex = count;
        var rows = new List<double[]>();
        for (int i = 0; i < count; i++)
        {
            if (m <= total - i * step)
                rows.Add(Enumerable.Range(i * step, m).SelectMany(r => inputs[r]).ToArray());
            else if (final)
                rows.Add(Enumerable.Range(i * step, m).SelectMany(r => inputs[Math.Min(r, total - 1)]).ToArray());
            else
            {
                spliceIndex = i;
                break;
            }
        }
        spliceIndex = Math.Min(total - 1, spliceIndex * step);
        return (rows, inputs.Skip(spliceIndex).Select(r => (double[])r.Clone()).ToList(), spliceIndex);
    }

    private Tensor<T> Normalize(List<double[]> rows)
    {
        var (shift, scale) = Encoder.Normalization;
        int dim = Encoder.Configuration.InputSize;
        var features = new Tensor<T>(new[] { rows.Count, dim });
        for (int r = 0; r < rows.Count; r++)
            for (int k = 0; k < dim; k++) features[r, k] = NumOps.FromDouble((rows[r][k] + shift[k]) * scale[k]);
        return features;
    }
}
