using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released DeepFilterNet3 neural graph, including grouped causal
/// convolutions, squeezed reset-after GRUs, ERB gains and five-tap complex filtering.
/// Waveform analysis/normalization/synthesis is a separate stateful frontend.</summary>
[AutoParameters]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
internal sealed partial class MetaVoiceDeepFilterNet<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels, _hidden, _erb, _df, _frequencies, _groups, _encoderGroups;
    private readonly List<string> _names = new(), _bufferNames = new();
    private readonly Dictionary<string, int> _indices = new(), _bufferIndices = new();
    [TrainableParameter] private readonly Tensor<T>[] _learnedTensors;
    private readonly Tensor<T>[] _buffers;
    private bool _loaded;
    public override bool SupportsTraining => true;

    internal IReadOnlyDictionary<string, Tensor<T>> NamedParameters => _names
        .Select((name, index) => (name, tensor: _learnedTensors[index])).ToDictionary(x => x.name, x => x.tensor);

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 3
        ? new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(2 * _frequencies)) } : null;

    internal MetaVoiceDeepFilterNet([LayerState] int channels = 64, [LayerState] int hidden = 256, [LayerState] int erb = 32,
        [LayerState] int df = 96, [LayerState] int frequencies = 481, [LayerState] int groups = 16, [LayerState] int encoderGroups = 32,
        [LayerState] bool loaded = false)
        : base(new[] { -1, frequencies * 2 + erb + df * 2 }, new[] { -1, frequencies * 2 })
    {
        if (channels <= 0 || hidden <= 0 || erb <= 0 || erb % 8 != 0 || df <= 0 || df % 2 != 0 ||
            frequencies <= df || groups <= 0 || encoderGroups <= 0)
            throw new ArgumentException("Invalid DeepFilterNet3 dimensions.");
        _channels = channels; _hidden = hidden; _erb = erb; _df = df; _frequencies = frequencies;
        _groups = groups; _encoderGroups = encoderGroups;
        _loaded = loaded;
        var parameters = new List<Tensor<T>>(); var buffers = new List<Tensor<T>>();
        void Parameter(string name, params int[] shape)
        {
            _indices.Add(name, parameters.Count); _names.Add(name);
            var value = new Tensor<T>(shape); parameters.Add(value); RegisterTrainableParameter(value, PersistentTensorRole.Weights);
        }
        void Buffer(string name, params int[] shape)
        {
            _bufferIndices.Add(name, buffers.Count); _bufferNames.Add(name);
            var value = new Tensor<T>(shape); buffers.Add(value); RegisterBuffer(value, name);
        }
        void Grouped(string name, int input, int output, int count)
        {
            if (input % count != 0 || output % count != 0) throw new ArgumentException("Grouped projection dimensions differ.");
            Parameter(name + ".weight", count, input / count, output / count);
        }
        void Gru(string name, int input, int output, int layers, int count, bool projection)
        {
            Grouped(name + ".linear_in.0", input, hidden, count);
            for (int i = 0; i < layers; i++)
            {
                Parameter(name + ".gru.weight_ih_l" + i, 3 * hidden, hidden);
                Parameter(name + ".gru.weight_hh_l" + i, 3 * hidden, hidden);
                Parameter(name + ".gru.bias_ih_l" + i, 3 * hidden);
                Parameter(name + ".gru.bias_hh_l" + i, 3 * hidden);
            }
            if (projection) Grouped(name + ".linear_out.0", hidden, output, count);
        }
        void Conv(string name, int input, int output, int kt, int kf, bool transpose = false)
        {
            int count = GreatestCommonDivisor(input, output), index = kt > 1 ? 1 : 0;
            // The author computes groups BEFORE disabling separability for 1x1.
            bool pointwise = count != 1 && Math.Max(kt, kf) != 1;
            Parameter(name + "." + index++ + ".weight", transpose ? input : output,
                (transpose ? output : input) / count, kt, kf);
            if (pointwise) Parameter(name + "." + index++ + ".weight", output, output, 1, 1);
            string bn = name + "." + index;
            Parameter(bn + ".weight", output); Parameter(bn + ".bias", output);
            Buffer(bn + ".running_mean", output); Buffer(bn + ".running_var", output);
            Buffer(bn + ".num_batches_tracked");
        }
        int embedding = channels * erb / 4;
        Buffer("erb_fb", frequencies, erb); Buffer("mask.erb_inv_fb", erb, frequencies);
        Conv("enc.erb_conv0", 1, channels, 3, 3);
        for (int i = 1; i < 4; i++) Conv("enc.erb_conv" + i, channels, channels, 1, 3);
        Conv("enc.df_conv0", 2, channels, 3, 3); Conv("enc.df_conv1", channels, channels, 1, 3);
        Grouped("enc.df_fc_emb.0", channels * df / 2, embedding, encoderGroups);
        Gru("enc.emb_gru", embedding, embedding, 1, groups, true);
        Parameter("enc.lsnr_fc.0.weight", 1, embedding); Parameter("enc.lsnr_fc.0.bias", 1);
        Gru("erb_dec.emb_gru", embedding, embedding, 2, groups, true);
        for (int i = 3; i >= 0; i--) Conv("erb_dec.conv" + i + "p", channels, channels, 1, 1);
        Conv("erb_dec.convt3", channels, channels, 1, 3);
        Conv("erb_dec.convt2", channels, channels, 1, 3, true);
        Conv("erb_dec.convt1", channels, channels, 1, 3, true);
        Conv("erb_dec.conv0_out", channels, 1, 1, 3);
        Conv("df_dec.df_convp", channels, 10, 5, 1);
        // SqueezedGRU_S's default is eight here: the author omits linear_groups.
        Gru("df_dec.df_gru", embedding, hidden, 2, 8, false);
        Grouped("df_dec.df_skip", embedding, hidden, groups);
        Grouped("df_dec.df_out.0", hidden, df * 10, groups);
        Parameter("df_dec.df_fc_a.0.weight", 1, hidden); Parameter("df_dec.df_fc_a.0.bias", 1);
        _learnedTensors = parameters.ToArray(); _buffers = buffers.ToArray();
    }

    private static int GreatestCommonDivisor(int a, int b)
    { while (b != 0) { int remainder = a % b; a = b; b = remainder; } return a; }

    internal void LoadTorch(Func<string, int[], double[]> read) => LoadTorch(AiDotNet.ModelLoading.TorchTensorSource.From(read));

    /// <summary>Copies every parameter and buffer straight from <paramref name="source"/> into its tensor.</summary>
    internal void LoadTorch(AiDotNet.ModelLoading.TorchTensorSource source)
    {
        foreach (string name in _names.Concat(_bufferNames))
        {
            var target = _indices.TryGetValue(name, out int index) ? _learnedTensors[index] : _buffers[_bufferIndices[name]];
            source.CopyTo(name, target.Shape.ToArray(), target);
            Engine.InvalidatePersistentTensor(target);
        }
        _loaded = true;
    }

    internal int[] ErbWidths()
    {
        if (!_loaded || _erb != 32 || _frequencies != 481)
            throw new InvalidOperationException("Load the released 32-band/481-bin DeepFilterNet3 before waveform enhancement.");
        var bank = B("mask.erb_inv_fb"); var widths = new int[_erb]; int frequency = 0;
        for (int band = 0; band < _erb; band++)
        {
            while (frequency < _frequencies && NumOps.ToDouble(bank[band, frequency]) == 1)
            { widths[band]++; frequency++; }
            if (widths[band] == 0) throw new InvalidDataException("DeepFilterNet ERB inverse bank is not the released contiguous partition.");
        }
        if (frequency != _frequencies) throw new InvalidDataException("DeepFilterNet ERB inverse bank does not cover the spectrum.");
        return widths;
    }

    private Tensor<T> P(string name) => _learnedTensors[_indices[name]];
    private Tensor<T> B(string name) => _buffers[_bufferIndices[name]];
    private Tensor<T> Slice(Tensor<T> value, int axis, int start, int length)
    {
        var starts = new int[value.Rank]; starts[axis] = start;
        var lengths = value.Shape.ToArray(); lengths[axis] = length;
        return Engine.TensorSlice(value, starts, lengths);
    }
    private Tensor<T> Cat(int axis, params Tensor<T>[] values) => Engine.TensorConcatenate(values, axis);
    private Tensor<T> Pad(Tensor<T> value, int axis, int before, int after)
    {
        var pieces = new List<Tensor<T>>(); var shape = value.Shape.ToArray();
        if (before > 0) { shape[axis] = before; pieces.Add(new Tensor<T>(shape)); }
        pieces.Add(value);
        if (after > 0) { shape[axis] = after; pieces.Add(new Tensor<T>(shape)); }
        return Cat(axis, pieces.ToArray());
    }
    private Tensor<T> TileFeature(Tensor<T> vector, Tensor<T> like)
        => Engine.TensorTile(Engine.Reshape(vector, new[] { 1, vector.Length, 1, 1 }),
            new[] { like.Shape[0], 1, like.Shape[2], like.Shape[3] });

    private Tensor<T> Convolution(string name, Tensor<T> input, int stride = 1, bool transpose = false, bool sigmoid = false)
    {
        int index = _indices.ContainsKey(name + ".0.weight") ? 0 : 1;
        var kernel = P(name + "." + index++ + ".weight"); int kt = kernel.Shape[2], kf = kernel.Shape[3];
        int output = transpose ? kernel.Shape[1] * input.Shape[1] : kernel.Shape[0];
        int groups = transpose ? input.Shape[1] : input.Shape[1] / kernel.Shape[1];
        input = Pad(input, 2, kt - 1, 0);
        var outputs = new List<Tensor<T>>();
        for (int group = 0; group < groups; group++)
        {
            var part = Slice(input, 1, group * input.Shape[1] / groups, input.Shape[1] / groups);
            var weights = Slice(kernel, 0, group * kernel.Shape[0] / groups, kernel.Shape[0] / groups);
            if (transpose)
            {
                // The current ConvTranspose2D backward rejects output_padding=1.
                // Exact zero insertion plus a flipped ordinary convolution preserves
                // the author's operation and gradients, including its trailing sample.
                if (kt != 1 || stride != 2 || kf != 3)
                    throw new InvalidOperationException("Unexpected released DeepFilterNet transposed kernel.");
                var shape = part.Shape;
                var expanded = Engine.Reshape(part, new[] { shape[0], shape[1], shape[2], shape[3], 1 });
                var inserted = Engine.Reshape(Cat(4, expanded, new Tensor<T>(expanded.Shape.ToArray())),
                    new[] { shape[0], shape[1], shape[2], 2 * shape[3] });
                var transposed = Engine.TensorPermute(weights, new[] { 1, 0, 2, 3 });
                var flipped = Cat(3, Enumerable.Range(0, kf).Select(i => Slice(transposed, 3, kf - 1 - i, 1)).ToArray());
                outputs.Add(Engine.Conv2D(inserted, flipped, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 }));
            }
            else outputs.Add(Engine.Conv2D(part, weights, new[] { 1, stride }, new[] { 0, kf / 2 }, new[] { 1, 1 }));
        }
        var value = Cat(1, outputs.ToArray());
        if (_indices.ContainsKey(name + "." + index + ".weight") && P(name + "." + index + ".weight").Rank == 4)
            value = Engine.Conv2D(value, P(name + "." + index++ + ".weight"), new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        string bn = name + "." + index;
        var centered = Engine.TensorSubtract(value, TileFeature(B(bn + ".running_mean"), value));
        var inverse = Engine.TensorPow(Engine.TensorAddScalar(B(bn + ".running_var"), NumOps.FromDouble(1e-5)), NumOps.FromDouble(-.5));
        value = Engine.TensorAdd(Engine.TensorMultiply(centered,
            TileFeature(Engine.TensorMultiply(P(bn + ".weight"), inverse), value)), TileFeature(P(bn + ".bias"), value));
        return sigmoid ? Engine.Sigmoid(value) : Engine.ReLU(value);
    }

    private Tensor<T> Grouped(string name, Tensor<T> input)
    {
        var weight = P(name + ".weight"); int groups = weight.Shape[0];
        int batch = input.Shape[0], time = input.Shape[1], width = weight.Shape[1];
        var values = new List<Tensor<T>>();
        for (int group = 0; group < groups; group++)
        {
            var x = Engine.Reshape(Slice(input, 2, group * width, width), new[] { batch * time, width });
            var w = Engine.Reshape(Slice(weight, 0, group, 1), new[] { width, weight.Shape[2] });
            values.Add(Engine.Reshape(Engine.TensorMatMul(x, w), new[] { batch, time, weight.Shape[2] }));
        }
        return Cat(2, values.ToArray());
    }

    private Tensor<T> Linear(string name, Tensor<T> input)
    {
        var weight = P(name + ".weight"); int rows = input.Length / input.Shape[input.Rank - 1];
        var result = Engine.TensorMatMul(Engine.Reshape(input, new[] { rows, weight.Shape[1] }), Engine.TensorTranspose(weight));
        result = Engine.TensorAdd(result, Engine.TensorTile(Engine.Reshape(P(name + ".bias"), new[] { 1, weight.Shape[0] }), new[] { rows, 1 }));
        var shape = input.Shape.ToArray(); shape[shape.Length - 1] = weight.Shape[0]; return Engine.Reshape(result, shape);
    }

    private Tensor<T> Gru(string name, Tensor<T> input, int layers, bool projection)
    {
        var value = Engine.ReLU(Grouped(name + ".linear_in.0", input)); int batch = value.Shape[0], time = value.Shape[1];
        for (int layer = 0; layer < layers; layer++)
        {
            var state = new Tensor<T>(new[] { batch, _hidden }); var sequence = new List<Tensor<T>>();
            Tensor<T> Affine(Tensor<T> x, string kind)
            {
                var w = P(name + ".gru.weight_" + kind + "_l" + layer);
                var b = P(name + ".gru.bias_" + kind + "_l" + layer);
                return Engine.TensorAdd(Engine.TensorMatMul(x, Engine.TensorTranspose(w)),
                    Engine.TensorTile(Engine.Reshape(b, new[] { 1, 3 * _hidden }), new[] { batch, 1 }));
            }
            for (int t = 0; t < time; t++)
            {
                var x = Affine(Engine.Reshape(Slice(value, 1, t, 1), new[] { batch, _hidden }), "ih");
                var h = Affine(state, "hh");
                var reset = Engine.Sigmoid(Engine.TensorAdd(Slice(x, 1, 0, _hidden), Slice(h, 1, 0, _hidden)));
                var update = Engine.Sigmoid(Engine.TensorAdd(Slice(x, 1, _hidden, _hidden), Slice(h, 1, _hidden, _hidden)));
                var candidate = Engine.Tanh(Engine.TensorAdd(Slice(x, 1, 2 * _hidden, _hidden),
                    Engine.TensorMultiply(reset, Slice(h, 1, 2 * _hidden, _hidden))));
                state = Engine.TensorAdd(candidate, Engine.TensorMultiply(update, Engine.TensorSubtract(state, candidate)));
                sequence.Add(Engine.Reshape(state, new[] { batch, 1, _hidden }));
            }
            value = Cat(1, sequence.ToArray());
        }
        return projection ? Engine.ReLU(Grouped(name + ".linear_out.0", value)) : value;
    }

    private Tensor<T> FlattenFeatures(Tensor<T> value) => Engine.Reshape(Engine.TensorPermute(value, new[] { 0, 2, 3, 1 }),
        new[] { value.Shape[0], value.Shape[2], value.Shape[3] * value.Shape[1] });

    internal (Tensor<T> Spectrum, Tensor<T> Mask, Tensor<T> Snr, Tensor<T> Coefficients) Forward(
        Tensor<T> spectrum, Tensor<T> erbFeatures, Tensor<T> spectralFeatures)
    {
        if (!_loaded) throw new InvalidOperationException("Load the original DeepFilterNet3 tensors before inference.");
        if (spectrum.Rank != 5 || spectrum.Shape[1] != 1 || spectrum.Shape[3] != _frequencies || spectrum.Shape[4] != 2 ||
            erbFeatures.Rank != 4 || erbFeatures.Shape[3] != _erb || spectralFeatures.Rank != 5 || spectralFeatures.Shape[3] != _df)
            throw new ArgumentException("DeepFilterNet3 expects spectrum [B,1,T,F,2], ERB [B,1,T,E], and complex features [B,1,T,D,2].");
        int batch = spectrum.Shape[0], time = spectrum.Shape[2], embedding = _channels * _erb / 4;
        if (time < 2 || erbFeatures.Shape[0] != batch || erbFeatures.Shape[2] != time || spectralFeatures.Shape[0] != batch || spectralFeatures.Shape[2] != time)
            throw new ArgumentException("DeepFilterNet3 batch/time axes differ or omit its two-frame lookahead.");
        using var scope = new WeightRegistry.MaterializeScope<T>(_learnedTensors);
        Tensor<T> Lookahead(Tensor<T> x) => Pad(Slice(x, 2, 2, time - 2), 2, 0, 2);
        var erbInput = Lookahead(erbFeatures);
        var specInput = Lookahead(Engine.TensorPermute(Engine.Reshape(spectralFeatures,
            new[] { batch, time, _df, 2 }), new[] { 0, 3, 1, 2 }));
        var e0 = Convolution("enc.erb_conv0", erbInput);
        var e1 = Convolution("enc.erb_conv1", e0, 2);
        var e2 = Convolution("enc.erb_conv2", e1, 2);
        var e3 = Convolution("enc.erb_conv3", e2);
        var c0 = Convolution("enc.df_conv0", specInput);
        var c1 = Convolution("enc.df_conv1", c0, 2);
        var encoded = Engine.TensorAdd(FlattenFeatures(e3), Engine.ReLU(Grouped("enc.df_fc_emb.0", FlattenFeatures(c1))));
        encoded = Gru("enc.emb_gru", encoded, 1, true);
        var snr = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.Sigmoid(Linear("enc.lsnr_fc.0", encoded)), NumOps.FromDouble(50)), NumOps.FromDouble(-15));
        var erb = Engine.TensorPermute(Engine.Reshape(Gru("erb_dec.emb_gru", encoded, 2, true),
            new[] { batch, time, _erb / 4, _channels }), new[] { 0, 3, 1, 2 });
        erb = Convolution("erb_dec.convt3", Engine.TensorAdd(Convolution("erb_dec.conv3p", e3), erb));
        erb = Convolution("erb_dec.convt2", Engine.TensorAdd(Convolution("erb_dec.conv2p", e2), erb), 2, true);
        erb = Convolution("erb_dec.convt1", Engine.TensorAdd(Convolution("erb_dec.conv1p", e1), erb), 2, true);
        var mask = Convolution("erb_dec.conv0_out", Engine.TensorAdd(Convolution("erb_dec.conv0p", e0), erb), sigmoid: true);
        var gain = Engine.TensorMatMul(Engine.Reshape(mask, new[] { batch * time, _erb }), B("mask.erb_inv_fb"));
        var masked = Engine.TensorMultiply(spectrum, Engine.TensorTile(Engine.Reshape(gain,
            new[] { batch, 1, time, _frequencies, 1 }), new[] { 1, 1, 1, 1, 2 }));
        var dfHidden = Engine.TensorAdd(Gru("df_dec.df_gru", encoded, 2, false), Grouped("df_dec.df_skip", encoded));
        var coefficients = Engine.TensorAdd(Engine.Reshape(Engine.Tanh(Grouped("df_dec.df_out.0", dfHidden)),
            new[] { batch, time, _df, 10 }), Engine.TensorPermute(Convolution("df_dec.df_convp", c0), new[] { 0, 2, 3, 1 }));
        coefficients = Engine.TensorPermute(Engine.Reshape(coefficients, new[] { batch, time, _df, 5, 2 }), new[] { 0, 3, 1, 2, 4 });
        var low = Slice(spectrum, 3, 0, _df); var padded = Pad(low, 2, 2, 2);
        var real = new Tensor<T>(new[] { batch, 1, time, _df, 1 }); var imaginary = new Tensor<T>(real.Shape.ToArray());
        for (int tap = 0; tap < 5; tap++)
        {
            var frame = Slice(padded, 2, tap, time);
            var filter = Engine.Reshape(Slice(coefficients, 1, tap, 1), new[] { batch, 1, time, _df, 2 });
            var xr = Slice(frame, 4, 0, 1); var xi = Slice(frame, 4, 1, 1);
            var cr = Slice(filter, 4, 0, 1); var ci = Slice(filter, 4, 1, 1);
            real = Engine.TensorAdd(real, Engine.TensorSubtract(Engine.TensorMultiply(xr, cr), Engine.TensorMultiply(xi, ci)));
            imaginary = Engine.TensorAdd(imaginary, Engine.TensorAdd(Engine.TensorMultiply(xr, ci), Engine.TensorMultiply(xi, cr)));
        }
        return (Cat(3, Cat(4, real, imaginary), Slice(masked, 3, _df, _frequencies - _df)), mask, snr, coefficients);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int width = 2 * _frequencies + _erb + 2 * _df;
        if (input.Rank != 3 || input.Shape[2] != width) throw new ArgumentException("Expected packed [batch,time,spectrum+ERB+complex-features].");
        int batch = input.Shape[0], time = input.Shape[1];
        var result = Forward(Engine.Reshape(Slice(input, 2, 0, 2 * _frequencies), new[] { batch, 1, time, _frequencies, 2 }),
            Engine.Reshape(Slice(input, 2, 2 * _frequencies, _erb), new[] { batch, 1, time, _erb }),
            Engine.Reshape(Slice(input, 2, 2 * _frequencies + _erb, 2 * _df), new[] { batch, 1, time, _df, 2 }));
        return Engine.Reshape(result.Spectrum, new[] { batch, time, 2 * _frequencies });
    }

    public override void ResetState() { }
}
