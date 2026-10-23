using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// StyleTTS's text aligner (§II-A, the transferable monotonic aligner; reference <c>Utils/ASR/models.py</c>
/// <c>ASRCNN</c>): MFCCs of the mel spectrogram (orthonormal DCT-II, 40 coefficients), a stride-2 conv, six blocks of
/// dilated residual convs with group norms, a projection, a CTC head, and a Tacotron 2-style attention decoder
/// (<c>ASRS2S</c>) whose location-sensitive attention over the frames gives the alignment.
/// </summary>
internal sealed class StyleTtsTextAligner<T>
{
    private readonly IEngine _engine;
    private readonly Tensor<T> _dct;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public StyleTtsTextAligner(IEngine engine, List<LayerBase<T>> layers, int mels, int hidden, int tokens, int tokenEmbedding,
        int convLayers = 6)
    {
        _engine = engine;
        Mels = mels;
        Hidden = hidden;
        Tokens = tokens;
        TokenEmbedding = tokenEmbedding;
        int mfcc = mels / 2;
        // torchaudio.functional.create_dct(n_mfcc, n_mels, "ortho"), [n_mels, n_mfcc].
        _dct = new Tensor<T>(new[] { mels, mfcc });
        for (int k = 0; k < mfcc; k++)
            for (int n = 0; n < mels; n++)
            {
                double v = Math.Cos(Math.PI / mels * (n + 0.5) * k) * Math.Sqrt(2.0 / mels) * (k == 0 ? 1 / Math.Sqrt(2.0) : 1.0);
                _dct[n, k] = NumOps.FromDouble(v);
            }
        InitialConv = StyleTtsOps.Own(layers, Plain(mfcc, hidden, 7, stride: 2, padding: 3));
        for (int l = 0; l < convLayers; l++)
        {
            var block = new List<(NormedConv1DLayer<T>, GroupNormalizationLayer<T>, NormedConv1DLayer<T>)>();
            for (int i = 0; i < 3; i++)
            {
                int dilation = (int)Math.Pow(3, i);
                block.Add((StyleTtsOps.Own(layers, Plain(hidden, hidden, 3, padding: dilation, dilation: dilation)),
                    StyleTtsOps.Own(layers, new GroupNormalizationLayer<T>(8, hidden, 1e-5)),
                    StyleTtsOps.Own(layers, Plain(hidden, hidden, 3, padding: 1))));
            }
            ConvBlocks.Add((block, StyleTtsOps.Own(layers, new GroupNormalizationLayer<T>(1, hidden, 1e-5))));
        }
        Projection = StyleTtsOps.Own(layers, Plain(hidden, hidden / 2, 1));
        CtcHidden = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(hidden));
        CtcOutput = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(tokens));

        int d = hidden / 2;
        Embedding = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(tokens, tokenEmbedding));
        ToSymbols = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(tokens));
        Query = StyleTtsOps.Own(layers, new BiasFreeLinearLayer<T>(d, d));
        Memory = StyleTtsOps.Own(layers, new BiasFreeLinearLayer<T>(d, d));
        Energy = StyleTtsOps.Own(layers, new BiasFreeLinearLayer<T>(d, 1));
        LocationConv = StyleTtsOps.Own(layers, new NormedConv1DLayer<T>(2, 32, 63, 1, 1, 1, 31, false, ConvolutionNormalization.None, useBias: false));
        LocationDense = StyleTtsOps.Own(layers, new BiasFreeLinearLayer<T>(32, d));
        DecoderCell = StyleTtsOps.Own(layers, new LSTMCellLayer<T>(d + tokenEmbedding, d));
        ToHidden = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(d));
    }

    private static NormedConv1DLayer<T> Plain(int input, int output, int kernel, int stride = 1, int padding = 0, int dilation = 1) =>
        new(input, output, kernel, stride, dilation, 1, padding, false, ConvolutionNormalization.None);

    public int Mels { get; }
    public int Hidden { get; }
    public int Tokens { get; }
    public int TokenEmbedding { get; }
    public NormedConv1DLayer<T> InitialConv { get; }
    public List<(List<(NormedConv1DLayer<T> Dilated, GroupNormalizationLayer<T> Norm, NormedConv1DLayer<T> Second)> Block, GroupNormalizationLayer<T> Norm)> ConvBlocks { get; } = new();
    public NormedConv1DLayer<T> Projection { get; }
    public DenseLayer<T> CtcHidden { get; }
    public DenseLayer<T> CtcOutput { get; }
    public TiedEmbeddingLayer<T> Embedding { get; }
    public DenseLayer<T> ToSymbols { get; }
    public BiasFreeLinearLayer<T> Query { get; }
    public BiasFreeLinearLayer<T> Memory { get; }
    public BiasFreeLinearLayer<T> Energy { get; }
    public NormedConv1DLayer<T> LocationConv { get; }
    public BiasFreeLinearLayer<T> LocationDense { get; }
    public LSTMCellLayer<T> DecoderCell { get; }
    public DenseLayer<T> ToHidden { get; }

    /// <summary>The start token of the attention decoder (1) and the token random masking writes (3).</summary>
    public const int StartToken = 1, UnknownToken = 3;

    /// <summary>The frame features <c>[1, hidden / 2, ⌈T / 2⌉]</c> of a mel spectrogram <c>[mels, T]</c> (the
    /// reference's <c>get_feature</c>).</summary>
    public Tensor<T> Features(Tensor<T> mel, bool training, Random random)
    {
        int frames = mel.Shape[1];
        var mfcc = _engine.TensorTranspose(_engine.TensorMatMul(_engine.TensorTranspose(mel), _dct));     // [n_mfcc, T]
        var x = InitialConv.Forward(_engine.Reshape(mfcc, new[] { 1, mfcc.Shape[0], frames }));
        foreach (var (block, norm) in ConvBlocks)
        {
            foreach (var (dilated, innerNorm, second) in block)
            {
                var h = GroupNorm(innerNorm, _engine.ReLU(dilated.Forward(x)));
                h = StyleTtsOps.Dropout(_engine, h, 0.2, training, random);
                h = StyleTtsOps.Dropout(_engine, _engine.ReLU(second.Forward(h)), 0.2, training, random);
                x = _engine.TensorAdd(h, x);
            }
            x = GroupNorm(norm, x);
        }
        return Projection.Forward(x);
    }

    // nn.GroupNorm on [1, C, T], through the layer's channels-first [1, C, T, 1] form.
    private Tensor<T> GroupNorm(GroupNormalizationLayer<T> norm, Tensor<T> x) =>
        _engine.Reshape(norm.Forward(_engine.Reshape(x, new[] { 1, x.Shape[1], x.Shape[2], 1 })), x._shape);

    /// <summary>The CTC head's logits <c>[⌈T / 2⌉, tokens]</c> of a mel spectrogram <c>[mels, T]</c>.</summary>
    public Tensor<T> CtcLogits(Tensor<T> mel, bool training, Random random)
    {
        var features = Features(mel, training, random);
        var rows = _engine.TensorTranspose(_engine.Reshape(features, new[] { Hidden / 2, features.Shape[2] }));
        return CtcOutput.Forward(_engine.ReLU(CtcHidden.Forward(rows)));
    }

    /// <summary>
    /// The attention decoder's outputs for a mel spectrogram <c>[mels, T]</c> and its phonemes <c>[L]</c>: the token
    /// logits <c>[L + 1, tokens]</c> and the attention energies <c>[L + 1, ⌈T / 2⌉]</c>, one row per decoding step
    /// (the start token's first). A tenth of the input tokens are replaced by the unknown token, as the reference's
    /// <c>ASRS2S.forward</c> does in training and evaluation alike.
    /// </summary>
    public (Tensor<T> Logits, Tensor<T> Energies) Decode(Tensor<T> mel, IReadOnlyList<int> text, bool training, Random random)
        => Decode(mel, text, training, random, MaskRate);

    /// <summary>The share of input tokens the decoder replaces by the unknown token (0.1; the reference masks in
    /// evaluation as well as training).</summary>
    public double MaskRate { get; set; } = 0.1;

    internal (Tensor<T> Logits, Tensor<T> Energies) Decode(Tensor<T> mel, IReadOnlyList<int> text, bool training, Random random, double maskRate)
    {
        var features = Features(mel, training, random);                                                    // [1, d, F]
        int d = Hidden / 2, frames = features.Shape[2];
        var memory = _engine.TensorTranspose(_engine.Reshape(features, new[] { d, frames }));              // [F, d]
        var processed = Memory.Forward(memory);
        var inputs = new int[text.Count + 1];
        inputs[0] = StartToken;
        for (int i = 0; i < text.Count; i++) inputs[i + 1] = random.NextDouble() < maskRate ? UnknownToken : text[i];
        var ids = new Tensor<T>(new[] { inputs.Length });
        for (int i = 0; i < inputs.Length; i++) ids[i] = NumOps.FromDouble(inputs[i]);
        var embedded = Embedding.Forward(ids);                                                             // [L + 1, e]

        var state = new Tensor<T>(new[] { 1, 2 * d });
        var weights = new Tensor<T>(new[] { 1, frames });
        var cumulative = new Tensor<T>(new[] { 1, frames });
        var context = new Tensor<T>(new[] { 1, d });
        var logits = new List<Tensor<T>>();
        var energies = new List<Tensor<T>>();
        for (int step = 0; step < inputs.Length; step++)
        {
            var input = _engine.TensorConcatenate(new[] { _engine.TensorSlice(embedded, new[] { step, 0 }, new[] { 1, TokenEmbedding }), context }, 1);
            state = DecoderCell.Forward(input, state);
            var hidden = DecoderCell.SplitState(state).Hidden;                                             // [1, d]
            var location = LocationConv.Forward(_engine.Reshape(_engine.TensorConcatenate(new[] { weights, cumulative }, 0), new[] { 1, 2, frames }));
            var processedLocation = LocationDense.Forward(_engine.TensorTranspose(_engine.Reshape(location, new[] { 32, frames })));   // [F, d]
            var query = _engine.TensorTile(Query.Forward(hidden), new[] { frames, 1 });
            var e = Energy.Forward(_engine.Tanh(_engine.TensorAdd(_engine.TensorAdd(query, processedLocation), processed)));       // [F, 1]
            var energy = _engine.Reshape(e, new[] { 1, frames });
            weights = _engine.Softmax(energy, axis: 1);
            cumulative = _engine.TensorAdd(cumulative, weights);
            context = _engine.TensorMatMul(weights, memory);                                               // [1, d]
            var projected = _engine.Tanh(ToHidden.Forward(_engine.TensorConcatenate(new[] { hidden, context }, 1)));
            logits.Add(ToSymbols.Forward(StyleTtsOps.Dropout(_engine, projected, 0.5, training, random)));
            energies.Add(energy);
        }
        return (_engine.TensorConcatenate(logits.ToArray(), 0), _engine.TensorConcatenate(energies.ToArray(), 0));
    }

    public void Load(TorchReader read, string name)
    {
        // The DCT matrix is a stored buffer (computed by torchaudio in single precision); read it as stored.
        var dct = read(name + ".to_mfcc.dct_mat", new[] { Mels, Mels / 2 });
        for (int i = 0; i < dct.Length; i++) _dct[i] = NumOps.FromDouble(dct[i]);
        StyleTtsOps.LoadPlain(InitialConv, read, name + ".init_cnn.conv");
        for (int l = 0; l < ConvBlocks.Count; l++)
        {
            var (block, norm) = ConvBlocks[l];
            for (int i = 0; i < block.Count; i++)
            {
                string p = $"{name}.cnns.{l}.0.blocks.{i}";
                StyleTtsOps.LoadPlain(block[i].Dilated, read, p + ".0.conv");
                StyleTtsOps.LoadAffine(_engine, block[i].Norm.GetGammaTensor(), block[i].Norm.GetBetaTensor(),
                    read(p + ".2.weight", new[] { Hidden }), read(p + ".2.bias", new[] { Hidden }));
                StyleTtsOps.LoadPlain(block[i].Second, read, p + ".4.conv");
            }
            StyleTtsOps.LoadAffine(_engine, norm.GetGammaTensor(), norm.GetBetaTensor(),
                read($"{name}.cnns.{l}.1.weight", new[] { Hidden }), read($"{name}.cnns.{l}.1.bias", new[] { Hidden }));
        }
        StyleTtsOps.LoadPlain(Projection, read, name + ".projection.conv");
        StyleTtsOps.LoadLinear(_engine, CtcHidden, read, name + ".ctc_linear.0.linear_layer", Hidden / 2, Hidden);
        StyleTtsOps.LoadLinear(_engine, CtcOutput, read, name + ".ctc_linear.2.linear_layer", Hidden, Tokens);
        int d = Hidden / 2;
        string s = name + ".asr_s2s";
        Embedding.LoadTable(read(s + ".embedding.weight", new[] { Tokens, TokenEmbedding }));
        StyleTtsOps.LoadLinear(_engine, ToSymbols, read, s + ".project_to_n_symbols", d, Tokens);
        TorchParameters.Linear(Query, d, d, read(s + ".attention_layer.query_layer.linear_layer.weight", new[] { d, d }));
        TorchParameters.Linear(Memory, d, d, read(s + ".attention_layer.memory_layer.linear_layer.weight", new[] { d, d }));
        TorchParameters.Linear(Energy, d, 1, read(s + ".attention_layer.v.linear_layer.weight", new[] { 1, d }));
        LocationConv.LoadTorchWeights(read(s + ".attention_layer.location_layer.location_conv.conv.weight", new[] { 32, 2, 63 }), null, new double[32]);
        TorchParameters.Linear(LocationDense, 32, d, read(s + ".attention_layer.location_layer.location_dense.linear_layer.weight", new[] { d, 32 }));
        DecoderCell.LoadTorchWeights(read(s + ".decoder_rnn.weight_ih", new[] { 4 * d, d + TokenEmbedding }), read(s + ".decoder_rnn.weight_hh", new[] { 4 * d, d }),
            read(s + ".decoder_rnn.bias_ih", new[] { 4 * d }), read(s + ".decoder_rnn.bias_hh", new[] { 4 * d }));
        StyleTtsOps.LoadLinear(_engine, ToHidden, read, s + ".project_to_hidden.0.linear_layer", 2 * d, d);
    }
}

/// <summary>
/// StyleTTS's pitch extractor (§II-A, Appendix C; reference <c>Utils/JDC/model.py</c>): the JDC network of Kum et al.
/// 2019 with one output — a conv block, three residual blocks pooling the frequency axis by 2, batch norm, LeakyReLU
/// (0.01), pooling by 4, dropout, a bidirectional LSTM and a linear head whose absolute value is the F0 of each frame.
/// </summary>
internal sealed class StyleTtsPitchExtractor<T>
{
    private readonly IEngine _engine;
    private const double Slope = 0.01;

    /// <param name="engine">The engine.</param>
    /// <param name="layers">Receives the layers.</param>
    /// <param name="channels">The conv block's width and the three residual blocks' widths (64, 128, 192, 256).</param>
    /// <param name="lstmHidden">Each LSTM direction's width (256).</param>
    /// <param name="mels">Mel bins of the input (80); they pool to a quarter of an eighth.</param>
    public StyleTtsPitchExtractor(IEngine engine, List<LayerBase<T>> layers, int[] channels, int lstmHidden, int mels)
    {
        _engine = engine;
        if (channels.Length != 4) throw new ArgumentException("The JDC network has a conv block and three residual blocks.", nameof(channels));
        Channels = channels;
        _pooledWidth = channels[3] * (mels / 8 / 4);
        Conv1 = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(1, channels[0], 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.None, useBias: false));
        Norm1 = StyleTtsOps.Own(layers, new BatchNormalizationLayer<T>(channels[0], 1e-5, 0.9));
        Conv2 = StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(channels[0], channels[0], 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.None, useBias: false));
        foreach (var (input, output) in new[] { (channels[0], channels[1]), (channels[1], channels[2]), (channels[2], channels[3]) })
            Blocks.Add((StyleTtsOps.Own(layers, new BatchNormalizationLayer<T>(input, 1e-5, 0.9)),
                StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(input, output, 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.None, useBias: false)),
                StyleTtsOps.Own(layers, new BatchNormalizationLayer<T>(output, 1e-5, 0.9)),
                StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(output, output, 3, 3, 1, 1, 1, 1, 1, ConvolutionNormalization.None, useBias: false)),
                StyleTtsOps.Own(layers, new NormedConv2DLayer<T>(input, output, 1, 1, 1, 1, 0, 0, 1, ConvolutionNormalization.None, useBias: false))));
        PoolNorm = StyleTtsOps.Own(layers, new BatchNormalizationLayer<T>(channels[3], 1e-5, 0.9));
        Lstm = new StyleTtsBiLstm<T>(engine, layers, _pooledWidth, lstmHidden);
        Classifier = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(1));
    }

    private readonly int _pooledWidth;
    public int[] Channels { get; }
    public NormedConv2DLayer<T> Conv1 { get; }
    public BatchNormalizationLayer<T> Norm1 { get; }
    public NormedConv2DLayer<T> Conv2 { get; }
    public List<(BatchNormalizationLayer<T> PreNorm, NormedConv2DLayer<T> Conv1, BatchNormalizationLayer<T> Norm, NormedConv2DLayer<T> Conv2, NormedConv2DLayer<T> Shortcut)> Blocks { get; } = new();
    public BatchNormalizationLayer<T> PoolNorm { get; }
    public StyleTtsBiLstm<T> Lstm { get; }
    public DenseLayer<T> Classifier { get; }

    // MaxPool2d((1, k)): the maximum of each run of k along the last axis.
    private Tensor<T> PoolLast(Tensor<T> x, int k)
    {
        int c = x.Shape[1], h = x.Shape[2], w = x.Shape[3] / k;
        if (x.Shape[3] != w * k) x = _engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { 1, c, h, w * k });
        var grouped = _engine.Reshape(x, new[] { 1, c, h, w, k });
        Tensor<T>? best = null;
        for (int j = 0; j < k; j++)
        {
            var part = _engine.TensorSlice(grouped, new[] { 0, 0, 0, 0, j }, new[] { 1, c, h, w, 1 });
            best = best is null ? part : _engine.TensorMax(best, part);
        }
        return _engine.Reshape(best!, new[] { 1, c, h, w });
    }

    /// <summary>The F0 of each frame <c>[1, T]</c> of a mel spectrogram <c>[mels, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> mel, bool training, Random random)
    {
        int mels = mel.Shape[0], frames = mel.Shape[1];
        var x = _engine.Reshape(_engine.TensorTranspose(mel), new[] { 1, 1, frames, mels });            // [1, 1, T, F]
        x = Conv2.Forward(StyleTtsOps.LeakyRelu(_engine, Norm1.Forward(Conv1.Forward(x)), Slope));
        foreach (var (preNorm, conv1, norm, conv2, shortcut) in Blocks)
        {
            x = PoolLast(StyleTtsOps.LeakyRelu(_engine, preNorm.Forward(x), Slope), 2);
            var residual = conv2.Forward(StyleTtsOps.LeakyRelu(_engine, norm.Forward(conv1.Forward(x)), Slope));
            x = _engine.TensorAdd(residual, shortcut.Forward(x));
        }
        x = PoolLast(StyleTtsOps.LeakyRelu(_engine, PoolNorm.Forward(x), Slope), 4);
        x = StyleTtsOps.Dropout(_engine, x, 0.2, training, random);                                        // [1, C, T, F/32]
        var rows = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { frames, _pooledWidth });
        var f0 = _engine.TensorAbs(Classifier.Forward(Lstm.Apply(rows)));                                  // [T, 1]
        return _engine.Reshape(f0, new[] { 1, frames });
    }

    public void Load(TorchReader read, string name)
    {
        void Conv(NormedConv2DLayer<T> conv, string n) => conv.LoadTorchWeights(read(n + ".weight", conv.TorchWeightShape), null);
        void Norm(BatchNormalizationLayer<T> norm, string n, int c) => TorchParameters.BatchNorm(norm, read(n + ".weight", new[] { c }),
            read(n + ".bias", new[] { c }), read(n + ".running_mean", new[] { c }), read(n + ".running_var", new[] { c }));
        Conv(Conv1, name + ".conv_block.0");
        Norm(Norm1, name + ".conv_block.1", Channels[0]);
        Conv(Conv2, name + ".conv_block.3");
        int[] inputs = { Channels[0], Channels[1], Channels[2] }, outputs = { Channels[1], Channels[2], Channels[3] };
        for (int i = 0; i < Blocks.Count; i++)
        {
            string b = $"{name}.res_block{i + 1}";
            Norm(Blocks[i].PreNorm, b + ".pre_conv.0", inputs[i]);
            Conv(Blocks[i].Conv1, b + ".conv.0");
            Norm(Blocks[i].Norm, b + ".conv.1", outputs[i]);
            Conv(Blocks[i].Conv2, b + ".conv.3");
            Conv(Blocks[i].Shortcut, b + ".conv1by1");
        }
        Norm(PoolNorm, name + ".pool_block.0", Channels[3]);
        Lstm.Load(read, name + ".bilstm_classifier");
        StyleTtsOps.LoadLinear(_engine, Classifier, read, name + ".classifier", 2 * Lstm.Hidden, 1);
    }
}
