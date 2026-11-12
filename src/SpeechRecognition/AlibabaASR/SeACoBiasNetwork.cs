using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// SeACo-Paraformer's bias modules (Shi et al. 2023, section 3.1; FunASR's <c>SeacoParaformer</c>): the bias encoder, the
/// bias decoder and the bias output layer.
/// </summary>
/// <remarks>
/// <para>The bias encoder embeds each hotword's tokens with the ASR decoder's embedding and runs a unidirectional LSTM over
/// them; the last step's state is the hotword's representation (Equation 2). The bias decoder is a SAN-M decoder whose
/// memory is the hotword representations: it reads the CIF acoustic embeddings and, separately, the ASR decoder's hidden
/// states (Equation 3; the release applies the same decoder to both). Their sum goes through the bias output layer, over
/// the ASR vocabulary plus the no-bias class (Equation 4).</para>
/// </remarks>
internal sealed class SeACoBiasNetwork<T>
{
    private readonly IEngine _engine;
    private readonly int _width;

    /// <param name="engine">The tensor engine.</param>
    /// <param name="layers">Receives every layer the bias modules own, in creation order.</param>
    /// <param name="width">The model width.</param>
    /// <param name="heads">The bias decoder's attention heads.</param>
    /// <param name="feedForward">The bias decoder's feed-forward width.</param>
    /// <param name="decoderLayers">The bias decoder's blocks with cross-attention.</param>
    /// <param name="kernel">The bias decoder's FSMN kernel.</param>
    /// <param name="encoderLayers">The bias encoder's LSTM layers.</param>
    /// <param name="classes">The bias output's classes: the ASR vocabulary plus the no-bias class.</param>
    public SeACoBiasNetwork(IEngine engine, List<LayerBase<T>> layers, int width, int heads, int feedForward, int decoderLayers, int kernel,
        int encoderLayers, int classes)
    {
        if (encoderLayers < 1) throw new ArgumentException("The bias encoder needs at least one LSTM layer.", nameof(encoderLayers));
        _engine = engine;
        _width = width;
        for (int i = 0; i < encoderLayers; i++)
        {
            var lstm = new LSTMLayer<T>(width);
            layers.Add(lstm);
            Encoder.Add(lstm);
        }
        Decoder = new SanmDecoder<T>(engine, layers, width, heads, feedForward, decoderLayers, kernel, 1e-12);
        Output = new DenseLayer<T>(classes, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(Output);
    }

    public List<LSTMLayer<T>> Encoder { get; } = new();
    public SanmDecoder<T> Decoder { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>The hotwords' representations <c>[n, width]</c>: the bias encoder's last state over each hotword's
    /// embeddings <c>[length, width]</c>.</summary>
    public Tensor<T> Hotwords(IReadOnlyList<Tensor<T>> embeddings)
    {
        if (embeddings.Count == 0) throw new ArgumentException("At least one hotword is required.", nameof(embeddings));
        var rows = new Tensor<T>[embeddings.Count];
        for (int h = 0; h < embeddings.Count; h++)
        {
            var x = embeddings[h];
            foreach (var lstm in Encoder) x = lstm.Forward(x);                                   // [length, width]
            rows[h] = _engine.TensorSlice(x, new[] { x.Shape[0] - 1, 0 }, new[] { 1, _width });
        }
        return rows.Length == 1 ? rows[0] : _engine.TensorConcatenate(rows, 0);
    }

    /// <summary>The bias logits <c>[n, classes]</c> for the acoustic embeddings and the ASR decoder's hidden states (both
    /// <c>[n, width]</c>) attending over the hotwords <c>[m, width]</c>.</summary>
    public Tensor<T> Logits(Tensor<T> hotwords, Tensor<T> acoustic, Tensor<T> hidden) =>
        Output.Forward(_engine.TensorAdd(Decoder.Forward(hotwords, acoustic), Decoder.Forward(hotwords, hidden)));

    /// <summary>Attention score filtering (section 3.2): the indices of the <paramref name="keep"/> hotwords the bias
    /// decoder's last layer attends to most from the ASR decoder's hidden states, followed by the last hotword (the
    /// default one), in their original order.</summary>
    public int[] Filter(Tensor<T> hotwords, Tensor<T> hidden, int keep)
    {
        var totals = Decoder.LastAttentionTotals(hotwords, hidden);
        int last = totals.Length - 1;
        return Enumerable.Range(0, last).OrderByDescending(i => totals[i]).Take(Math.Min(keep, last)).OrderBy(i => i)
            .Concat(new[] { last }).ToArray();
    }
}
