using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>
/// OpenVoice's tone-color reference encoder (myshell-ai/OpenVoice, models.ReferenceEncoder):
/// linear-magnitude spectrogram, optional frequency layer normalization, six weight-normalized
/// stride-two convolutions, a PyTorch reset-after GRU, and an unnormalized speaker projection.
/// </summary>
internal sealed class OpenVoiceReferenceEncoder<T>
{
    private readonly IEngine _engine;
    private readonly int _frequencies, _speakerChannels, _gruInputs;
    private readonly LayerNormalizationLayer<T>? _normalization;
    private readonly List<NormedConv2DLayer<T>> _convolutions = new();
    private readonly DenseLayer<T> _input, _recurrent, _projection;
    private readonly List<LayerBase<T>> _layers = new();
    private static readonly int[] Filters = { 1, 32, 32, 64, 64, 128, 128 };
    private const int Hidden = 128;

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    public OpenVoiceReferenceEncoder(IEngine engine, int frequencies, int speakerChannels = 256,
        bool layerNormalization = true)
    {
        if (frequencies <= 0) throw new ArgumentOutOfRangeException(nameof(frequencies));
        if (speakerChannels <= 0) throw new ArgumentOutOfRangeException(nameof(speakerChannels));
        _engine = engine;
        _frequencies = frequencies;
        _speakerChannels = speakerChannels;
        if (layerNormalization)
        {
            _normalization = new LayerNormalizationLayer<T>(frequencies, 1e-5);
            _layers.Add(_normalization);
        }
        int reduced = frequencies;
        for (int i = 0; i < 6; i++)
        {
            var convolution = new NormedConv2DLayer<T>(Filters[i], Filters[i + 1], 3, 3,
                2, 2, 1, 1, 1, ConvolutionNormalization.Weight);
            _convolutions.Add(convolution);
            _layers.Add(convolution);
            reduced = (reduced + 1) / 2;
        }
        _gruInputs = 128 * reduced;
        _input = new DenseLayer<T>(3 * Hidden, new IdentityActivation<T>() as IActivationFunction<T>);
        _recurrent = new DenseLayer<T>(3 * Hidden, new IdentityActivation<T>() as IActivationFunction<T>);
        _projection = new DenseLayer<T>(speakerChannels, new IdentityActivation<T>() as IActivationFunction<T>);
        _layers.Add(_input);
        _layers.Add(_recurrent);
        _layers.Add(_projection);
    }

    /// <summary>Maps [batch, time, FFT/2+1] linear magnitudes to [batch, speakerChannels].</summary>
    public Tensor<T> Forward(Tensor<T> spectrogram)
    {
        if (spectrogram.Rank != 3 || spectrogram.Shape[2] != _frequencies || spectrogram.Shape[1] <= 0)
            throw new ArgumentException("Expected a nonempty [batch, time, frequency] spectrogram.", nameof(spectrogram));
        int batch = spectrogram.Shape[0];
        var features = _engine.Reshape(spectrogram, new[] { batch, 1, spectrogram.Shape[1], _frequencies });
        if (_normalization is not null) features = _normalization.Forward(features);
        foreach (var convolution in _convolutions) features = _engine.ReLU(convolution.Forward(features));
        int time = features.Shape[2];
        var sequence = _engine.Reshape(_engine.TensorPermute(features, new[] { 0, 2, 1, 3 }),
            new[] { batch, time, _gruInputs });
        var state = new Tensor<T>(new[] { batch, Hidden });
        for (int t = 0; t < time; t++)
        {
            var step = _engine.Reshape(_engine.TensorSlice(sequence, new[] { 0, t, 0 },
                new[] { batch, 1, _gruInputs }), new[] { batch, _gruInputs });
            var input = _input.Forward(step);
            var recurrent = _recurrent.Forward(state);
            Tensor<T> Gate(Tensor<T> value, int index) => _engine.TensorSlice(value,
                new[] { 0, index * Hidden }, new[] { batch, Hidden });
            var reset = _engine.Sigmoid(_engine.TensorAdd(Gate(input, 0), Gate(recurrent, 0)));
            var update = _engine.Sigmoid(_engine.TensorAdd(Gate(input, 1), Gate(recurrent, 1)));
            // PyTorch applies reset AFTER the recurrent affine, including its own candidate bias.
            // GRUCellLayer uses reset-before and cannot represent these checkpoint weights.
            var candidate = _engine.Tanh(_engine.TensorAdd(Gate(input, 2),
                _engine.TensorMultiply(reset, Gate(recurrent, 2))));
            state = _engine.TensorAdd(candidate,
                _engine.TensorMultiply(update, _engine.TensorSubtract(state, candidate)));
        }
        return _projection.Forward(state);
    }

    public void LoadTorch(Func<string, int[], double[]> read, string prefix = "ref_enc.") =>
        LoadTorch(AiDotNet.ModelLoading.TorchTensorSource.From(read), prefix);

    /// <summary>Loads the reference encoder straight from <paramref name="source"/>.</summary>
    internal void LoadTorch(AiDotNet.ModelLoading.TorchTensorSource source, string prefix = "ref_enc.")
    {
        if (_normalization is not null)
            TorchParameters.LayerNorm(_engine, _normalization, source, prefix + "layernorm.weight", prefix + "layernorm.bias");
        for (int i = 0; i < _convolutions.Count; i++)
        {
            string key = prefix + "convs." + i;
            _convolutions[i].LoadTorchWeights(source, key + ".weight_v", key + ".weight_g", key + ".bias");
        }
        TorchParameters.Linear(_engine, _input, _gruInputs, 3 * Hidden, source, prefix + "gru.weight_ih_l0", prefix + "gru.bias_ih_l0");
        TorchParameters.Linear(_engine, _recurrent, Hidden, 3 * Hidden, source, prefix + "gru.weight_hh_l0", prefix + "gru.bias_hh_l0");
        TorchParameters.Linear(_engine, _projection, Hidden, _speakerChannels, source, prefix + "proj.weight", prefix + "proj.bias");
    }
}
