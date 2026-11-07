using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>OpenVoice's masked, speaker-conditioned, weight-normalized WaveNet residual/skip stack.</summary>
internal sealed class OpenVoiceWaveNet<T>
{
    private readonly IEngine _engine;
    private readonly int _hidden, _conditionChannels, _kernel;
    private readonly List<NormedConv1DLayer<T>> _inputs = new(), _residuals = new();
    private readonly NormedConv1DLayer<T> _condition;
    private readonly List<LayerBase<T>> _layers = new();
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    internal OpenVoiceWaveNet(IEngine engine, int hidden, int kernel, int count, int conditionChannels)
    {
        if (hidden <= 0 || kernel <= 0 || kernel % 2 == 0 || count <= 0 || conditionChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(hidden));
        _engine = engine; _hidden = hidden; _conditionChannels = conditionChannels; _kernel = kernel;
        _condition = Add(conditionChannels, 2 * hidden * count, 1);
        for (int i = 0; i < count; i++)
        {
            _inputs.Add(Add(hidden, 2 * hidden, kernel));
            _residuals.Add(Add(hidden, i == count - 1 ? hidden : 2 * hidden, 1));
        }
    }

    private NormedConv1DLayer<T> Add(int input, int output, int kernel)
    {
        var layer = new NormedConv1DLayer<T>(input, output, kernel, 1, 1, 1, kernel / 2,
            false, ConvolutionNormalization.Weight);
        _layers.Add(layer); return layer;
    }

    public Tensor<T> Forward(Tensor<T> input, Tensor<T> mask, Tensor<T> condition)
    {
        int batch = input.Shape[0], time = input.Shape[2];
        var projected = _condition.Forward(condition);
        if (projected.Shape[2] == 1 && time != 1)
            projected = _engine.TensorTile(projected, new[] { 1, 1, time });
        var hiddenMask = _engine.TensorTile(mask, new[] { 1, _hidden, 1 });
        Tensor<T> Slice(Tensor<T> x, int offset, int width) => _engine.TensorSlice(x,
            new[] { 0, offset, 0 }, new[] { batch, width, time });
        var x = input;
        Tensor<T>? output = null;
        for (int i = 0; i < _inputs.Count; i++)
        {
            var activation = _engine.TensorAdd(_inputs[i].Forward(x), Slice(projected, 2 * _hidden * i, 2 * _hidden));
            var gated = _engine.TensorMultiply(_engine.Tanh(Slice(activation, 0, _hidden)),
                _engine.Sigmoid(Slice(activation, _hidden, _hidden)));
            var residual = _residuals[i].Forward(gated);
            Tensor<T> skip;
            if (i < _inputs.Count - 1)
            {
                x = _engine.TensorMultiply(_engine.TensorAdd(x, Slice(residual, 0, _hidden)), hiddenMask);
                skip = Slice(residual, _hidden, _hidden);
            }
            else skip = residual;
            output = output is null ? skip : _engine.TensorAdd(output, skip);
        }
        return _engine.TensorMultiply(output!, hiddenMask);
    }

    public void LoadTorch(Func<string, int[], double[]> read, string prefix)
    {
        void Load(NormedConv1DLayer<T> layer, string name, int input, int output, int kernel)
            => layer.LoadTorchWeights(read(name + ".weight_v", new[] { output, input, kernel }),
                read(name + ".weight_g", new[] { output, 1, 1 }), read(name + ".bias", new[] { output }));
        Load(_condition, prefix + "cond_layer", _conditionChannels, 2 * _hidden * _inputs.Count, 1);
        for (int i = 0; i < _inputs.Count; i++)
        {
            Load(_inputs[i], prefix + "in_layers." + i, _hidden, 2 * _hidden, _kernel);
            Load(_residuals[i], prefix + "res_skip_layers." + i, _hidden,
                i < _inputs.Count - 1 ? 2 * _hidden : _hidden, 1);
        }
    }
}
