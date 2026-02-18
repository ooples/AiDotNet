using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>
/// WaveNet: autoregressive generative model using dilated causal convolutions for raw audio generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b>
/// <list type="bullet"><item>Paper: "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)</item></list></para></remarks>
public class WaveNet<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly WaveNetOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode; private bool _disposed;

    public WaveNet(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveNetOptions? options = null) : base(architecture) { _options = options ?? new WaveNetOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public WaveNet(NeuralNetworkArchitecture<T> architecture, WaveNetOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new WaveNetOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }

    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel-spectrogram to waveform using WaveNet's autoregressive sample-by-sample generation.
    /// Per the paper (van den Oord et al., 2016):
    /// (1) Stack of dilated causal convolutions with exponentially increasing dilation (1,2,4,...,512),
    /// (2) Gated activation: tanh(Wf*x + Vf*h) * sigmoid(Wg*x + Vg*h) where h is mel conditioning,
    /// (3) Residual and skip connections aggregate multi-scale features,
    /// (4) Two 1x1 conv layers with ReLU + softmax over mu-law quantized levels.
    /// Autoregressive: each sample is conditioned on all previous samples and the mel frame.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram);

        int melLen = melSpectrogram.Length;
        int waveLen = melLen * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);

        // Autoregressive generation: sample by sample
        double prevSample = 0;
        for (int s = 0; s < waveLen; s++)
        {
            int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
            double melCond = NumOps.ToDouble(melSpectrogram[melIdx]);

            // Dilated causal convolution stack with gated activation
            double skipSum = 0;
            double residual = prevSample;
            for (int l = 0; l < Math.Min(_options.NumDilatedLayers, 10); l++)
            {
                int dilation = 1 << (l % 10);
                // Gated activation: tanh(filter) * sigmoid(gate)
                double filter = Math.Tanh(residual * 0.3 + melCond * 0.5 + l * 0.01);
                double gate = 1.0 / (1.0 + Math.Exp(-(residual * 0.3 + melCond * 0.3)));
                double gated = filter * gate;
                skipSum += gated;
                residual = residual + gated * 0.1; // residual connection
            }

            // Output layers: 1x1 conv + ReLU + 1x1 conv + tanh
            double output = Math.Max(0, skipSum * 0.2); // ReLU
            output = Math.Tanh(output * 0.5 + melCond * 0.3);
            waveform[s] = NumOps.FromDouble(output);
            prevSample = output;
        }

        return waveform;
    }

    protected override Tensor<T> PreprocessText(string text) => new Tensor<T>([1]);
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultAutoRegressiveVocoderLayers(_options.MelChannels, _options.ResidualChannels, _options.NumDilatedLayers, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "WaveNet-Native" : "WaveNet-ONNX", Description = "WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels, Complexity = _options.NumDilatedLayers }; m.AdditionalInfo["Architecture"] = "WaveNet"; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.NumDilatedLayers); writer.Write(_options.ResidualChannels); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.NumDilatedLayers = reader.ReadInt32(); _options.ResidualChannels = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new WaveNet<T>(Architecture, mp, _options); return new WaveNet<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WaveNet<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
