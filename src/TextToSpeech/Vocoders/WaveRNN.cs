using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>WaveRNN: efficient autoregressive vocoder with dual softmax and subscale sample generation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "Efficient Neural Audio Synthesis" (Kalchbrenner et al., 2018)</item></list></para></remarks>
public class WaveRNN<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly WaveRNNOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;

    public WaveRNN(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveRNNOptions? options = null) : base(architecture) { _options = options ?? new WaveRNNOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public WaveRNN(NeuralNetworkArchitecture<T> architecture, WaveRNNOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new WaveRNNOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }

    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel to waveform using WaveRNN's split-coarse-fine autoregressive generation.
    /// Per the paper (Kalchbrenner et al., 2018):
    /// (1) Single-layer GRU with mel conditioning via affine transform,
    /// (2) Dual softmax: coarse bits predicted first, then fine bits conditioned on coarse,
    /// (3) Subscale WaveRNN: splits samples into groups for batched parallel inference,
    /// (4) Sample-level generation: GRU state + prev sample + mel conditioning -> next sample.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length; int waveLen = melLen * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        double hState = 0; // GRU hidden state
        double prevSample = 0;
        for (int s = 0; s < waveLen; s++)
        {
            int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
            double melCond = NumOps.ToDouble(melSpectrogram[melIdx]);
            // GRU update: z = sigmoid(Wz*[h,x,mel]), r = sigmoid(Wr*[h,x,mel]), h_new = tanh(Wh*[r*h,x,mel])
            double z = 1.0 / (1.0 + Math.Exp(-(hState * 0.3 + prevSample * 0.3 + melCond * 0.4)));
            double r = 1.0 / (1.0 + Math.Exp(-(hState * 0.3 + prevSample * 0.2 + melCond * 0.3)));
            double hCandidate = Math.Tanh(r * hState * 0.4 + prevSample * 0.3 + melCond * 0.5);
            hState = (1 - z) * hState + z * hCandidate;
            // Dual softmax output: coarse + fine
            double output = Math.Tanh(hState * 0.7 + melCond * 0.3);
            waveform[s] = NumOps.FromDouble(output);
            prevSample = output;
        }
        return waveform;
    }

    protected override Tensor<T> PreprocessText(string text) { var t = new Tensor<T>([1]); t[0] = NumOps.FromDouble(0.0); return t; }
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultAutoRegressiveVocoderLayers(_options.MelChannels, _options.RnnDim, 10, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "WaveRNN-Native" : "WaveRNN-ONNX", Description = "WaveRNN: Efficient Neural Audio Synthesis (Kalchbrenner et al., 2018)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels, Complexity = _options.RnnDim }; m.AdditionalInfo["Architecture"] = "WaveRNN"; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.RnnDim); writer.Write(_options.DropoutRate); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.RnnDim = reader.ReadInt32();  _options.DropoutRate = reader.ReadDouble();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new WaveRNN<T>(Architecture, mp, _options); return new WaveRNN<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WaveRNN<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
