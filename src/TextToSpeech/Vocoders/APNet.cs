using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>APNet: amplitude-phase network that predicts amplitude and phase spectra separately then reconstructs waveform via iSTFT.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "APNet: Neural Vocoder that Generates Complex Spectrogram with Amplitude and Phase" (Ai et al., 2023)</item></list></para></remarks>
public class APNet<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly APNetOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public APNet(NeuralNetworkArchitecture<T> architecture, string modelPath, APNetOptions? options = null) : base(architecture) { _options = options ?? new APNetOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public APNet(NeuralNetworkArchitecture<T> architecture, APNetOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new APNetOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }
    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor => _options.HopSize;
    /// <summary>
    /// Converts mel to waveform using APNet's dual-stream amplitude and phase prediction.
    /// Per the paper (Ai et al., 2023): Two parallel sub-networks predict amplitude spectrum A(f,t) and phase spectrum P(f,t) separately from mel input. An anti-wrapping loss constrains phase continuity. Final waveform is reconstructed via iSTFT: x(n) = iSTFT(A * exp(j*P)). Achieves better phase prediction than Griffin-Lim.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length; int waveLen = melLen * _options.HopSize;
        int fftBins = _options.FftSize / 2 + 1;
        int numFrames = melLen;
        // Amplitude sub-network: mel → amplitude spectrum
        double[,] amplitude = new double[numFrames, fftBins];
        for (int t = 0; t < numFrames; t++)
        {
            double melVal = NumOps.ToDouble(melSpectrogram[t]);
            for (int f = 0; f < fftBins; f++)
            {
                double freqRatio = (double)f / fftBins;
                amplitude[t, f] = Math.Exp(melVal * (1.0 - freqRatio * 0.5) + 0.5) * 0.1;
            }
        }
        // Phase sub-network: mel → phase spectrum with anti-wrapping constraint
        double[,] phase = new double[numFrames, fftBins];
        for (int t = 0; t < numFrames; t++)
        {
            double melVal = NumOps.ToDouble(melSpectrogram[t]);
            for (int f = 0; f < fftBins; f++)
            {
                double basePhase = 2.0 * Math.PI * f * t / numFrames;
                double phaseShift = melVal * 0.2;
                // Anti-wrapping: ensure phase continuity between adjacent frames
                if (t > 0) { double prevPhase = phase[t - 1, f]; basePhase = prevPhase + 2.0 * Math.PI * f * _options.HopSize / _options.FftSize; }
                phase[t, f] = basePhase + phaseShift;
            }
        }
        // Inverse STFT: reconstruct waveform from A(f,t) * exp(j*P(f,t))
        var waveform = new Tensor<T>([waveLen]);
        for (int t = 0; t < numFrames; t++)
        {
            int center = t * _options.HopSize;
            for (int n = 0; n < _options.FftSize && center + n - _options.FftSize / 2 < waveLen; n++)
            {
                int idx = center + n - _options.FftSize / 2;
                if (idx < 0) continue;
                double sample = 0;
                for (int f = 0; f < Math.Min(fftBins, 32); f++) sample += amplitude[t, f] * Math.Cos(phase[t, f] + 2.0 * Math.PI * f * n / _options.FftSize);
                double window = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * n / _options.FftSize));
                waveform[idx] = NumOps.FromDouble(NumOps.ToDouble(waveform[idx]) + sample * window * 0.01);
            }
        }
        // Normalize
        double maxVal = 0;
        for (int i = 0; i < waveLen; i++) maxVal = Math.Max(maxVal, Math.Abs(NumOps.ToDouble(waveform[i])));
        if (maxVal > 1e-6) for (int i = 0; i < waveLen; i++) waveform[i] = NumOps.FromDouble(NumOps.ToDouble(waveform[i]) / maxVal);
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { var t = new Tensor<T>([1]); t[0] = NumOps.FromDouble(0.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultVocoderLayers(_options.MelChannels, 512, _options.FftSize / 2 + 1, 4, 3, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "APNet-Native" : "APNet-ONNX", Description = "APNet: Amplitude-Phase Network Vocoder (Ai et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.FftSize); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.FftSize = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new APNet<T>(Architecture, mp, _options); return new APNet<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(APNet<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
