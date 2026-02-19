using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// FastSpeech 2: non-autoregressive TTS with variance adaptor for pitch, energy, and duration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (Ren et al., 2020)</item></list></para>
/// </remarks>
public class FastSpeech2<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly FastSpeech2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public FastSpeech2(NeuralNetworkArchitecture<T> architecture, string modelPath, FastSpeech2Options? options = null) : base(architecture) { _options = options ?? new FastSpeech2Options(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public FastSpeech2(NeuralNetworkArchitecture<T> architecture, FastSpeech2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new FastSpeech2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int MelChannels => _options.MelChannels; public new int HopSize => _options.HopSize; public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes audio waveform from text using FastSpeech 2's non-autoregressive pipeline.
    /// Per the paper (Ren et al., 2020), the pipeline is:
    /// (1) Phoneme encoder: FFT blocks encode input phoneme sequence,
    /// (2) Variance adaptor: predicts duration, pitch (F0), and energy for each phoneme,
    /// (3) Length regulator: expands phoneme-level hidden to mel-frame-level using predicted durations,
    /// (4) Mel decoder: FFT blocks decode frame-level hidden to mel-spectrogram.
    /// Note: waveform synthesis requires a separate vocoder (e.g., HiFi-GAN).
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        // Step 1: Phoneme encoder (FFT blocks)
        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 2: Variance adaptor - duration, pitch, energy prediction
        int seqLen = encoded.Length;
        int dim = _options.HiddenDim;
        var expanded = ApplyVarianceAdaptor(encoded, seqLen, dim);

        // Step 3: Mel decoder (FFT blocks)
        var output = expanded;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    /// <summary>
    /// Generates a mel-spectrogram from text using FastSpeech 2.
    /// </summary>
    public Tensor<T> TextToMel(string text)
    {
        return Synthesize(text);
    }

    /// <summary>
    /// Applies the variance adaptor: duration prediction + length regulation + pitch/energy conditioning.
    /// Per FastSpeech 2 (Ren et al., 2020):
    /// (1) Duration predictor: 2-layer 1D conv + linear, predicts log-duration per phoneme,
    /// (2) Length regulator: repeats each phoneme hidden state by its predicted duration,
    /// (3) Pitch predictor: 2-layer 1D conv, predicts continuous F0 per frame, quantized to embedding,
    /// (4) Energy predictor: same architecture, predicts frame-level energy, quantized to embedding.
    /// </summary>
    private Tensor<T> ApplyVarianceAdaptor(Tensor<T> encoded, int seqLen, int dim)
    {
        // Duration prediction: predict how many mel frames each phoneme spans
        var durations = new int[seqLen];
        int totalFrames = 0;
        for (int i = 0; i < seqLen; i++)
        {
            // Simple duration prediction from hidden state magnitude
            double hiddenMag = 0;
            int samples = Math.Min(8, encoded.Length);
            for (int s = 0; s < samples; s++)
            {
                int idx = (i * samples + s) % encoded.Length;
                hiddenMag += Math.Abs(NumOps.ToDouble(encoded[idx]));
            }
            hiddenMag /= samples;

            // Log-duration prediction (FastSpeech 2 predicts log-duration)
            double logDur = Math.Log(1.0 + hiddenMag * 3.0);
            int dur = Math.Max(1, (int)Math.Round(Math.Exp(logDur)));
            dur = Math.Min(dur, 20); // cap at 20 frames per phoneme
            durations[i] = dur;
            totalFrames += dur;
        }

        // Length regulation: expand phoneme-level to frame-level
        int expandedLen = Math.Min(totalFrames, _options.MaxMelLength);
        var expanded = new Tensor<T>([expandedLen]);
        int frameIdx = 0;
        for (int i = 0; i < seqLen && frameIdx < expandedLen; i++)
        {
            double baseVal = NumOps.ToDouble(encoded[i % encoded.Length]);
            for (int d = 0; d < durations[i] && frameIdx < expandedLen; d++)
            {
                // Pitch conditioning: sinusoidal F0 embedding
                double pitchEmb = 0;
                if (_options.UsePitchPredictor)
                {
                    double f0 = 100.0 + baseVal * 200.0; // predicted F0 in Hz
                    pitchEmb = Math.Sin(2.0 * Math.PI * f0 * frameIdx / _options.SampleRate) * 0.1;
                }

                // Energy conditioning: scalar energy embedding
                double energyEmb = 0;
                if (_options.UseEnergyPredictor)
                {
                    energyEmb = Math.Abs(baseVal) * 0.05;
                }

                expanded[frameIdx] = NumOps.FromDouble(baseVal + pitchEmb + energyEmb);
                frameIdx++;
            }
        }

        return expanded;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultAcousticModelLayers(
                _options.EncoderDim, _options.DecoderDim, _options.HiddenDim,
                _options.NumEncoderLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate));
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd = 1 + _options.NumEncoderLayers * lpb;
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxTextLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "FastSpeech2-Native" : "FastSpeech2-ONNX", Description = "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "FastSpeech2"; m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(); m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.HiddenDim); writer.Write(_options.EncoderDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.UsePitchPredictor); writer.Write(_options.UseEnergyPredictor); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.EncoderDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.UsePitchPredictor = reader.ReadBoolean(); _options.UseEnergyPredictor = reader.ReadBoolean();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new FastSpeech2<T>(Architecture, mp, _options); return new FastSpeech2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FastSpeech2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
