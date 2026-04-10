using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Models;

/// <summary>
/// The Harmonic Resonance Engine — a novel neural architecture that replaces the traditional
/// neuron-weight-bias paradigm with spectral communication via orthogonal frequency carriers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional neural networks have neurons connected by weights — each neuron
/// computes a weighted sum of its inputs and applies an activation function. The HRE works differently:
///
/// 1. Mellin-Fourier Layer (optional): Makes the input scale-and-shift-invariant
/// 2. OFDM Layers: Encode features onto frequency carriers and create cross-feature
///    interactions via intermodulation distortion
/// 3. IMD Attention: Computes attention scores at O(N log N) via wave interference
/// 4. Spectral Sparsity: Keeps only the K most important frequency components
/// 5. Output Projection: Maps spectral features to the desired output dimension
///
/// Key advantages over traditional architectures:
/// - O(N log N) complexity for attention-like operations (vs O(N^2))
/// - Single-pass Hebbian learning (vs multi-epoch backpropagation)
/// - Spectral compression (K coefficients vs N^2 weights)
/// - Built-in periodicity awareness (natural for cyclical data)
/// </para>
/// </remarks>
public class HREModel<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly HREModelOptions _options;

    // Pipeline stages
    private MellinFourierLayer<T>? _mellinFourierLayer;
    private readonly List<OFDMLayer<T>> _ofdmLayers = [];
    private IMDAttentionLayer<T>? _attentionLayer;
    private readonly SpectralSparsityMask<T> _sparsityMask;

    // Output projection (simple linear layer for mapping spectral features to output)
    private Vector<T> _outputWeights;
    private T _outputBias;

    /// <summary>
    /// Gets the model configuration options.
    /// </summary>
    public HREModelOptions Options => _options;

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            int count = _options.OutputSize * _options.CarrierCount + _options.OutputSize; // Output projection
            if (_mellinFourierLayer is not null) count += _mellinFourierLayer.ParameterCount;
            foreach (var layer in _ofdmLayers) count += layer.ParameterCount;
            if (_attentionLayer is not null) count += _attentionLayer.ParameterCount;
            return count;
        }
    }

    /// <summary>
    /// Gets the last computed attention weights (for visualization).
    /// </summary>
    public Matrix<T>? LastAttentionWeights => _attentionLayer?.LastAttentionWeights;

    /// <summary>
    /// Creates a new Harmonic Resonance Engine model.
    /// </summary>
    /// <param name="options">Model configuration options.</param>
    public HREModel(HREModelOptions options)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options;
        _sparsityMask = new SpectralSparsityMask<T>();

        BuildPipeline();

        // Initialize output projection
        _outputWeights = new Vector<T>(options.OutputSize * options.CarrierCount);
        _outputBias = _numOps.Zero;
        InitializeOutputWeights();
    }

    /// <summary>
    /// Forward pass through the full HRE pipeline.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor of shape [OutputSize].</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;

        // Stage 1: Mellin-Fourier invariance (optional)
        if (_mellinFourierLayer is not null)
        {
            current = _mellinFourierLayer.Forward(current);
        }

        // Stage 2: OFDM layers (spectral encode → nonlinearity → decode)
        foreach (var ofdmLayer in _ofdmLayers)
        {
            current = ofdmLayer.Forward(current);
        }

        // Stage 3: IMD attention
        if (_attentionLayer is not null)
        {
            current = _attentionLayer.Forward(current);
        }

        // Stage 4: Output projection (linear map from carrier features to output)
        var output = new Tensor<T>([_options.OutputSize]);
        int carrierCount = Math.Min(current.Length, _options.CarrierCount);

        for (int o = 0; o < _options.OutputSize; o++)
        {
            T sum = _outputBias;
            for (int i = 0; i < carrierCount; i++)
            {
                sum = _numOps.Add(sum,
                    _numOps.Multiply(_outputWeights[o * _options.CarrierCount + i], current[i]));
            }
            output[o] = sum;
        }

        return output;
    }

    /// <summary>
    /// Sets the model to training mode (enables Hebbian updates during forward pass).
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        _mellinFourierLayer?.SetTrainingMode(isTraining);
        foreach (var layer in _ofdmLayers) layer.SetTrainingMode(isTraining);
        _attentionLayer?.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Gets a summary of the model architecture.
    /// </summary>
    public string GetSummary()
    {
        var lines = new List<string>
        {
            "=== Harmonic Resonance Engine ===",
            $"Input size: {_options.InputSize}",
            $"Output size: {_options.OutputSize}",
            $"Carriers: {_options.CarrierCount}",
            $"FFT size: {_options.FftSize}",
            $"Nonlinearity: {_options.Nonlinearity}",
            $"Learning rule: {_options.LearningRule}",
            $"Sparsity K: {(_options.UseMDLAutoK ? "auto (MDL)" : _options.SparsityK.ToString())}",
            $"Mellin-Fourier: {_options.UseMellinFourier}",
            ""
        };

        if (_mellinFourierLayer is not null)
            lines.Add($"  MellinFourierLayer: {_mellinFourierLayer.LayerName} ({_mellinFourierLayer.ParameterCount} params)");

        for (int i = 0; i < _ofdmLayers.Count; i++)
            lines.Add($"  OFDMLayer[{i}]: {_ofdmLayers[i].LayerName} ({_ofdmLayers[i].ParameterCount} params)");

        if (_attentionLayer is not null)
            lines.Add($"  IMDAttentionLayer: {_attentionLayer.LayerName} ({_attentionLayer.ParameterCount} params)");

        lines.Add($"  OutputProjection: {_options.CarrierCount} -> {_options.OutputSize}");
        lines.Add($"");
        lines.Add($"Total parameters: {ParameterCount}");

        return string.Join(Environment.NewLine, lines);
    }

    private void BuildPipeline()
    {
        // Stage 1: Mellin-Fourier invariance layer
        if (_options.UseMellinFourier)
        {
            // Input signal → invariant fingerprint → carrier features
            int fingerPrintSize = Math.Min(_options.InputSize, _options.CarrierCount);
            _mellinFourierLayer = new MellinFourierLayer<T>(_options.InputSize, fingerPrintSize);
        }

        // Stage 2: OFDM layers
        int currentSize = _options.UseMellinFourier
            ? Math.Min(_options.InputSize, _options.CarrierCount)
            : _options.InputSize;

        for (int i = 0; i < _options.NumOFDMLayers; i++)
        {
            var ofdmLayer = new OFDMLayer<T>(
                currentSize, _options.CarrierCount, _options.FftSize, _options.Nonlinearity);
            _ofdmLayers.Add(ofdmLayer);
            currentSize = _options.CarrierCount;
        }

        // Stage 3: IMD attention
        if (_options.NumAttentionLayers > 0)
        {
            _attentionLayer = new IMDAttentionLayer<T>(_options.CarrierCount, _options.FftSize);
        }
    }

    private void InitializeOutputWeights()
    {
        // Xavier initialization
        double scale = Math.Sqrt(2.0 / (_options.CarrierCount + _options.OutputSize));
        var rng = _options.Seed.HasValue
            ? new Random(_options.Seed.Value)
            : Random.Shared;

        for (int i = 0; i < _outputWeights.Length; i++)
        {
            _outputWeights[i] = _numOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * scale);
        }
    }
}
