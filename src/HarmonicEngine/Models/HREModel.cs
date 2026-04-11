using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;

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
[ModelDomain(ModelDomain.General)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelDomain(ModelDomain.Finance)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "The Harmonic Resonance Engine: A Spectral Architecture for Neural Computation via Intermodulation",
    "https://github.com/ooples/AiDotNet/blob/master/src/HarmonicEngine/Paper/theorems.tex",
    Year = 2026,
    Authors = "AiDotNet Contributors")]
public class HREModel<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{
    private readonly HREModelOptions _options;

    // Pipeline stages
    private MellinFourierLayer<T>? _mellinFourierLayer;
    private readonly List<OFDMLayer<T>> _ofdmLayers = [];
    private readonly List<IMDAttentionLayer<T>> _attentionLayers = [];
    private readonly SpectralSparsityMask<T> _sparsityMask;

    // Output projection (simple linear layer for mapping spectral features to output).
    // Stored as a Matrix [OutputSize x CarrierCount] for efficient vectorized mat-vec.
    private Matrix<T> _outputWeights;
    private T _outputBias;

    // Cached features from last forward pass (needed for gradient-based weight update in Train)
    private Vector<T>? _lastFeatures;

    /// <summary>
    /// Gets the model configuration options.
    /// </summary>
    public HREModelOptions Options => _options;

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            // Output projection: weights (OutputSize * CarrierCount) + 1 scalar bias
            int count = _options.OutputSize * _options.CarrierCount + 1;
            if (_mellinFourierLayer is not null) count += _mellinFourierLayer.ParameterCount;
            foreach (var layer in _ofdmLayers) count += layer.ParameterCount;
            foreach (var layer in _attentionLayers) count += layer.ParameterCount;
            return count;
        }
    }

    /// <inheritdoc/>
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Gets the last computed attention weights from the first attention layer (for visualization).
    /// </summary>
    public Matrix<T>? LastAttentionWeights => _attentionLayers.Count > 0 ? _attentionLayers[0].LastAttentionWeights : null;

    /// <summary>
    /// Creates a new Harmonic Resonance Engine model.
    /// </summary>
    /// <param name="options">Model configuration options.</param>
    public HREModel(HREModelOptions options)
    {
        _options = options;
        _sparsityMask = new SpectralSparsityMask<T>();

        BuildPipeline();

        // Initialize output projection
        _outputWeights = new Matrix<T>(options.OutputSize, options.CarrierCount);
        _outputBias = NumOps.Zero;
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

        // Stage 2: OFDM layers (spectral encode -> nonlinearity -> decode)
        foreach (var ofdmLayer in _ofdmLayers)
        {
            current = ofdmLayer.Forward(current);
        }

        // Stage 3: IMD attention
        foreach (var attnLayer in _attentionLayers)
        {
            current = attnLayer.Forward(current);
        }

        // Stage 4: Spectral sparsity — keep only top-K strongest components
        current = ApplySparsity(current);

        // Stage 5: Output projection via a single vectorized matrix-vector multiply.
        // Build features vector once at full CarrierCount size (zero-padded if needed),
        // then do one mat-vec instead of per-row allocations.
        _lastFeatures = new Vector<T>(_options.CarrierCount);
        int carrierCount = Math.Min(current.Length, _options.CarrierCount);
        for (int i = 0; i < carrierCount; i++) _lastFeatures[i] = current[i];

        var outputVec = _outputWeights.Multiply(_lastFeatures);
        var output = new Tensor<T>([_options.OutputSize]);
        for (int o = 0; o < _options.OutputSize; o++)
        {
            output[o] = NumOps.Add(outputVec[o], _outputBias);
        }

        return output;
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var prediction = Forward(input);

        // Gradient descent on output projection: dL/dw_oj = error_o * feature_j
        // Normalize by ||features||² for stability (NLMS-style update) — prevents
        // divergence when MellinFourier or other layers produce large-magnitude features.
        var lr = NumOps.FromDouble(_options.HebbianLearningRate);
        int carrierCount = _lastFeatures is not null ? _lastFeatures.Length : _options.CarrierCount;

        // Compute ||features||² for normalization
        double featurePower = 0;
        if (_lastFeatures is not null)
        {
            for (int j = 0; j < _lastFeatures.Length; j++)
            {
                double fj = NumOps.ToDouble(_lastFeatures[j]);
                featurePower += fj * fj;
            }
        }
        var normFactor = NumOps.FromDouble(1.0 / (featurePower + 1e-8));

        for (int o = 0; o < _options.OutputSize; o++)
        {
            T error = NumOps.Subtract(prediction[o], expectedOutput[o]);
            T scaledError = NumOps.Multiply(NumOps.Multiply(lr, error), normFactor);

            // Update output weights: w_oj -= (lr/||f||²) * error * feature_j
            if (_lastFeatures is not null)
            {
                for (int j = 0; j < _options.CarrierCount; j++)
                {
                    T grad = NumOps.Multiply(scaledError, _lastFeatures[j]);
                    _outputWeights[o, j] = NumOps.Subtract(_outputWeights[o, j], grad);
                }
            }

            // Update bias: b -= lr * error (unnormalized — bias is scalar)
            _outputBias = NumOps.Subtract(_outputBias, NumOps.Multiply(lr, error));
        }
    }

    /// <summary>
    /// Sets the model to training mode (enables Hebbian updates during forward pass).
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        _mellinFourierLayer?.SetTrainingMode(isTraining);
        foreach (var layer in _ofdmLayers) layer.SetTrainingMode(isTraining);
        foreach (var layer in _attentionLayers) layer.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int rows = _outputWeights.Rows;
        int cols = _outputWeights.Columns;
        int totalParams = rows * cols + 1; // weights + bias
        var parameters = new Vector<T>(totalParams);
        int idx = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                parameters[idx++] = _outputWeights[r, c];
            }
        }
        parameters[idx] = _outputBias;
        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int rows = _outputWeights.Rows;
        int cols = _outputWeights.Columns;
        int weightCount = rows * cols;
        int idx = 0;
        for (int r = 0; r < rows && idx < parameters.Length; r++)
        {
            for (int c = 0; c < cols && idx < parameters.Length; c++)
            {
                _outputWeights[r, c] = parameters[idx++];
            }
        }
        if (parameters.Length > weightCount)
        {
            _outputBias = parameters[weightCount];
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = new HREModel<T>(_options);
        clone.SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var clone = new HREModel<T>(_options);
        clone.SetParameters(GetParameters());
        return clone;
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

        for (int i = 0; i < _attentionLayers.Count; i++)
            lines.Add($"  IMDAttentionLayer[{i}]: {_attentionLayers[i].LayerName} ({_attentionLayers[i].ParameterCount} params)");

        lines.Add($"  OutputProjection: {_options.CarrierCount} -> {_options.OutputSize}");
        lines.Add($"");
        lines.Add($"Total parameters: {ParameterCount}");

        return string.Join(Environment.NewLine, lines);
    }

    private Tensor<T> ApplySparsity(Tensor<T> current)
    {
        int n = current.Length;

        // Convert tensor to complex spectrum for sparsity mask
        var spectrum = new Vector<Complex<T>>(n);
        for (int i = 0; i < n; i++)
        {
            spectrum[i] = new Complex<T>(current[i], NumOps.Zero);
        }

        // Determine K. SparsityK=0 with UseMDLAutoK=false means "no sparsity" (keep all).
        int k;
        if (_options.UseMDLAutoK)
        {
            k = _sparsityMask.SelectK(spectrum);
        }
        else if (_options.SparsityK <= 0)
        {
            return current; // No sparsity requested
        }
        else
        {
            k = Math.Min(_options.SparsityK, n);
        }

        if (k >= n) return current; // No sparsity needed

        // Apply top-K sparsity
        var sparse = _sparsityMask.Apply(spectrum, k);

        // Convert back to tensor (take real part)
        var result = new Tensor<T>([n]);
        for (int i = 0; i < n; i++)
        {
            result[i] = sparse[i].Real;
        }
        return result;
    }

    private void BuildPipeline()
    {
        // Stage 1: Mellin-Fourier invariance layer
        if (_options.UseMellinFourier)
        {
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

        // Stage 3: IMD attention layers
        for (int i = 0; i < _options.NumAttentionLayers; i++)
        {
            _attentionLayers.Add(new IMDAttentionLayer<T>(_options.CarrierCount, _options.FftSize));
        }
    }

    private void InitializeOutputWeights()
    {
        double scale = Math.Sqrt(2.0 / (_options.CarrierCount + _options.OutputSize));
        var rng = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        for (int r = 0; r < _outputWeights.Rows; r++)
        {
            for (int c = 0; c < _outputWeights.Columns; c++)
            {
                _outputWeights[r, c] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * scale);
            }
        }
    }
}
