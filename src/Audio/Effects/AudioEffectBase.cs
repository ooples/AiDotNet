using AiDotNet.Interfaces;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Base class for audio effects processors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class AudioEffectBase<T> : IAudioEffect<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    #region IAudioEffect Properties

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public int SampleRate { get; protected set; }

    /// <inheritdoc/>
    public bool Bypass { get; set; }

    /// <inheritdoc/>
    public double Mix { get; set; }

    /// <inheritdoc/>
    public virtual int LatencySamples => 0;

    /// <inheritdoc/>
    public virtual int TailSamples => 0;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, AudioEffectParameter<T>> Parameters => _parameters;

    #endregion

    /// <summary>
    /// Mutable parameters dictionary.
    /// </summary>
    protected readonly Dictionary<string, AudioEffectParameter<T>> _parameters = new();

    /// <summary>
    /// Initializes a new AudioEffectBase.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate.</param>
    /// <param name="mix">Dry/wet mix (0-1).</param>
    protected AudioEffectBase(int sampleRate = 44100, double mix = 1.0)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        SampleRate = sampleRate;
        Mix = mix;
        Bypass = false;
    }

    #region Abstract Methods

    /// <summary>
    /// Processes a single sample through the effect.
    /// </summary>
    /// <param name="input">Input sample.</param>
    /// <returns>Processed sample (wet signal only).</returns>
    protected abstract T ProcessSampleInternal(T input);

    #endregion

    #region IAudioEffect Implementation

    /// <inheritdoc/>
    public virtual Tensor<T> Process(Tensor<T> input)
    {
        if (Bypass) return input;

        var samples = input.ToVector().ToArray();
        var output = new T[samples.Length];

        for (int i = 0; i < samples.Length; i++)
        {
            output[i] = ProcessSample(samples[i]);
        }

        // Create tensor and copy output data
        var result = new Tensor<T>([output.Length]);
        var resultVector = result.ToVector();
        for (int i = 0; i < output.Length; i++)
        {
            resultVector[i] = output[i];
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual T ProcessSample(T sample)
    {
        if (Bypass) return sample;

        var wet = ProcessSampleInternal(sample);

        // Apply dry/wet mix
        var dryAmount = NumOps.FromDouble(1.0 - Mix);
        var wetAmount = NumOps.FromDouble(Mix);

        return NumOps.Add(
            NumOps.Multiply(sample, dryAmount),
            NumOps.Multiply(wet, wetAmount));
    }

    /// <inheritdoc/>
    public virtual void ProcessInPlace(Span<T> buffer)
    {
        if (Bypass) return;

        for (int i = 0; i < buffer.Length; i++)
        {
            buffer[i] = ProcessSample(buffer[i]);
        }
    }

    /// <inheritdoc/>
    public abstract void Reset();

    /// <inheritdoc/>
    public virtual void SetParameter(string name, T value)
    {
        if (_parameters.TryGetValue(name, out var param))
        {
            param.CurrentValue = value;
            OnParameterChanged(name, value);
        }
    }

    /// <inheritdoc/>
    public virtual T GetParameter(string name)
    {
        if (_parameters.TryGetValue(name, out var param))
        {
            return param.CurrentValue;
        }
        return NumOps.Zero;
    }

    #endregion

    #region Protected Methods

    /// <summary>
    /// Called when a parameter value changes.
    /// </summary>
    /// <param name="name">Parameter name.</param>
    /// <param name="value">New value.</param>
    protected virtual void OnParameterChanged(string name, T value) { }

    /// <summary>
    /// Adds a parameter to the effect.
    /// </summary>
    protected void AddParameter(
        string name,
        string displayName,
        T minValue,
        T maxValue,
        T defaultValue,
        string unit = "",
        string description = "")
    {
        _parameters[name] = new AudioEffectParameter<T>
        {
            Name = name,
            DisplayName = displayName,
            MinValue = minValue,
            MaxValue = maxValue,
            DefaultValue = defaultValue,
            CurrentValue = defaultValue,
            Unit = unit,
            Description = description
        };
    }

    /// <summary>
    /// Converts decibels to linear amplitude.
    /// </summary>
    protected double DbToLinear(double db)
    {
        return Math.Pow(10, db / 20.0);
    }

    /// <summary>
    /// Converts linear amplitude to decibels.
    /// </summary>
    protected double LinearToDb(double linear)
    {
        return 20.0 * Math.Log10(Math.Max(linear, 1e-10));
    }

    #endregion
}
