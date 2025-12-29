using AiDotNet.Interfaces;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Base class for algorithmic audio enhancement (non-neural network based).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Provides common functionality for all audio enhancers including:
/// <list type="bullet">
/// <item><description>Frame-based processing with overlap-add</description></item>
/// <item><description>Streaming mode with state management</description></item>
/// <item><description>STFT-based analysis/synthesis</description></item>
/// </list>
/// </para>
/// </remarks>
public abstract class AudioEnhancerBase<T> : IAudioEnhancer<T>
{
    #region Numeric Operations

    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    #endregion

    #region Configuration

    /// <summary>
    /// Audio sample rate.
    /// </summary>
    public int SampleRate { get; protected set; }

    /// <summary>
    /// FFT size for spectral analysis.
    /// </summary>
    protected readonly int _fftSize;

    /// <summary>
    /// Hop size between frames.
    /// </summary>
    protected readonly int _hopSize;

    /// <summary>
    /// Window function coefficients.
    /// </summary>
    protected readonly T[] _window;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; }

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; }

    /// <inheritdoc/>
    public int LatencySamples => _fftSize;

    #endregion

    #region Streaming State

    /// <summary>
    /// Input buffer for streaming mode.
    /// </summary>
    protected T[]? _inputBuffer;

    /// <summary>
    /// Output buffer for overlap-add.
    /// </summary>
    protected T[]? _outputBuffer;

    /// <summary>
    /// Current position in input buffer.
    /// </summary>
    protected int _bufferPosition;

    /// <summary>
    /// Estimated noise profile for spectral subtraction.
    /// </summary>
    protected T[]? _noiseProfile;

    #endregion

    /// <summary>
    /// Initializes a new instance of the AudioEnhancerBase class.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="numChannels">Number of audio channels.</param>
    /// <param name="fftSize">FFT size for spectral analysis.</param>
    /// <param name="hopSize">Hop size between frames.</param>
    /// <param name="enhancementStrength">Enhancement strength (0-1).</param>
    protected AudioEnhancerBase(
        int sampleRate = 16000,
        int numChannels = 1,
        int fftSize = 512,
        int hopSize = 128,
        double enhancementStrength = 0.7)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        SampleRate = sampleRate;
        NumChannels = numChannels;
        _fftSize = fftSize;
        _hopSize = hopSize;
        EnhancementStrength = enhancementStrength;

        // Initialize Hann window
        _window = CreateHannWindow(fftSize);

        // Initialize streaming buffers
        ResetState();
    }

    #region Abstract Methods

    /// <summary>
    /// Processes a single spectral frame.
    /// </summary>
    /// <param name="magnitudes">Magnitude spectrum.</param>
    /// <param name="phases">Phase spectrum.</param>
    /// <returns>Enhanced magnitude spectrum.</returns>
    protected abstract T[] ProcessSpectralFrame(T[] magnitudes, T[] phases);

    #endregion

    #region IAudioEnhancer Implementation

    /// <inheritdoc/>
    public virtual Tensor<T> Enhance(Tensor<T> audio)
    {
        // Process entire audio using overlap-add STFT
        var samples = audio.ToVector().ToArray();
        var enhanced = ProcessOverlapAdd(samples);
        // Create tensor and copy enhanced data
        var result = new Tensor<T>([enhanced.Length]);
        var resultVector = result.ToVector();
        for (int i = 0; i < enhanced.Length; i++)
        {
            resultVector[i] = enhanced[i];
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // Default implementation: ignore reference and just enhance
        // Subclasses can override for echo cancellation
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        var samples = audioChunk.ToVector().ToArray();
        var enhanced = ProcessStreamingChunk(samples);
        // Create tensor and copy enhanced data
        var chunkResult = new Tensor<T>([enhanced.Length]);
        var chunkVector = chunkResult.ToVector();
        for (int i = 0; i < enhanced.Length; i++)
        {
            chunkVector[i] = enhanced[i];
        }
        return chunkResult;
    }

    /// <inheritdoc/>
    public virtual void ResetState()
    {
        _inputBuffer = new T[_fftSize];
        _outputBuffer = new T[_fftSize];
        _bufferPosition = 0;
    }

    /// <inheritdoc/>
    public virtual void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        var samples = noiseOnlyAudio.ToVector().ToArray();
        _noiseProfile = EstimateNoiseSpectrum(samples);
    }

    #endregion

    #region Protected Helper Methods

    /// <summary>
    /// Creates a Hann window of the specified size.
    /// </summary>
    protected T[] CreateHannWindow(int size)
    {
        var window = new T[size];
        for (int i = 0; i < size; i++)
        {
            var value = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (size - 1)));
            window[i] = NumOps.FromDouble(value);
        }
        return window;
    }

    /// <summary>
    /// Processes audio using overlap-add method.
    /// </summary>
    protected T[] ProcessOverlapAdd(T[] input)
    {
        int numFrames = (input.Length - _fftSize) / _hopSize + 1;
        var output = new T[input.Length];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            // Extract and window frame
            var frameData = new T[_fftSize];
            for (int i = 0; i < _fftSize && start + i < input.Length; i++)
            {
                frameData[i] = NumOps.Multiply(input[start + i], _window[i]);
            }

            // Compute FFT (simplified - real implementation would use FFT library)
            var (magnitudes, phases) = ComputeFFT(frameData);

            // Process in frequency domain
            var enhancedMagnitudes = ProcessSpectralFrame(magnitudes, phases);

            // Inverse FFT
            var enhanced = ComputeIFFT(enhancedMagnitudes, phases);

            // Window and overlap-add
            for (int i = 0; i < _fftSize && start + i < output.Length; i++)
            {
                var windowed = NumOps.Multiply(enhanced[i], _window[i]);
                output[start + i] = NumOps.Add(output[start + i], windowed);
            }
        }

        return output;
    }

    /// <summary>
    /// Processes a streaming chunk of audio.
    /// </summary>
    protected T[] ProcessStreamingChunk(T[] chunk)
    {
        if (_inputBuffer is null || _outputBuffer is null)
            ResetState();

        var output = new T[chunk.Length];
        int outputPos = 0;

        for (int i = 0; i < chunk.Length; i++)
        {
            _inputBuffer![_bufferPosition] = chunk[i];
            _bufferPosition++;

            if (_bufferPosition >= _hopSize)
            {
                // Process frame
                var frameData = new T[_fftSize];
                for (int j = 0; j < _fftSize; j++)
                {
                    int idx = (j + _bufferPosition - _hopSize) % _fftSize;
                    frameData[j] = NumOps.Multiply(_inputBuffer[idx], _window[j]);
                }

                var (magnitudes, phases) = ComputeFFT(frameData);
                var enhancedMagnitudes = ProcessSpectralFrame(magnitudes, phases);
                var enhanced = ComputeIFFT(enhancedMagnitudes, phases);

                // Overlap-add to output buffer
                for (int j = 0; j < _fftSize; j++)
                {
                    var windowed = NumOps.Multiply(enhanced[j], _window[j]);
                    _outputBuffer![j] = NumOps.Add(_outputBuffer[j], windowed);
                }

                // Output hop samples
                for (int j = 0; j < _hopSize && outputPos < output.Length; j++)
                {
                    output[outputPos++] = _outputBuffer![j];
                }

                // Shift output buffer
                Array.Copy(_outputBuffer!, _hopSize, _outputBuffer!, 0, _fftSize - _hopSize);
                Array.Clear(_outputBuffer!, _fftSize - _hopSize, _hopSize);

                _bufferPosition = 0;
            }
        }

        return output;
    }

    /// <summary>
    /// Estimates noise spectrum from noise-only audio.
    /// </summary>
    protected T[] EstimateNoiseSpectrum(T[] noiseAudio)
    {
        int numFrames = (noiseAudio.Length - _fftSize) / _hopSize + 1;
        var avgMagnitudes = new T[_fftSize / 2 + 1];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;
            var frameData = new T[_fftSize];

            for (int i = 0; i < _fftSize && start + i < noiseAudio.Length; i++)
            {
                frameData[i] = NumOps.Multiply(noiseAudio[start + i], _window[i]);
            }

            var (magnitudes, _) = ComputeFFT(frameData);

            for (int i = 0; i < avgMagnitudes.Length; i++)
            {
                avgMagnitudes[i] = NumOps.Add(avgMagnitudes[i], magnitudes[i]);
            }
        }

        // Average
        var divisor = NumOps.FromDouble(numFrames);
        for (int i = 0; i < avgMagnitudes.Length; i++)
        {
            avgMagnitudes[i] = NumOps.Divide(avgMagnitudes[i], divisor);
        }

        return avgMagnitudes;
    }

    /// <summary>
    /// Computes FFT of audio frame (simplified implementation).
    /// </summary>
    protected virtual (T[] Magnitudes, T[] Phases) ComputeFFT(T[] frame)
    {
        // Simplified DFT - real implementation should use FFT library
        int numBins = _fftSize / 2 + 1;
        var magnitudes = new T[numBins];
        var phases = new T[numBins];

        for (int k = 0; k < numBins; k++)
        {
            double real = 0, imag = 0;
            for (int n = 0; n < _fftSize; n++)
            {
                double angle = -2 * Math.PI * k * n / _fftSize;
                real += NumOps.ToDouble(frame[n]) * Math.Cos(angle);
                imag += NumOps.ToDouble(frame[n]) * Math.Sin(angle);
            }

            magnitudes[k] = NumOps.FromDouble(Math.Sqrt(real * real + imag * imag));
            phases[k] = NumOps.FromDouble(Math.Atan2(imag, real));
        }

        return (magnitudes, phases);
    }

    /// <summary>
    /// Computes inverse FFT (simplified implementation).
    /// </summary>
    protected virtual T[] ComputeIFFT(T[] magnitudes, T[] phases)
    {
        var output = new T[_fftSize];

        for (int n = 0; n < _fftSize; n++)
        {
            double sum = 0;
            for (int k = 0; k < magnitudes.Length; k++)
            {
                double mag = NumOps.ToDouble(magnitudes[k]);
                double phase = NumOps.ToDouble(phases[k]);
                double angle = 2 * Math.PI * k * n / _fftSize + phase;
                sum += mag * Math.Cos(angle);

                // Add conjugate for negative frequencies
                if (k > 0 && k < magnitudes.Length - 1)
                {
                    sum += mag * Math.Cos(-angle);
                }
            }
            output[n] = NumOps.FromDouble(sum / _fftSize);
        }

        return output;
    }

    #endregion
}
