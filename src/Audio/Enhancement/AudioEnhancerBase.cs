using System.Numerics;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

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
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class AudioEnhancerBase<T> : IAudioEnhancer<T>
{
    /// <summary>
    /// Gets the hardware-accelerated computation engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

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
        // Write DIRECTLY into the tensor's storage. Tensor<T>.ToVector() materializes a COPY, not a
        // view, so the previous `result.ToVector()[i] = ...` filled a throwaway vector and returned
        // the freshly-allocated (all-zero) tensor — every enhancer emitted SILENCE.
        var result = new Tensor<T>([enhanced.Length]);
        var resultSpan = result.Data.Span;
        for (int i = 0; i < enhanced.Length; i++)
        {
            resultSpan[i] = enhanced[i];
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
        // Same ToVector()-copy defect as Enhance above: write straight into the tensor's storage so
        // streaming chunks carry the enhanced samples instead of zeros.
        var chunkResult = new Tensor<T>([enhanced.Length]);
        var chunkSpan = chunkResult.Data.Span;
        for (int i = 0; i < enhanced.Length; i++)
        {
            chunkSpan[i] = enhanced[i];
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
        return AudioEngineOps.SymmetricHannWindow<T>(AiDotNetEngine.Current, size).ToArray();
    }

    /// <summary>
    /// Processes audio using overlap-add method.
    /// </summary>
    protected T[] ProcessOverlapAdd(T[] input)
    {
        var engine = AiDotNetEngine.Current;
        int numFrames = (input.Length - _fftSize) / _hopSize + 1;
        var signal = new Tensor<T>(input, [input.Length]);
        var window = new Tensor<T>(_window, [_window.Length]);
        var synthesized = new List<Tensor<T>>();
        var starts = new List<int>();

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            // Extract and window frame (zero beyond the end of the input)
            var frameData = WindowedFrame(engine, signal, window, start);

            // Compute FFT (simplified - real implementation would use FFT library)
            var (magnitudes, phases) = ComputeFFT(frameData);

            // Process in frequency domain
            var enhancedMagnitudes = ProcessSpectralFrame(magnitudes, phases);

            // Inverse FFT
            var enhanced = ComputeIFFT(enhancedMagnitudes, phases);

            // Window for overlap-add
            synthesized.Add(engine.TensorMultiply(AudioEngineOps.FitLength(engine, new Tensor<T>(enhanced, [enhanced.Length]), _fftSize), window));
            starts.Add(start);
        }

        return AudioEngineOps.OverlapAdd(engine, synthesized, starts, input.Length).ToArray();
    }

    /// <summary>
    /// The <see cref="_fftSize"/>-sample frame at <paramref name="start"/> multiplied by the analysis
    /// window; samples past the end of the signal are zero.
    /// </summary>
    private T[] WindowedFrame(IEngine engine, Tensor<T> signal, Tensor<T> window, int start)
    {
        int available = Math.Max(0, Math.Min(_fftSize, signal.Length - start));
        var samples = available > 0
            ? AudioEngineOps.FitLength(engine, engine.TensorSlice(signal, [start], [available]), _fftSize)
            : new Tensor<T>([_fftSize]);
        return engine.TensorMultiply(samples, window).ToArray();
    }

    /// <summary>
    /// Processes a streaming chunk of audio.
    /// </summary>
    protected T[] ProcessStreamingChunk(T[] chunk)
    {
        if (_inputBuffer is null || _outputBuffer is null)
            ResetState();

        var inputBuf = _inputBuffer ?? throw new InvalidOperationException("Input buffer not initialized.");
        var outputBuf = _outputBuffer ?? throw new InvalidOperationException("Output buffer not initialized.");

        var output = new T[chunk.Length];
        int outputPos = 0;

        for (int i = 0; i < chunk.Length; i++)
        {
            inputBuf[_bufferPosition] = chunk[i];
            _bufferPosition++;

            if (_bufferPosition >= _hopSize)
            {
                // Process frame. _bufferPosition == _hopSize here, so the frame reads the input buffer
                // in order: frame[j] = inputBuf[j] * window[j].
                var engine = AiDotNetEngine.Current;
                var window = new Tensor<T>(_window, [_window.Length]);
                var frameData = engine.TensorMultiply(new Tensor<T>(inputBuf, [inputBuf.Length]), window).ToArray();

                var (magnitudes, phases) = ComputeFFT(frameData);
                var enhancedMagnitudes = ProcessSpectralFrame(magnitudes, phases);
                var enhanced = ComputeIFFT(enhancedMagnitudes, phases);

                // Overlap-add to output buffer
                engine.TensorAdd(
                    new Tensor<T>(outputBuf, [outputBuf.Length]),
                    engine.TensorMultiply(AudioEngineOps.FitLength(engine, new Tensor<T>(enhanced, [enhanced.Length]), _fftSize), window))
                    .CopyTo(outputBuf.AsSpan());

                // Output hop samples
                for (int j = 0; j < _hopSize && outputPos < output.Length; j++)
                {
                    output[outputPos++] = outputBuf[j];
                }

                // Shift output buffer
                Array.Copy(outputBuf, _hopSize, outputBuf, 0, _fftSize - _hopSize);
                Array.Clear(outputBuf, _fftSize - _hopSize, _hopSize);

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
        var engine = AiDotNetEngine.Current;
        int numFrames = (noiseAudio.Length - _fftSize) / _hopSize + 1;
        int bins = _fftSize / 2 + 1;
        var signal = new Tensor<T>(noiseAudio, [noiseAudio.Length]);
        var window = new Tensor<T>(_window, [_window.Length]);
        var sumMagnitudes = new Tensor<T>([bins]);

        for (int frame = 0; frame < numFrames; frame++)
        {
            var (magnitudes, _) = ComputeFFT(WindowedFrame(engine, signal, window, frame * _hopSize));
            var head = new Tensor<T>(magnitudes, [magnitudes.Length]);
            sumMagnitudes = engine.TensorAdd(sumMagnitudes, head.Length == bins ? head : engine.TensorSlice(head, [0], [bins]));
        }

        // Average
        return engine.TensorDivideScalar(sumMagnitudes, NumOps.FromDouble(numFrames)).ToArray();
    }

    /// <summary>
    /// Computes FFT of audio frame using FftSharp library (O(N log N) algorithm).
    /// </summary>
    protected virtual (T[] Magnitudes, T[] Phases) ComputeFFT(T[] frame)
    {
        // O(N log N) complex FFT of the (zero-padded) frame; magnitude and phase of the positive
        // frequencies (N/2+1 bins).
        var (magnitudes, phases) = AudioEngineOps.HalfSpectrum(AiDotNetEngine.Current, new Tensor<T>(frame, [frame.Length]), _fftSize);
        return (magnitudes.ToArray(), phases.ToArray());
    }

    /// <summary>
    /// Computes inverse FFT using FftSharp library.
    /// </summary>
    protected virtual T[] ComputeIFFT(T[] magnitudes, T[] phases)
    {
        // Reconstruct full complex spectrum from magnitude/phase (N/2+1 positive frequency bins,
        // negative frequencies by conjugate symmetry X[N-k] = conj(X[k])), then the real part of the
        // 1/N-normalised inverse FFT.
        var engine = AiDotNetEngine.Current;
        var phaseTensor = new Tensor<T>(phases, [phases.Length]);
        if (phases.Length != magnitudes.Length)
            phaseTensor = AudioEngineOps.FitLength(engine, phaseTensor, magnitudes.Length);
        return AudioEngineOps.InverseHalfSpectrum(engine, new Tensor<T>(magnitudes, [magnitudes.Length]), phaseTensor, _fftSize).ToArray();
    }

    #endregion
}
