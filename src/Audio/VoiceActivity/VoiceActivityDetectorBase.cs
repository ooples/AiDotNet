using AiDotNet.Interfaces;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Base class for algorithmic voice activity detection implementations (non-neural network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Voice Activity Detection (VAD) determines whether audio contains speech or silence.
/// This is fundamental to many audio applications including speech recognition,
/// communication systems, and noise reduction.
/// </para>
/// <para><b>For Beginners:</b> VAD answers a simple question: "Is someone speaking right now?"
///
/// Common uses:
/// - Skip silence during transcription
/// - Reduce transmission bandwidth in VoIP
/// - Trigger recording only when speech is detected
/// - Segment audio into speaker turns
///
/// This base class provides:
/// - Frame-based processing with hangover logic
/// - Streaming mode with state management
/// - Segment detection across entire audio files
///
/// For neural network-based VAD (like Silero), see classes that extend AudioNeuralNetworkBase.
/// </para>
/// </remarks>
public abstract class VoiceActivityDetectorBase<T> : IVoiceActivityDetector<T>
{
    #region Numeric Operations

    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    #endregion

    #region Configuration

    /// <inheritdoc/>
    public int SampleRate { get; protected set; }

    /// <inheritdoc/>
    public int FrameSize { get; protected set; }

    /// <inheritdoc/>
    public double Threshold { get; set; }

    /// <inheritdoc/>
    public int MinSpeechDurationMs { get; set; }

    /// <inheritdoc/>
    public int MinSilenceDurationMs { get; set; }

    #endregion

    #region Streaming State

    /// <summary>
    /// Number of consecutive speech frames.
    /// </summary>
    protected int _speechFrameCount;

    /// <summary>
    /// Number of consecutive silence frames.
    /// </summary>
    protected int _silenceFrameCount;

    /// <summary>
    /// Current speech state.
    /// </summary>
    protected bool _inSpeech;

    #endregion

    /// <summary>
    /// Initializes a new instance of VoiceActivityDetectorBase.
    /// </summary>
    /// <param name="sampleRate">Audio sample rate.</param>
    /// <param name="frameSize">Frame size in samples.</param>
    /// <param name="threshold">Detection threshold (0-1).</param>
    /// <param name="minSpeechDurationMs">Minimum speech duration in ms.</param>
    /// <param name="minSilenceDurationMs">Minimum silence duration in ms.</param>
    protected VoiceActivityDetectorBase(
        int sampleRate = 16000,
        int frameSize = 480,
        double threshold = 0.5,
        int minSpeechDurationMs = 250,
        int minSilenceDurationMs = 300)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        SampleRate = sampleRate;
        FrameSize = frameSize;
        Threshold = threshold;
        MinSpeechDurationMs = minSpeechDurationMs;
        MinSilenceDurationMs = minSilenceDurationMs;

        ResetState();
    }

    #region Abstract Methods

    /// <summary>
    /// Computes speech probability for a single frame.
    /// </summary>
    /// <param name="frame">Audio frame data.</param>
    /// <returns>Speech probability (0-1).</returns>
    protected abstract T ComputeFrameProbability(T[] frame);

    #endregion

    #region IVoiceActivityDetector Implementation

    /// <inheritdoc/>
    public virtual bool DetectSpeech(Tensor<T> audioFrame)
    {
        var prob = GetSpeechProbability(audioFrame);
        return NumOps.ToDouble(prob) >= Threshold;
    }

    /// <inheritdoc/>
    public virtual T GetSpeechProbability(Tensor<T> audioFrame)
    {
        var frame = audioFrame.ToVector().ToArray();
        return ComputeFrameProbability(frame);
    }

    /// <inheritdoc/>
    public virtual IReadOnlyList<(int StartSample, int EndSample)> DetectSpeechSegments(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        var segments = new List<(int, int)>();

        int minSpeechFrames = (MinSpeechDurationMs * SampleRate) / (1000 * FrameSize);
        int minSilenceFrames = (MinSilenceDurationMs * SampleRate) / (1000 * FrameSize);

        int? segmentStart = null;
        int speechCount = 0;
        int silenceCount = 0;
        bool inSpeech = false;

        for (int i = 0; i + FrameSize <= samples.Length; i += FrameSize)
        {
            var frame = new T[FrameSize];
            Array.Copy(samples, i, frame, 0, FrameSize);

            var prob = ComputeFrameProbability(frame);
            bool isSpeech = NumOps.ToDouble(prob) >= Threshold;

            if (isSpeech)
            {
                speechCount++;
                silenceCount = 0;

                if (!inSpeech && speechCount >= minSpeechFrames)
                {
                    // Speech segment starts
                    inSpeech = true;
                    segmentStart = i - (speechCount - 1) * FrameSize;
                }
            }
            else
            {
                silenceCount++;
                speechCount = 0;

                if (inSpeech && silenceCount >= minSilenceFrames)
                {
                    // Speech segment ends
                    inSpeech = false;
                    if (segmentStart.HasValue)
                    {
                        segments.Add((segmentStart.Value, i - (silenceCount - 1) * FrameSize));
                        segmentStart = null;
                    }
                }
            }
        }

        // Handle segment at end of audio
        if (inSpeech && segmentStart.HasValue)
        {
            segments.Add((segmentStart.Value, samples.Length));
        }

        return segments;
    }

    /// <inheritdoc/>
    public virtual T[] GetFrameProbabilities(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        int numFrames = samples.Length / FrameSize;
        var probabilities = new T[numFrames];

        for (int i = 0; i < numFrames; i++)
        {
            var frame = new T[FrameSize];
            Array.Copy(samples, i * FrameSize, frame, 0, FrameSize);
            probabilities[i] = ComputeFrameProbability(frame);
        }

        return probabilities;
    }

    /// <inheritdoc/>
    public virtual (bool IsSpeech, T Probability) ProcessChunk(Tensor<T> audioChunk)
    {
        var prob = GetSpeechProbability(audioChunk);
        var isSpeech = NumOps.ToDouble(prob) >= Threshold;

        // Apply hangover logic for smooth transitions
        if (isSpeech)
        {
            _speechFrameCount++;
            _silenceFrameCount = 0;
        }
        else
        {
            _silenceFrameCount++;
            _speechFrameCount = 0;
        }

        int minSpeechFrames = (MinSpeechDurationMs * SampleRate) / (1000 * FrameSize);
        int minSilenceFrames = (MinSilenceDurationMs * SampleRate) / (1000 * FrameSize);

        if (!_inSpeech && _speechFrameCount >= minSpeechFrames)
        {
            _inSpeech = true;
        }
        else if (_inSpeech && _silenceFrameCount >= minSilenceFrames)
        {
            _inSpeech = false;
        }

        return (_inSpeech, prob);
    }

    /// <inheritdoc/>
    public virtual void ResetState()
    {
        _speechFrameCount = 0;
        _silenceFrameCount = 0;
        _inSpeech = false;
    }

    #endregion
}
