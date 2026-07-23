using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <summary>
/// Audio event detection and classification extensions for AiModelBuilder.
/// </summary>
/// <remarks>
/// <para>
/// These methods provide audio-specific operations through the facade pattern.
/// Configure any model that implements <see cref="IAudioEventDetector{T}"/> via
/// <see cref="AiModelBuilder{T, TInput, TOutput}.ConfigureModel"/> and then use
/// these methods for audio event detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio event detection identifies sounds in audio recordings
/// or real-time streams. After configuring a model (like BEATs or AudioEventDetector),
/// use these methods to detect sounds, get probabilities, or start real-time monitoring.
/// </para>
/// </remarks>
public static class AudioBuilderExtensions
{
    /// <summary>
    /// Detects audio events in the given audio using the configured model's default threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="audio">Raw audio waveform tensor [samples]. Should be mono audio at the
    /// model's expected sample rate (typically 16000 Hz) with values in [-1.0, 1.0].</param>
    /// <returns>Detection result containing all events above the model's confidence threshold,
    /// with timing information and per-class statistics.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no model is configured or the
    /// configured model does not implement <see cref="IAudioEventDetector{T}"/>.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for detecting sounds in audio. Pass in
    /// audio data and get back a list of all sounds the model detected, including what they are,
    /// how confident the model is, and when they occurred.
    ///
    /// Example:
    /// <code>
    /// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new BEATs&lt;float&gt;(architecture, "beats_iter3.onnx"));
    ///
    /// var result = builder.DetectAudioEvents(audioTensor);
    /// foreach (var evt in result.Events)
    /// {
    ///     Console.WriteLine($"{evt.EventType}: {evt.Confidence:P1} at {evt.StartTime:F1}s");
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public static AudioEventResult<T> DetectAudioEvents<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> audio)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.Detect(audio);
    }

    /// <summary>
    /// Detects audio events with a custom confidence threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="threshold">Confidence threshold in range [0, 1]. Only events at or above
    /// this confidence are included. Lower values detect more sounds but with more false positives.</param>
    /// <returns>Detection result containing events above the specified threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to control detection sensitivity:
    /// <list type="bullet">
    /// <item><b>0.1-0.2</b>: Very sensitive - catches most sounds, some false alarms</item>
    /// <item><b>0.3</b>: Balanced (typical default)</item>
    /// <item><b>0.5-0.7</b>: Conservative - only confident detections</item>
    /// <item><b>0.8+</b>: Very strict - only near-certain detections</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static AudioEventResult<T> DetectAudioEvents<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> audio,
        T threshold)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.Detect(audio, threshold);
    }

    /// <summary>
    /// Detects audio events asynchronously without blocking the calling thread.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="cancellationToken">Token to cancel the detection operation.</param>
    /// <returns>Detection result (same as <see cref="DetectAudioEvents{T}(AiModelBuilder{T,Tensor{T},Tensor{T}},Tensor{T})"/>
    /// but non-blocking).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this in UI applications or web APIs where you don't want to
    /// freeze the interface while processing audio. The await keyword lets other work continue
    /// while detection runs in the background.
    /// </para>
    /// </remarks>
    public static Task<AudioEventResult<T>> DetectAudioEventsAsync<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.DetectAsync(audio, cancellationToken);
    }

    /// <summary>
    /// Detects only specific event types, filtering out everything else.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="eventTypes">Event type names to detect (case-insensitive matching).</param>
    /// <returns>Detection result filtered to only the specified event types.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only care about specific sounds:
    /// <code>
    /// // Only detect safety-related sounds
    /// var result = builder.DetectSpecificAudioEvents(audio,
    ///     new[] { "Gunshot", "Glass breaking", "Siren" });
    /// </code>
    /// </para>
    /// </remarks>
    public static AudioEventResult<T> DetectSpecificAudioEvents<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> audio,
        IReadOnlyList<string> eventTypes)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.DetectSpecific(audio, eventTypes);
    }

    /// <summary>
    /// Detects specific event types with a custom confidence threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="eventTypes">Event type names to detect.</param>
    /// <param name="threshold">Confidence threshold [0, 1].</param>
    /// <returns>Detection result filtered to specified event types above the threshold.</returns>
    public static AudioEventResult<T> DetectSpecificAudioEvents<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> audio,
        IReadOnlyList<string> eventTypes,
        T threshold)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.DetectSpecific(audio, eventTypes, threshold);
    }

    /// <summary>
    /// Gets raw frame-level event probabilities for all classes without thresholding.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <returns>A 2D tensor [time_frames, num_events] where each value is a probability in [0, 1].
    /// Row i = time frame i, column j = probability of event class j.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives the full probability matrix without any filtering.
    /// Useful for visualization (heatmaps), custom thresholding per class, or analysis
    /// of temporal patterns in the audio.
    /// </para>
    /// </remarks>
    public static Tensor<T> GetAudioEventProbabilities<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> audio)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.GetEventProbabilities(audio);
    }

    /// <summary>
    /// Starts a streaming event detection session for real-time audio monitoring.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <returns>A streaming session that accepts audio chunks and emits events in real-time.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Streaming mode processes audio in real-time from a microphone
    /// or other live source. Feed audio chunks as they arrive and get event notifications:
    /// <code>
    /// using var session = builder.StartAudioEventStreaming();
    /// session.EventDetected += (s, evt) =>
    ///     Console.WriteLine($"[LIVE] {evt.EventType}: {evt.Confidence:P1}");
    ///
    /// while (recording)
    /// {
    ///     var chunk = microphone.ReadSamples(4096);
    ///     session.FeedAudio(chunk);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public static IStreamingEventDetectionSession<T> StartAudioEventStreaming<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.StartStreamingSession();
    }

    /// <summary>
    /// Starts a streaming event detection session with custom settings.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <param name="sampleRate">Sample rate of incoming audio in Hz.</param>
    /// <param name="threshold">Confidence threshold for event detection [0, 1].</param>
    /// <returns>A streaming session configured with the specified settings.</returns>
    public static IStreamingEventDetectionSession<T> StartAudioEventStreaming<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        int sampleRate,
        T threshold)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.StartStreamingSession(sampleRate, threshold);
    }

    /// <summary>
    /// Gets the list of event types the configured audio model can detect.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <returns>List of supported event type names (e.g., "Speech", "Dog barking", "Music").</returns>
    public static IReadOnlyList<string> GetSupportedAudioEvents<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.SupportedEvents;
    }

    /// <summary>
    /// Gets the time resolution of the configured audio event detection model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured audio event detection model.</param>
    /// <returns>Time resolution in seconds (determines how precisely event boundaries are detected).</returns>
    public static double GetAudioEventTimeResolution<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var detector = GetAudioEventDetector(builder);
        return detector.TimeResolution;
    }

    /// <summary>
    /// Extracts and casts the configured model to <see cref="IAudioEventDetector{T}"/>.
    /// </summary>
    private static IAudioEventDetector<T> GetAudioEventDetector<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IAudioEventDetector<T> detector)
        {
            return detector;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with an audio event detection model " +
                "(e.g., BEATs, AudioEventDetector) before calling audio detection methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support " +
            $"audio event detection. Use ConfigureModel() with a model that implements " +
            $"IAudioEventDetector<{typeof(T).Name}> (e.g., BEATs, AudioEventDetector).");
    }
}
