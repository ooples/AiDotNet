global using AiDotNet.WindowFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for spectral analysis of time series data, which transforms time-domain signals
/// into the frequency domain to identify periodic components and patterns.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the spectral analysis.</typeparam>
/// <remarks>
/// <para>
/// Spectral analysis is a technique used to decompose a time series signal into its constituent frequency 
/// components, revealing periodic patterns that might not be apparent in the original time domain representation. 
/// This is typically accomplished using the Fast Fourier Transform (FFT) algorithm, which efficiently computes 
/// the Discrete Fourier Transform of a signal. The resulting frequency spectrum shows the amplitude and phase 
/// of different frequency components present in the signal. This class provides configuration options for 
/// spectral analysis, including the FFT size, window function selection, and overlap settings for spectrograms 
/// or short-time Fourier transforms. These options allow fine-tuning of the spectral analysis to balance 
/// frequency resolution, time resolution, and spectral leakage based on the specific characteristics of the 
/// time series being analyzed.
/// </para>
/// <para><b>For Beginners:</b> Spectral analysis helps identify repeating patterns in your time series data.
/// 
/// Time series data shows how values change over time, but sometimes it's hard to see patterns:
/// - A stock price might fluctuate in ways that look random
/// - Sensor readings might contain multiple overlapping cycles
/// - Audio signals contain many frequencies mixed together
/// 
/// Spectral analysis transforms this data from the time domain to the frequency domain:
/// - Instead of seeing "what happened when"
/// - You see "which cycles/frequencies are present and how strong they are"
/// 
/// This is like:
/// - Breaking down a musical chord into individual notes
/// - Identifying that a stock has both weekly and quarterly patterns
/// - Finding that a machine vibrates at specific frequencies when it's about to fail
/// 
/// This class lets you configure how this transformation happens, controlling the
/// trade-offs between frequency precision, time precision, and other factors.
/// </para>
/// </remarks>
public class SpectralAnalysisOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of points used in the Fast Fourier Transform (FFT).
    /// </summary>
    /// <value>A positive integer, typically a power of 2, defaulting to 512.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of points used in the Fast Fourier Transform (FFT) computation. The FFT 
    /// is most efficient when the number of points is a power of 2, which is why the default value is 512 (2^9). 
    /// If the input signal has fewer points than this value, it is typically zero-padded to reach this length. 
    /// If it has more points, it may be segmented or truncated. A larger FFT size provides better frequency 
    /// resolution (the ability to distinguish between closely spaced frequency components) but requires more 
    /// computational resources. A smaller FFT size reduces computational requirements but provides coarser 
    /// frequency resolution. The appropriate value depends on the specific application requirements, the length 
    /// of the input signal, and the desired balance between frequency resolution and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the precision of frequency detection and computational efficiency.
    /// 
    /// The NFFT value determines:
    /// - How many frequency bins the analysis will produce
    /// - The resolution or precision of frequency detection
    /// - How computationally intensive the calculation will be
    /// 
    /// The default value of 512 means:
    /// - The FFT will use 512 points (ideally a power of 2 for efficiency)
    /// - This provides a good balance between precision and computational cost
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 1024, 2048): More precise frequency detection, but more computation
    /// - Lower values (e.g., 256, 128): Faster computation, but less precise frequency detection
    /// 
    /// When to adjust this value:
    /// - Increase it when you need to distinguish between very similar frequencies
    /// - Decrease it when processing speed is more important than frequency precision
    /// 
    /// For example, in audio processing, higher NFFT values help distinguish between notes that
    /// are very close in pitch, while lower values might be sufficient for detecting broader
    /// frequency bands like bass vs. treble.
    /// </para>
    /// </remarks>
    public int NFFT { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to apply a window function to the signal before spectral analysis.
    /// </summary>
    /// <value>A boolean value, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This property determines whether a window function is applied to the signal before performing spectral 
    /// analysis. Window functions help reduce spectral leakage, which occurs when the signal being analyzed is 
    /// not perfectly periodic within the observation interval. Spectral leakage causes energy from a single 
    /// frequency component to spread across multiple frequency bins, potentially obscuring weaker components. 
    /// The default value of true enables windowing, which is recommended for most applications to improve the 
    /// accuracy of the spectral analysis. However, windowing also modifies the signal, potentially affecting 
    /// amplitude measurements and time-domain characteristics. In some specialized applications where preserving 
    /// the original signal characteristics is more important than reducing spectral leakage, windowing might be 
    /// disabled by setting this property to false.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether to use a special technique to improve frequency analysis accuracy.
    /// 
    /// When analyzing a signal:
    /// - The FFT assumes the signal repeats perfectly (is periodic)
    /// - Real-world signals rarely meet this assumption
    /// - This mismatch causes "spectral leakage" - frequencies bleeding into neighboring bins
    /// 
    /// Window functions solve this problem by:
    /// - Gradually tapering the signal at the edges of each analysis segment
    /// - Reducing the artificial discontinuities that cause leakage
    /// - Improving the accuracy of frequency detection
    /// 
    /// The default value of true means:
    /// - Window functions will be applied before spectral analysis
    /// - This provides more accurate frequency information for most applications
    /// 
    /// When to adjust this value:
    /// - Keep it true (default) for most applications
    /// - Set it to false only in specialized cases where preserving the original signal amplitude
    ///   is more important than frequency precision
    /// 
    /// For example, in audio spectrum analyzers, window functions are almost always used to
    /// provide cleaner, more accurate frequency displays.
    /// </para>
    /// </remarks>
    public bool UseWindowFunction { get; set; } = true;

    /// <summary>
    /// Gets or sets the window function to apply to the signal before spectral analysis.
    /// </summary>
    /// <value>An implementation of IWindowFunction&lt;T&gt;, defaulting to a Hanning window.</value>
    /// <remarks>
    /// <para>
    /// This property specifies which window function is applied to the signal before spectral analysis, if 
    /// UseWindowFunction is set to true. Different window functions have different characteristics in terms of 
    /// main lobe width, side lobe height, and spectral leakage. The Hanning window (the default) provides a good 
    /// balance between frequency resolution and amplitude accuracy for many applications. Other common options 
    /// include the Hamming window (similar to Hanning but with different coefficients), the Blackman window 
    /// (better side lobe suppression but wider main lobe), and the rectangular window (no windowing, preserves 
    /// signal amplitude but has poor spectral leakage properties). The choice of window function should be based 
    /// on the specific requirements of the application, particularly the relative importance of frequency resolution, 
    /// amplitude accuracy, and side lobe suppression. [0]
    /// </para>
    /// <para><b>For Beginners:</b> This setting selects which specific window function to use for improving spectral analysis.
    /// 
    /// Different window functions have different trade-offs:
    /// - They all reduce spectral leakage, but in different ways
    /// - Some preserve amplitude better, others provide better frequency separation
    /// - The choice depends on what aspects of the signal you care most about
    /// 
    /// The default Hanning window:
    /// - Provides excellent frequency resolution
    /// - Has good reduction of spectral leakage
    /// - Works well for a wide range of applications
    /// - Is one of the most commonly used window functions
    /// 
    /// Other common window functions include:
    /// - Hamming: Similar to Hanning, slightly different trade-offs
    /// - Blackman: Better at suppressing distant frequency leakage
    /// - Rectangular: No windowing (preserves amplitude but poor leakage properties)
    /// 
    /// When to change this value:
    /// - For most applications, the default Hanning window works well
    /// - Change it only if you have specific requirements or are familiar with the
    ///   trade-offs between different window functions
    /// 
    /// For example, in audio analysis, the Hanning window is often used for general spectral
    /// analysis, while Blackman might be preferred when trying to detect very quiet sounds
    /// in the presence of loud ones. [0]
    /// </para>
    /// </remarks>
    public IWindowFunction<T> WindowFunction { get; set; } = new HanningWindow<T>();

    /// <summary>
    /// Gets or sets the percentage of overlap between adjacent segments in spectrograms or short-time Fourier transforms.
    /// </summary>
    /// <value>An integer between 0 and 99, defaulting to 50 (50% overlap).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the percentage of overlap between adjacent segments when performing spectral analysis 
    /// on consecutive portions of a signal, as in spectrograms or short-time Fourier transforms. Overlapping segments 
    /// helps mitigate the effects of windowing, which reduces the amplitude of the signal at the edges of each segment. 
    /// The default value of 50 (representing 50% overlap) is a common choice that provides a good balance between 
    /// computational efficiency and accurate representation of the signal. Higher overlap percentages provide smoother 
    /// time-frequency representations and better capture transient events but require more computational resources. 
    /// Lower overlap percentages reduce computational requirements but may miss short-duration events and produce 
    /// less smooth spectrograms. The appropriate value depends on the specific application requirements and the 
    /// characteristics of the signal being analyzed.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much consecutive analysis windows overlap when analyzing how a signal changes over time.
    /// 
    /// When analyzing how frequencies change over time (like in a spectrogram):
    /// - The signal is divided into multiple segments
    /// - Each segment is analyzed separately
    /// - Overlapping these segments improves accuracy and smoothness
    /// 
    /// The default value of 50 means:
    /// - Each segment shares 50% of its data with the next segment
    /// - This provides a good balance between time resolution and computational efficiency
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 75%): Smoother results, better detection of short events, more computation
    /// - Lower values (e.g., 25%): Less computation, but potentially missing short events
    /// - 0%: No overlap, most efficient but lowest quality
    /// 
    /// When to adjust this value:
    /// - Increase it when analyzing signals with rapid changes or short events
    /// - Decrease it when computational efficiency is more important than capturing every detail
    /// 
    /// For example, in speech analysis, higher overlap percentages help capture the rapid
    /// transitions between phonemes, while in long-term climate data analysis, lower
    /// overlap might be sufficient.
    /// </para>
    /// </remarks>
    public int OverlapPercentage { get; set; } = 50;

    /// <summary>
    /// Gets or sets the sampling rate of the time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sampling rate is the number of samples per unit time (typically seconds).
    /// For example, a sampling rate of 1000 means 1000 samples per second (1 kHz).
    /// This is used to convert normalized frequencies to actual frequencies in the original units.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// The sampling rate tells the model how frequently your data was collected. For example:
    /// - For hourly temperature data, the sampling rate would be 1/3600 (one sample per 3600 seconds)
    /// - For audio data, common sampling rates are 44100 Hz (CD quality) or 48000 Hz (DVD quality)
    /// - For ECG data, sampling rates might be around 250 Hz or 500 Hz
    /// 
    /// When you provide the sampling rate, the model can express frequencies in real-world units
    /// (like Hz or cycles per day) rather than just normalized values between 0 and 0.5.
    /// 
    /// If you don't specify a sampling rate, the default value of 1.0 is used, meaning frequencies
    /// will be expressed as cycles per sample. To get frequencies in Hz, set this to your actual
    /// sampling rate in Hz.
    /// </para>
    /// </remarks>
    public double SamplingRate { get; set; } = 1.0; // Default to 1.0 (cycles per sample)
}
