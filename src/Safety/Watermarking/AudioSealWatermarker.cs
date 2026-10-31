using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Localized keyed audio watermark: the spread-spectrum mark of <see cref="AudioWatermarker{T}"/>, detected segment by
/// segment so it reports which parts of a recording carry the mark.
/// </summary>
/// <remarks>
/// <para>
/// AudioSeal (San Roman et al. 2024) introduced localized detection: a detector that says, for each part of a clip,
/// whether that part is watermarked, so a recording that splices generated speech into real speech is caught and the
/// generated part located. AudioSeal reaches this with a jointly trained generator and detector network, which this
/// library does not ship. This class provides the same localization with a keyed spread-spectrum mark instead: the
/// audio is marked frame by frame as in <see cref="AudioWatermarker{T}"/>, and <see cref="DetectSegments"/> runs the
/// correlation test on each segment of <c>segmentSamples</c> samples separately.
/// </para>
/// <para>
/// <see cref="AudioWatermarkerBase{T}.Detect"/> asks whether any part is marked: it takes the strongest of the pooled
/// test and the per-segment tests and applies a Bonferroni correction for their number, so the false-positive rate of
/// a single test holds. Segments are on the original frame grid, so localization covers edits that replace audio in
/// place; cropping or inserting audio shifts the grid and is not covered.
/// </para>
/// <para>
/// <b>For Beginners:</b> this watermarker can tell you which seconds of a recording carry your watermark, not just
/// whether the recording does, which helps when only part of a clip was AI-generated.
/// </para>
/// <para>
/// <b>References:</b>
/// - San Roman, Fernandez, Elsahar, Defossez, Furon, Tran. Proactive Detection of Voice Cloning with Localized
///   Watermarking (AudioSeal, ICML 2024) — the localization objective, not the network
/// - Kirovski, Malvar. Spread-Spectrum Watermarking of Audio Signals (IEEE TSP, 2003) — the mark itself
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Spread-Spectrum Watermarking of Audio Signals",
    "https://doi.org/10.1109/TSP.2003.809384",
    Year = 2003,
    Authors = "Darko Kirovski, Henrique S. Malvar")]
public class AudioSealWatermarker<T> : AudioWatermarkerBase<T>
{
    private readonly int _segmentSize;

    /// <inheritdoc />
    public override string ModuleName => "AudioSealWatermarker";

    /// <inheritdoc />
    protected override string WatermarkDescription => "Localized spread-spectrum audio watermark";

    /// <inheritdoc />
    protected override int PatternSalt => 0x4C4F4331; // "LOC1"

    /// <summary>
    /// Initializes a new localized audio watermarker.
    /// </summary>
    /// <param name="watermarkStrength">
    /// Embedding strength, positive, as in <see cref="AudioWatermarker{T}"/>. Default: 0.5.
    /// </param>
    /// <param name="segmentSamples">
    /// Segment length in samples for localized detection; frames are assigned to the segment their first sample
    /// falls in. Default: 4096 (0.26 s at 16 kHz).
    /// </param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    public AudioSealWatermarker(double watermarkStrength = 0.5, int segmentSamples = 4096, string? key = null)
        : base(watermarkStrength, key)
    {
        if (segmentSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(segmentSamples), "The segment length must be positive.");
        _segmentSize = segmentSamples;
    }

    /// <summary>
    /// Runs the correlation test on each segment separately.
    /// </summary>
    /// <param name="audioSamples">The audio samples to check.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>
    /// One result per segment, in order, up to the last segment a frame starts in; segment i starts at sample
    /// i x segmentSamples. A segment shorter than a frame can contain no frame start; its result has
    /// <see cref="WatermarkResult.ScoredUnits"/> 0. Each result's confidence is that of its own test, without a
    /// correction for the number of segments.
    /// </returns>
    public IReadOnlyList<WatermarkResult> DetectSegments(Vector<T> audioSamples, int sampleRate) =>
        Segments(FrameStatistics(audioSamples, sampleRate));

    /// <inheritdoc />
    public override WatermarkResult Detect(Vector<T> audioSamples, int sampleRate)
    {
        var frames = FrameStatistics(audioSamples, sampleRate);
        int units = 0;
        foreach (var f in frames) units += f.Count;
        double best = PooledZ(frames);
        // Only segments that scored a frame are tests; counting empty ones would over-correct the Bonferroni bound.
        var scored = Segments(frames).Where(segment => segment.ScoredUnits > 0).ToList();
        foreach (var segment in scored)
            if (segment.ZScore > best) best = segment.ZScore;
        int tests = scored.Count > 1 ? scored.Count + 1 : 1;
        return AudioResult(WatermarkStatistics.BonferroniZ(best, tests), units);
    }

    private List<WatermarkResult> Segments(IReadOnlyList<(int StartSample, double Numerator, double Energy, int Count)> frames)
    {
        var results = new List<WatermarkResult>();
        if (frames.Count == 0) return results;
        int segments = (frames[frames.Count - 1].StartSample / _segmentSize) + 1;
        var numerators = new double[segments];
        var energies = new double[segments];
        var units = new int[segments];
        foreach (var f in frames)
        {
            int s = f.StartSample / _segmentSize;
            numerators[s] += f.Numerator;
            energies[s] += f.Energy;
            units[s] += f.Count;
        }

        for (int s = 0; s < segments; s++)
            results.Add(AudioResult(WatermarkStatistics.CorrelationZ(numerators[s], energies[s]), units[s]));
        return results;
    }

    /// <inheritdoc />
    protected override void EmbedFrame(double[] band, double[] pattern, double frameRms) => AdditiveEmbed(band, pattern, frameRms);

    /// <inheritdoc />
    protected override bool DetectionWeights(double[] band, double[] weights) => AdditiveWeights(band, weights);
}
