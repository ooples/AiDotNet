// -----------------------------------------------------------------------------
// Port of DIO and StoneMask from WORLD (https://github.com/mmorise/World).
//
// Copyright (c) 2010 M. Morise. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
// - Redistributions of source code must retain the above copyright notice, this list of conditions and the
//   following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
//   following disclaimer in the documentation and/or other materials provided with the distribution.
// - Neither the name of the M. Morise nor the names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
// -----------------------------------------------------------------------------
using AiDotNet.Attributes;
using AiDotNet.Enums;
using EnginePadMode = AiDotNet.Tensors.Engines.PadMode;

namespace AiDotNet.Audio.Pitch;

/// <summary>
/// F0 estimation with WORLD's DIO (Distributed Inline-filter Operation), refined by StoneMask.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the F0 extractor FastSpeech 2 (Ren et al. 2021, Appendix C.2) and the acoustic models built on it take
/// their pitch targets from: PyWorld's <c>dio</c> followed by <c>stonemask</c>. The port follows WORLD's C++
/// source line for line, including its in-place spectrum handling, so it reproduces PyWorld's contours rather
/// than approximating them. Frequencies are in Hz; unvoiced frames are 0.
/// </para>
/// <para><b>For Beginners:</b> DIO looks at the signal through a bank of low-pass filters and measures the
/// spacing of zero crossings and peaks in each one; the most self-consistent spacing becomes a rough pitch.
/// StoneMask then sharpens each rough pitch using the instantaneous frequency of its first harmonics.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.SignalProcessing)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications", "https://doi.org/10.1587/transinf.2015EDP7457", Year = 2016, Authors = "Masanori Morise, Fumiya Yokomori, Kenji Ozawa")]
public class WorldPitchDetector<T> : PitchDetectorBase<T>
{
    // WORLD constantnumbers.h
    private const double CutOff = 50.0;
    private const double FloorF0StoneMask = 40.0;
    private const double SafeGuardMinimum = 0.000000000001;
    private const double Log2 = 0.69314718055994529;
    private const double MaximumValue = 100000.0;

    // The smallest positive normal Half, 2^-14: Half cannot hold WORLD's 1e-12 safeguard, which rounds to 0.
    private const double HalfSmallestNormal = 6.103515625e-05;

    // Power-of-two pre-scale of the F0 deviations DIO squares (exact in binary floating point, so float and double results
    // are unchanged, while Half's squares stay below 65504).
    private const double DeviationScale = 1.0 / 1024.0;

    // Elements per batched refinement or filtering block, to bound memory on long recordings.
    private const int BlockElements = 1 << 20;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly double _channelsInOctave;
    private readonly int _speed;
    private readonly double _allowedRange;
    private readonly bool _refineWithStoneMask;

    /// <summary>
    /// Creates a DIO + StoneMask detector with WORLD's defaults (F0 71-800 Hz, 2 channels per octave, speed 1,
    /// allowed range 0.1).
    /// </summary>
    /// <param name="sampleRate">Sampling rate of the audio, in Hz.</param>
    /// <param name="minPitch">DIO's F0 floor, in Hz.</param>
    /// <param name="maxPitch">DIO's F0 ceiling, in Hz.</param>
    /// <param name="channelsInOctave">Number of DIO filter bands per octave.</param>
    /// <param name="speed">DIO decimation ratio, 1 to 12; 1 is the most accurate.</param>
    /// <param name="allowedRange">Threshold DIO uses when fixing the contour.</param>
    /// <param name="refineWithStoneMask">Whether to refine DIO's estimate with StoneMask, as PyWorld pipelines do.</param>
    public WorldPitchDetector(
        int sampleRate = 22050,
        double minPitch = 71.0,
        double maxPitch = 800.0,
        double channelsInOctave = 2.0,
        int speed = 1,
        double allowedRange = 0.1,
        bool refineWithStoneMask = true)
        : base(sampleRate, minPitch, maxPitch)
    {
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (minPitch <= 0 || maxPitch <= minPitch)
            throw new ArgumentOutOfRangeException(nameof(maxPitch), "Require 0 < minPitch < maxPitch.");
        if (channelsInOctave <= 0) throw new ArgumentOutOfRangeException(nameof(channelsInOctave));
        _channelsInOctave = channelsInOctave;
        _speed = speed;
        _allowedRange = allowedRange;
        _refineWithStoneMask = refineWithStoneMask;
    }

    private static IEngine E => AiDotNetEngine.Current;

    /// <summary>
    /// Estimates the F0 contour of <paramref name="audio"/> with one value per <paramref name="framePeriodMs"/>,
    /// starting at time 0 (PyWorld's <c>dio</c>, then <c>stonemask</c> when enabled).
    /// </summary>
    /// <returns>F0 in Hz per frame (0 where unvoiced) and each frame's time in seconds, both <c>[frames]</c>.</returns>
    public (Tensor<T> F0, Tensor<T> TemporalPositions) EstimateF0(Tensor<T> audio, double framePeriodMs)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (framePeriodMs <= 0) throw new ArgumentOutOfRangeException(nameof(framePeriodMs));
        if (audio.Length < 2) throw new ArgumentException("DIO needs at least two samples.", nameof(audio));

        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var x = Flat(audio);
        int f0Length = GetSamplesForDio(SampleRate, x.Length, framePeriodMs);
        var temporalPositions = Positions(f0Length, framePeriodMs);
        var f0 = DioGeneralBody(x, SampleRate, framePeriodMs, MinPitch, MaxPitch, _channelsInOctave, _speed,
            _allowedRange, temporalPositions);
        if (!_refineWithStoneMask) return (f0, temporalPositions);
        return (StoneMaskCore(x, SampleRate, f0, temporalPositions), temporalPositions);
    }

    /// <inheritdoc/>
    public override IReadOnlyList<PitchFrame<T>> ExtractDetailedPitchContour(Tensor<T> audio, int hopSizeMs = 10)
    {
        var (f0, _) = EstimateF0(audio, hopSizeMs);
        var voiced = E.TensorGreaterThan(f0, Ops.Zero);
        var pitch = f0.ToArray();
        var confidence = voiced.ToArray();
        var isVoiced = voiced.Cast<int>().ToArray();
        var frames = new PitchFrame<T>[pitch.Length];
        for (int i = 0; i < frames.Length; i++)
        {
            frames[i] = new PitchFrame<T>
            {
                Time = i * (double)hopSizeMs / 1000.0,
                Pitch = pitch[i],
                Confidence = confidence[i],
                IsVoiced = isVoiced[i] != 0
            };
        }
        return frames;
    }

    /// <inheritdoc/>
    /// <remarks>DIO is an utterance-level method; a single frame is analysed as a short utterance and the
    /// median of its voiced estimates is returned, with the voiced fraction as the confidence.</remarks>
    public override (T Pitch, T Confidence)? DetectPitchWithConfidence(Tensor<T> audioFrame)
    {
        if (audioFrame is null) throw new ArgumentNullException(nameof(audioFrame));
        return DetectFramePitch(Flat(audioFrame));
    }

    /// <inheritdoc/>
    protected override (double Pitch, double Confidence)? DetectPitchInternal(double[] frame)
    {
        if (frame is null) throw new ArgumentNullException(nameof(frame));
        // The base class hands decoded double samples over; they are cast to T once.
        var result = DetectFramePitch(new Tensor<double>(new[] { frame.Length }, new Vector<double>((double[])frame.Clone())).Cast<T>());
        if (!result.HasValue) return null;
        return (Ops.ToDouble(result.Value.Pitch), Ops.ToDouble(result.Value.Confidence));
    }

    private (T Pitch, T Confidence)? DetectFramePitch(Tensor<T> frame)
    {
        double framePeriodMs = 1000.0 * frame.Length / SampleRate / 4.0;
        var (f0, _) = EstimateF0(frame, Math.Max(1.0, framePeriodMs));
        int voicedCount = E.TensorCountNonzero(E.TensorGreaterThan(f0, Ops.Zero));
        if (voicedCount == 0) return null;
        // F0 is 0 where unvoiced and positive elsewhere, so the voiced values are the top of the ascending sort.
        var (sorted, _) = E.TensorSort(f0, 0, false);
        int unvoiced = f0.Length - voicedCount;
        return (sorted[unvoiced + voicedCount / 2], Ops.FromDouble((double)voicedCount / f0.Length));
    }

    #region DIO (dio.cpp)

    /// <summary>Number of frames DIO produces (WORLD GetSamplesForDIO).</summary>
    public static int GetSamplesForDio(int fs, int xLength, double framePeriod)
        => (int)(1000.0 * xLength / fs / framePeriod) + 1;

    private static Tensor<T> DioGeneralBody(Tensor<T> x, int fs, double framePeriod, double f0Floor, double f0Ceil,
        double channelsInOctave, int speed, double allowedRange, Tensor<T> temporalPositions)
    {
        int xLength = x.Length;
        int f0Length = temporalPositions.Length;
        int numberOfBands = 1 + (int)(Math.Log(f0Ceil / f0Floor) / Log2 * channelsInOctave);
        var boundaryF0List = new double[numberOfBands];
        for (int i = 0; i < numberOfBands; i++)
            boundaryF0List[i] = f0Floor * Math.Pow(2.0, (i + 1) / channelsInOctave);

        int decimationRatio = Math.Max(Math.Min(speed, 12), 1);
        int yLength = 1 + xLength / decimationRatio;
        double actualFs = (double)fs / decimationRatio;
        int fftSize = GetSuitableFftSize(yLength + MatlabRound(actualFs / CutOff) * 2 + 1
            + 4 * (int)(1.0 + actualFs / boundaryF0List[0] / 2.0));

        var (ySpecRe, ySpecIm) = GetSpectrumForEstimation(x, yLength, actualFs, fftSize, decimationRatio);

        // Per band, the candidate and its score (the deviation of the four interval estimates over the candidate);
        // the best band per frame is the first with the smallest score, as WORLD's strict '>' scan keeps it.
        var candidates = new Tensor<T>[numberOfBands];
        Tensor<T> bestF0Contour = Zeros(f0Length);
        Tensor<T> bestScore = Zeros(f0Length);
        var guard = Guard(SafeGuardMinimum);
        for (int i = 0; i < numberOfBands; i++)
        {
            var (score, candidate) = GetF0CandidateFromRawEvent(boundaryF0List[i], actualFs, ySpecRe, ySpecIm,
                yLength, fftSize, f0Floor, f0Ceil, temporalPositions);
            var bandScore = E.TensorDivide(score, E.TensorAddScalar(candidate, guard));
            candidates[i] = candidate;
            if (i == 0)
            {
                bestF0Contour = candidate;
                bestScore = bandScore;
                continue;
            }
            var better = E.TensorGreaterThan(bestScore, bandScore);
            bestF0Contour = E.TensorWhere(better, candidate, bestF0Contour);
            bestScore = E.TensorWhere(better, bandScore, bestScore);
        }

        var candidatesByFrame = E.TensorTranspose(E.TensorStack(candidates, 0));    // [frames, bands]
        return FixF0Contour(framePeriod, candidatesByFrame, bestF0Contour, f0Length, f0Floor, allowedRange);
    }

    // The mean-removed (decimated) signal's spectrum through DIO's low-cut filter, bins 0 … fftSize/2.
    private static (Tensor<T> Re, Tensor<T> Im) GetSpectrumForEstimation(Tensor<T> x, int yLength, double actualFs,
        int fftSize, int decimationRatio)
    {
        // WORLD fills y from a zeroed buffer of fftSize: the decimated (or copied) samples, and zeros after them up to
        // yLength, all of which enter the mean.
        var source = decimationRatio != 1 ? Decimate(x, decimationRatio) : x;
        var y = source.Length >= yLength ? Slice(source, 0, yLength) : Pad(source, 0, yLength - source.Length);
        y = E.TensorSubtractScalar(y, ScaledMean(y));
        var (specRe, specIm) = Fft(Pad(y, 0, fftSize - yLength));

        int cutoffInSample = MatlabRound(actualFs / CutOff);
        var (filtRe, filtIm) = Fft(DesignLowCutFilter(cutoffInSample * 2 + 1, fftSize));

        int bins = fftSize / 2 + 1;
        return ComplexMultiply(Slice(specRe, 0, bins), Slice(specIm, 0, bins), Slice(filtRe, 0, bins), Slice(filtIm, 0, bins));
    }

    // WORLD's DesignLowCutFilter, step by step on the fftSize buffer: the normalized negative Hann window, its first half
    // copied to the end, everything shifted left by half the window, and 1 added at index 0.
    private static Tensor<T> DesignLowCutFilter(int n, int fftSize)
    {
        var window = E.TensorAddScalar(E.TensorMultiplyScalar(E.TensorCos(E.TensorDivideScalar(
            E.TensorMultiplyScalar(E.TensorMultiplyScalar(Range(1, n), C(2.0)), C(Math.PI)), C(n + 1.0))), C(-0.5)), C(0.5));
        var filter = Pad(E.TensorDivideScalar(E.TensorNegate(window), E.TensorSum(window)), 0, fftSize - n);
        int half = (n - 1) / 2;
        if (half > 0) filter = E.TensorSetSlice(filter, Slice(filter, 0, half), new[] { fftSize - half });
        filter = Cat(Slice(filter, half, n), Slice(filter, n, fftSize - n));
        return E.TensorSetSlice(filter, E.TensorAddScalar(Slice(filter, 0, 1), Ops.One), new[] { 0 });
    }

    private static Tensor<T> FixF0Contour(double framePeriod, Tensor<T> candidatesByFrame, Tensor<T> bestF0Contour,
        int f0Length, double f0Floor, double allowedRange)
    {
        int voiceRangeMinimum = (int)(0.5 + 1000.0 / framePeriod / f0Floor) * 2 + 1;
        if (f0Length <= voiceRangeMinimum) return Zeros(f0Length);

        // FixStep1: eliminate unnatural jumps.
        int inner = f0Length - 2 * voiceRangeMinimum;
        var f0Base = inner > 0 ? Pad(Slice(bestF0Contour, voiceRangeMinimum, inner), voiceRangeMinimum, voiceRangeMinimum)
            : Zeros(f0Length);
        var previous = Pad(Slice(f0Base, 0, f0Length - 1), 1, 0);
        var jump = E.TensorAbs(E.TensorDivide(E.TensorSubtract(f0Base, previous),
            E.TensorAddScalar(f0Base, Guard(SafeGuardMinimum))));
        var f0Tmp1 = E.TensorWhere(E.TensorLessThan(jump, C(allowedRange)), f0Base, Zeros(f0Length));
        f0Tmp1 = Pad(Slice(f0Tmp1, voiceRangeMinimum, f0Length - voiceRangeMinimum), voiceRangeMinimum, 0);

        // FixStep2: eliminate suspected F0 at the start and end of voiced sections (a zero anywhere in the window).
        int center = (voiceRangeMinimum - 1) / 2;
        int middle = f0Length - 2 * center;
        var zerosSoFar = E.TensorCumSum(PadInt(E.TensorEquals(f0Tmp1, Ops.Zero).Cast<int>(), 1, 0), 0);
        var windowZeros = E.TensorSubtract(SliceInt(zerosSoFar, 2 * center + 1, middle), SliceInt(zerosSoFar, 0, middle));
        var hasZero = E.TensorGreaterThan(windowZeros, 0).Cast<T>();
        var f0Tmp2 = Cat(Slice(f0Tmp1, 0, center),
            E.TensorWhere(hasZero, Zeros(middle), Slice(f0Tmp1, center, middle)),
            Slice(f0Tmp1, f0Length - center, center));

        // Section ends (voiced to unvoiced) and starts (unvoiced to voiced).
        var voiced = E.TensorNotEquals(f0Tmp2, Ops.Zero).Cast<int>();
        var change = E.TensorSubtract(SliceInt(voiced, 1, f0Length - 1), SliceInt(voiced, 0, f0Length - 1));
        int[] negativeIndex = E.TensorNonzero(E.TensorEquals(change, -1)).ToArray();
        int[] positiveIndex = E.TensorAddScalar(E.TensorNonzero(E.TensorEquals(change, 1)), 1).ToArray();

        // FixStep3: extend voiced sections forward. Each frame's choice depends on the two before it, so the extension is
        // a scan over frames; each step picks the nearest band candidate with engine ops.
        var extended = f0Tmp2.Clone();
        for (int i = 0; i < negativeIndex.Length; i++)
        {
            int limit = i == negativeIndex.Length - 1 ? f0Length - 1 : negativeIndex[i + 1];
            for (int j = negativeIndex[i]; j < limit; j++)
                if (!ExtendVoicing(extended, candidatesByFrame, j + 1, j, j - 1, allowedRange)) break;
        }

        // FixStep4: extend voiced sections backward.
        var fixedF0Contour = extended.Clone();
        for (int i = positiveIndex.Length - 1; i >= 0; i--)
        {
            int limit = i == 0 ? 1 : positiveIndex[i - 1];
            for (int j = positiveIndex[i]; j > limit; j--)
                if (!ExtendVoicing(fixedF0Contour, candidatesByFrame, j - 1, j, j + 1, allowedRange)) break;
        }
        return fixedF0Contour;
    }

    // One step of FixStep3/4 (WORLD SelectBestF0): the band candidate nearest the linear extrapolation of the contour,
    // or 0 when it is further than the allowed range. Writes the target frame; returns whether it is voiced.
    private static bool ExtendVoicing(Tensor<T> contour, Tensor<T> candidatesByFrame, int target, int current, int past,
        double allowedRange)
    {
        T referenceF0 = Ops.Divide(Ops.Subtract(Ops.Multiply(contour[current], C(3.0)), contour[past]), C(2.0));
        var row = Row(candidatesByFrame, target);
        var error = E.TensorAbs(E.TensorSubtractScalar(row, referenceF0));
        int best = FirstIndexOf(error, E.TensorMinValue(error));
        T bestF0 = row[best];
        if (Ops.GreaterThan(Ops.Abs(Ops.Subtract(Ops.One, Ops.Divide(bestF0, referenceF0))), C(allowedRange)))
            bestF0 = Ops.Zero;
        contour[target] = bestF0;
        return !Ops.Equals(bestF0, Ops.Zero);
    }

    private static (Tensor<T> Score, Tensor<T> Candidate) GetF0CandidateFromRawEvent(double boundaryF0, double fs,
        Tensor<T> ySpecRe, Tensor<T> ySpecIm, int yLength, int fftSize, double f0Floor, double f0Ceil,
        Tensor<T> temporalPositions)
    {
        int f0Length = temporalPositions.Length;
        var filteredSignal = GetFilteredSignal(MatlabRound(fs / boundaryF0 / 2.0), fftSize, ySpecRe, ySpecIm, yLength);
        var sets = FourIntervalContours(filteredSignal, fs, temporalPositions);
        if (sets is null)
            return (Tensor<T>.CreateDefault(new[] { f0Length }, C(MaximumValue)), Zeros(f0Length));

        var candidate = E.TensorDivideScalar(E.TensorAdd(E.TensorAdd(E.TensorAdd(sets[0], sets[1]), sets[2]), sets[3]), C(4.0));
        Tensor<T> Deviation(Tensor<T> set) => E.TensorSquare(E.TensorMultiplyScalar(E.TensorSubtract(set, candidate), C(DeviationScale)));
        var score = E.TensorMultiplyScalar(E.TensorSqrt(E.TensorDivideScalar(E.TensorAdd(E.TensorAdd(E.TensorAdd(
            Deviation(sets[0]), Deviation(sets[1])), Deviation(sets[2])), Deviation(sets[3])), C(3.0))), C(1.0 / DeviationScale));

        var outOfRange = OutOfRange(candidate, boundaryF0 / 2.0, boundaryF0, f0Floor, f0Ceil);
        return (E.TensorWhere(outOfRange, Tensor<T>.CreateDefault(new[] { f0Length }, C(MaximumValue)), score),
            E.TensorWhere(outOfRange, Zeros(f0Length), candidate));
    }

    // 1 where the candidate is above either upper bound or below either lower bound (NaN compares false, as in WORLD).
    private static Tensor<T> OutOfRange(Tensor<T> candidate, double lower, double upper, double f0Floor, double f0Ceil)
    {
        var one = Tensor<T>.CreateDefault(candidate._shape, Ops.One);
        var outside = E.TensorWhere(E.TensorGreaterThan(candidate, C(upper)), one, E.TensorLessThan(candidate, C(lower)));
        outside = E.TensorWhere(E.TensorGreaterThan(candidate, C(f0Ceil)), one, outside);
        return E.TensorWhere(E.TensorLessThan(candidate, C(f0Floor)), one, outside);
    }

    // The signal through DIO's Nuttall low-pass of length 4·halfAverageLength, delay compensated.
    private static Tensor<T> GetFilteredSignal(int halfAverageLength, int fftSize, Tensor<T> ySpecRe, Tensor<T> ySpecIm,
        int yLength)
    {
        int length = halfAverageLength * 4;
        var (lpRe, lpIm) = Fft(Pad(NuttallWindow(length), 0, fftSize - length));
        var (re, im) = ConvolveHalfSpectrum(ySpecRe, ySpecIm, lpRe, lpIm, fftSize);
        return Slice(InverseRealFft(re, im, fftSize), halfAverageLength * 2, yLength);
    }

    /// <summary>
    /// WORLD's frequency-domain convolution (dio.cpp and harvest.cpp GetFilteredSignal): bins 0 … fftSize/2 of the
    /// signal's spectrum times the filter's. WORLD writes each product to bin fftSize − i − 1 as well while it goes, and at
    /// i = fftSize/2 − 1 that is bin fftSize/2, which the next iteration multiplies by the signal again; the inverse
    /// transform then reads both bins fftSize/2 − 1 and fftSize/2 as Y[N/2]·Y[N/2−1]·F[N/2−1].
    /// </summary>
    internal static (Tensor<T> Re, Tensor<T> Im) ConvolveHalfSpectrum(Tensor<T> ySpecRe, Tensor<T> ySpecIm,
        Tensor<T> filterRe, Tensor<T> filterIm, int fftSize)
    {
        int half = fftSize / 2;
        var (pRe, pIm) = ComplexMultiply(ySpecRe, ySpecIm, Slice(filterRe, 0, half + 1), Slice(filterIm, 0, half + 1));
        var (qRe, qIm) = ComplexMultiply(Slice(ySpecRe, half, 1), Slice(ySpecIm, half, 1), Slice(pRe, half - 1, 1), Slice(pIm, half - 1, 1));
        return (Cat(Slice(pRe, 0, half - 1), qRe, qRe), Cat(Slice(pIm, 0, half - 1), qIm, qIm));
    }

    // GetFourZeroCrossingIntervals, each interpolated to the frame times; null when any has fewer than three intervals.
    private static Tensor<T>[]? FourIntervalContours(Tensor<T> filtered, double fs, Tensor<T> temporalPositions)
    {
        int length = filtered.Length;
        var negated = E.TensorNegate(filtered);
        var difference = E.TensorSubtract(Slice(negated, 0, length - 1), Slice(negated, 1, length - 1));
        var series = new[]
        {
            ZeroCrossingEngine(filtered, fs),
            ZeroCrossingEngine(negated, fs),
            ZeroCrossingEngine(difference, fs),
            ZeroCrossingEngine(E.TensorNegate(difference), fs),
        };
        for (int k = 0; k < series.Length; k++)
            if (series[k].Count <= 2) return null;
        var sets = new Tensor<T>[series.Length];
        for (int k = 0; k < series.Length; k++)
            sets[k] = Interp1(series[k].Locations, series[k].Intervals, series[k].Count, temporalPositions);
        return sets;
    }

    // Negative-going zero crossings, refined linearly between samples; the interval between consecutive crossings as a
    // frequency and its midpoint in seconds. Count is the number of intervals (0 with fewer than two crossings).
    private static (Tensor<T> Locations, Tensor<T> Intervals, int Count) ZeroCrossingEngine(Tensor<T> signal, double fs)
    {
        int length = signal.Length;
        var none = (Zeros(1), Zeros(1), 0);
        if (length < 2) return none;
        var current = Slice(signal, 0, length - 1);
        var next = Slice(signal, 1, length - 1);
        var one = Tensor<T>.CreateDefault(current._shape, Ops.One);
        var notPositive = E.TensorWhere(E.TensorLessThan(next, Ops.Zero), one, E.TensorEquals(next, Ops.Zero));
        var crossing = E.TensorWhere(E.TensorGreaterThan(current, Ops.Zero), notPositive, Zeros(length - 1));
        var before = E.TensorNonzero(crossing);
        int count = before.Shape[0];
        if (count < 2) return none;
        before = E.Reshape(before, new[] { count });
        var edges = E.TensorAddScalar(before, 1);
        var atBefore = E.TensorIndexSelect(signal, before, 0);
        var atEdge = E.TensorIndexSelect(signal, edges, 0);
        var fineEdges = E.TensorSubtract(edges.Cast<T>(), E.TensorDivide(atBefore, E.TensorSubtract(atEdge, atBefore)));
        var lower = Slice(fineEdges, 0, count - 1);
        var upper = Slice(fineEdges, 1, count - 1);
        var intervals = Over(fs, E.TensorSubtract(upper, lower));
        var locations = E.TensorDivideScalar(E.TensorDivideScalar(E.TensorAdd(lower, upper), C(2.0)), C(fs));
        return (locations, intervals, count - 1);
    }

    #endregion

    #region StoneMask (stonemask.cpp)

    /// <summary>StoneMask's refinement (PyWorld's <c>stonemask</c>) of an F0 contour at its frames' times.</summary>
    /// <returns>The refined F0, <c>[frames]</c>.</returns>
    public Tensor<T> StoneMask(Tensor<T> x, Tensor<T> f0, Tensor<T> temporalPositions)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (f0 is null) throw new ArgumentNullException(nameof(f0));
        if (temporalPositions is null) throw new ArgumentNullException(nameof(temporalPositions));
        if (f0.Length != temporalPositions.Length) throw new ArgumentException("Each F0 value needs its time.", nameof(temporalPositions));
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return StoneMaskCore(Flat(x), SampleRate, Flat(f0), Flat(temporalPositions));
    }

    // GetRefinedF0 for every frame: frames whose F0 is outside (40 Hz, fs/12] give 0; the rest are refined in blocks of
    // equal FFT size, then put back in frame order.
    private static Tensor<T> StoneMaskCore(Tensor<T> x, int fs, Tensor<T> f0, Tensor<T> temporalPositions)
    {
        int frames = f0.Length;
        var one = Tensor<T>.CreateDefault(f0._shape, Ops.One);
        var tooHigh = E.TensorWhere(E.TensorGreaterThan(f0, C(fs / 12.0)), Zeros(frames), one);
        var valid = E.TensorWhere(E.TensorGreaterThan(f0, C(FloorF0StoneMask)), tooHigh, Zeros(frames));
        int[] validFrames = E.TensorNonzero(valid).ToArray();
        if (validFrames.Length == 0) return Zeros(frames);

        var validIndex = IntTensor(validFrames);
        var initialF0 = E.TensorIndexSelect(f0, validIndex, 0);
        var positions = E.TensorIndexSelect(temporalPositions, validIndex, 0);
        int[] halfWindow = E.TensorFloor(E.TensorAddScalar(Over(1.5 * fs, initialF0), Ops.One)).Cast<int>().ToArray();

        var order = new List<int>();
        var refined = new List<Tensor<T>>();
        foreach (var block in RefinementBlocks(halfWindow))
        {
            var pairs = IntTensor(block.Pairs);
            var blockF0 = E.TensorIndexSelect(initialF0, pairs, 0);
            var (power, numeratorI) = InstantaneousSpectra(x, fs, E.TensorIndexSelect(positions, pairs, 0),
                block.HalfWindow, block.FftSize, RefinementWindow.StoneMask);
            var guard = Guard(SafeGuardMinimum / block.FftSize);
            var tentative = FixF0(power, numeratorI, block.FftSize, fs, blockF0, 2, guard);
            // GetMeanF0 gives 0 when the two-harmonic estimate is not in (0, 2·F0]; GetRefinedF0 then keeps the input F0.
            var none = Zeros(block.Pairs.Length);
            var ok = E.TensorWhere(E.TensorGreaterThan(tentative, Ops.Zero),
                E.TensorWhere(E.TensorGreaterThan(tentative, E.TensorMultiplyScalar(blockF0, C(2.0))), none,
                    Tensor<T>.CreateDefault(none._shape, Ops.One)), none);
            var mean = E.TensorWhere(ok, FixF0(power, numeratorI, block.FftSize, fs, E.TensorWhere(ok, tentative, blockF0), 6, guard), none);
            var far = E.TensorGreaterThan(E.TensorAbs(E.TensorSubtract(mean, blockF0)), E.TensorMultiplyScalar(blockF0, C(0.2)));
            refined.Add(E.TensorWhere(far, blockF0, mean));
            foreach (int pair in block.Pairs) order.Add(validFrames[pair]);
        }
        return E.TensorIndexPut(Zeros(frames), new[] { IntTensor(order.ToArray()) }, Cat(refined.ToArray()), false);
    }

    // WORLD FixF0 (StoneMask's): the amplitude-weighted mean of the first harmonics' instantaneous frequencies over
    // harmonic number. power and numeratorI are [pairs, fftSize/2 + 1], f0 is [pairs].
    private static Tensor<T> FixF0(Tensor<T> power, Tensor<T> numeratorI, int fftSize, double fs, Tensor<T> f0,
        int numberOfHarmonics, T guard)
    {
        var (harmonic, instantaneous, amplitude) = Harmonics(power, numeratorI, fftSize, fs, f0, numberOfHarmonics);
        var numerator = E.ReduceSum(E.TensorMultiply(amplitude, instantaneous), new[] { 1 }, false);
        var denominator = E.ReduceSum(E.TensorMultiply(amplitude, harmonic), new[] { 1 }, false);
        return E.TensorDivide(numerator, E.TensorAddScalar(denominator, guard));
    }

    // Per pair and harmonic k = 1 … H: k, and the instantaneous frequency (0 where the power is 0) and amplitude of bin
    // round(f0·N/fs·k), at most N/2 (StoneMask's min; Harvest's harmonics below Nyquist never reach it, and the bins of
    // the harmonics Harvest does not use are only read, never counted).
    private static (Tensor<T> Harmonic, Tensor<T> Instantaneous, Tensor<T> Amplitude) Harmonics(
        Tensor<T> power, Tensor<T> numeratorI, int fftSize, double fs, Tensor<T> f0, int numberOfHarmonics)
    {
        int pairs = f0.Length;
        var shape = new[] { pairs, numberOfHarmonics };
        var harmonic = E.TensorBroadcastTo(E.Reshape(Range(1, numberOfHarmonics), new[] { 1, numberOfHarmonics }), shape);
        var perPair = E.TensorBroadcastTo(E.Reshape(f0, new[] { pairs, 1 }), shape);
        var bin = MatlabRound(E.TensorMultiply(E.TensorDivideScalar(E.TensorMultiplyScalar(perPair, C(fftSize)), C(fs)), harmonic));
        var index = E.TensorClamp(bin.Cast<int>(), 0, fftSize / 2);
        var p = E.TensorTakeAlongDim(power, index, 1);
        var q = E.TensorTakeAlongDim(numeratorI, index, 1);
        var frequency = E.TensorAdd(
            E.TensorDivideScalar(E.TensorMultiplyScalar(index.Cast<T>(), C(fs)), C(fftSize)),
            E.TensorDivideScalar(E.TensorDivideScalar(E.TensorMultiplyScalar(E.TensorDivide(q, p), C(fs)), C(2.0)), C(Math.PI)));
        var instantaneous = E.TensorWhere(E.TensorEquals(p, Ops.Zero), new Tensor<T>(shape), frequency);
        return (harmonic, instantaneous, E.TensorSqrt(p));
    }

    #endregion

    #region Instantaneous-frequency spectra (stonemask.cpp GetMeanF0, harvest.cpp GetMeanF0)

    /// <summary>How the analysis window's sample positions are derived.</summary>
    private enum RefinementWindow
    {
        /// <summary>StoneMask: each sample index rounded from its own time.</summary>
        StoneMask,
        /// <summary>Harvest: the first index rounded (with WORLD's +0.001), the rest consecutive.</summary>
        Harvest,
    }

    private sealed class RefinementBlock
    {
        public RefinementBlock(int fftSize, int[] pairs, int[] halfWindow)
        {
            FftSize = fftSize;
            Pairs = pairs;
            HalfWindow = halfWindow;
        }

        public int FftSize { get; }
        public int[] Pairs { get; }
        public int[] HalfWindow { get; }
    }

    // Groups the pairs by WORLD's FFT size 2^(2 + floor(log2(2h + 1))) and splits each group into bounded blocks.
    private static IEnumerable<RefinementBlock> RefinementBlocks(int[] halfWindow)
    {
        var groups = new SortedDictionary<int, List<int>>();
        for (int p = 0; p < halfWindow.Length; p++)
        {
            int fftSize = (int)Math.Pow(2.0, 2.0 + (int)(Math.Log(halfWindow[p] * 2.0 + 1.0) / Log2));
            if (!groups.TryGetValue(fftSize, out var members)) groups[fftSize] = members = new List<int>();
            members.Add(p);
        }
        foreach (var group in groups)
        {
            int blockSize = Math.Max(1, BlockElements / group.Key);
            for (int start = 0; start < group.Value.Count; start += blockSize)
            {
                var pairs = group.Value.GetRange(start, Math.Min(blockSize, group.Value.Count - start)).ToArray();
                var half = new int[pairs.Length];
                for (int i = 0; i < pairs.Length; i++) half[i] = halfWindow[pairs[i]];
                yield return new RefinementBlock(group.Key, pairs, half);
            }
        }
    }

    /// <summary>
    /// The windowed spectra GetMeanF0 reads, for a block of pairs sharing one FFT size: the power spectrum of the
    /// Blackman-windowed waveform and the cross term with its time-derivative window, bins 0 … fftSize/2. Both spectra are
    /// scaled by 1/fftSize (a power of two, so float and double ratios are unchanged and Half does not overflow).
    /// </summary>
    private static (Tensor<T> Power, Tensor<T> NumeratorI) InstantaneousSpectra(Tensor<T> x, double fs,
        Tensor<T> positions, int[] halfWindow, int fftSize, RefinementWindow kind)
    {
        int pairs = halfWindow.Length;
        var shape = new[] { pairs, fftSize };
        var column = E.TensorBroadcastTo(E.Reshape(Range(0, fftSize), new[] { 1, fftSize }), shape);
        var half = IntTensor(halfWindow).Cast<T>();
        var halfB = PerPair(half, shape);
        var position = PerPair(positions, shape);

        Tensor<T> sampleIndex, windowTime;
        if (kind == RefinementWindow.StoneMask)
        {
            // index_raw[i] = round((t + (i − h)/fs)·fs); the window at (index_raw − 1)/fs − t.
            var baseTime = E.TensorDivideScalar(E.TensorSubtract(column, halfB), C(fs));
            var indexRaw = MatlabRound(E.TensorMultiplyScalar(E.TensorAdd(position, baseTime), C(fs)));
            sampleIndex = E.TensorSubtractScalar(indexRaw, Ops.One);
            windowTime = E.TensorSubtract(E.TensorDivideScalar(sampleIndex, C(fs)), position);
        }
        else
        {
            // basic_index = round((t − h/fs)·fs + 0.001); sample basic_index + i − 1.
            var basicIndex = MatlabRound(E.TensorAddScalar(E.TensorMultiplyScalar(
                E.TensorAdd(positions, E.TensorDivideScalar(E.TensorNegate(half), C(fs))), C(fs)), C(0.001)));
            sampleIndex = E.TensorSubtractScalar(E.TensorAdd(PerPair(basicIndex, shape), column), Ops.One);
            windowTime = E.TensorSubtract(E.TensorDivideScalar(sampleIndex, C(fs)), position);
        }

        var clamped = E.TensorClamp(sampleIndex.Cast<int>(), 0, x.Length - 1);
        var samples = E.Reshape(E.TensorTake(x, clamped), shape);
        var windowLength = PerPair(E.TensorDivideScalar(E.TensorAddScalar(E.TensorMultiplyScalar(half, C(2.0)), Ops.One), C(fs)), shape);
        var inside = E.TensorLessThan(column, E.TensorAddScalar(E.TensorMultiplyScalar(halfB, C(2.0)), Ops.One));
        var zero = new Tensor<T>(shape);

        // Blackman window 0.42 + 0.5 cos(2πτ/L) + 0.08 cos(4πτ/L) and its central difference, zero past the window.
        var mainWindow = E.TensorAdd(
            E.TensorAddScalar(E.TensorMultiplyScalar(E.TensorCos(E.TensorDivide(E.TensorMultiplyScalar(windowTime, C(2.0 * Math.PI)), windowLength)), C(0.5)), C(0.42)),
            E.TensorMultiplyScalar(E.TensorCos(E.TensorDivide(E.TensorMultiplyScalar(windowTime, C(4.0 * Math.PI)), windowLength)), C(0.08)));
        mainWindow = E.TensorWhere(inside, mainWindow, zero);
        var nextSample = E.PadNd(E.TensorSlice(mainWindow, new[] { 0, 1 }, new[] { pairs, fftSize - 1 }), new[] { 0, 1 }, EnginePadMode.Constant, Ops.Zero);
        var previousSample = E.PadNd(E.TensorSlice(mainWindow, new[] { 0, 0 }, new[] { pairs, fftSize - 1 }), new[] { 1, 0 }, EnginePadMode.Constant, Ops.Zero);
        var diffWindow = E.TensorWhere(inside, E.TensorDivideScalar(E.TensorNegate(E.TensorSubtract(nextSample, previousSample)), C(2.0)), zero);

        var (mainRe, mainIm) = Fft(E.TensorWhere(inside, E.TensorMultiply(samples, mainWindow), zero));
        var (diffRe, diffIm) = Fft(E.TensorWhere(inside, E.TensorMultiply(samples, diffWindow), zero));
        int bins = fftSize / 2 + 1;
        var scale = C(1.0 / fftSize);
        Tensor<T> Bins(Tensor<T> spectrum) => E.TensorMultiplyScalar(E.TensorSlice(spectrum, new[] { 0, 0 }, new[] { pairs, bins }), scale);
        mainRe = Bins(mainRe);
        mainIm = Bins(mainIm);
        diffRe = Bins(diffRe);
        diffIm = Bins(diffIm);
        var numeratorI = E.TensorSubtract(E.TensorMultiply(mainRe, diffIm), E.TensorMultiply(mainIm, diffRe));
        var power = E.TensorAdd(E.TensorMultiply(mainRe, mainRe), E.TensorMultiply(mainIm, mainIm));
        return (power, numeratorI);
    }

    private static Tensor<T> PerPair(Tensor<T> values, int[] shape)
        => E.TensorBroadcastTo(E.Reshape(values, new[] { shape[0], 1 }), shape);

    #endregion

    #region Harvest (harvest.cpp)

    /// <summary>Number of frames Harvest produces (WORLD GetSamplesForHarvest).</summary>
    public static int GetSamplesForHarvest(int fs, int xLength, double framePeriod)
        => (int)(1000.0 * xLength / fs / framePeriod) + 1;

    /// <summary>
    /// F0 by WORLD's Harvest (Morise 2017; PyWorld's <c>harvest</c>), one value per <paramref name="framePeriodMs"/>
    /// from time 0, between this detector's F0 floor and ceiling. Unlike DIO it searches 40 bands per octave on audio
    /// resampled near 8 kHz, refines each candidate by instantaneous frequency and links the contour across frames; it
    /// is the extractor CosyVoice's HiFT training reads its F0 targets from.
    /// </summary>
    /// <remarks>Follows WORLD's <c>harvest.cpp</c> (as PyWorld 0.3.5 bundles it), keeping its quirks: the 1 ms basic
    /// contour sampled at the requested period, the in-place spectrum convolution, and <c>ExtendSub</c>'s mean F0 carried
    /// from one voiced section to the next.</remarks>
    /// <returns>F0 in Hz per frame (0 where unvoiced) and each frame's time in seconds, both <c>[frames]</c>.</returns>
    public (Tensor<T> F0, Tensor<T> TemporalPositions) Harvest(Tensor<T> x, double framePeriodMs)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (framePeriodMs <= 0) throw new ArgumentOutOfRangeException(nameof(framePeriodMs));
        if (x.Length < 2) throw new ArgumentException("Harvest needs at least two samples.", nameof(x));
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var signal = Flat(x);
        int fs = SampleRate;
        int dimensionRatio = MatlabRound(fs / 8000.0);
        const double channelsInOctave = 40;

        int basicLength = GetSamplesForHarvest(fs, signal.Length, 1);
        var basicPositions = Positions(basicLength, 1.0);
        var basicF0 = HarvestGeneralBody(signal, fs, MinPitch, MaxPitch, channelsInOctave, dimensionRatio, basicPositions);
        if (framePeriodMs == 1.0) return (basicF0, basicPositions);

        int f0Length = GetSamplesForHarvest(fs, signal.Length, framePeriodMs);
        var pick = new int[f0Length];
        for (int i = 0; i < f0Length; i++) pick[i] = Math.Min(basicLength - 1, MatlabRound(i * framePeriodMs / 1000.0 * 1000.0));
        return (E.TensorIndexSelect(basicF0, IntTensor(pick), 0), Positions(f0Length, framePeriodMs));
    }

    private static Tensor<T> HarvestGeneralBody(Tensor<T> x, int fs, double f0Floor, double f0Ceil,
        double channelsInOctave, int speed, Tensor<T> temporalPositions)
    {
        int xLength = x.Length;
        int f0Length = temporalPositions.Length;
        double adjustedF0Floor = f0Floor * 0.9;
        double adjustedF0Ceil = f0Ceil * 1.1;
        int numberOfChannels = 1 + (int)(Math.Log(adjustedF0Ceil / adjustedF0Floor) / Log2 * channelsInOctave);
        var boundaryF0List = new double[numberOfChannels];
        for (int i = 0; i < numberOfChannels; i++)
            boundaryF0List[i] = adjustedF0Floor * Math.Pow(2.0, (i + 1) / channelsInOctave);

        int decimationRatio = Math.Max(Math.Min(speed, 12), 1);
        int yLength = (int)Math.Ceiling((double)xLength / decimationRatio);
        double actualFs = (double)fs / decimationRatio;
        int fftSize = GetSuitableFftSize(yLength + 5 + 2 * (int)(2.0 * actualFs / boundaryF0List[0]));

        var (y, ySpecRe, ySpecIm) = HarvestWaveformAndSpectrum(x, yLength, fftSize, decimationRatio);

        var raw = new Tensor<T>[numberOfChannels];
        for (int i = 0; i < numberOfChannels; i++)
            raw[i] = HarvestF0CandidateFromRawEvent(boundaryF0List[i], actualFs, ySpecRe, ySpecIm, yLength, fftSize,
                f0Floor, f0Ceil, temporalPositions);
        var rawByFrame = E.TensorTranspose(E.TensorStack(raw, 0));                         // [frames, channels]

        var (official, numberOfCandidates) = DetectOfficialF0Candidates(rawByFrame, numberOfChannels, f0Length);
        if (numberOfCandidates == 0) return Zeros(f0Length);
        var f0Candidates = OverlapF0Candidates(official, f0Length, numberOfCandidates);
        int candidates = numberOfCandidates * 7;

        var shape = new[] { f0Length, candidates };
        var (refined, scores) = HarvestRefinedF0(y, actualFs,
            E.Reshape(f0Candidates, new[] { f0Length * candidates }),
            E.Reshape(PerPair(temporalPositions, shape), new[] { f0Length * candidates }), f0Floor, f0Ceil);
        f0Candidates = E.Reshape(refined, shape);
        var f0Scores = E.Reshape(scores, shape);
        (f0Candidates, f0Scores) = RemoveUnreliableCandidates(f0Candidates, f0Scores, f0Length, candidates);

        var bestF0Contour = HarvestFixF0Contour(f0Candidates, f0Scores, f0Length, candidates);
        return SmoothF0Contour(bestF0Contour);
    }

    // GetWaveformAndSpectrum (with its Sub): the signal decimated after edge extension, its mean removed, its spectrum.
    private static (Tensor<T> Y, Tensor<T> Re, Tensor<T> Im) HarvestWaveformAndSpectrum(Tensor<T> x, int yLength,
        int fftSize, int decimationRatio)
    {
        Tensor<T> y;
        if (decimationRatio == 1)
        {
            y = x;
        }
        else
        {
            int lag = (int)(Math.Ceiling(140.0 / decimationRatio) * decimationRatio);
            int xLength = x.Length;
            var extended = Cat(E.TensorTile(Slice(x, 0, 1), new[] { lag }), x, E.TensorTile(Slice(x, xLength - 1, 1), new[] { lag }));
            y = Slice(Decimate(extended, decimationRatio), lag / decimationRatio, yLength);
        }
        y = E.TensorSubtractScalar(y, ScaledMean(y));
        var (re, im) = Fft(Pad(y, 0, fftSize - yLength));
        int bins = fftSize / 2 + 1;
        return (y, Slice(re, 0, bins), Slice(im, 0, bins));
    }

    /// <summary>
    /// Harvest's GetFilteredSignal: the signal through a Nuttall-windowed cosine band-pass at <paramref name="boundaryF0"/>,
    /// convolved as WORLD does (<see cref="ConvolveHalfSpectrum"/>) and delay compensated.
    /// </summary>
    internal static Tensor<T> HarvestFilteredSignal(double boundaryF0, int fftSize, double fs, Tensor<T> ySpecRe,
        Tensor<T> ySpecIm, int yLength)
    {
        int filterLengthHalf = MatlabRound(fs / boundaryF0 * 2.0);
        int length = filterLengthHalf * 2 + 1;
        var carrier = E.TensorCos(E.TensorDivideScalar(
            E.TensorMultiplyScalar(Range(-filterLengthHalf, length), C(2 * Math.PI * boundaryF0)), C(fs)));
        var bandPass = Pad(E.TensorMultiply(NuttallWindow(length), carrier), 0, fftSize - length);
        var (re, im) = Fft(bandPass);
        var (convolvedRe, convolvedIm) = ConvolveHalfSpectrum(ySpecRe, ySpecIm, re, im, fftSize);
        return Slice(InverseRealFft(convolvedRe, convolvedIm, fftSize), filterLengthHalf + 1, yLength);
    }

    // GetF0CandidateFromRawEvent: the four zero-crossing interval series, interpolated to the frames and averaged.
    private static Tensor<T> HarvestF0CandidateFromRawEvent(double boundaryF0, double fs, Tensor<T> ySpecRe,
        Tensor<T> ySpecIm, int yLength, int fftSize, double f0Floor, double f0Ceil, Tensor<T> temporalPositions)
    {
        int f0Length = temporalPositions.Length;
        var filtered = HarvestFilteredSignal(boundaryF0, fftSize, fs, ySpecRe, ySpecIm, yLength);
        var sets = FourIntervalContours(filtered, fs, temporalPositions);
        if (sets is null) return Zeros(f0Length);
        var candidate = E.TensorDivideScalar(E.TensorAdd(E.TensorAdd(E.TensorAdd(sets[0], sets[1]), sets[2]), sets[3]), C(4.0));
        var outOfRange = OutOfRange(candidate, boundaryF0 * 0.9, boundaryF0 * 1.1, f0Floor, f0Ceil);
        return E.TensorWhere(outOfRange, Zeros(f0Length), candidate);
    }

    // DetectOfficialF0Candidates (with Sub1, Sub2): per frame, the mean of each run of at least 10 voiced channels
    // (the first and last channel counted unvoiced), in channel order. Returns [frames, max runs] and that maximum.
    private static (Tensor<T> Candidates, int Count) DetectOfficialF0Candidates(Tensor<T> rawByFrame, int numberOfChannels,
        int f0Length)
    {
        var vuv = E.TensorGreaterThan(rawByFrame, Ops.Zero).Cast<int>();
        vuv = E.PadNd(E.TensorSlice(vuv, new[] { 0, 1 }, new[] { f0Length, numberOfChannels - 2 }), new[] { 1, 1 },
            EnginePadMode.Constant, 0);
        var change = E.TensorSubtract(E.TensorSlice(vuv, new[] { 0, 1 }, new[] { f0Length, numberOfChannels - 1 }),
            E.TensorSlice(vuv, new[] { 0, 0 }, new[] { f0Length, numberOfChannels - 1 }));
        // Row-major (frame, channel) order: within a frame each run's start precedes its end, so the k-th start and the
        // k-th end bound the same run.
        int[] starts = E.TensorNonzero(E.TensorEquals(change, 1)).ToArray();
        int[] ends = E.TensorNonzero(E.TensorEquals(change, -1)).ToArray();

        var frame = new List<int>();
        var first = new List<int>();
        var last = new List<int>();
        var rank = new List<int>();
        int count = 0, currentFrame = -1, maximum = 0;
        for (int s = 0; s < starts.Length / 2; s++)
        {
            int st = starts[2 * s + 1] + 1, ed = ends[2 * s + 1] + 1;
            if (ed - st < 10) continue;
            int f = starts[2 * s];
            if (f != currentFrame)
            {
                currentFrame = f;
                count = 0;
            }
            frame.Add(f);
            first.Add(st);
            last.Add(ed);
            rank.Add(count++);
            maximum = Math.Max(maximum, count);
        }
        if (maximum == 0) return (Zeros(1), 0);

        int runs = frame.Count;
        var shape = new[] { runs, numberOfChannels };
        var rows = E.TensorIndexSelect(rawByFrame, IntTensor(frame.ToArray()), 0);
        var channel = E.TensorBroadcastTo(E.Reshape(IntRange(0, numberOfChannels), new[] { 1, numberOfChannels }), shape);
        var lower = E.TensorBroadcastTo(E.Reshape(IntTensor(first.ToArray()), new[] { runs, 1 }), shape);
        var upper = E.TensorBroadcastTo(E.Reshape(IntTensor(last.ToArray()), new[] { runs, 1 }), shape);
        var zero = new Tensor<int>(shape);
        var inRun = E.TensorWhere(E.TensorGreaterThan(lower, channel), zero,
            E.TensorWhere(E.TensorGreaterThan(upper, channel), Tensor<int>.CreateDefault(shape, 1), zero)).Cast<T>();
        // The sum is pre-scaled by 2^-8 (exact) so that Half does not overflow on 150 channels of Hz.
        var sums = E.ReduceSum(E.TensorWhere(inRun, E.TensorMultiplyScalar(rows, C(1.0 / 256.0)), new Tensor<T>(shape)), new[] { 1 }, false);
        var lengths = E.TensorSubtract(IntTensor(last.ToArray()), IntTensor(first.ToArray())).Cast<T>();
        var means = E.TensorDivide(E.TensorMultiplyScalar(sums, C(256.0)), lengths);
        var candidates = E.TensorIndexPut(new Tensor<T>(new[] { f0Length, maximum }),
            new[] { IntTensor(frame.ToArray()), IntTensor(rank.ToArray()) }, means, false);
        return (candidates, maximum);
    }

    // OverlapF0Candidates: each frame also takes the candidates of the three frames before and after it.
    private static Tensor<T> OverlapF0Candidates(Tensor<T> candidates, int f0Length, int numberOfCandidates)
    {
        const int n = 3;
        var blocks = new Tensor<T>[2 * n + 1];
        blocks[0] = candidates;
        for (int i = 1; i <= n; i++)
        {
            if (f0Length <= i)
            {
                blocks[i] = blocks[i + n] = new Tensor<T>(new[] { f0Length, numberOfCandidates });
                continue;
            }
            var earlier = E.TensorSlice(candidates, new[] { 0, 0 }, new[] { f0Length - i, numberOfCandidates });
            var later = E.TensorSlice(candidates, new[] { i, 0 }, new[] { f0Length - i, numberOfCandidates });
            blocks[i] = E.PadNd(earlier, new[] { 0, 0, i, 0 }, EnginePadMode.Constant, Ops.Zero);
            blocks[i + n] = E.PadNd(later, new[] { 0, 0, 0, i }, EnginePadMode.Constant, Ops.Zero);
        }
        return E.Concat(blocks, 1);
    }

    // GetRefinedF0 (Harvest's) for every candidate: instantaneous-frequency refinement and its reliability score; (0, 0)
    // for a non-positive candidate or a refinement outside the floor/ceiling or scoring under 2.5.
    private static (Tensor<T> F0, Tensor<T> Score) HarvestRefinedF0(Tensor<T> y, double fs, Tensor<T> f0, Tensor<T> positions,
        double f0Floor, double f0Ceil)
    {
        int count = f0.Length;
        int[] validPairs = E.TensorNonzero(E.TensorGreaterThan(f0, Ops.Zero)).ToArray();
        if (validPairs.Length == 0) return (Zeros(count), Zeros(count));
        var validIndex = IntTensor(validPairs);
        var currentF0 = E.TensorIndexSelect(f0, validIndex, 0);
        var currentPositions = E.TensorIndexSelect(positions, validIndex, 0);
        int[] halfWindow = E.TensorFloor(E.TensorAddScalar(Over(1.5 * fs, currentF0), Ops.One)).Cast<int>().ToArray();

        var order = new List<int>();
        var refinedF0 = new List<Tensor<T>>();
        var refinedScore = new List<Tensor<T>>();
        foreach (var block in RefinementBlocks(halfWindow))
        {
            var pairs = IntTensor(block.Pairs);
            var blockF0 = E.TensorIndexSelect(currentF0, pairs, 0);
            var (power, numeratorI) = InstantaneousSpectra(y, fs, E.TensorIndexSelect(currentPositions, pairs, 0),
                block.HalfWindow, block.FftSize, RefinementWindow.Harvest);

            // FixF0 (Harvest's): up to six harmonics below Nyquist; the weighted harmonic mean and the inverse of the
            // mean relative deviation of each harmonic's frequency over its number from the candidate.
            const int maximumHarmonics = 6;
            var harmonicsUsed = E.TensorMin(E.TensorFloor(Over(fs / 2.0, blockF0)), C(maximumHarmonics));
            var (harmonic, instantaneous, amplitude) = Harmonics(power, numeratorI, block.FftSize, fs, blockF0,
                maximumHarmonics);
            var shape = new[] { block.Pairs.Length, maximumHarmonics };
            var used = E.TensorWhere(E.TensorGreaterThan(harmonic, PerPair(harmonicsUsed, shape)), new Tensor<T>(shape),
                Tensor<T>.CreateDefault(shape, Ops.One));
            var zero = new Tensor<T>(shape);
            var perPairF0 = PerPair(blockF0, shape);
            var numerator = E.ReduceSum(E.TensorWhere(used, E.TensorMultiply(amplitude, instantaneous), zero), new[] { 1 }, false);
            var denominator = E.ReduceSum(E.TensorWhere(used, E.TensorMultiply(amplitude, harmonic), zero), new[] { 1 }, false);
            var deviation = E.TensorAbs(E.TensorDivide(E.TensorSubtract(E.TensorDivide(instantaneous, harmonic), perPairF0), perPairF0));
            var score = E.ReduceSum(E.TensorWhere(used, deviation, zero), new[] { 1 }, false);

            var refined = E.TensorDivide(numerator, E.TensorAddScalar(denominator, Guard(SafeGuardMinimum / block.FftSize)));
            var reliability = Over(1.0, E.TensorAddScalar(E.TensorDivide(score, harmonicsUsed), Guard(SafeGuardMinimum)));
            var one = Tensor<T>.CreateDefault(refined._shape, Ops.One);
            var rejected = E.TensorWhere(E.TensorLessThan(refined, C(f0Floor)), one,
                E.TensorWhere(E.TensorGreaterThan(refined, C(f0Ceil)), one, E.TensorLessThan(reliability, C(2.5))));
            var none = Zeros(block.Pairs.Length);
            refinedF0.Add(E.TensorWhere(rejected, none, refined));
            refinedScore.Add(E.TensorWhere(rejected, none, reliability));
            foreach (int pair in block.Pairs) order.Add(validPairs[pair]);
        }
        var index = new[] { IntTensor(order.ToArray()) };
        return (E.TensorIndexPut(Zeros(count), index, Cat(refinedF0.ToArray()), false),
            E.TensorIndexPut(Zeros(count), index, Cat(refinedScore.ToArray()), false));
    }

    // RemoveUnreliableCandidates: a candidate with no candidate within 5 % in the frame before or after (judged on the
    // candidates as they were before any removal) is dropped with its score.
    private static (Tensor<T> Candidates, Tensor<T> Scores) RemoveUnreliableCandidates(Tensor<T> f0Candidates,
        Tensor<T> f0Scores, int f0Length, int numberOfCandidates)
    {
        if (f0Length < 3) return (f0Candidates, f0Scores);
        int c = numberOfCandidates;
        int rowsPerBlock = Math.Max(1, BlockElements / (c * c));
        var candidateParts = new List<Tensor<T>> { E.TensorSlice(f0Candidates, new[] { 0, 0 }, new[] { 1, c }) };
        var scoreParts = new List<Tensor<T>> { E.TensorSlice(f0Scores, new[] { 0, 0 }, new[] { 1, c }) };
        for (int start = 1; start < f0Length - 1; start += rowsPerBlock)
        {
            int rows = Math.Min(rowsPerBlock, f0Length - 1 - start);
            var reference = E.TensorSlice(f0Candidates, new[] { start, 0 }, new[] { rows, c });
            var cube = new[] { rows, c, c };
            var referenceB = E.TensorBroadcastTo(E.Reshape(reference, new[] { rows, c, 1 }), cube);
            Tensor<T> NearestError(int neighbourStart)
            {
                var neighbour = E.TensorSlice(f0Candidates, new[] { neighbourStart, 0 }, new[] { rows, c });
                var neighbourB = E.TensorBroadcastTo(E.Reshape(neighbour, new[] { rows, 1, c }), cube);
                var error = E.TensorDivide(E.TensorAbs(E.TensorSubtract(referenceB, neighbourB)), referenceB);
                return E.TensorMin(E.TensorNegate(E.ReduceMax(E.TensorNegate(error), new[] { 2 }, false)), Ops.One);
            }
            var nearest = E.TensorMin(NearestError(start + 1), NearestError(start - 1));
            var zero = new Tensor<T>(reference._shape);
            var drop = E.TensorWhere(E.TensorEquals(reference, Ops.Zero), zero, E.TensorGreaterThan(nearest, C(0.05)));
            candidateParts.Add(E.TensorWhere(drop, zero, reference));
            scoreParts.Add(E.TensorWhere(drop, zero, E.TensorSlice(f0Scores, new[] { start, 0 }, new[] { rows, c })));
        }
        candidateParts.Add(E.TensorSlice(f0Candidates, new[] { f0Length - 1, 0 }, new[] { 1, c }));
        scoreParts.Add(E.TensorSlice(f0Scores, new[] { f0Length - 1, 0 }, new[] { 1, c }));
        return (E.Concat(candidateParts, 0), E.Concat(scoreParts, 0));
    }

    // FixF0Contour: the best-scoring candidate per frame, then the four fixing steps.
    private static Tensor<T> HarvestFixF0Contour(Tensor<T> f0Candidates, Tensor<T> f0Scores, int f0Length, int numberOfCandidates)
    {
        // SearchF0Base: the first candidate with the highest positive score (WORLD's strict '>' from 0).
        var contour = Zeros(f0Length);
        var best = Zeros(f0Length);
        for (int j = 0; j < numberOfCandidates; j++)
        {
            var candidate = Column(f0Candidates, j);
            var score = Column(f0Scores, j);
            var better = E.TensorGreaterThan(score, best);
            contour = E.TensorWhere(better, candidate, contour);
            best = E.TensorWhere(better, score, best);
        }
        // These parameters are optimized by speech databases.
        var step1 = HarvestFixStep1(contour, 0.008);
        var step2 = HarvestFixStep2(step1, 6);
        var step3 = HarvestFixStep3(step2, f0Candidates, f0Scores, numberOfCandidates, 0.18);
        return HarvestFixStep4(step3, 9);
    }

    // Step 1: a rapid change of the contour is replaced by 0.
    private static Tensor<T> HarvestFixStep1(Tensor<T> f0Base, double allowedRange)
    {
        int n = f0Base.Length;
        if (n < 3) return Zeros(n);
        var current = Slice(f0Base, 2, n - 2);
        var previous = Slice(f0Base, 1, n - 2);
        var reference = E.TensorSubtract(E.TensorMultiplyScalar(previous, C(2.0)), Slice(f0Base, 0, n - 2));
        var range = C(allowedRange);
        var zero = Zeros(n - 2);
        var extrapolationJump = E.TensorGreaterThan(E.TensorAbs(E.TensorDivide(E.TensorSubtract(current, reference), reference)), range);
        var stepJump = E.TensorGreaterThan(E.TensorDivide(E.TensorAbs(E.TensorSubtract(current, previous)), previous), range);
        var kept = E.TensorWhere(extrapolationJump, E.TensorWhere(stepJump, zero, current), current);
        return Pad(E.TensorWhere(E.TensorEquals(current, Ops.Zero), zero, kept), 2, 0);
    }

    // GetBoundaryList: the first and last frame of each voiced section, the ends of the contour counted unvoiced.
    private static int[] HarvestBoundaryList(Tensor<T> f0)
    {
        int n = f0.Length;
        if (n < 3) return new int[0];
        var vuv = PadInt(SliceInt(E.TensorGreaterThan(f0, Ops.Zero).Cast<int>(), 1, n - 2), 1, 1);
        var change = E.TensorSubtract(SliceInt(vuv, 1, n - 1), SliceInt(vuv, 0, n - 1));
        int[] positions = E.TensorNonzero(E.TensorNotEquals(change, 0)).ToArray();
        var boundaries = new int[positions.Length];
        for (int k = 0; k < positions.Length; k++) boundaries[k] = positions[k] + 1 - k % 2;
        return boundaries;
    }

    // 1 inside the given sections (inclusive bounds), as a [length] mask of T.
    private static Tensor<T> SectionMask(int length, IReadOnlyList<(int Start, int End)> sections)
    {
        var marker = new int[length + 1];
        foreach (var (start, end) in sections)
        {
            marker[start]++;
            marker[end + 1]--;
        }
        var running = E.TensorCumSum(new Tensor<int>(marker, new[] { length + 1 }), 0);
        return E.TensorGreaterThan(SliceInt(running, 0, length), 0).Cast<T>();
    }

    // Step 2: voiced sections shorter than voiceRangeMinimum frames are removed.
    private static Tensor<T> HarvestFixStep2(Tensor<T> f0Step1, int voiceRangeMinimum)
    {
        int[] boundaries = HarvestBoundaryList(f0Step1);
        var shortSections = new List<(int, int)>();
        for (int i = 0; i < boundaries.Length / 2; i++)
            if (boundaries[i * 2 + 1] - boundaries[i * 2] < voiceRangeMinimum)
                shortSections.Add((boundaries[i * 2], boundaries[i * 2 + 1]));
        if (shortSections.Count == 0) return f0Step1;
        return E.TensorWhere(SectionMask(f0Step1.Length, shortSections), Zeros(f0Step1.Length), f0Step1);
    }

    // Step 3: voiced sections are extended by the continuity of the contour, kept when long enough for their mean F0, and
    // merged by their candidates' scores.
    private static Tensor<T> HarvestFixStep3(Tensor<T> f0Step2, Tensor<T> f0Candidates, Tensor<T> f0Scores,
        int numberOfCandidates, double allowedRange)
    {
        int f0Length = f0Step2.Length;
        int[] boundaryList = HarvestBoundaryList(f0Step2);
        int sections = boundaryList.Length / 2;
        // GetMultiChannelF0: each voiced section in a channel of its own.
        var multiChannelF0 = new Tensor<T>[sections];
        for (int i = 0; i < sections; i++)
        {
            int st = boundaryList[i * 2], ed = boundaryList[i * 2 + 1];
            multiChannelF0[i] = Pad(Slice(f0Step2, st, ed - st + 1), st, f0Length - ed - 1);
        }

        int numberOfChannels = HarvestExtend(multiChannelF0, sections, f0Length, boundaryList, f0Candidates, allowedRange);
        if (numberOfChannels == 0) return f0Step2;
        return HarvestMergeF0(multiChannelF0, boundaryList, numberOfChannels, f0Candidates, f0Scores);
    }

    // Extend, with ExtendSub: every section extended both ways, then only those long enough for their mean F0 kept (moved
    // to the front). WORLD passes the same channel and boundary arrays as input and output, and that aliasing is kept.
    private static int HarvestExtend(Tensor<T>[] multiChannelF0, int numberOfSections, int f0Length, int[] boundaryList,
        Tensor<T> f0Candidates, double allowedRange)
    {
        const int threshold = 100;
        for (int i = 0; i < numberOfSections; i++)
        {
            boundaryList[i * 2 + 1] = HarvestExtendF0(boundaryList[i * 2 + 1],
                Math.Min(f0Length - 2, boundaryList[i * 2 + 1] + threshold), 1, f0Candidates, allowedRange, multiChannelF0[i]);
            boundaryList[i * 2] = HarvestExtendF0(boundaryList[i * 2], Math.Max(1, boundaryList[i * 2] - threshold), -1,
                f0Candidates, allowedRange, multiChannelF0[i]);
        }

        int count = 0;
        // The reference never resets the mean between sections.
        T meanF0 = Ops.Zero;
        for (int i = 0; i < numberOfSections; i++)
        {
            if (!LongEnoughForMean(multiChannelF0[i], boundaryList[i * 2], boundaryList[i * 2 + 1], ref meanF0)) continue;
            (multiChannelF0[count], multiChannelF0[i]) = (multiChannelF0[i], multiChannelF0[count]);
            (boundaryList[count * 2], boundaryList[i * 2]) = (boundaryList[i * 2], boundaryList[count * 2]);
            (boundaryList[count * 2 + 1], boundaryList[i * 2 + 1]) = (boundaryList[i * 2 + 1], boundaryList[count * 2 + 1]);
            count++;
        }
        return count;
    }

    // ExtendSub's test for one section: the carried mean F0 accumulates the section's values over [st, ed) and is divided
    // by its length; the section is kept when 2200 / mean is below that length.
    private static bool LongEnoughForMean(Tensor<T> channel, int st, int ed, ref T meanF0)
    {
        const double meanThreshold = 2200.0;
        T sum = ed > st ? E.TensorSum(Slice(channel, st, ed - st)) : Ops.Zero;
        meanF0 = Ops.Divide(Ops.Add(meanF0, sum), C(ed - st));
        return Ops.LessThan(Ops.Divide(C(meanThreshold), meanF0), C(ed - st));
    }

    // ExtendF0: extends a section from origin by following the nearest candidates until four frames in a row find none.
    // Each frame's choice depends on the last voiced value, so this is a scan over frames.
    private static int HarvestExtendF0(int origin, int lastPoint, int shift, Tensor<T> f0Candidates, double allowedRange,
        Tensor<T> extendedF0)
    {
        const int threshold = 4;
        T referenceF0 = extendedF0[origin];
        int shiftedOrigin = origin;
        int distance = Math.Abs(lastPoint - origin);
        int misses = 0;
        for (int i = 0; i <= distance; i++)
        {
            int target = origin + shift * i + shift;
            if (HarvestExtendStep(extendedF0, f0Candidates, target, ref referenceF0, allowedRange))
            {
                misses = 0;
                shiftedOrigin = target;
            }
            else
            {
                misses++;
            }
            if (misses == threshold) break;
        }
        return shiftedOrigin;
    }

    // One ExtendF0 step: the target frame takes the candidate nearest the reference (SelectBestF0); a voiced pick becomes
    // the new reference.
    private static bool HarvestExtendStep(Tensor<T> extendedF0, Tensor<T> f0Candidates, int target, ref T referenceF0,
        double allowedRange)
    {
        T best = HarvestSelectBestF0(referenceF0, Row(f0Candidates, target), allowedRange);
        extendedF0[target] = best;
        if (Ops.Equals(best, Ops.Zero)) return false;
        referenceF0 = best;
        return true;
    }

    // SelectBestF0 (Harvest's): the candidate with the smallest relative error from the reference, within the allowed
    // range; WORLD's '>' scan keeps the last of equal errors. 0 when none is within range.
    private static T HarvestSelectBestF0(T referenceF0, Tensor<T> candidates, double allowedRange)
    {
        var error = E.TensorDivideScalar(E.TensorAbs(E.TensorSubtractScalar(candidates, referenceF0)), referenceF0);
        T minimum = E.TensorMinValue(error);
        if (!Ops.LessThanOrEquals(minimum, C(allowedRange))) return Ops.Zero;
        int[] at = E.TensorNonzero(E.TensorEquals(error, minimum)).ToArray();
        return at.Length == 0 ? Ops.Zero : candidates[at[at.Length - 1]];
    }

    // MergeF0 with MakeSortedOrder, SearchScore and MergeF0Sub: overlapping channels merged by their candidates' scores.
    private static Tensor<T> HarvestMergeF0(Tensor<T>[] multiChannelF0, int[] boundaryList, int numberOfChannels,
        Tensor<T> f0Candidates, Tensor<T> f0Scores)
    {
        // MakeSortedOrder, exactly as WORLD swaps.
        var order = new int[numberOfChannels];
        for (int i = 0; i < numberOfChannels; i++) order[i] = i;
        for (int i = 1; i < numberOfChannels; i++)
            for (int j = i - 1; j >= 0; j--)
            {
                if (boundaryList[order[j] * 2] > boundaryList[order[i] * 2]) (order[i], order[j]) = (order[j], order[i]);
                else break;
            }

        var mergedF0 = multiChannelF0[0].Clone();
        for (int i = 1; i < numberOfChannels; i++)
        {
            int channel = order[i];
            int st = boundaryList[channel * 2], ed = boundaryList[channel * 2 + 1];
            if (st - boundaryList[1] > 0)
            {
                mergedF0 = CopyRange(mergedF0, multiChannelF0[channel], st, ed);
                boundaryList[0] = st;
                boundaryList[1] = ed;
            }
            else
            {
                (mergedF0, boundaryList[1]) = HarvestMergeF0Sub(mergedF0, boundaryList[0], boundaryList[1],
                    multiChannelF0[channel], st, ed, f0Candidates, f0Scores);
            }
        }
        return mergedF0;
    }

    private static (Tensor<T> Merged, int End) HarvestMergeF0Sub(Tensor<T> f01, int st1, int ed1, Tensor<T> f02, int st2,
        int ed2, Tensor<T> f0Candidates, Tensor<T> f0Scores)
    {
        if (st1 <= st2 && ed1 >= ed2) return (f01, ed1);
        bool firstScoresHigher = ed1 >= st2
            && Ops.GreaterThan(SearchScoreSum(f01, st2, ed1, f0Candidates, f0Scores), SearchScoreSum(f02, st2, ed1, f0Candidates, f0Scores));
        return (CopyRange(f01, f02, firstScoresHigher ? ed1 : st2, ed2), ed2);
    }

    // Σ over frames st … ed of SearchScore: the highest score among the frame's candidates equal to the contour, or 0.
    private static T SearchScoreSum(Tensor<T> f0, int st, int ed, Tensor<T> f0Candidates, Tensor<T> f0Scores)
    {
        int rows = ed - st + 1, c = f0Candidates.Shape[1];
        var shape = new[] { rows, c };
        var candidates = E.TensorSlice(f0Candidates, new[] { st, 0 }, shape);
        var scores = E.TensorSlice(f0Scores, new[] { st, 0 }, shape);
        var contour = PerPair(Slice(f0, st, rows), shape);
        var matching = E.TensorWhere(E.TensorEquals(candidates, contour), scores, new Tensor<T>(shape));
        return E.TensorSum(E.TensorMax(E.ReduceMax(matching, new[] { 1 }, false), Ops.Zero));
    }

    // destination with frames from … to (inclusive) taken from source; unchanged when the range is empty.
    private static Tensor<T> CopyRange(Tensor<T> destination, Tensor<T> source, int from, int to)
        => to < from ? destination : E.TensorSetSlice(destination, Slice(source, from, to - from + 1), new[] { from });

    // Step 4: F0 in a short unvoiced gap is interpolated across it, from one past the last voiced value to one below the
    // next (WORLD's ±1).
    private static Tensor<T> HarvestFixStep4(Tensor<T> f0Step3, int threshold)
    {
        int[] boundaryList = HarvestBoundaryList(f0Step3);
        var gapFrames = new List<int>();
        var gapIds = new List<int>();
        var steps = new List<int>();
        var ends = new List<int>();
        var starts = new List<int>();
        var spans = new List<int>();
        for (int i = 0; i < boundaryList.Length / 2 - 1; i++)
        {
            int distance = boundaryList[(i + 1) * 2] - boundaryList[i * 2 + 1] - 1;
            if (distance >= threshold) continue;
            int gap = ends.Count;
            ends.Add(boundaryList[i * 2 + 1]);
            starts.Add(boundaryList[(i + 1) * 2]);
            spans.Add(distance + 1);
            int count = 1;
            for (int j = boundaryList[i * 2 + 1] + 1; j <= boundaryList[(i + 1) * 2] - 1; j++)
            {
                gapFrames.Add(j);
                gapIds.Add(gap);
                steps.Add(count++);
            }
        }
        if (gapFrames.Count == 0) return f0Step3;
        var tmp0 = E.TensorAddScalar(E.TensorIndexSelect(f0Step3, IntTensor(ends.ToArray()), 0), Ops.One);
        var tmp1 = E.TensorSubtractScalar(E.TensorIndexSelect(f0Step3, IntTensor(starts.ToArray()), 0), Ops.One);
        var coefficient = E.TensorDivide(E.TensorSubtract(tmp1, tmp0), IntTensor(spans.ToArray()).Cast<T>());
        var ids = IntTensor(gapIds.ToArray());
        var values = E.TensorAdd(E.TensorIndexSelect(tmp0, ids, 0),
            E.TensorMultiply(E.TensorIndexSelect(coefficient, ids, 0), IntTensor(steps.ToArray()).Cast<T>()));
        return E.TensorIndexPut(f0Step3.Clone(), new[] { IntTensor(gapFrames.ToArray()) }, values, false);
    }

    // SmoothF0Contour: each voiced section, held constant outside its bounds over the contour padded by 300 frames of
    // zeros, through the zero-lag (forward-backward) second-order Butterworth filter; frames outside every section are 0.
    private static Tensor<T> SmoothF0Contour(Tensor<T> f0)
    {
        double[] b = { 0.0078202080334971724, 0.015640416066994345 };
        double[] a = { 1.7347257688092754, -0.76600660094326412 };
        const int lag = 300;
        int f0Length = f0.Length;
        int length = f0Length + lag * 2;
        var f0Contour = Pad(f0, lag, lag);
        int[] boundaryList = HarvestBoundaryList(f0Contour);
        int sections = boundaryList.Length / 2;
        if (sections == 0) return Zeros(f0Length);

        var frames = new List<int>();
        var values = new List<Tensor<T>>();
        int perBlock = Math.Max(1, BlockElements / length);
        for (int first = 0; first < sections; first += perBlock)
        {
            int count = Math.Min(perBlock, sections - first);
            var index = new int[count * length];
            var picks = new List<int>();
            for (int s = 0; s < count; s++)
            {
                int st = boundaryList[(first + s) * 2], ed = boundaryList[(first + s) * 2 + 1];
                for (int j = 0; j < length; j++) index[s * length + j] = Math.Min(Math.Max(j, st), ed);
                for (int j = st; j <= ed; j++)
                {
                    picks.Add(s * length + j);
                    frames.Add(j - lag);
                }
            }
            var rows = E.Reshape(E.TensorTake(f0Contour, new Tensor<int>(index, new[] { count * length })), new[] { count, length });
            var filtered = ZeroPhase(rows, new[] { b[0], b[1], b[0] }, new[] { -a[0], -a[1] });
            values.Add(E.TensorTake(E.Reshape(filtered, new[] { count * length }), IntTensor(picks.ToArray())));
        }
        return E.TensorIndexPut(Zeros(f0Length), new[] { IntTensor(frames.ToArray()) }, Cat(values.ToArray()), false);
    }

    #endregion

    #region WORLD helpers (common.cpp, matlabfunctions.cpp)

    private static int MatlabRound(double x) => x > 0 ? (int)(x + 0.5) : (int)(x - 0.5);

    // matlab_round on every element: floor(x + 0.5) above 0, −floor(0.5 − x) otherwise (C's truncation of x ∓ 0.5).
    private static Tensor<T> MatlabRound(Tensor<T> x)
        => E.TensorWhere(E.TensorGreaterThan(x, Ops.Zero),
            E.TensorFloor(E.TensorAddScalar(x, C(0.5))),
            E.TensorNegate(E.TensorFloor(E.TensorAddScalar(E.TensorNegate(x), C(0.5)))));

    private static int GetSuitableFftSize(int sample)
        => (int)Math.Pow(2.0, (int)(Math.Log(sample) / Log2) + 1.0);

    private static Tensor<T> NuttallWindow(int length)
    {
        var t = E.TensorDivideScalar(Range(0, length), C(length - 1.0));
        Tensor<T> Cosine(double multiple) => E.TensorCos(E.TensorMultiplyScalar(t, C(multiple * Math.PI)));
        var window = E.TensorAddScalar(E.TensorMultiplyScalar(Cosine(2.0), C(-0.487396)), C(0.355768));
        window = E.TensorAdd(window, E.TensorMultiplyScalar(Cosine(4.0), C(0.144232)));
        return E.TensorSubtract(window, E.TensorMultiplyScalar(Cosine(6.0), C(0.012604)));
    }

    // interp1 with WORLD's histc: each query takes the segment k = clamp(#{x ≤ xi}, 1, n − 1) and interpolates linearly.
    private static Tensor<T> Interp1(Tensor<T> x, Tensor<T> y, int xLength, Tensor<T> xi)
    {
        var k = E.TensorClamp(E.TensorSearchSorted(x, xi, true), 1, xLength - 1);
        var previous = E.TensorAddScalar(k, -1);
        var x0 = E.TensorIndexSelect(x, previous, 0);
        var y0 = E.TensorIndexSelect(y, previous, 0);
        var s = E.TensorDivide(E.TensorSubtract(xi, x0), E.TensorSubtract(E.TensorIndexSelect(x, k, 0), x0));
        return E.TensorAdd(y0, E.TensorMultiply(s, E.TensorSubtract(E.TensorIndexSelect(y, k, 0), y0)));
    }

    // decimate: 9 samples of odd reflection each side, the order-3 low-pass forward and backward, every r-th sample from
    // nbeg (all the samples WORLD writes, which can run past (n − 1)/r + 1).
    private static Tensor<T> Decimate(Tensor<T> x, int r)
    {
        const int nFact = 9;
        int xLength = x.Length;
        var front = E.TensorSubtract(E.TensorTile(E.TensorMultiplyScalar(Slice(x, 0, 1), C(2.0)), new[] { nFact }),
            E.TensorFlip(Slice(x, 1, nFact), new[] { 0 }));
        var back = E.TensorSubtract(E.TensorTile(E.TensorMultiplyScalar(Slice(x, xLength - 1, 1), C(2.0)), new[] { nFact }),
            E.TensorFlip(Slice(x, xLength - 1 - nFact, nFact), new[] { 0 }));
        var padded = E.Reshape(Cat(front, x, back), new[] { 1, xLength + 2 * nFact });

        var (numerator, denominator) = DecimationFilter(r);
        var filtered = E.Reshape(ZeroPhase(padded, numerator, denominator), new[] { xLength + 2 * nFact });

        int nout = (xLength - 1) / r + 1;
        int nbeg = r - r * nout + xLength;
        var picks = new List<int>();
        for (int i = nbeg; i < xLength + nFact; i += r) picks.Add(i + nFact - 1);
        return E.TensorIndexSelect(filtered, IntTensor(picks.ToArray()), 0);
    }

    // FilterForDecimate's order-3 IIR as (b0, b1, b1, b0) over 1 − a0 z⁻¹ − a1 z⁻² − a2 z⁻³.
    private static (double[] Numerator, double[] DenominatorTail) DecimationFilter(int r)
    {
        double a0, a1, a2, b0, b1;
        switch (r)
        {
            case 11: a0 = 2.450743295230728; a1 = -2.06794904601978; a2 = 0.59574774438332101; b0 = 0.0026822508007163792; b1 = 0.0080467524021491377; break;
            case 12: a0 = 2.4981398605924205; a1 = -2.1368928194784025; a2 = 0.62187513816221485; b0 = 0.0021097275904709001; b1 = 0.0063291827714127002; break;
            case 10: a0 = 2.3936475118069387; a1 = -1.9873904075111861; a2 = 0.5658879979027055; b0 = 0.0034818622251927556; b1 = 0.010445586675578267; break;
            case 9: a0 = 2.3236003491759578; a1 = -1.8921545617463598; a2 = 0.53148928133729068; b0 = 0.0046331164041389372; b1 = 0.013899349212416812; break;
            case 8: a0 = 2.2357462340187593; a1 = -1.7780899984041358; a2 = 0.49152555365968692; b0 = 0.0063522763407111993; b1 = 0.019056829022133598; break;
            case 7: a0 = 2.1225239019534703; a1 = -1.6395144861046302; a2 = 0.44469707800587366; b0 = 0.0090366882681608418; b1 = 0.027110064804482525; break;
            case 6: a0 = 1.9715352749512141; a1 = -1.4686795689225347; a2 = 0.3893908434965701; b0 = 0.013469181309343825; b1 = 0.040407543928031475; break;
            case 5: a0 = 1.7610939654280557; a1 = -1.2554914843859768; a2 = 0.3237186507788215; b0 = 0.021334858522387423; b1 = 0.06400457556716227; break;
            case 4: a0 = 1.4499664446880227; a1 = -0.98943497080950582; a2 = 0.24578252340690215; b0 = 0.036710750339322612; b1 = 0.11013225101796784; break;
            case 3: a0 = 0.95039378983237421; a1 = -0.67429146741526791; a2 = 0.15412211621346475; b0 = 0.071221945171178636; b1 = 0.21366583551353591; break;
            case 2: a0 = 0.041156734567757189; a1 = -0.42599112459189636; a2 = 0.041037215479961225; b0 = 0.16797464681802227; b1 = 0.50392394045406674; break;
            default: a0 = 0; a1 = 0; a2 = 0; b0 = 0; b1 = 0; break;
        }
        return (new[] { b0, b1, b1, b0 }, new[] { -a0, -a1, -a2 });
    }

    // The filter forward, the result reversed, forward again and reversed back (WORLD's FilteringF0 and decimate), over
    // each row of x [rows, n].
    private static Tensor<T> ZeroPhase(Tensor<T> x, double[] numerator, double[] denominatorTail)
    {
        var once = E.TensorFlip(LinearFilter(x, numerator, denominatorTail), new[] { 1 });
        return E.TensorFlip(LinearFilter(once, numerator, denominatorTail), new[] { 1 });
    }

    /// <summary>
    /// A causal IIR with zero initial state over each row of <paramref name="x"/> [rows, n]: the zeros Σ b_k x[t − k], then
    /// the poles of 1 + d1 z⁻¹ + … through the engine's diagonal-pole scan, a real pole as one state and a conjugate pair as
    /// two. The poles are the roots of the fixed coefficient polynomial, computed once.
    /// </summary>
    private static Tensor<T> LinearFilter(Tensor<T> x, double[] numerator, double[] denominatorTail)
    {
        int rows = x.Shape[0], n = x.Shape[1];
        Tensor<T> zeros = E.TensorMultiplyScalar(x, C(numerator[0]));
        for (int k = 1; k < numerator.Length && k < n; k++)
        {
            if (numerator[k] == 0.0) continue;
            var delayed = E.PadNd(E.TensorSlice(x, new[] { 0, 0 }, new[] { rows, n - k }), new[] { k, 0 }, EnginePadMode.Constant, Ops.Zero);
            zeros = E.TensorAdd(zeros, E.TensorMultiplyScalar(delayed, C(numerator[k])));
        }

        double c1, c0;
        if (denominatorTail.Length == 3)
        {
            // z³ + d1 z² + d2 z + d3: its real root by bisection (the poles lie inside the unit circle, so the cubic is
            // negative at −1.5 and positive at 1.5), then the quadratic factor z² + c1 z + c0.
            double d1 = denominatorTail[0], d2 = denominatorTail[1], d3 = denominatorTail[2];
            double lo = -1.5, hi = 1.5;
            for (int iteration = 0; iteration < 200; iteration++)
            {
                double mid = 0.5 * (lo + hi);
                if (((mid + d1) * mid + d2) * mid + d3 < 0) lo = mid; else hi = mid;
            }
            double realPole = 0.5 * (lo + hi);
            zeros = PoleScan(zeros, new[] { realPole }, new[] { 0.0 }, new[] { 1.0 }, new[] { 0.0 });
            c1 = d1 + realPole;
            c0 = d2 + realPole * c1;
        }
        else
        {
            c1 = denominatorTail[0];
            c0 = denominatorTail[1];
        }

        double discriminant = c1 * c1 - 4 * c0;
        if (discriminant < -1e-9 * Math.Max(1.0, c1 * c1))
        {
            double re = -c1 / 2, im = Math.Sqrt(-discriminant) / 2;
            // 1 / ((1 − p z⁻¹)(1 − p̄ z⁻¹)) = r / (1 − p z⁻¹) + r̄ / (1 − p̄ z⁻¹), r = p / (p − p̄) = ½ − i·re / (2 im).
            return PoleScan(zeros, new[] { re, re }, new[] { im, -im }, new[] { 0.5, 0.5 }, new[] { -re / (2 * im), re / (2 * im) });
        }
        double root = Math.Sqrt(Math.Max(discriminant, 0.0)) / 2;
        var first = PoleScan(zeros, new[] { -c1 / 2 + root }, new[] { 0.0 }, new[] { 1.0 }, new[] { 0.0 });
        return PoleScan(first, new[] { -c1 / 2 - root }, new[] { 0.0 }, new[] { 1.0 }, new[] { 0.0 });
    }

    // y[t] = Σ_s Re(C_s h_s[t]), h_s[t] = A_s h_s[t − 1] + u[t], over each row of u [rows, n].
    private static Tensor<T> PoleScan(Tensor<T> u, double[] transitionRe, double[] transitionIm, double[] outputRe,
        double[] outputIm)
    {
        int rows = u.Shape[0], n = u.Shape[1], states = transitionRe.Length;
        var ones = new double[states];
        for (int s = 0; s < states; s++) ones[s] = 1.0;
        var output = E.ComplexDiagonalSsmScanForward(
            E.Reshape(u, new[] { rows, n, 1, 1 }),
            Constant(transitionRe, new[] { 1, states }),
            Constant(transitionIm, new[] { 1, states }),
            Constant(ones, new[] { 1, states, 1 }),
            Constant(new double[states], new[] { 1, states, 1 }),
            Constant(outputRe, new[] { 1, 1, states }),
            Constant(outputIm, new[] { 1, 1, states }),
            new Tensor<T>(new[] { 1, 1 }));
        return E.Reshape(output, new[] { rows, n });
    }

    // A fixed table of filter constants, converted to T once.
    private static Tensor<T> Constant(double[] values, int[] shape)
        => new Tensor<double>(shape, new Vector<double>((double[])values.Clone())).Cast<T>();

    #endregion

    #region Tensor helpers

    private static T C(double value) => Ops.FromDouble(value);

    // epsilon as a T, raised to Half's smallest normal value where Half would round it to 0.
    private static T Guard(double epsilon)
        => Ops.FromDouble(typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);

    private static Tensor<T> Flat(Tensor<T> x) => E.Reshape(x, new[] { x.Length });

    private static Tensor<T> Zeros(int length) => new Tensor<T>(new[] { length });

    private static Tensor<T> Range(int start, int count) => E.TensorArange<T>(start, start + count, 1);

    // i · period / 1000 for i = 0 … count − 1: each frame's time in seconds.
    private static Tensor<T> Positions(int count, double framePeriodMs)
        => E.TensorDivideScalar(E.TensorMultiplyScalar(Range(0, count), C(framePeriodMs)), C(1000.0));

    private static Tensor<int> IntTensor(int[] values) => new Tensor<int>(values, new[] { values.Length });

    private static Tensor<int> IntRange(int start, int count)
    {
        var values = new int[count];
        for (int i = 0; i < count; i++) values[i] = start + i;
        return IntTensor(values);
    }

    private static Tensor<T> Slice(Tensor<T> x, int start, int length)
        => length <= 0 ? Zeros(0) : E.TensorSlice(x, new[] { start }, new[] { length });

    private static Tensor<int> SliceInt(Tensor<int> x, int start, int length) => E.TensorSlice(x, new[] { start }, new[] { length });

    private static Tensor<T> Pad(Tensor<T> x, int before, int after)
        => before == 0 && after == 0 ? x : E.PadNd(x, new[] { before, after }, EnginePadMode.Constant, Ops.Zero);

    private static Tensor<int> PadInt(Tensor<int> x, int before, int after)
        => E.PadNd(x, new[] { before, after }, EnginePadMode.Constant, 0);

    private static Tensor<T> Cat(params Tensor<T>[] parts)
    {
        var nonEmpty = new List<Tensor<T>>();
        foreach (var part in parts) if (part.Length > 0) nonEmpty.Add(part);
        return nonEmpty.Count == 1 ? nonEmpty[0] : E.Concat(nonEmpty, 0);
    }

    private static Tensor<T> Row(Tensor<T> matrix, int row)
    {
        int columns = matrix.Shape[1];
        return E.Reshape(E.TensorSlice(matrix, new[] { row, 0 }, new[] { 1, columns }), new[] { columns });
    }

    private static Tensor<T> Column(Tensor<T> matrix, int column)
    {
        int rows = matrix.Shape[0];
        return E.Reshape(E.TensorSlice(matrix, new[] { 0, column }, new[] { rows, 1 }), new[] { rows });
    }

    // value / t elementwise.
    private static Tensor<T> Over(double value, Tensor<T> t) => E.TensorDivide(Tensor<T>.CreateDefault(t._shape, C(value)), t);

    // The first index at which values equals target (0 when none does, as WORLD's scan then keeps its first entry).
    private static int FirstIndexOf(Tensor<T> values, T target)
    {
        int[] at = E.TensorNonzero(E.TensorEquals(values, target)).ToArray();
        return at.Length == 0 ? 0 : at[0];
    }

    // The mean, summed with a power-of-two pre-scale (exact) so that Half does not overflow on long signals.
    private static T ScaledMean(Tensor<T> y)
    {
        double scale = Math.Pow(2.0, Math.Ceiling(Math.Log(Math.Max(1, y.Length), 2.0)));
        return Ops.Multiply(E.TensorMean(E.TensorMultiplyScalar(y, C(1.0 / scale))), C(scale));
    }

    private static (Tensor<T> Re, Tensor<T> Im) ComplexMultiply(Tensor<T> aRe, Tensor<T> aIm, Tensor<T> bRe, Tensor<T> bIm)
        => (E.TensorSubtract(E.TensorMultiply(aRe, bRe), E.TensorMultiply(aIm, bIm)),
            E.TensorAdd(E.TensorMultiply(aRe, bIm), E.TensorMultiply(aIm, bRe)));

    /// <summary>Unnormalized forward DFT of a real signal (FFTW / WORLD r2c convention) along the last axis; all N bins.</summary>
    private static (Tensor<T> Re, Tensor<T> Im) Fft(Tensor<T> signal)
    {
        E.FFT(signal, new Tensor<T>(signal._shape), out var re, out var im);
        return (re, im);
    }

    /// <summary>
    /// Unnormalized inverse of a real signal's spectrum (FFTW / WORLD c2r convention): only bins 0..N/2 are read,
    /// the imaginary parts of the DC and Nyquist bins are ignored, and the result is not divided by N.
    /// </summary>
    private static Tensor<T> InverseRealFft(Tensor<T> re, Tensor<T> im, int n)
    {
        int half = n / 2;
        var innerRe = Slice(re, 1, half - 1);
        var innerIm = Slice(im, 1, half - 1);
        var fullRe = Cat(Slice(re, 0, half + 1), E.TensorFlip(innerRe, new[] { 0 }));
        var fullIm = Cat(Zeros(1), innerIm, Zeros(1), E.TensorNegate(E.TensorFlip(innerIm, new[] { 0 })));
        E.IFFT(fullRe, fullIm, out var outRe, out _);
        return E.TensorMultiplyScalar(outRe, C(n));
    }

    #endregion
}
