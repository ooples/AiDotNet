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

    /// <summary>
    /// Estimates the F0 contour of <paramref name="audio"/> with one value per <paramref name="framePeriodMs"/>,
    /// starting at time 0 (PyWorld's <c>dio</c>, then <c>stonemask</c> when enabled).
    /// </summary>
    /// <returns>F0 in Hz per frame (0 where unvoiced) and each frame's time in seconds.</returns>
    public (double[] F0, double[] TemporalPositions) EstimateF0(Tensor<T> audio, double framePeriodMs)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var span = audio.AsSpan();
        var x = new double[span.Length];
        for (int i = 0; i < x.Length; i++) x[i] = NumOps.ToDouble(span[i]);
        return EstimateF0(x, framePeriodMs);
    }

    /// <inheritdoc cref="EstimateF0(Tensor{T}, double)"/>
    public (double[] F0, double[] TemporalPositions) EstimateF0(double[] x, double framePeriodMs)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (framePeriodMs <= 0) throw new ArgumentOutOfRangeException(nameof(framePeriodMs));
        if (x.Length < 2) throw new ArgumentException("DIO needs at least two samples.", nameof(x));

        int f0Length = GetSamplesForDio(SampleRate, x.Length, framePeriodMs);
        var temporalPositions = new double[f0Length];
        var f0 = new double[f0Length];
        DioGeneralBody(x, SampleRate, framePeriodMs, MinPitch, MaxPitch, _channelsInOctave, _speed,
            _allowedRange, temporalPositions, f0);
        if (!_refineWithStoneMask) return (f0, temporalPositions);

        var refined = new double[f0Length];
        for (int i = 0; i < f0Length; i++)
            refined[i] = GetRefinedF0(x, SampleRate, temporalPositions[i], f0[i]);
        return (refined, temporalPositions);
    }

    /// <inheritdoc/>
    public override IReadOnlyList<PitchFrame<T>> ExtractDetailedPitchContour(Tensor<T> audio, int hopSizeMs = 10)
    {
        var (f0, times) = EstimateF0(audio, hopSizeMs);
        var frames = new PitchFrame<T>[f0.Length];
        for (int i = 0; i < f0.Length; i++)
        {
            bool voiced = f0[i] > 0;
            frames[i] = new PitchFrame<T>
            {
                Time = times[i],
                Pitch = NumOps.FromDouble(f0[i]),
                Confidence = NumOps.FromDouble(voiced ? 1.0 : 0.0),
                IsVoiced = voiced
            };
        }
        return frames;
    }

    /// <inheritdoc/>
    /// <remarks>DIO is an utterance-level method; a single frame is analysed as a short utterance and the
    /// median of its voiced estimates is returned.</remarks>
    protected override (double Pitch, double Confidence)? DetectPitchInternal(double[] frame)
    {
        double framePeriodMs = 1000.0 * frame.Length / SampleRate / 4.0;
        var (f0, _) = EstimateF0(frame, Math.Max(1.0, framePeriodMs));
        var voiced = f0.Where(v => v > 0).OrderBy(v => v).ToArray();
        if (voiced.Length == 0) return null;
        return (voiced[voiced.Length / 2], (double)voiced.Length / f0.Length);
    }

    #region DIO (dio.cpp)

    /// <summary>Number of frames DIO produces (WORLD GetSamplesForDIO).</summary>
    public static int GetSamplesForDio(int fs, int xLength, double framePeriod)
        => (int)(1000.0 * xLength / fs / framePeriod) + 1;

    private void DioGeneralBody(double[] x, int fs, double framePeriod, double f0Floor, double f0Ceil,
        double channelsInOctave, int speed, double allowedRange, double[] temporalPositions, double[] f0)
    {
        int xLength = x.Length;
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

        int f0Length = f0.Length;
        var f0Candidates = new double[numberOfBands][];
        var f0Scores = new double[numberOfBands][];
        for (int i = 0; i < f0Length; i++) temporalPositions[i] = i * framePeriod / 1000.0;

        for (int i = 0; i < numberOfBands; i++)
        {
            var candidate = new double[f0Length];
            var score = new double[f0Length];
            GetF0CandidateFromRawEvent(boundaryF0List[i], actualFs, ySpecRe, ySpecIm, yLength, fftSize, f0Floor,
                f0Ceil, temporalPositions, f0Length, score, candidate);
            f0Candidates[i] = new double[f0Length];
            f0Scores[i] = new double[f0Length];
            for (int j = 0; j < f0Length; j++)
            {
                f0Scores[i][j] = score[j] / (candidate[j] + SafeGuardMinimum);
                f0Candidates[i][j] = candidate[j];
            }
        }

        var bestF0Contour = new double[f0Length];
        for (int i = 0; i < f0Length; i++)
        {
            double tmp = f0Scores[0][i];
            bestF0Contour[i] = f0Candidates[0][i];
            for (int j = 1; j < numberOfBands; j++)
            {
                if (tmp > f0Scores[j][i])
                {
                    tmp = f0Scores[j][i];
                    bestF0Contour[i] = f0Candidates[j][i];
                }
            }
        }

        FixF0Contour(framePeriod, numberOfBands, f0Candidates, bestF0Contour, f0Length, f0Floor, allowedRange, f0);
    }

    private (double[] Re, double[] Im) GetSpectrumForEstimation(double[] x, int yLength, double actualFs,
        int fftSize, int decimationRatio)
    {
        int xLength = x.Length;
        var y = new double[fftSize];
        if (decimationRatio != 1)
            Decimate(x, xLength, decimationRatio, y);
        else
            for (int i = 0; i < xLength; i++) y[i] = x[i];

        double meanY = 0.0;
        for (int i = 0; i < yLength; i++) meanY += y[i];
        meanY /= yLength;
        for (int i = 0; i < yLength; i++) y[i] -= meanY;
        for (int i = yLength; i < fftSize; i++) y[i] = 0.0;

        var (specRe, specIm) = ForwardFft(y);

        int cutoffInSample = MatlabRound(actualFs / CutOff);
        DesignLowCutFilter(cutoffInSample * 2 + 1, fftSize, y);
        var (filtRe, filtIm) = ForwardFft(y);

        for (int i = 0; i <= fftSize / 2; i++)
        {
            double tmp = specRe[i] * filtRe[i] - specIm[i] * filtIm[i];
            specIm[i] = specRe[i] * filtIm[i] + specIm[i] * filtRe[i];
            specRe[i] = tmp;
        }
        return (specRe, specIm);
    }

    private static void DesignLowCutFilter(int n, int fftSize, double[] lowCutFilter)
    {
        for (int i = 1; i <= n; i++)
            lowCutFilter[i - 1] = 0.5 - 0.5 * Math.Cos(i * 2.0 * Math.PI / (n + 1));
        for (int i = n; i < fftSize; i++) lowCutFilter[i] = 0.0;
        double sumOfAmplitude = 0.0;
        for (int i = 0; i < n; i++) sumOfAmplitude += lowCutFilter[i];
        for (int i = 0; i < n; i++) lowCutFilter[i] = -lowCutFilter[i] / sumOfAmplitude;
        for (int i = 0; i < (n - 1) / 2; i++)
            lowCutFilter[fftSize - (n - 1) / 2 + i] = lowCutFilter[i];
        for (int i = 0; i < n; i++) lowCutFilter[i] = lowCutFilter[i + (n - 1) / 2];
        lowCutFilter[0] += 1.0;
    }

    private void FixF0Contour(double framePeriod, int numberOfCandidates, double[][] f0Candidates,
        double[] bestF0Contour, int f0Length, double f0Floor, double allowedRange, double[] fixedF0Contour)
    {
        int voiceRangeMinimum = (int)(0.5 + 1000.0 / framePeriod / f0Floor) * 2 + 1;
        if (f0Length <= voiceRangeMinimum) return;

        var f0Tmp1 = new double[f0Length];
        var f0Tmp2 = new double[f0Length];

        // FixStep1: eliminate unnatural jumps.
        var f0Base = new double[f0Length];
        for (int i = voiceRangeMinimum; i < f0Length - voiceRangeMinimum; i++) f0Base[i] = bestF0Contour[i];
        for (int i = voiceRangeMinimum; i < f0Length; i++)
            f0Tmp1[i] = Math.Abs((f0Base[i] - f0Base[i - 1]) / (SafeGuardMinimum + f0Base[i])) < allowedRange
                ? f0Base[i] : 0.0;

        // FixStep2: eliminate suspected F0 at the start and end of voiced sections.
        Array.Copy(f0Tmp1, f0Tmp2, f0Length);
        int center = (voiceRangeMinimum - 1) / 2;
        for (int i = center; i < f0Length - center; i++)
        {
            for (int j = -center; j <= center; j++)
            {
                if (f0Tmp1[i + j] == 0)
                {
                    f0Tmp2[i] = 0.0;
                    break;
                }
            }
        }

        var positiveIndex = new int[f0Length];
        var negativeIndex = new int[f0Length];
        int positiveCount = 0, negativeCount = 0;
        for (int i = 1; i < f0Length; i++)
        {
            if (f0Tmp2[i] == 0 && f0Tmp2[i - 1] != 0)
                negativeIndex[negativeCount++] = i - 1;
            else if (f0Tmp2[i - 1] == 0 && f0Tmp2[i] != 0)
                positiveIndex[positiveCount++] = i;
        }

        // FixStep3: extend voiced sections forward.
        Array.Copy(f0Tmp2, f0Tmp1, f0Length);
        for (int i = 0; i < negativeCount; i++)
        {
            int limit = i == negativeCount - 1 ? f0Length - 1 : negativeIndex[i + 1];
            for (int j = negativeIndex[i]; j < limit; j++)
            {
                f0Tmp1[j + 1] = SelectBestF0(f0Tmp1[j], f0Tmp1[j - 1], f0Candidates, numberOfCandidates, j + 1,
                    allowedRange);
                if (f0Tmp1[j + 1] == 0) break;
            }
        }

        // FixStep4: extend voiced sections backward.
        Array.Copy(f0Tmp1, fixedF0Contour, f0Length);
        for (int i = positiveCount - 1; i >= 0; i--)
        {
            int limit = i == 0 ? 1 : positiveIndex[i - 1];
            for (int j = positiveIndex[i]; j > limit; j--)
            {
                fixedF0Contour[j - 1] = SelectBestF0(fixedF0Contour[j], fixedF0Contour[j + 1], f0Candidates,
                    numberOfCandidates, j - 1, allowedRange);
                if (fixedF0Contour[j - 1] == 0) break;
            }
        }
    }

    private static double SelectBestF0(double currentF0, double pastF0, double[][] f0Candidates,
        int numberOfCandidates, int targetIndex, double allowedRange)
    {
        double referenceF0 = (currentF0 * 3.0 - pastF0) / 2.0;
        double minimumError = Math.Abs(referenceF0 - f0Candidates[0][targetIndex]);
        double bestF0 = f0Candidates[0][targetIndex];
        for (int i = 1; i < numberOfCandidates; i++)
        {
            double currentError = Math.Abs(referenceF0 - f0Candidates[i][targetIndex]);
            if (currentError < minimumError)
            {
                minimumError = currentError;
                bestF0 = f0Candidates[i][targetIndex];
            }
        }
        if (Math.Abs(1.0 - bestF0 / referenceF0) > allowedRange) return 0.0;
        return bestF0;
    }

    private void GetF0CandidateFromRawEvent(double boundaryF0, double fs, double[] ySpecRe, double[] ySpecIm,
        int yLength, int fftSize, double f0Floor, double f0Ceil, double[] temporalPositions, int f0Length,
        double[] f0Score, double[] f0Candidate)
    {
        var filteredSignal = GetFilteredSignal(MatlabRound(fs / boundaryF0 / 2.0), fftSize, ySpecRe, ySpecIm,
            yLength);

        var negLoc = new double[yLength]; var negInt = new double[yLength];
        var posLoc = new double[yLength]; var posInt = new double[yLength];
        var peakLoc = new double[yLength]; var peakInt = new double[yLength];
        var dipLoc = new double[yLength]; var dipInt = new double[yLength];

        int numberOfNegatives = ZeroCrossingEngine(filteredSignal, yLength, fs, negLoc, negInt);
        for (int i = 0; i < yLength; i++) filteredSignal[i] = -filteredSignal[i];
        int numberOfPositives = ZeroCrossingEngine(filteredSignal, yLength, fs, posLoc, posInt);
        for (int i = 0; i < yLength - 1; i++) filteredSignal[i] = filteredSignal[i] - filteredSignal[i + 1];
        int numberOfPeaks = ZeroCrossingEngine(filteredSignal, yLength - 1, fs, peakLoc, peakInt);
        for (int i = 0; i < yLength - 1; i++) filteredSignal[i] = -filteredSignal[i];
        int numberOfDips = ZeroCrossingEngine(filteredSignal, yLength - 1, fs, dipLoc, dipInt);

        if (CheckEvent(numberOfNegatives - 2) * CheckEvent(numberOfPositives - 2)
            * CheckEvent(numberOfPeaks - 2) * CheckEvent(numberOfDips - 2) == 0)
        {
            for (int i = 0; i < f0Length; i++)
            {
                f0Score[i] = MaximumValue;
                f0Candidate[i] = 0.0;
            }
            return;
        }

        var set0 = new double[f0Length]; var set1 = new double[f0Length];
        var set2 = new double[f0Length]; var set3 = new double[f0Length];
        Interp1(negLoc, negInt, numberOfNegatives, temporalPositions, f0Length, set0);
        Interp1(posLoc, posInt, numberOfPositives, temporalPositions, f0Length, set1);
        Interp1(peakLoc, peakInt, numberOfPeaks, temporalPositions, f0Length, set2);
        Interp1(dipLoc, dipInt, numberOfDips, temporalPositions, f0Length, set3);

        for (int i = 0; i < f0Length; i++)
        {
            f0Candidate[i] = (set0[i] + set1[i] + set2[i] + set3[i]) / 4.0;
            f0Score[i] = Math.Sqrt(((set0[i] - f0Candidate[i]) * (set0[i] - f0Candidate[i])
                + (set1[i] - f0Candidate[i]) * (set1[i] - f0Candidate[i])
                + (set2[i] - f0Candidate[i]) * (set2[i] - f0Candidate[i])
                + (set3[i] - f0Candidate[i]) * (set3[i] - f0Candidate[i])) / 3.0);
            if (f0Candidate[i] > boundaryF0 || f0Candidate[i] < boundaryF0 / 2.0
                || f0Candidate[i] > f0Ceil || f0Candidate[i] < f0Floor)
            {
                f0Candidate[i] = 0.0;
                f0Score[i] = MaximumValue;
            }
        }
    }

    private static int CheckEvent(int x) => x > 0 ? 1 : 0;

    private double[] GetFilteredSignal(int halfAverageLength, int fftSize, double[] ySpecRe, double[] ySpecIm,
        int yLength)
    {
        var lowPassFilter = new double[fftSize];
        NuttallWindow(halfAverageLength * 4, lowPassFilter);
        for (int i = halfAverageLength * 4; i < fftSize; i++) lowPassFilter[i] = 0.0;

        var (lpRe, lpIm) = ForwardFft(lowPassFilter);

        // Convolution in the frequency domain. WORLD writes the conjugate-side copy into the same array as it
        // goes, so its last two iterations overwrite bins N/2-1 and N/2, which the inverse transform reads. The
        // sequence is kept as WORLD has it; its effect is below the 1e-6 test tolerance because the Nuttall
        // low-pass spectrum is effectively zero that close to Nyquist (verified by removing it: still matches).
        double tmp = ySpecRe[0] * lpRe[0] - ySpecIm[0] * lpIm[0];
        lpIm[0] = ySpecRe[0] * lpIm[0] + ySpecIm[0] * lpRe[0];
        lpRe[0] = tmp;
        for (int i = 1; i <= fftSize / 2; i++)
        {
            tmp = ySpecRe[i] * lpRe[i] - ySpecIm[i] * lpIm[i];
            lpIm[i] = ySpecRe[i] * lpIm[i] + ySpecIm[i] * lpRe[i];
            lpRe[i] = tmp;
            lpRe[fftSize - i - 1] = lpRe[i];
            lpIm[fftSize - i - 1] = lpIm[i];
        }

        var filteredSignal = InverseRealFft(lpRe, lpIm, fftSize);

        int indexBias = halfAverageLength * 2;
        for (int i = 0; i < yLength; i++) filteredSignal[i] = filteredSignal[i + indexBias];
        return filteredSignal;
    }

    private static int ZeroCrossingEngine(double[] filteredSignal, int yLength, double fs,
        double[] intervalLocations, double[] intervals)
    {
        var negativeGoingPoints = new int[yLength];
        for (int i = 0; i < yLength - 1; i++)
            negativeGoingPoints[i] = 0.0 < filteredSignal[i] && filteredSignal[i + 1] <= 0.0 ? i + 1 : 0;
        negativeGoingPoints[yLength - 1] = 0;

        var edges = new int[yLength];
        int count = 0;
        for (int i = 0; i < yLength; i++)
            if (negativeGoingPoints[i] > 0) edges[count++] = negativeGoingPoints[i];
        if (count < 2) return 0;

        var fineEdges = new double[count];
        for (int i = 0; i < count; i++)
            fineEdges[i] = edges[i] - filteredSignal[edges[i] - 1]
                / (filteredSignal[edges[i]] - filteredSignal[edges[i] - 1]);

        for (int i = 0; i < count - 1; i++)
        {
            intervals[i] = fs / (fineEdges[i + 1] - fineEdges[i]);
            intervalLocations[i] = (fineEdges[i] + fineEdges[i + 1]) / 2.0 / fs;
        }
        return count - 1;
    }

    #endregion

    #region StoneMask (stonemask.cpp)

    private double GetRefinedF0(double[] x, int fs, double currentPosition, double initialF0)
    {
        if (initialF0 <= FloorF0StoneMask || initialF0 > fs / 12.0) return 0.0;

        int halfWindowLength = (int)(1.5 * fs / initialF0 + 1.0);
        double windowLengthInTime = (2.0 * halfWindowLength + 1.0) / fs;
        int baseTimeLength = halfWindowLength * 2 + 1;
        var baseTime = new double[baseTimeLength];
        for (int i = 0; i < baseTimeLength; i++) baseTime[i] = (double)(-halfWindowLength + i) / fs;
        int fftSize = (int)Math.Pow(2.0, 2.0 + (int)(Math.Log(halfWindowLength * 2.0 + 1.0) / Log2));

        double meanF0 = GetMeanF0(x, fs, currentPosition, initialF0, fftSize, windowLengthInTime, baseTime);
        if (Math.Abs(meanF0 - initialF0) > initialF0 * 0.2) meanF0 = initialF0;
        return meanF0;
    }

    private double GetMeanF0(double[] x, int fs, double currentPosition, double initialF0, int fftSize,
        double windowLengthInTime, double[] baseTime)
    {
        int baseTimeLength = baseTime.Length;
        var indexRaw = new int[baseTimeLength];
        for (int i = 0; i < baseTimeLength; i++) indexRaw[i] = MatlabRound((currentPosition + baseTime[i]) * fs);

        var mainWindow = new double[baseTimeLength];
        for (int i = 0; i < baseTimeLength; i++)
        {
            double tmp = (indexRaw[i] - 1.0) / fs - currentPosition;
            mainWindow[i] = 0.42 + 0.5 * Math.Cos(2.0 * Math.PI * tmp / windowLengthInTime)
                + 0.08 * Math.Cos(4.0 * Math.PI * tmp / windowLengthInTime);
        }

        var diffWindow = new double[baseTimeLength];
        diffWindow[0] = -mainWindow[1] / 2.0;
        for (int i = 1; i < baseTimeLength - 1; i++) diffWindow[i] = -(mainWindow[i + 1] - mainWindow[i - 1]) / 2.0;
        diffWindow[baseTimeLength - 1] = mainWindow[baseTimeLength - 2] / 2.0;

        int xLength = x.Length;
        var waveform = new double[fftSize];
        for (int i = 0; i < baseTimeLength; i++)
        {
            int index = Math.Max(0, Math.Min(xLength - 1, indexRaw[i] - 1));
            waveform[i] = x[index] * mainWindow[i];
        }
        var (mainRe, mainIm) = ForwardFft(waveform);
        for (int i = 0; i < baseTimeLength; i++)
        {
            int index = Math.Max(0, Math.Min(xLength - 1, indexRaw[i] - 1));
            waveform[i] = x[index] * diffWindow[i];
        }
        for (int i = baseTimeLength; i < fftSize; i++) waveform[i] = 0.0;
        var (diffRe, diffIm) = ForwardFft(waveform);

        var powerSpectrum = new double[fftSize / 2 + 1];
        var numeratorI = new double[fftSize / 2 + 1];
        for (int j = 0; j <= fftSize / 2; j++)
        {
            numeratorI[j] = mainRe[j] * diffIm[j] - mainIm[j] * diffRe[j];
            powerSpectrum[j] = mainRe[j] * mainRe[j] + mainIm[j] * mainIm[j];
        }

        double tentativeF0 = FixF0(powerSpectrum, numeratorI, fftSize, fs, initialF0, 2);
        if (tentativeF0 <= 0.0 || tentativeF0 > initialF0 * 2) return 0.0;
        return FixF0(powerSpectrum, numeratorI, fftSize, fs, tentativeF0, 6);
    }

    private static double FixF0(double[] powerSpectrum, double[] numeratorI, int fftSize, int fs, double initialF0,
        int numberOfHarmonics)
    {
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < numberOfHarmonics; i++)
        {
            int index = Math.Min(MatlabRound(initialF0 * fftSize / fs * (i + 1)), fftSize / 2);
            double instantaneousFrequency = powerSpectrum[index] == 0.0 ? 0.0
                : (double)index * fs / fftSize + numeratorI[index] / powerSpectrum[index] * fs / 2.0 / Math.PI;
            double amplitude = Math.Sqrt(powerSpectrum[index]);
            numerator += amplitude * instantaneousFrequency;
            denominator += amplitude * (i + 1);
        }
        return numerator / (denominator + SafeGuardMinimum);
    }

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
    /// <remarks>Ported line for line from WORLD's <c>harvest.cpp</c> (as PyWorld 0.3.5 bundles it), keeping its
    /// quirks: the 1 ms basic contour sampled at the requested period, and <c>ExtendSub</c>'s mean F0 carried from one
    /// voiced section to the next.</remarks>
    /// <returns>F0 in Hz per frame (0 where unvoiced) and each frame's time in seconds.</returns>
    public (double[] F0, double[] TemporalPositions) Harvest(double[] x, double framePeriodMs)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (framePeriodMs <= 0) throw new ArgumentOutOfRangeException(nameof(framePeriodMs));
        if (x.Length < 2) throw new ArgumentException("Harvest needs at least two samples.", nameof(x));
        int fs = SampleRate;
        int dimensionRatio = MatlabRound(fs / 8000.0);
        const double channelsInOctave = 40;

        int basicLength = GetSamplesForHarvest(fs, x.Length, 1);
        var basicF0 = new double[basicLength];
        var basicPositions = new double[basicLength];
        HarvestGeneralBody(x, fs, 1, MinPitch, MaxPitch, channelsInOctave, dimensionRatio, basicPositions, basicF0);
        if (framePeriodMs == 1.0) return (basicF0, basicPositions);

        int f0Length = GetSamplesForHarvest(fs, x.Length, framePeriodMs);
        var positions = new double[f0Length];
        var f0 = new double[f0Length];
        for (int i = 0; i < f0Length; i++)
        {
            positions[i] = i * framePeriodMs / 1000.0;
            f0[i] = basicF0[Math.Min(basicLength - 1, MatlabRound(positions[i] * 1000.0))];
        }
        return (f0, positions);
    }

    /// <summary>StoneMask's refinement (PyWorld's <c>stonemask</c>) of an F0 contour at its frames' times.</summary>
    public double[] StoneMask(double[] x, double[] f0, double[] temporalPositions)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (f0 is null) throw new ArgumentNullException(nameof(f0));
        if (temporalPositions is null) throw new ArgumentNullException(nameof(temporalPositions));
        if (f0.Length != temporalPositions.Length) throw new ArgumentException("Each F0 value needs its time.", nameof(temporalPositions));
        var refined = new double[f0.Length];
        for (int i = 0; i < f0.Length; i++) refined[i] = GetRefinedF0(x, SampleRate, temporalPositions[i], f0[i]);
        return refined;
    }

    private void HarvestGeneralBody(double[] x, int fs, int framePeriod, double f0Floor, double f0Ceil,
        double channelsInOctave, int speed, double[] temporalPositions, double[] f0)
    {
        int xLength = x.Length;
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

        var y = new double[fftSize];
        var (ySpecRe, ySpecIm) = HarvestWaveformAndSpectrum(x, xLength, yLength, fftSize, decimationRatio, y);

        int f0Length = GetSamplesForHarvest(fs, xLength, framePeriod);
        for (int i = 0; i < f0Length; i++)
        {
            temporalPositions[i] = (double)i * framePeriod / 1000.0;
            f0[i] = 0.0;
        }

        const int overlapParameter = 7;
        int maxCandidates = MatlabRound(numberOfChannels / 10.0) * overlapParameter;
        var f0Candidates = new double[f0Length][];
        var f0Scores = new double[f0Length][];
        for (int i = 0; i < f0Length; i++)
        {
            f0Candidates[i] = new double[maxCandidates];
            f0Scores[i] = new double[maxCandidates];
        }

        var rawF0Candidates = new double[numberOfChannels][];
        for (int i = 0; i < numberOfChannels; i++)
        {
            rawF0Candidates[i] = new double[f0Length];
            HarvestF0CandidateFromRawEvent(boundaryF0List[i], actualFs, ySpecRe, ySpecIm, yLength, fftSize, f0Floor, f0Ceil,
                temporalPositions, f0Length, rawF0Candidates[i]);
        }
        int numberOfCandidates = DetectOfficialF0Candidates(rawF0Candidates, numberOfChannels, f0Length, maxCandidates, f0Candidates);
        OverlapF0Candidates(f0Length, numberOfCandidates, f0Candidates);
        numberOfCandidates *= overlapParameter;

        for (int i = 0; i < f0Length; i++)
            for (int j = 0; j < numberOfCandidates; j++)
            {
                var (refined, score) = HarvestRefinedF0(y, yLength, actualFs, temporalPositions[i], f0Candidates[i][j], f0Floor, f0Ceil);
                f0Candidates[i][j] = refined;
                f0Scores[i][j] = score;
            }
        RemoveUnreliableCandidates(f0Length, numberOfCandidates, f0Candidates, f0Scores);

        var bestF0Contour = new double[f0Length];
        HarvestFixF0Contour(f0Candidates, f0Scores, f0Length, numberOfCandidates, bestF0Contour);
        SmoothF0Contour(bestF0Contour, f0Length, f0);
    }

    // GetWaveformAndSpectrum (with its Sub): the signal decimated after edge extension, its mean removed, its spectrum.
    private (double[] Re, double[] Im) HarvestWaveformAndSpectrum(double[] x, int xLength, int yLength, int fftSize,
        int decimationRatio, double[] y)
    {
        if (decimationRatio == 1)
        {
            for (int i = 0; i < xLength; i++) y[i] = x[i];
        }
        else
        {
            int lag = (int)(Math.Ceiling(140.0 / decimationRatio) * decimationRatio);
            int newXLength = xLength + lag * 2;
            var newY = new double[newXLength];
            var newX = new double[newXLength];
            for (int i = 0; i < lag; i++) newX[i] = x[0];
            for (int i = lag; i < lag + xLength; i++) newX[i] = x[i - lag];
            for (int i = lag + xLength; i < newXLength; i++) newX[i] = x[xLength - 1];
            Decimate(newX, newXLength, decimationRatio, newY);
            for (int i = 0; i < yLength; i++) y[i] = newY[lag / decimationRatio + i];
        }

        double meanY = 0.0;
        for (int i = 0; i < yLength; i++) meanY += y[i];
        meanY /= yLength;
        for (int i = 0; i < yLength; i++) y[i] -= meanY;
        for (int i = yLength; i < fftSize; i++) y[i] = 0.0;
        return ForwardFft(y);
    }

    // GetFilteredSignal: the signal through a Nuttall-windowed cosine band-pass at boundaryF0, delay compensated.
    private double[] HarvestFilteredSignal(double boundaryF0, int fftSize, double fs, double[] ySpecRe, double[] ySpecIm,
        int yLength)
    {
        int filterLengthHalf = MatlabRound(fs / boundaryF0 * 2.0);
        var bandPassFilter = new double[fftSize];
        NuttallWindow(filterLengthHalf * 2 + 1, bandPassFilter);
        for (int i = -filterLengthHalf; i <= filterLengthHalf; i++)
            bandPassFilter[i + filterLengthHalf] *= Math.Cos(2 * Math.PI * boundaryF0 * i / fs);

        var (re, im) = ForwardFft(bandPassFilter);
        // The convolution; c2r reads bins 0 … fft/2 only, so the reference's mirrored upper half does not matter.
        for (int i = 0; i <= fftSize / 2; i++)
        {
            double tmp = ySpecRe[i] * re[i] - ySpecIm[i] * im[i];
            im[i] = ySpecRe[i] * im[i] + ySpecIm[i] * re[i];
            re[i] = tmp;
        }
        var filtered = InverseRealFft(re, im, fftSize);
        int indexBias = filterLengthHalf + 1;
        for (int i = 0; i < yLength; i++) filtered[i] = filtered[i + indexBias];
        return filtered;
    }

    // GetF0CandidateFromRawEvent: the four zero-crossing interval series, interpolated to the frames and averaged.
    private void HarvestF0CandidateFromRawEvent(double boundaryF0, double fs, double[] ySpecRe, double[] ySpecIm, int yLength,
        int fftSize, double f0Floor, double f0Ceil, double[] temporalPositions, int f0Length, double[] f0Candidate)
    {
        var filtered = HarvestFilteredSignal(boundaryF0, fftSize, fs, ySpecRe, ySpecIm, yLength);
        var locations = new double[4][];
        var intervals = new double[4][];
        var counts = new int[4];
        for (int k = 0; k < 4; k++)
        {
            locations[k] = new double[yLength];
            intervals[k] = new double[yLength];
        }
        // GetFourZeroCrossingIntervals: negative-going, positive-going, peaks and dips, each by the same engine.
        counts[0] = ZeroCrossingEngine(filtered, yLength, fs, locations[0], intervals[0]);
        for (int i = 0; i < yLength; i++) filtered[i] = -filtered[i];
        counts[1] = ZeroCrossingEngine(filtered, yLength, fs, locations[1], intervals[1]);
        for (int i = 0; i < yLength - 1; i++) filtered[i] = filtered[i] - filtered[i + 1];
        counts[2] = ZeroCrossingEngine(filtered, yLength - 1, fs, locations[2], intervals[2]);
        for (int i = 0; i < yLength - 1; i++) filtered[i] = -filtered[i];
        counts[3] = ZeroCrossingEngine(filtered, yLength - 1, fs, locations[3], intervals[3]);

        if (CheckEvent(counts[0] - 2) * CheckEvent(counts[1] - 2) * CheckEvent(counts[2] - 2) * CheckEvent(counts[3] - 2) == 0)
        {
            for (int i = 0; i < f0Length; i++) f0Candidate[i] = 0.0;
            return;
        }
        var interpolated = new double[4][];
        for (int k = 0; k < 4; k++)
        {
            interpolated[k] = new double[f0Length];
            Interp1(locations[k], intervals[k], counts[k], temporalPositions, f0Length, interpolated[k]);
        }
        double upper = boundaryF0 * 1.1, lower = boundaryF0 * 0.9;
        for (int i = 0; i < f0Length; i++)
        {
            f0Candidate[i] = (interpolated[0][i] + interpolated[1][i] + interpolated[2][i] + interpolated[3][i]) / 4.0;
            if (f0Candidate[i] > upper || f0Candidate[i] < lower || f0Candidate[i] > f0Ceil || f0Candidate[i] < f0Floor)
                f0Candidate[i] = 0.0;
        }
    }

    // DetectOfficialF0Candidates (with Sub1, Sub2): per frame, the mean of each run of at least 10 voiced bands.
    private static int DetectOfficialF0Candidates(double[][] rawF0Candidates, int numberOfChannels, int f0Length,
        int maxCandidates, double[][] f0Candidates)
    {
        int numberOfCandidates = 0;
        var vuv = new int[numberOfChannels];
        var st = new int[numberOfChannels];
        var ed = new int[numberOfChannels];
        for (int i = 0; i < f0Length; i++)
        {
            for (int j = 0; j < numberOfChannels; j++) vuv[j] = rawF0Candidates[j][i] > 0 ? 1 : 0;
            vuv[0] = vuv[numberOfChannels - 1] = 0;
            int numberOfVoicedSections = 0;
            for (int j = 1; j < numberOfChannels; j++)
            {
                int tmp = vuv[j] - vuv[j - 1];
                if (tmp == 1) st[numberOfVoicedSections] = j;
                if (tmp == -1) ed[numberOfVoicedSections++] = j;
            }

            int count = 0;
            for (int s = 0; s < numberOfVoicedSections; s++)
            {
                if (ed[s] - st[s] < 10) continue;
                double tmpF0 = 0.0;
                for (int j = st[s]; j < ed[s]; j++) tmpF0 += rawF0Candidates[j][i];
                tmpF0 /= ed[s] - st[s];
                f0Candidates[i][count++] = tmpF0;
            }
            for (int s = count; s < maxCandidates; s++) f0Candidates[i][s] = 0.0;
            numberOfCandidates = Math.Max(numberOfCandidates, count);
        }
        return numberOfCandidates;
    }

    // OverlapF0Candidates: each frame also takes the candidates of the three frames on either side.
    private static void OverlapF0Candidates(int f0Length, int numberOfCandidates, double[][] f0Candidates)
    {
        const int n = 3;
        for (int i = 1; i <= n; i++)
            for (int j = 0; j < numberOfCandidates; j++)
            {
                for (int k = i; k < f0Length; k++)
                    f0Candidates[k][j + numberOfCandidates * i] = f0Candidates[k - i][j];
                for (int k = 0; k < f0Length - i; k++)
                    f0Candidates[k][j + numberOfCandidates * (i + n)] = f0Candidates[k + i][j];
            }
    }

    // GetRefinedF0 (Harvest's): instantaneous-frequency refinement of a candidate, and its reliability score.
    private (double F0, double Score) HarvestRefinedF0(double[] x, int xLength, double fs, double currentPosition,
        double currentF0, double f0Floor, double f0Ceil)
    {
        if (currentF0 <= 0.0) return (0.0, 0.0);

        int halfWindowLength = (int)(1.5 * fs / currentF0 + 1.0);
        double windowLengthInTime = (2.0 * halfWindowLength + 1.0) / fs;
        int baseTimeLength = halfWindowLength * 2 + 1;
        var baseTime = new double[baseTimeLength];
        for (int i = 0; i < baseTimeLength; i++) baseTime[i] = (-halfWindowLength + i) / fs;
        int fftSize = (int)Math.Pow(2.0, 2.0 + (int)(Math.Log(halfWindowLength * 2.0 + 1.0) / Log2));

        // GetMeanF0 with GetBaseIndex, GetMainWindow, GetDiffWindow and GetSpectra.
        int basicIndex = MatlabRound((currentPosition + baseTime[0]) * fs + 0.001);
        var mainWindow = new double[baseTimeLength];
        for (int i = 0; i < baseTimeLength; i++)
        {
            double tmp = (basicIndex + i - 1.0) / fs - currentPosition;
            mainWindow[i] = 0.42 + 0.5 * Math.Cos(2.0 * Math.PI * tmp / windowLengthInTime)
                + 0.08 * Math.Cos(4.0 * Math.PI * tmp / windowLengthInTime);
        }
        var diffWindow = new double[baseTimeLength];
        diffWindow[0] = -mainWindow[1] / 2.0;
        for (int i = 1; i < baseTimeLength - 1; i++) diffWindow[i] = -(mainWindow[i + 1] - mainWindow[i - 1]) / 2.0;
        diffWindow[baseTimeLength - 1] = mainWindow[baseTimeLength - 2] / 2.0;

        var waveform = new double[fftSize];
        for (int i = 0; i < baseTimeLength; i++)
            waveform[i] = x[Math.Max(0, Math.Min(xLength - 1, basicIndex + i - 1))] * mainWindow[i];
        var (mainRe, mainIm) = ForwardFft(waveform);
        for (int i = 0; i < baseTimeLength; i++)
            waveform[i] = x[Math.Max(0, Math.Min(xLength - 1, basicIndex + i - 1))] * diffWindow[i];
        var (diffRe, diffIm) = ForwardFft(waveform);

        var powerSpectrum = new double[fftSize / 2 + 1];
        var numeratorI = new double[fftSize / 2 + 1];
        for (int j = 0; j <= fftSize / 2; j++)
        {
            numeratorI[j] = mainRe[j] * diffIm[j] - mainIm[j] * diffRe[j];
            powerSpectrum[j] = mainRe[j] * mainRe[j] + mainIm[j] * mainIm[j];
        }

        // FixF0 (Harvest's): the amplitude-weighted harmonic frequencies, and the inverse of their mean deviation.
        int numberOfHarmonics = Math.Min((int)(fs / 2.0 / currentF0), 6);
        double numerator = 0.0, denominator = 0.0, score = 0.0;
        for (int i = 0; i < numberOfHarmonics; i++)
        {
            int index = MatlabRound(currentF0 * fftSize / fs * (i + 1));
            double instantaneousFrequency = powerSpectrum[index] == 0.0 ? 0.0
                : (double)index * fs / fftSize + numeratorI[index] / powerSpectrum[index] * fs / 2.0 / Math.PI;
            double amplitude = Math.Sqrt(powerSpectrum[index]);
            numerator += amplitude * instantaneousFrequency;
            denominator += amplitude * (i + 1.0);
            score += Math.Abs((instantaneousFrequency / (i + 1.0) - currentF0) / currentF0);
        }
        double refinedF0 = numerator / (denominator + SafeGuardMinimum);
        double refinedScore = 1.0 / (score / numberOfHarmonics + SafeGuardMinimum);
        if (refinedF0 < f0Floor || refinedF0 > f0Ceil || refinedScore < 2.5) return (0.0, 0.0);
        return (refinedF0, refinedScore);
    }

    // SelectBestF0 (Harvest's): the candidate nearest the reference within the allowed relative error.
    private static double HarvestSelectBestF0(double referenceF0, double[] f0Candidates, int numberOfCandidates,
        double allowedRange, out double bestError)
    {
        double bestF0 = 0.0;
        bestError = allowedRange;
        for (int i = 0; i < numberOfCandidates; i++)
        {
            double tmp = Math.Abs(referenceF0 - f0Candidates[i]) / referenceF0;
            if (tmp > bestError) continue;
            bestF0 = f0Candidates[i];
            bestError = tmp;
        }
        return bestF0;
    }

    // RemoveUnreliableCandidates: a candidate with no close candidate in the frame before or after is dropped.
    private static void RemoveUnreliableCandidates(int f0Length, int numberOfCandidates, double[][] f0Candidates, double[][] f0Scores)
    {
        var tmp = new double[f0Length][];
        for (int i = 0; i < f0Length; i++)
        {
            tmp[i] = new double[numberOfCandidates];
            Array.Copy(f0Candidates[i], tmp[i], numberOfCandidates);
        }
        for (int i = 1; i < f0Length - 1; i++)
            for (int j = 0; j < numberOfCandidates; j++)
            {
                double referenceF0 = f0Candidates[i][j];
                if (referenceF0 == 0) continue;
                HarvestSelectBestF0(referenceF0, tmp[i + 1], numberOfCandidates, 1.0, out double error1);
                HarvestSelectBestF0(referenceF0, tmp[i - 1], numberOfCandidates, 1.0, out double error2);
                if (Math.Min(error1, error2) <= 0.05) continue;
                f0Candidates[i][j] = 0;
                f0Scores[i][j] = 0;
            }
    }

    // FixF0Contour: the best-scoring candidate per frame, then the four fixing steps.
    private static void HarvestFixF0Contour(double[][] f0Candidates, double[][] f0Scores, int f0Length, int numberOfCandidates,
        double[] bestF0Contour)
    {
        var contour1 = new double[f0Length];
        var contour2 = new double[f0Length];
        // SearchF0Base.
        for (int i = 0; i < f0Length; i++)
        {
            double best = 0.0;
            contour1[i] = 0.0;
            for (int j = 0; j < numberOfCandidates; j++)
                if (f0Scores[i][j] > best)
                {
                    contour1[i] = f0Candidates[i][j];
                    best = f0Scores[i][j];
                }
        }
        // These parameters are optimized by speech databases.
        HarvestFixStep1(contour1, f0Length, 0.008, contour2);
        HarvestFixStep2(contour2, f0Length, 6, contour1);
        HarvestFixStep3(contour1, f0Length, numberOfCandidates, f0Candidates, 0.18, f0Scores, contour2);
        HarvestFixStep4(contour2, f0Length, 9, bestF0Contour);
    }

    // Step 1: a rapid change of the contour is replaced by 0.
    private static void HarvestFixStep1(double[] f0Base, int f0Length, double allowedRange, double[] f0Step1)
    {
        for (int i = 0; i < f0Length; i++) f0Step1[i] = 0.0;
        for (int i = 2; i < f0Length; i++)
        {
            if (f0Base[i] == 0.0) continue;
            double referenceF0 = f0Base[i - 1] * 2 - f0Base[i - 2];
            f0Step1[i] = Math.Abs((f0Base[i] - referenceF0) / referenceF0) > allowedRange
                && Math.Abs(f0Base[i] - f0Base[i - 1]) / f0Base[i - 1] > allowedRange ? 0.0 : f0Base[i];
        }
    }

    // GetBoundaryList: the first and last frame of each voiced section (the ends treated as unvoiced).
    private static int HarvestBoundaryList(double[] f0, int f0Length, int[] boundaryList)
    {
        int numberOfBoundaries = 0;
        var vuv = new int[f0Length];
        for (int i = 0; i < f0Length; i++) vuv[i] = f0[i] > 0 ? 1 : 0;
        vuv[0] = vuv[f0Length - 1] = 0;
        for (int i = 1; i < f0Length; i++)
            if (vuv[i] - vuv[i - 1] != 0)
            {
                boundaryList[numberOfBoundaries] = i - numberOfBoundaries % 2;
                numberOfBoundaries++;
            }
        return numberOfBoundaries;
    }

    // Step 2: voiced sections shorter than voiceRangeMinimum frames are removed.
    private static void HarvestFixStep2(double[] f0Step1, int f0Length, int voiceRangeMinimum, double[] f0Step2)
    {
        Array.Copy(f0Step1, f0Step2, f0Length);
        var boundaryList = new int[f0Length];
        int numberOfBoundaries = HarvestBoundaryList(f0Step1, f0Length, boundaryList);
        for (int i = 0; i < numberOfBoundaries / 2; i++)
        {
            if (boundaryList[i * 2 + 1] - boundaryList[i * 2] >= voiceRangeMinimum) continue;
            for (int j = boundaryList[i * 2]; j <= boundaryList[i * 2 + 1]; j++) f0Step2[j] = 0.0;
        }
    }

    // GetMultiChannelF0: each voiced section in a channel of its own.
    private static void HarvestMultiChannelF0(double[] f0, int f0Length, int[] boundaryList, int numberOfBoundaries,
        double[][] multiChannelF0)
    {
        for (int i = 0; i < numberOfBoundaries / 2; i++)
        {
            for (int j = 0; j < boundaryList[i * 2]; j++) multiChannelF0[i][j] = 0.0;
            for (int j = boundaryList[i * 2]; j <= boundaryList[i * 2 + 1]; j++) multiChannelF0[i][j] = f0[j];
            for (int j = boundaryList[i * 2 + 1] + 1; j < f0Length; j++) multiChannelF0[i][j] = 0.0;
        }
    }

    // ExtendF0: extends a section from origin by following the nearest candidates until four frames find none.
    private static int HarvestExtendF0(int origin, int lastPoint, int shift, double[][] f0Candidates, int numberOfCandidates,
        double allowedRange, double[] extendedF0)
    {
        const int threshold = 4;
        double tmpF0 = extendedF0[origin];
        int shiftedOrigin = origin;
        int distance = Math.Abs(lastPoint - origin);
        int count = 0;
        for (int i = 0; i <= distance; i++)
        {
            int index = origin + shift * i;
            extendedF0[index + shift] = HarvestSelectBestF0(tmpF0, f0Candidates[index + shift], numberOfCandidates, allowedRange, out _);
            if (extendedF0[index + shift] == 0.0)
            {
                count++;
            }
            else
            {
                tmpF0 = extendedF0[index + shift];
                count = 0;
                shiftedOrigin = index + shift;
            }
            if (count == threshold) break;
        }
        return shiftedOrigin;
    }

    // Swap: exchanges two channels' contours and boundaries.
    private static void HarvestSwap(int index1, int index2, double[][] f0, int[] boundary)
    {
        (f0[index1], f0[index2]) = (f0[index2], f0[index1]);
        (boundary[index1 * 2], boundary[index2 * 2]) = (boundary[index2 * 2], boundary[index1 * 2]);
        (boundary[index1 * 2 + 1], boundary[index2 * 2 + 1]) = (boundary[index2 * 2 + 1], boundary[index1 * 2 + 1]);
    }

    // Extend, with ExtendSub: every section extended both ways, then only those long enough for their mean F0 kept.
    // FixStep3 passes the same arrays as input and output, and that aliasing is kept.
    private static int HarvestExtend(double[][] multiChannelF0, int numberOfSections, int f0Length, int[] boundaryList,
        double[][] f0Candidates, int numberOfCandidates, double allowedRange, double[][] extendedF0, int[] shiftedBoundaryList)
    {
        const int threshold = 100;
        for (int i = 0; i < numberOfSections; i++)
        {
            shiftedBoundaryList[i * 2 + 1] = HarvestExtendF0(boundaryList[i * 2 + 1],
                Math.Min(f0Length - 2, boundaryList[i * 2 + 1] + threshold), 1, f0Candidates, numberOfCandidates, allowedRange,
                extendedF0[i]);
            shiftedBoundaryList[i * 2] = HarvestExtendF0(boundaryList[i * 2], Math.Max(1, boundaryList[i * 2] - threshold), -1,
                f0Candidates, numberOfCandidates, allowedRange, extendedF0[i]);
        }

        const double meanThreshold = 2200.0;
        int count = 0;
        // The reference never resets the mean between sections.
        double meanF0 = 0.0;
        for (int i = 0; i < numberOfSections; i++)
        {
            int st = shiftedBoundaryList[i * 2];
            int ed = shiftedBoundaryList[i * 2 + 1];
            for (int j = st; j < ed; j++) meanF0 += multiChannelF0[i][j];
            meanF0 /= ed - st;
            if (meanThreshold / meanF0 < ed - st) HarvestSwap(count++, i, extendedF0, shiftedBoundaryList);
        }
        return count;
    }

    // MergeF0 with MakeSortedOrder, SearchScore and MergeF0Sub: overlapping channels merged by their candidates' scores.
    private static void HarvestMergeF0(double[][] multiChannelF0, int[] boundaryList, int numberOfChannels, int f0Length,
        double[][] f0Candidates, double[][] f0Scores, int numberOfCandidates, double[] mergedF0)
    {
        var order = new int[numberOfChannels];
        for (int i = 0; i < numberOfChannels; i++) order[i] = i;
        for (int i = 1; i < numberOfChannels; i++)
            for (int j = i - 1; j >= 0; j--)
            {
                if (boundaryList[order[j] * 2] > boundaryList[order[i] * 2]) (order[i], order[j]) = (order[j], order[i]);
                else break;
            }

        for (int i = 0; i < f0Length; i++) mergedF0[i] = multiChannelF0[0][i];
        for (int i = 1; i < numberOfChannels; i++)
        {
            if (boundaryList[order[i] * 2] - boundaryList[1] > 0)
            {
                for (int j = boundaryList[order[i] * 2]; j <= boundaryList[order[i] * 2 + 1]; j++)
                    mergedF0[j] = multiChannelF0[order[i]][j];
                boundaryList[0] = boundaryList[order[i] * 2];
                boundaryList[1] = boundaryList[order[i] * 2 + 1];
            }
            else
            {
                boundaryList[1] = HarvestMergeF0Sub(mergedF0, boundaryList[0], boundaryList[1], multiChannelF0[order[i]],
                    boundaryList[order[i] * 2], boundaryList[order[i] * 2 + 1], f0Candidates, f0Scores, numberOfCandidates, mergedF0);
            }
        }
    }

    private static double HarvestSearchScore(double f0, double[] f0Candidates, double[] f0Scores, int numberOfCandidates)
    {
        double score = 0.0;
        for (int i = 0; i < numberOfCandidates; i++)
            if (f0 == f0Candidates[i] && score < f0Scores[i]) score = f0Scores[i];
        return score;
    }

    private static int HarvestMergeF0Sub(double[] f01, int st1, int ed1, double[] f02, int st2, int ed2, double[][] f0Candidates,
        double[][] f0Scores, int numberOfCandidates, double[] mergedF0)
    {
        if (st1 <= st2 && ed1 >= ed2) return ed1;
        double score1 = 0.0, score2 = 0.0;
        for (int i = st2; i <= ed1; i++)
        {
            score1 += HarvestSearchScore(f01[i], f0Candidates[i], f0Scores[i], numberOfCandidates);
            score2 += HarvestSearchScore(f02[i], f0Candidates[i], f0Scores[i], numberOfCandidates);
        }
        if (score1 > score2)
            for (int i = ed1; i <= ed2; i++) mergedF0[i] = f02[i];
        else
            for (int i = st2; i <= ed2; i++) mergedF0[i] = f02[i];
        return ed2;
    }

    // Step 3: voiced sections are extended by the continuity of the contour.
    private static void HarvestFixStep3(double[] f0Step2, int f0Length, int numberOfCandidates, double[][] f0Candidates,
        double allowedRange, double[][] f0Scores, double[] f0Step3)
    {
        Array.Copy(f0Step2, f0Step3, f0Length);
        var boundaryList = new int[f0Length];
        int numberOfBoundaries = HarvestBoundaryList(f0Step2, f0Length, boundaryList);
        var multiChannelF0 = new double[numberOfBoundaries / 2][];
        for (int i = 0; i < numberOfBoundaries / 2; i++) multiChannelF0[i] = new double[f0Length];
        HarvestMultiChannelF0(f0Step2, f0Length, boundaryList, numberOfBoundaries, multiChannelF0);

        int numberOfChannels = HarvestExtend(multiChannelF0, numberOfBoundaries / 2, f0Length, boundaryList, f0Candidates,
            numberOfCandidates, allowedRange, multiChannelF0, boundaryList);
        if (numberOfChannels != 0)
            HarvestMergeF0(multiChannelF0, boundaryList, numberOfChannels, f0Length, f0Candidates, f0Scores, numberOfCandidates, f0Step3);
    }

    // Step 4: F0 in a short unvoiced gap is interpolated across it.
    private static void HarvestFixStep4(double[] f0Step3, int f0Length, int threshold, double[] f0Step4)
    {
        Array.Copy(f0Step3, f0Step4, f0Length);
        var boundaryList = new int[f0Length];
        int numberOfBoundaries = HarvestBoundaryList(f0Step3, f0Length, boundaryList);
        for (int i = 0; i < numberOfBoundaries / 2 - 1; i++)
        {
            int distance = boundaryList[(i + 1) * 2] - boundaryList[i * 2 + 1] - 1;
            if (distance >= threshold) continue;
            double tmp0 = f0Step3[boundaryList[i * 2 + 1]] + 1;
            double tmp1 = f0Step3[boundaryList[(i + 1) * 2]] - 1;
            double coefficient = (tmp1 - tmp0) / (distance + 1.0);
            int count = 1;
            for (int j = boundaryList[i * 2 + 1] + 1; j <= boundaryList[(i + 1) * 2] - 1; j++) f0Step4[j] = tmp0 + coefficient * count++;
        }
    }

    // FilteringF0: a zero-lag (forward-backward) second-order Butterworth filter over one section.
    private static void HarvestFilteringF0(double[] a, double[] b, double[] x, int xLength, int st, int ed, double[] y)
    {
        var tmpX = new double[xLength];
        for (int i = 0; i < st; i++) x[i] = x[st];
        for (int i = ed + 1; i < xLength; i++) x[i] = x[ed];

        double w0 = 0.0, w1 = 0.0;
        for (int i = 0; i < xLength; i++)
        {
            double wt = x[i] + a[0] * w0 + a[1] * w1;
            tmpX[xLength - i - 1] = b[0] * wt + b[1] * w0 + b[0] * w1;
            w1 = w0;
            w0 = wt;
        }
        w0 = w1 = 0.0;
        for (int i = 0; i < xLength; i++)
        {
            double wt = tmpX[i] + a[0] * w0 + a[1] * w1;
            y[xLength - i - 1] = b[0] * wt + b[1] * w0 + b[0] * w1;
            w1 = w0;
            w0 = wt;
        }
    }

    // SmoothF0Contour: each voiced section through the zero-lag filter, padded by 300 frames of zeros.
    private static void SmoothF0Contour(double[] f0, int f0Length, double[] smoothedF0)
    {
        double[] b = { 0.0078202080334971724, 0.015640416066994345 };
        double[] a = { 1.7347257688092754, -0.76600660094326412 };
        const int lag = 300;
        int newF0Length = f0Length + lag * 2;
        var f0Contour = new double[newF0Length];
        for (int i = lag; i < lag + f0Length; i++) f0Contour[i] = f0[i - lag];

        var boundaryList = new int[newF0Length];
        int numberOfBoundaries = HarvestBoundaryList(f0Contour, newF0Length, boundaryList);
        var multiChannelF0 = new double[numberOfBoundaries / 2][];
        for (int i = 0; i < numberOfBoundaries / 2; i++) multiChannelF0[i] = new double[newF0Length];
        HarvestMultiChannelF0(f0Contour, newF0Length, boundaryList, numberOfBoundaries, multiChannelF0);

        for (int i = 0; i < numberOfBoundaries / 2; i++)
        {
            HarvestFilteringF0(a, b, multiChannelF0[i], newF0Length, boundaryList[i * 2], boundaryList[i * 2 + 1], f0Contour);
            for (int j = boundaryList[i * 2]; j <= boundaryList[i * 2 + 1]; j++) smoothedF0[j - lag] = f0Contour[j];
        }
    }

    #endregion

    #region WORLD helpers (common.cpp, matlabfunctions.cpp)

    private static int MatlabRound(double x) => x > 0 ? (int)(x + 0.5) : (int)(x - 0.5);

    private static int GetSuitableFftSize(int sample)
        => (int)Math.Pow(2.0, (int)(Math.Log(sample) / Log2) + 1.0);

    private static void NuttallWindow(int yLength, double[] y)
    {
        for (int i = 0; i < yLength; i++)
        {
            double tmp = i / (yLength - 1.0);
            y[i] = 0.355768 - 0.487396 * Math.Cos(2.0 * Math.PI * tmp) + 0.144232 * Math.Cos(4.0 * Math.PI * tmp)
                - 0.012604 * Math.Cos(6.0 * Math.PI * tmp);
        }
    }

    private static void Interp1(double[] x, double[] y, int xLength, double[] xi, int xiLength, double[] yi)
    {
        var h = new double[xLength - 1];
        var k = new int[xiLength];
        for (int i = 0; i < xLength - 1; i++) h[i] = x[i + 1] - x[i];
        Histc(x, xLength, xi, xiLength, k);
        for (int i = 0; i < xiLength; i++)
        {
            double s = (xi[i] - x[k[i] - 1]) / h[k[i] - 1];
            yi[i] = y[k[i] - 1] + s * (y[k[i]] - y[k[i] - 1]);
        }
    }

    private static void Histc(double[] x, int xLength, double[] edges, int edgesLength, int[] index)
    {
        int count = 1;
        int i = 0;
        for (; i < edgesLength; i++)
        {
            index[i] = 1;
            if (edges[i] >= x[0]) break;
        }
        for (; i < edgesLength; i++)
        {
            if (edges[i] < x[count])
            {
                index[i] = count;
            }
            else
            {
                index[i--] = count++;
            }
            if (count == xLength) break;
        }
        count--;
        for (i++; i < edgesLength; i++) index[i] = count;
    }

    private static void Decimate(double[] x, int xLength, int r, double[] y)
    {
        const int nFact = 9;
        var tmp1 = new double[xLength + nFact * 2];
        var tmp2 = new double[xLength + nFact * 2];

        for (int i = 0; i < nFact; i++) tmp1[i] = 2 * x[0] - x[nFact - i];
        for (int i = nFact; i < nFact + xLength; i++) tmp1[i] = x[i - nFact];
        for (int i = nFact + xLength; i < 2 * nFact + xLength; i++)
            tmp1[i] = 2 * x[xLength - 1] - x[xLength - 2 - (i - (nFact + xLength))];

        FilterForDecimate(tmp1, 2 * nFact + xLength, r, tmp2);
        for (int i = 0; i < 2 * nFact + xLength; i++) tmp1[i] = tmp2[2 * nFact + xLength - i - 1];
        FilterForDecimate(tmp1, 2 * nFact + xLength, r, tmp2);
        for (int i = 0; i < 2 * nFact + xLength; i++) tmp1[i] = tmp2[2 * nFact + xLength - i - 1];

        int nout = (xLength - 1) / r + 1;
        int nbeg = r - r * nout + xLength;
        int count = 0;
        for (int i = nbeg; i < xLength + nFact; i += r) y[count++] = tmp1[i + nFact - 1];
    }

    private static void FilterForDecimate(double[] x, int xLength, int r, double[] y)
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

        double w0 = 0.0, w1 = 0.0, w2 = 0.0;
        for (int i = 0; i < xLength; i++)
        {
            double wt = x[i] + a0 * w0 + a1 * w1 + a2 * w2;
            y[i] = b0 * wt + b1 * w0 + b1 * w1 + b0 * w2;
            w2 = w1;
            w1 = w0;
            w0 = wt;
        }
    }

    #endregion

    #region FFT

    /// <summary>Unnormalized forward DFT of a real signal (FFTW / WORLD r2c convention); all N bins returned.</summary>
    private (double[] Re, double[] Im) ForwardFft(double[] signal)
    {
        int n = signal.Length;
        var re = new Tensor<double>(new[] { n }, new Vector<double>((double[])signal.Clone()));
        var im = new Tensor<double>(new[] { n });
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current.FFT(re, im, out var outRe, out var outIm);
        return (outRe.AsSpan().ToArray(), outIm.AsSpan().ToArray());
    }

    /// <summary>
    /// Unnormalized inverse of a real signal's spectrum (FFTW / WORLD c2r convention): only bins 0..N/2 are read,
    /// the imaginary parts of the DC and Nyquist bins are ignored, and the result is not divided by N.
    /// </summary>
    private static double[] InverseRealFft(double[] re, double[] im, int n)
    {
        var fullRe = new double[n];
        var fullIm = new double[n];
        fullRe[0] = re[0];
        fullRe[n / 2] = re[n / 2];
        for (int k = 1; k < n / 2; k++)
        {
            fullRe[k] = re[k];
            fullIm[k] = im[k];
            fullRe[n - k] = re[k];
            fullIm[n - k] = -im[k];
        }
        var tRe = new Tensor<double>(new[] { n }, new Vector<double>(fullRe));
        var tIm = new Tensor<double>(new[] { n }, new Vector<double>(fullIm));
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current.IFFT(tRe, tIm, out var outRe, out _);
        var result = outRe.AsSpan().ToArray();
        for (int i = 0; i < n; i++) result[i] *= n;
        return result;
    }

    #endregion
}
