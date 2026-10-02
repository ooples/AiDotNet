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
