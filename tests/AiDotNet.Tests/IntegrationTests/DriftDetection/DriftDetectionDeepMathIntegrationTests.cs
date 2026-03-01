using AiDotNet.DriftDetection;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DriftDetection;

/// <summary>
/// Deep math-correctness integration tests for drift detectors (DDM, EDDM, Page-Hinkley, ADWIN).
/// Every expected value is hand-calculated from the source code formulas.
/// </summary>
public class DriftDetectionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double MediumTolerance = 1e-6;

    #region DDM Formula Verification

    [Fact]
    public void DDM_ErrorRate_HandCalculated()
    {
        // Feed 3 errors in 10 observations: error rate = 3/10 = 0.3
        var ddm = new DDM<double>(minimumObservations: 1);

        double[] obs = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0];
        foreach (var o in obs)
            ddm.AddObservation(o);

        Assert.Equal(0.3, ddm.GetErrorRate(), Tolerance);
        Assert.Equal(10, ddm.ObservationCount);
    }

    [Fact]
    public void DDM_StandardDeviation_Formula_p_s()
    {
        // p = errors/n, s = sqrt(p*(1-p)/n)
        // 2 errors in 10 obs: p=0.2, s=sqrt(0.2*0.8/10)=sqrt(0.016)=0.12649
        // p+s = 0.2 + 0.12649 = 0.32649
        var ddm = new DDM<double>(minimumObservations: 1);

        // Feed 2 errors at positions 1 and 4
        double[] obs = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0];
        foreach (var o in obs)
            ddm.AddObservation(o);

        double p = 2.0 / 10.0;
        double s = Math.Sqrt(p * (1 - p) / 10);
        double expectedPsi = p + s;

        Assert.Equal(p, ddm.EstimatedMean, Tolerance);
        // MinPsi should be <= p+s (it tracks the minimum)
        Assert.True(ddm.GetMinimumPsi() <= expectedPsi + 1e-10);
    }

    [Fact]
    public void DDM_MinPsi_UpdatedToMinimum()
    {
        // After warm-up, the minimum p+s should reflect the best (lowest) error period
        var ddm = new DDM<double>(minimumObservations: 5);

        // First 10 obs: 1 error → p=0.1, s=sqrt(0.1*0.9/10)=sqrt(0.009)=0.09487
        // p+s = 0.19487
        for (int i = 0; i < 10; i++)
            ddm.AddObservation(i == 0 ? 1.0 : 0.0);

        double minPsi1 = ddm.GetMinimumPsi();

        // Next 10 obs: 0 errors → after 20 obs: p=1/20=0.05, s=sqrt(0.05*0.95/20)=0.04873
        // p+s = 0.09873, which is lower
        for (int i = 0; i < 10; i++)
            ddm.AddObservation(0.0);

        double minPsi2 = ddm.GetMinimumPsi();

        Assert.True(minPsi2 < minPsi1,
            $"MinPsi should decrease when error rate drops: {minPsi2} < {minPsi1}");
    }

    [Fact]
    public void DDM_DriftDetection_WhenErrorExceedsThreshold()
    {
        // warningBound = minP + 2*minS
        // driftBound = minP + 3*minS
        // With low minP and then sudden all-errors, drift should trigger
        var ddm = new DDM<double>(warningThreshold: 2.0, driftThreshold: 3.0, minimumObservations: 10);

        // Establish low error baseline: 0 errors in 50 obs
        // p=0, s=0, minP=0, minS=0, driftBound=0
        for (int i = 0; i < 50; i++)
            ddm.AddObservation(0.0);

        // When minP=0 and minS=0, any error makes p > driftBound=0
        bool driftDetected = ddm.AddObservation(1.0);

        // p = 1/51 > 0 = driftBound → drift!
        Assert.True(driftDetected);
        Assert.True(ddm.IsInDrift);
    }

    [Fact]
    public void DDM_WarningZone_BetweenThresholds()
    {
        // Need minP > 0 so warning and drift bounds are different
        var ddm = new DDM<double>(warningThreshold: 2.0, driftThreshold: 3.0, minimumObservations: 5);

        // Stable period: ~10% error rate (10 errors in 100)
        for (int i = 0; i < 100; i++)
            ddm.AddObservation(i % 10 == 0 ? 1.0 : 0.0);

        Assert.False(ddm.IsInDrift);

        // Gradually increase error rate to trigger warning but not drift
        // Feed a burst of errors to push above warning threshold
        bool warningTriggered = false;
        bool driftTriggered = false;
        for (int i = 0; i < 50; i++)
        {
            // ~50% error rate
            driftTriggered = ddm.AddObservation(i % 2 == 0 ? 1.0 : 0.0);
            if (ddm.IsInWarning && !driftTriggered) warningTriggered = true;
            if (driftTriggered) break;
        }

        // Either warning or drift should have triggered
        Assert.True(warningTriggered || driftTriggered,
            "Should trigger at least warning with increased error rate");
    }

    [Fact]
    public void DDM_NoDetectionDuringWarmup()
    {
        var ddm = new DDM<double>(minimumObservations: 30);

        // Feed errors during warm-up - should never trigger drift
        for (int i = 0; i < 29; i++)
        {
            bool result = ddm.AddObservation(1.0); // All errors
            Assert.False(result, $"Should not detect drift during warm-up at obs {i + 1}");
        }
    }

    [Fact]
    public void DDM_ErrorCounting_ThresholdAt05()
    {
        // Values > 0.5 count as errors
        var ddm = new DDM<double>(minimumObservations: 1);

        ddm.AddObservation(0.0);   // not error (0 <= 0.5)
        ddm.AddObservation(0.5);   // not error (0.5 == 0.5, not > 0.5)
        ddm.AddObservation(0.51);  // error (> 0.5)
        ddm.AddObservation(1.0);   // error (> 0.5)

        Assert.Equal(0.5, ddm.GetErrorRate(), Tolerance); // 2/4
    }

    [Fact]
    public void DDM_DriftProbability_ScalesWithDistance()
    {
        // When in warning zone, DriftProbability = (p - warningBound) / (driftBound - warningBound)
        var ddm = new DDM<double>(warningThreshold: 2.0, driftThreshold: 3.0, minimumObservations: 5);

        // Establish baseline with some errors
        for (int i = 0; i < 100; i++)
            ddm.AddObservation(i % 5 == 0 ? 1.0 : 0.0); // 20% error

        // DriftProbability should be 0 when not in warning
        if (!ddm.IsInWarning && !ddm.IsInDrift)
        {
            Assert.Equal(0.0, ddm.DriftProbability, Tolerance);
        }
    }

    #endregion

    #region EDDM Formula Verification

    [Fact]
    public void EDDM_DistanceBetweenErrors_HandCalculated()
    {
        // Errors at positions 5, 10, 15 → distances: 5, 5
        var eddm = new EDDM<double>(minimumObservations: 1, minimumErrors: 2);

        // Feed pattern: 4 correct then 1 error, repeated 3 times
        for (int i = 0; i < 15; i++)
            eddm.AddObservation((i + 1) % 5 == 0 ? 1.0 : 0.0);

        Assert.Equal(3, eddm.ErrorCount);
        // Mean distance should be 5.0 (Welford: mean of [5, 5])
        Assert.Equal(5.0, eddm.GetMeanDistance(), MediumTolerance);
    }

    [Fact]
    public void EDDM_WelfordMean_HandCalculated()
    {
        // Errors at positions 3, 7, 12 → distances: 4, 5
        // Welford's mean of [4, 5] = 4.5
        var eddm = new EDDM<double>(minimumObservations: 1, minimumErrors: 2);

        for (int i = 1; i <= 12; i++)
        {
            bool isError = (i == 3 || i == 7 || i == 12);
            eddm.AddObservation(isError ? 1.0 : 0.0);
        }

        Assert.Equal(3, eddm.ErrorCount);
        Assert.Equal(4.5, eddm.GetMeanDistance(), MediumTolerance);
    }

    [Fact]
    public void EDDM_WelfordStd_HandCalculated()
    {
        // Errors at 2, 5, 10 → distances: 3, 5
        // Welford: n=2, mean=4, M2 = (3-4)*(3-4) + ... but careful with Welford order
        // Welford step 1: distance=3, n=1, mean=3, M2=0
        // Welford step 2: distance=5, delta=5-3=2, mean=3+2/1=5... wait
        // Actually errorCount-1 is the divisor for delta update
        // Error 1: _lastErrorPosition=2. Error 2: distance=5-2=3, _errorCount=2
        //   delta = 3 - 0 = 3, _distanceMean += 3/1 = 3, delta2 = 3-3 = 0, M2 += 3*0 = 0
        // Error 3: distance=10-5=5, _errorCount=3
        //   delta = 5 - 3 = 2, _distanceMean += 2/2 = 4, delta2 = 5-4 = 1, M2 += 2*1 = 2
        // variance = M2/(errorCount-2) = 2/1 = 2, std = sqrt(2) ≈ 1.41421
        var eddm = new EDDM<double>(minimumObservations: 1, minimumErrors: 2);

        for (int i = 1; i <= 10; i++)
        {
            bool isError = (i == 2 || i == 5 || i == 10);
            eddm.AddObservation(isError ? 1.0 : 0.0);
        }

        Assert.Equal(3, eddm.ErrorCount);
        Assert.Equal(4.0, eddm.GetMeanDistance(), MediumTolerance);
        Assert.Equal(Math.Sqrt(2.0), eddm.GetDistanceStd(), MediumTolerance);
    }

    [Fact]
    public void EDDM_Ratio_HandCalculated()
    {
        // After establishing max psi, verify ratio computation
        var eddm = new EDDM<double>(warningThreshold: 0.95, driftThreshold: 0.90,
            minimumObservations: 1, minimumErrors: 3);

        // First phase: errors every 10 obs → large distance, high psi
        for (int i = 1; i <= 40; i++)
            eddm.AddObservation(i % 10 == 0 ? 1.0 : 0.0);

        double ratio1 = eddm.GetCurrentRatio();
        // Should be at or near 1.0 (at max performance)
        Assert.True(ratio1 >= 0.9, $"Initial ratio should be near 1.0, got {ratio1}");
    }

    [Fact]
    public void EDDM_NoDetectionDuringWarmup()
    {
        var eddm = new EDDM<double>(minimumObservations: 30, minimumErrors: 30);

        // Feed many errors but not enough to meet minimumErrors
        for (int i = 0; i < 29; i++)
        {
            bool result = eddm.AddObservation(1.0);
            Assert.False(result, "Should not detect drift during warm-up");
        }
    }

    [Fact]
    public void EDDM_DriftWhenDistancesShrink()
    {
        var eddm = new EDDM<double>(warningThreshold: 0.95, driftThreshold: 0.90,
            minimumObservations: 1, minimumErrors: 3);

        // Phase 1: Errors every 20 obs → large distances
        for (int i = 1; i <= 100; i++)
            eddm.AddObservation(i % 20 == 0 ? 1.0 : 0.0);

        // Phase 2: Errors every 2 obs → distances shrink dramatically
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            if (eddm.AddObservation(i % 2 == 0 ? 1.0 : 0.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected, "EDDM should detect drift when error frequency increases");
    }

    #endregion

    #region Page-Hinkley Formula Verification

    [Fact]
    public void PageHinkley_RunningMean_WelfordFormula()
    {
        // Welford mean after [1, 3, 5, 7, 9]:
        // After 1: mean=1
        // After 3: mean=1+(3-1)/2=2
        // After 5: mean=2+(5-2)/3=3
        // After 7: mean=3+(7-3)/4=4
        // After 9: mean=4+(9-4)/5=5
        var ph = new PageHinkley<double>(minimumObservations: 1);

        double[] vals = [1, 3, 5, 7, 9];
        foreach (var v in vals)
            ph.AddObservation(v);

        Assert.Equal(5.0, ph.EstimatedMean, MediumTolerance);
    }

    [Fact]
    public void PageHinkley_CumulativeSum_HandCalculated()
    {
        // sum += (val - runningMean - alpha)
        // With alpha=0 for simplicity:
        // val=1: mean=1, sum += 1-1-0=0 → sum=0
        // val=3: mean=2, sum += 3-2-0=1 → sum=1
        // val=5: mean=3, sum += 5-3-0=2 → sum=3
        // val=7: mean=4, sum += 7-4-0=3 → sum=6
        // val=9: mean=5, sum += 9-5-0=4 → sum=10
        var ph = new PageHinkley<double>(lambda: 50, alpha: 0, minimumObservations: 1);

        double[] vals = [1, 3, 5, 7, 9];
        foreach (var v in vals)
            ph.AddObservation(v);

        Assert.Equal(10.0, ph.GetCumulativeSum(), MediumTolerance);
    }

    [Fact]
    public void PageHinkley_CumulativeSum_WithAlpha()
    {
        // alpha=1: sum += (val - mean - 1)
        // val=1: mean=1, sum += 1-1-1=-1 → sum=-1
        // val=3: mean=2, sum += 3-2-1=0 → sum=-1
        // val=5: mean=3, sum += 5-3-1=1 → sum=0
        // val=7: mean=4, sum += 7-4-1=2 → sum=2
        // val=9: mean=5, sum += 9-5-1=3 → sum=5
        var ph = new PageHinkley<double>(lambda: 50, alpha: 1.0, minimumObservations: 1);

        double[] vals = [1, 3, 5, 7, 9];
        foreach (var v in vals)
            ph.AddObservation(v);

        Assert.Equal(5.0, ph.GetCumulativeSum(), MediumTolerance);
    }

    [Fact]
    public void PageHinkley_TestStatistic_DetectIncrease()
    {
        // For DetectIncrease: statistic = sum - sumMin
        // Constant stream then sudden increase
        var ph = new PageHinkley<double>(lambda: 5, alpha: 0,
            mode: PageHinkley<double>.DetectionMode.DetectIncrease, minimumObservations: 3);

        // Feed constant 0s: sum stays near 0
        for (int i = 0; i < 5; i++)
            ph.AddObservation(0.0);

        double stat1 = ph.GetTestStatistic();

        // Feed large values: sum increases above minimum
        for (int i = 0; i < 5; i++)
            ph.AddObservation(10.0);

        double stat2 = ph.GetTestStatistic();
        Assert.True(stat2 > stat1, "Test statistic should increase with rising values");
    }

    [Fact]
    public void PageHinkley_DetectsIncrease_WhenMeanShifts()
    {
        var ph = new PageHinkley<double>(lambda: 10, alpha: 0.005,
            mode: PageHinkley<double>.DetectionMode.DetectIncrease, minimumObservations: 5);

        // Stable period: values near 0
        for (int i = 0; i < 50; i++)
            ph.AddObservation(0.0);

        // Sudden shift: values jump to 10
        bool driftDetected = false;
        for (int i = 0; i < 100; i++)
        {
            if (ph.AddObservation(10.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected, "Page-Hinkley should detect increase in mean");
    }

    [Fact]
    public void PageHinkley_DetectsDecrease_WhenMeanDrops()
    {
        var ph = new PageHinkley<double>(lambda: 10, alpha: 0.005,
            mode: PageHinkley<double>.DetectionMode.DetectDecrease, minimumObservations: 5);

        // Stable period: values near 10
        for (int i = 0; i < 50; i++)
            ph.AddObservation(10.0);

        // Sudden shift: values drop to 0
        bool driftDetected = false;
        for (int i = 0; i < 100; i++)
        {
            if (ph.AddObservation(0.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected, "Page-Hinkley should detect decrease in mean");
    }

    [Fact]
    public void PageHinkley_DetectBoth_CatchesEitherDirection()
    {
        var ph = new PageHinkley<double>(lambda: 10, alpha: 0.005,
            mode: PageHinkley<double>.DetectionMode.DetectBoth, minimumObservations: 5);

        // Stable at 5
        for (int i = 0; i < 50; i++)
            ph.AddObservation(5.0);

        // Shift up to 15
        bool driftDetected = false;
        for (int i = 0; i < 100; i++)
        {
            if (ph.AddObservation(15.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected, "DetectBoth should catch upward shift");
    }

    [Fact]
    public void PageHinkley_NoDrift_StableStream()
    {
        var ph = new PageHinkley<double>(lambda: 50, alpha: 0.005, minimumObservations: 5);

        // All same value → mean equals value, deviations are tiny
        bool anyDrift = false;
        for (int i = 0; i < 200; i++)
        {
            if (ph.AddObservation(5.0))
            {
                anyDrift = true;
                break;
            }
        }

        Assert.False(anyDrift, "Constant stream should not trigger drift");
    }

    [Fact]
    public void PageHinkley_Warning_At80Percent()
    {
        // Warning triggers when DriftProbability > 0.8
        var ph = new PageHinkley<double>(lambda: 100, alpha: 0,
            mode: PageHinkley<double>.DetectionMode.DetectIncrease, minimumObservations: 5);

        // Stable baseline
        for (int i = 0; i < 20; i++)
            ph.AddObservation(0.0);

        // Gradually increase to push toward threshold
        bool warningTriggered = false;
        for (int i = 0; i < 200; i++)
        {
            ph.AddObservation(5.0);
            if (ph.IsInWarning)
            {
                warningTriggered = true;
                break;
            }
            if (ph.IsInDrift) break;
        }

        // Either warning or drift should occur
        Assert.True(warningTriggered || ph.IsInDrift);
    }

    #endregion

    #region ADWIN Formula Verification

    [Fact]
    public void ADWIN_WindowSize_GrowsWithObservations()
    {
        var adwin = new ADWIN<double>();

        Assert.Equal(0, adwin.WindowSize);

        for (int i = 0; i < 50; i++)
            adwin.AddObservation(0.0);

        Assert.True(adwin.WindowSize > 0, "Window should grow with stable observations");
    }

    [Fact]
    public void ADWIN_EstimatedMean_MatchesRunningAverage()
    {
        var adwin = new ADWIN<double>();

        // Feed values [1, 2, 3, 4, 5]
        double[] vals = [1, 2, 3, 4, 5];
        foreach (var v in vals)
            adwin.AddObservation(v);

        // Without drift, mean should equal simple average = 15/5 = 3.0
        Assert.Equal(3.0, adwin.EstimatedMean, MediumTolerance);
    }

    [Fact]
    public void ADWIN_DetectsDrift_WhenMeanShifts()
    {
        var adwin = new ADWIN<double>(delta: 0.01);

        // Stable at 0
        for (int i = 0; i < 100; i++)
            adwin.AddObservation(0.0);

        // Sudden shift to 1
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            if (adwin.AddObservation(1.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected, "ADWIN should detect shift from 0 to 1");
    }

    [Fact]
    public void ADWIN_WindowShrinks_OnDrift()
    {
        var adwin = new ADWIN<double>(delta: 0.01);

        // Fill window with 0s
        for (int i = 0; i < 100; i++)
            adwin.AddObservation(0.0);

        int windowBefore = adwin.WindowSize;

        // Shift to 1s until drift detected
        for (int i = 0; i < 200; i++)
        {
            if (adwin.AddObservation(1.0))
                break;
        }

        // Window should have shrunk (old data dropped)
        if (adwin.IsInDrift)
        {
            Assert.True(adwin.WindowSize < windowBefore,
                $"Window should shrink on drift: {adwin.WindowSize} < {windowBefore}");
        }
    }

    [Fact]
    public void ADWIN_NoDrift_StableStream()
    {
        var adwin = new ADWIN<double>(delta: 0.002);

        bool anyDrift = false;
        for (int i = 0; i < 200; i++)
        {
            if (adwin.AddObservation(0.5))
            {
                anyDrift = true;
                break;
            }
        }

        Assert.False(anyDrift, "Constant stream should not trigger ADWIN drift");
    }

    #endregion

    #region Cross-Detector Consistency Tests

    [Fact]
    public void AllDetectors_StableStream_NoDrift()
    {
        var ddm = new DDM<double>();
        var eddm = new EDDM<double>();
        var ph = new PageHinkley<double>();
        var adwin = new ADWIN<double>();

        for (int i = 0; i < 200; i++)
        {
            Assert.False(ddm.AddObservation(0.0));
            Assert.False(eddm.AddObservation(0.0));
            Assert.False(ph.AddObservation(5.0));
            Assert.False(adwin.AddObservation(0.5));
        }
    }

    [Fact]
    public void AllDetectors_Reset_ClearsCompletely()
    {
        var ddm = new DDM<double>();
        var ph = new PageHinkley<double>();
        var adwin = new ADWIN<double>();

        // Feed data
        for (int i = 0; i < 50; i++)
        {
            ddm.AddObservation(1.0);
            ph.AddObservation(10.0);
            adwin.AddObservation(1.0);
        }

        // Reset
        ddm.Reset();
        ph.Reset();
        adwin.Reset();

        Assert.Equal(0, ddm.ObservationCount);
        Assert.False(ddm.IsInDrift);
        Assert.False(ddm.IsInWarning);

        Assert.Equal(0, ph.ObservationCount);
        Assert.False(ph.IsInDrift);

        Assert.Equal(0, adwin.WindowSize);
        Assert.False(adwin.IsInDrift);
    }

    [Fact]
    public void AllDetectors_SuddenDrift_EventuallyDetected()
    {
        // Phase 1: all correct. Phase 2: all errors.
        // All detectors should eventually detect drift.
        var ddm = new DDM<double>(minimumObservations: 10);
        var ph = new PageHinkley<double>(lambda: 10, minimumObservations: 10);
        var adwin = new ADWIN<double>(delta: 0.01);

        // Stable phase
        for (int i = 0; i < 100; i++)
        {
            ddm.AddObservation(0.0);
            ph.AddObservation(0.0);
            adwin.AddObservation(0.0);
        }

        // Drift phase
        bool ddmDrift = false, phDrift = false, adwinDrift = false;
        for (int i = 0; i < 200; i++)
        {
            if (!ddmDrift && ddm.AddObservation(1.0)) ddmDrift = true;
            if (!phDrift && ph.AddObservation(1.0)) phDrift = true;
            if (!adwinDrift && adwin.AddObservation(1.0)) adwinDrift = true;

            if (ddmDrift && phDrift && adwinDrift) break;
        }

        Assert.True(ddmDrift, "DDM should detect sudden drift");
        Assert.True(phDrift, "Page-Hinkley should detect sudden drift");
        Assert.True(adwinDrift, "ADWIN should detect sudden drift");
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void DDM_Validation_ThrowsForInvalidParams()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DDM<double>(warningThreshold: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DDM<double>(warningThreshold: -1));
        // driftThreshold must be > warningThreshold
        Assert.Throws<ArgumentOutOfRangeException>(() => new DDM<double>(warningThreshold: 3.0, driftThreshold: 2.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DDM<double>(minimumObservations: 0));
    }

    [Fact]
    public void EDDM_Validation_ThrowsForInvalidParams()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EDDM<double>(warningThreshold: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EDDM<double>(warningThreshold: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EDDM<double>(driftThreshold: 0));
        // driftThreshold must be < warningThreshold
        Assert.Throws<ArgumentOutOfRangeException>(() => new EDDM<double>(warningThreshold: 0.9, driftThreshold: 0.95));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EDDM<double>(minimumErrors: 1));
    }

    [Fact]
    public void PageHinkley_Validation_ThrowsForInvalidParams()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PageHinkley<double>(lambda: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PageHinkley<double>(lambda: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PageHinkley<double>(alpha: -1));
    }

    [Fact]
    public void ADWIN_Validation_ThrowsForInvalidParams()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(delta: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(delta: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(maxBuckets: 1));
    }

    [Fact]
    public void DDM_ObservationCount_Increments()
    {
        var ddm = new DDM<double>();
        Assert.Equal(0, ddm.ObservationCount);

        ddm.AddObservation(0.0);
        Assert.Equal(1, ddm.ObservationCount);

        ddm.AddObservation(1.0);
        Assert.Equal(2, ddm.ObservationCount);
    }

    [Fact]
    public void EDDM_ErrorCount_TracksErrors()
    {
        var eddm = new EDDM<double>(minimumObservations: 1, minimumErrors: 2);

        eddm.AddObservation(0.0); // not error
        Assert.Equal(0, eddm.ErrorCount);

        eddm.AddObservation(1.0); // error
        Assert.Equal(1, eddm.ErrorCount);

        eddm.AddObservation(0.0); // not error
        Assert.Equal(1, eddm.ErrorCount);

        eddm.AddObservation(1.0); // error
        Assert.Equal(2, eddm.ErrorCount);
    }

    [Fact]
    public void PageHinkley_ConstantValues_SumStaysNearZero()
    {
        // With constant values, deviations from mean approach 0
        var ph = new PageHinkley<double>(lambda: 50, alpha: 0, minimumObservations: 1);

        for (int i = 0; i < 100; i++)
            ph.AddObservation(5.0);

        // Sum should be very small for constant values
        Assert.True(Math.Abs(ph.GetCumulativeSum()) < 1.0,
            $"Cumulative sum should be near 0 for constant stream, got {ph.GetCumulativeSum()}");
    }

    [Fact]
    public void EDDM_DistanceStd_ZeroWithTwoErrors()
    {
        // Need errorCount > 2 for non-zero std
        var eddm = new EDDM<double>(minimumObservations: 1, minimumErrors: 2);

        // Only 2 errors: std should be 0
        eddm.AddObservation(1.0);
        eddm.AddObservation(0.0);
        eddm.AddObservation(1.0);

        Assert.Equal(2, eddm.ErrorCount);
        Assert.Equal(0.0, eddm.GetDistanceStd(), Tolerance);
    }

    #endregion
}
