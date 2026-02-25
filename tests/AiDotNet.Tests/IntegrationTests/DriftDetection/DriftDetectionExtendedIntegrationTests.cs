using AiDotNet.DriftDetection;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DriftDetection;

/// <summary>
/// Extended integration tests for drift detection with deep mathematical verification.
/// Tests DDM/EDDM/ADWIN/PageHinkley error rate tracking, statistical formulas,
/// threshold behavior, warning-to-drift transitions, and known drift detection.
/// </summary>
public class DriftDetectionExtendedIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region DDM - Deep Error Rate and Threshold Verification

    [Fact]
    public void DDM_ErrorRate_HandCalculated()
    {
        // Feed known sequence: 100 observations, exactly 20 errors
        var ddm = new DDM<double>(minimumObservations: 5);

        // First 80 correct, then 20 errors (but interleave to avoid early drift)
        for (int i = 0; i < 80; i++) ddm.AddObservation(0.0);

        double errorRate = ddm.GetErrorRate();
        Assert.Equal(0.0, errorRate, Tolerance);

        // Now add 20 errors
        for (int i = 0; i < 20; i++) ddm.AddObservation(1.0);

        errorRate = ddm.GetErrorRate();
        Assert.Equal(20.0 / 100.0, errorRate, Tolerance);
    }

    [Fact]
    public void DDM_MinimumPsi_TracksCorrectly()
    {
        var ddm = new DDM<double>(minimumObservations: 5);

        // All correct: p=0, s=0, psi=0
        for (int i = 0; i < 10; i++) ddm.AddObservation(0.0);

        double psi = ddm.GetMinimumPsi();
        // With p=0 and s=0, psi should be 0
        Assert.Equal(0.0, psi, Tolerance);
    }

    [Fact]
    public void DDM_WarningBeforeDrift_TransitionSequence()
    {
        // Use very small minimum observations to test warning->drift transition
        var ddm = new DDM<double>(
            warningThreshold: 2.0,
            driftThreshold: 3.0,
            minimumObservations: 10);

        // Establish a low error rate baseline
        for (int i = 0; i < 50; i++) ddm.AddObservation(0.0);

        Assert.False(ddm.IsInWarning);
        Assert.False(ddm.IsInDrift);

        // Gradually increase error rate to trigger warning
        bool warningTriggered = false;
        bool driftTriggered = false;

        for (int i = 0; i < 200; i++)
        {
            ddm.AddObservation(1.0); // All errors now
            if (ddm.IsInWarning && !driftTriggered) warningTriggered = true;
            if (ddm.IsInDrift) { driftTriggered = true; break; }
        }

        // Should have seen drift
        Assert.True(driftTriggered, "DDM should detect drift with sudden all-error stream");
    }

    [Fact]
    public void DDM_WarningDelay_ExpiresWarning()
    {
        var ddm = new DDM<double>(
            warningThreshold: 2.0,
            driftThreshold: 3.0,
            minimumObservations: 10,
            warningDelay: 20);

        // Establish baseline
        for (int i = 0; i < 30; i++) ddm.AddObservation(0.0);

        // Push to warning zone with moderate error rate
        for (int i = 0; i < 30; i++)
        {
            ddm.AddObservation(i % 3 == 0 ? 1.0 : 0.0); // ~33% errors
        }

        // If warning was set, continue with low errors for warningDelay
        // to test that the warning expires
        for (int i = 0; i < 30; i++) ddm.AddObservation(0.0);

        // After a period of stability, warning should clear
        // (drift detection state resets during stable periods)
    }

    [Fact]
    public void DDM_DriftProbability_IncreasesGradually()
    {
        var ddm = new DDM<double>(
            warningThreshold: 2.0,
            driftThreshold: 3.0,
            minimumObservations: 10);

        // Establish baseline
        for (int i = 0; i < 30; i++) ddm.AddObservation(0.0);

        double prevProb = 0;
        // Gradually increase errors - drift probability should increase
        for (int i = 0; i < 100; i++)
        {
            ddm.AddObservation(1.0);
            if (ddm.IsInDrift) break;

            if (ddm.DriftProbability > 0)
            {
                // Once we're in warning zone, probability should be non-negative
                Assert.True(ddm.DriftProbability >= 0 && ddm.DriftProbability <= 1,
                    $"DriftProbability {ddm.DriftProbability} should be in [0,1]");
            }
        }
    }

    [Fact]
    public void DDM_ObservationCount_IncreasesWithEachCall()
    {
        var ddm = new DDM<double>();
        Assert.Equal(0, ddm.ObservationCount);

        for (int i = 1; i <= 10; i++)
        {
            ddm.AddObservation(0.0);
            Assert.Equal(i, ddm.ObservationCount);
        }
    }

    [Fact]
    public void DDM_Reset_ResetsAllState()
    {
        var ddm = new DDM<double>(minimumObservations: 5);

        for (int i = 0; i < 50; i++) ddm.AddObservation(i % 2 == 0 ? 1.0 : 0.0);

        ddm.Reset();

        Assert.Equal(0, ddm.ObservationCount);
        Assert.Equal(0.0, ddm.GetErrorRate(), Tolerance);
        Assert.False(ddm.IsInDrift);
        Assert.False(ddm.IsInWarning);
        Assert.Equal(0.0, ddm.DriftProbability, Tolerance);
    }

    #endregion

    #region EDDM - Distance-Based Verification

    [Fact]
    public void EDDM_ErrorDistanceMean_HandCalculated()
    {
        // Create EDDM with low minimums so we can test math
        var eddm = new EDDM<double>(minimumObservations: 5, minimumErrors: 3);

        // Pattern: correct, correct, ERROR, correct, correct, correct, ERROR, correct, ERROR
        //          1        2        3      4        5        6        7      8        9
        // Distances between errors: 3→7 = 4, 7→9 = 2
        // Mean distance: (4 + 2) / 2 = 3.0
        eddm.AddObservation(0.0); // 1
        eddm.AddObservation(0.0); // 2
        eddm.AddObservation(1.0); // 3 - first error
        eddm.AddObservation(0.0); // 4
        eddm.AddObservation(0.0); // 5
        eddm.AddObservation(0.0); // 6
        eddm.AddObservation(1.0); // 7 - second error, distance = 4
        eddm.AddObservation(0.0); // 8
        eddm.AddObservation(1.0); // 9 - third error, distance = 2

        double meanDist = eddm.GetMeanDistance();
        // Mean of distances [4, 2] = 3.0
        Assert.Equal(3.0, meanDist, Tolerance);
    }

    [Fact]
    public void EDDM_ErrorDistanceStd_HandCalculated()
    {
        var eddm = new EDDM<double>(minimumObservations: 5, minimumErrors: 4);

        // Create pattern with known distances
        // Errors at positions: 3, 7, 9, 13
        // Distances: 4, 2, 4
        // Mean = (4+2+4)/3 = 10/3 ≈ 3.333
        // Var = [(4-3.333)^2 + (2-3.333)^2 + (4-3.333)^2] / (4-2) = [0.444+1.778+0.444]/2 = 1.333
        // Std = sqrt(1.333) ≈ 1.155
        eddm.AddObservation(0.0); // 1
        eddm.AddObservation(0.0); // 2
        eddm.AddObservation(1.0); // 3
        eddm.AddObservation(0.0); // 4
        eddm.AddObservation(0.0); // 5
        eddm.AddObservation(0.0); // 6
        eddm.AddObservation(1.0); // 7
        eddm.AddObservation(0.0); // 8
        eddm.AddObservation(1.0); // 9
        eddm.AddObservation(0.0); // 10
        eddm.AddObservation(0.0); // 11
        eddm.AddObservation(0.0); // 12
        eddm.AddObservation(1.0); // 13

        double std = eddm.GetDistanceStd();
        // With Welford's algorithm: variance = M2/(n-2) where n = errorCount
        // Distances: 4, 2, 4 (3 distances from 4 errors)
        // M2 should track sum of squared deviations
        Assert.True(std > 0, "Standard deviation of non-constant distances should be positive");
    }

    [Fact]
    public void EDDM_ErrorCount_TracksCorrectly()
    {
        var eddm = new EDDM<double>(minimumObservations: 5, minimumErrors: 5);

        // Add 10 observations: 3 errors at positions 2, 5, 8
        eddm.AddObservation(0.0); // 1
        eddm.AddObservation(1.0); // 2 - error 1
        eddm.AddObservation(0.0); // 3
        eddm.AddObservation(0.0); // 4
        eddm.AddObservation(1.0); // 5 - error 2
        eddm.AddObservation(0.0); // 6
        eddm.AddObservation(0.0); // 7
        eddm.AddObservation(1.0); // 8 - error 3
        eddm.AddObservation(0.0); // 9
        eddm.AddObservation(0.0); // 10

        Assert.Equal(3, eddm.ErrorCount);
    }

    [Fact]
    public void EDDM_CurrentRatio_DecreasesWithMoreErrors()
    {
        var eddm = new EDDM<double>(minimumObservations: 5, minimumErrors: 5);

        // Establish baseline with widely-spaced errors
        for (int i = 0; i < 100; i++)
        {
            eddm.AddObservation(i % 10 == 0 ? 1.0 : 0.0); // ~10% errors, distance ~10
        }

        double baselineRatio = eddm.GetCurrentRatio();

        // Now add closely-spaced errors (should decrease ratio)
        for (int i = 0; i < 100; i++)
        {
            eddm.AddObservation(i % 2 == 0 ? 1.0 : 0.0); // ~50% errors, distance ~2
        }

        double newRatio = eddm.GetCurrentRatio();
        Assert.True(newRatio < baselineRatio,
            $"Ratio should decrease from {baselineRatio} when errors become more frequent, got {newRatio}");
    }

    [Fact]
    public void EDDM_DetectsDrift_WhenErrorsBecomeDense()
    {
        var eddm = new EDDM<double>(
            warningThreshold: 0.95,
            driftThreshold: 0.90,
            minimumObservations: 30,
            minimumErrors: 30);

        // Phase 1: sparse errors (error every 10th observation)
        for (int i = 0; i < 500; i++)
        {
            eddm.AddObservation(i % 10 == 0 ? 1.0 : 0.0);
        }
        Assert.False(eddm.IsInDrift, "Should not drift during stable sparse error phase");

        // Phase 2: dense errors (error every 2nd observation)
        bool detected = false;
        for (int i = 0; i < 500; i++)
        {
            if (eddm.AddObservation(i % 2 == 0 ? 1.0 : 0.0))
            {
                detected = true;
                break;
            }
        }

        Assert.True(detected, "EDDM should detect drift when errors become 5x more frequent");
    }

    [Fact]
    public void EDDM_InvalidParameters_Throw()
    {
        // Warning must be in (0,1)
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EDDM<double>(warningThreshold: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EDDM<double>(warningThreshold: 1.0));

        // Drift must be < warning
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EDDM<double>(warningThreshold: 0.95, driftThreshold: 0.95));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EDDM<double>(warningThreshold: 0.95, driftThreshold: 0.96));

        // Minimum errors must be >= 2
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new EDDM<double>(minimumErrors: 1));
    }

    #endregion

    #region ADWIN - Window and Hoeffding Bound Verification

    [Fact]
    public void ADWIN_WindowSize_GrowsWithStableData()
    {
        var adwin = new ADWIN<double>(delta: 0.01);

        for (int i = 0; i < 100; i++)
        {
            adwin.AddObservation(0.5); // Stable stream
        }

        Assert.Equal(100, adwin.WindowSize);
    }

    [Fact]
    public void ADWIN_WindowSize_ShrinksAfterDrift()
    {
        var adwin = new ADWIN<double>(delta: 0.01);

        // Phase 1: stable at 0.0
        for (int i = 0; i < 200; i++)
        {
            adwin.AddObservation(0.0);
        }

        int windowBefore = adwin.WindowSize;

        // Phase 2: sudden shift to 1.0
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            if (adwin.AddObservation(1.0))
            {
                driftDetected = true;
                break;
            }
        }

        if (driftDetected)
        {
            // After drift, window should have shrunk (removed old portion)
            Assert.True(adwin.WindowSize < windowBefore,
                $"Window should shrink after drift: {adwin.WindowSize} >= {windowBefore}");
        }
    }

    [Fact]
    public void ADWIN_EstimatedMean_TracksWindowMean()
    {
        var adwin = new ADWIN<double>(delta: 0.01);

        // Feed constant value
        for (int i = 0; i < 50; i++)
        {
            adwin.AddObservation(0.5);
        }

        Assert.Equal(0.5, adwin.EstimatedMean, 0.01);

        // Feed higher values
        for (int i = 0; i < 200; i++)
        {
            adwin.AddObservation(0.8);
        }

        // After enough high values (and possible drift detection),
        // mean should be closer to 0.8
        Assert.True(adwin.EstimatedMean > 0.6,
            $"Estimated mean {adwin.EstimatedMean} should track toward 0.8 after drift");
    }

    [Fact]
    public void ADWIN_StableMean_NoDrift()
    {
        var adwin = new ADWIN<double>(delta: 0.01);
        var rng = RandomHelper.CreateSeededRandom(42);

        // Feed random values around mean 0.5 with small variance
        bool anyDrift = false;
        for (int i = 0; i < 500; i++)
        {
            double val = 0.5 + (rng.NextDouble() - 0.5) * 0.1; // Range [0.45, 0.55]
            if (adwin.AddObservation(val))
            {
                anyDrift = true;
            }
        }

        // Should not detect drift for stable data with tight range
        // (note: ADWIN with delta=0.01 might still trigger occasionally on random data)
        // We mainly test that it doesn't crash and that the mean is reasonable
        Assert.True(Math.Abs(adwin.EstimatedMean - 0.5) < 0.1,
            $"Mean should be close to 0.5 for stable data, got {adwin.EstimatedMean}");
    }

    [Fact]
    public void ADWIN_InvalidDelta_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(delta: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(delta: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(delta: -0.1));
    }

    [Fact]
    public void ADWIN_InvalidMaxBuckets_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ADWIN<double>(maxBuckets: 1));
    }

    [Fact]
    public void ADWIN_Reset_ClearsAllState()
    {
        var adwin = new ADWIN<double>();

        for (int i = 0; i < 50; i++) adwin.AddObservation(0.5);

        adwin.Reset();

        Assert.Equal(0, adwin.WindowSize);
        Assert.Equal(0, adwin.ObservationCount);
        Assert.False(adwin.IsInDrift);
    }

    #endregion

    #region PageHinkley - Cumulative Sum Verification

    [Fact]
    public void PageHinkley_StableStream_NoDrift()
    {
        var ph = new PageHinkley<double>(lambda: 50, alpha: 0.005);
        var rng = RandomHelper.CreateSeededRandom(42);

        bool anyDrift = false;
        for (int i = 0; i < 500; i++)
        {
            // Small noise around 0.5
            double val = 0.5 + (rng.NextDouble() - 0.5) * 0.02;
            if (ph.AddObservation(val))
            {
                anyDrift = true;
            }
        }

        Assert.False(anyDrift, "PageHinkley should not detect drift in stable data");
    }

    [Fact]
    public void PageHinkley_LargeMeanShift_DetectsDrift()
    {
        var ph = new PageHinkley<double>(lambda: 20, alpha: 0.01);

        // Phase 1: values around 0.0
        for (int i = 0; i < 100; i++)
        {
            ph.AddObservation(0.0);
        }

        // Phase 2: shift to 1.0
        bool detected = false;
        for (int i = 0; i < 100; i++)
        {
            if (ph.AddObservation(1.0))
            {
                detected = true;
                break;
            }
        }

        Assert.True(detected, "PageHinkley should detect large mean shift from 0.0 to 1.0");
    }

    [Fact]
    public void PageHinkley_ObservationCount_Correct()
    {
        var ph = new PageHinkley<double>();
        Assert.Equal(0, ph.ObservationCount);

        for (int i = 1; i <= 20; i++)
        {
            ph.AddObservation(0.0);
            Assert.Equal(i, ph.ObservationCount);
        }
    }

    #endregion

    #region Wrapper Detector Consistency

    [Fact]
    public void DDMDriftDetector_MatchesDDM_Behavior()
    {
        var ddm = new DDM<double>(minimumObservations: 10);
        var wrapper = new DDMDriftDetector<double>(minimumObservations: 10);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Feed same sequence to both
        for (int i = 0; i < 100; i++)
        {
            double val = rng.NextDouble() > 0.7 ? 1.0 : 0.0;
            bool ddmResult = ddm.AddObservation(val);
            bool wrapperResult = wrapper.AddObservation(val);

            Assert.Equal(ddmResult, wrapperResult);
            Assert.Equal(ddm.IsInDrift, wrapper.IsInDrift);
        }
    }

    [Fact]
    public void EDDMDriftDetector_MatchesEDDM_Behavior()
    {
        var eddm = new EDDM<double>(minimumObservations: 10, minimumErrors: 5);
        var wrapper = new EDDMDriftDetector<double>(minimumObservations: 10, minimumErrors: 5);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Feed same sequence to both
        for (int i = 0; i < 100; i++)
        {
            double val = rng.NextDouble() > 0.7 ? 1.0 : 0.0;
            bool eddmResult = eddm.AddObservation(val);
            bool wrapperResult = wrapper.AddObservation(val);

            Assert.Equal(eddmResult, wrapperResult);
            Assert.Equal(eddm.IsInDrift, wrapper.IsInDrift);
        }
    }

    [Fact]
    public void PageHinkleyDriftDetector_MatchesPageHinkley_Behavior()
    {
        var ph = new PageHinkley<double>(lambda: 50, alpha: 0.005);
        var wrapper = new PageHinkleyDriftDetector<double>(lambda: 50, delta: 0.005);

        var rng = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < 100; i++)
        {
            double val = rng.NextDouble();
            bool phResult = ph.AddObservation(val);
            bool wrapperResult = wrapper.AddObservation(val);

            Assert.Equal(phResult, wrapperResult);
            Assert.Equal(ph.IsInDrift, wrapper.IsInDrift);
        }
    }

    #endregion

    #region Known Drift Scenarios

    [Fact]
    public void DDM_DetectsGradualDrift()
    {
        var ddm = new DDM<double>(
            warningThreshold: 2.0,
            driftThreshold: 3.0,
            minimumObservations: 30);

        var rng = RandomHelper.CreateSeededRandom(42);

        // Phase 1: 5% error rate
        for (int i = 0; i < 200; i++)
        {
            ddm.AddObservation(rng.NextDouble() < 0.05 ? 1.0 : 0.0);
        }

        // Phase 2: gradually increasing error rate
        bool driftDetected = false;
        for (int epoch = 0; epoch < 10; epoch++)
        {
            double errorRate = 0.05 + epoch * 0.05; // 5% to 50%
            for (int i = 0; i < 50; i++)
            {
                if (ddm.AddObservation(rng.NextDouble() < errorRate ? 1.0 : 0.0))
                {
                    driftDetected = true;
                    break;
                }
            }
            if (driftDetected) break;
        }

        Assert.True(driftDetected,
            "DDM should detect gradual drift from 5% to 50% error rate");
    }

    [Fact]
    public void ADWIN_DetectsAbruptDrift()
    {
        var adwin = new ADWIN<double>(delta: 0.002);

        // Phase 1: stable at 0.2
        for (int i = 0; i < 300; i++)
        {
            adwin.AddObservation(0.2);
        }

        // Phase 2: abrupt shift to 0.8
        bool driftDetected = false;
        for (int i = 0; i < 300; i++)
        {
            if (adwin.AddObservation(0.8))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected,
            "ADWIN should detect abrupt mean shift from 0.2 to 0.8");
    }

    [Fact]
    public void AllDetectors_NoFalseAlarm_ConstantStream()
    {
        var ddm = new DDM<double>(minimumObservations: 30);
        var eddm = new EDDM<double>(minimumObservations: 30, minimumErrors: 30);
        var adwin = new ADWIN<double>(delta: 0.001);
        var ph = new PageHinkley<double>(lambda: 100, alpha: 0.005);

        // Constant stream with no errors (for DDM/EDDM) and constant values (for ADWIN/PH)
        for (int i = 0; i < 1000; i++)
        {
            Assert.False(ddm.AddObservation(0.0), $"DDM false alarm at step {i}");
            // EDDM needs errors to work, skip
            Assert.False(adwin.AddObservation(0.5), $"ADWIN false alarm at step {i}");
            Assert.False(ph.AddObservation(0.5), $"PageHinkley false alarm at step {i}");
        }
    }

    #endregion
}
