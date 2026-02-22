using AiDotNet.DriftDetection;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DriftDetection;

/// <summary>
/// Integration tests for drift detection classes: DDM, EDDM, ADWIN, PageHinkley, and their wrappers.
/// </summary>
public class DriftDetectionIntegrationTests
{
    #region DDM Tests

    [Fact]
    public void DDM_StableStream_NoDrift()
    {
        var ddm = new DDM<double>();
        // Feed stable low-error stream
        for (int i = 0; i < 200; i++)
        {
            ddm.AddObservation(0.0); // Correct predictions
        }

        Assert.False(ddm.IsInDrift);
        Assert.False(ddm.IsInWarning);
    }

    [Fact]
    public void DDM_HighErrorStream_DetectsDrift()
    {
        var ddm = new DDM<double>(warningThreshold: 2.0, driftThreshold: 3.0, minimumObservations: 30);

        // Stable period with low errors
        for (int i = 0; i < 100; i++)
        {
            ddm.AddObservation(i % 20 == 0 ? 1.0 : 0.0); // ~5% error rate
        }

        // Drift period: high errors
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            if (ddm.AddObservation(1.0)) // All errors
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected || ddm.IsInDrift);
    }

    [Fact]
    public void DDM_Reset_ClearsState()
    {
        var ddm = new DDM<double>();
        for (int i = 0; i < 50; i++)
            ddm.AddObservation(1.0);

        ddm.Reset();
        Assert.False(ddm.IsInDrift);
        Assert.False(ddm.IsInWarning);
        Assert.Equal(0, ddm.ObservationCount);
    }

    [Fact]
    public void DDM_GetErrorRate_IsAccurate()
    {
        var ddm = new DDM<double>();
        // 20 errors out of 100 observations
        for (int i = 0; i < 100; i++)
        {
            ddm.AddObservation(i < 20 ? 1.0 : 0.0);
        }

        Assert.Equal(0.20, ddm.GetErrorRate(), 0.01);
    }

    [Fact]
    public void DDM_GetMinimumPsi_ReturnsValue()
    {
        var ddm = new DDM<double>();
        for (int i = 0; i < 50; i++)
            ddm.AddObservation(0.0);

        var minPsi = ddm.GetMinimumPsi();
        Assert.True(minPsi >= 0);
    }

    [Fact]
    public void DDM_NegativeWarningThreshold_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DDM<double>(warningThreshold: -1.0));
    }

    [Fact]
    public void DDM_DriftLessThanWarning_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DDM<double>(warningThreshold: 3.0, driftThreshold: 2.0));
    }

    [Fact]
    public void DDM_Properties_SetCorrectly()
    {
        var ddm = new DDM<double>(warningThreshold: 1.5, driftThreshold: 2.5, minimumObservations: 50, warningDelay: 200);
        Assert.Equal(1.5, ddm.WarningThreshold);
        Assert.Equal(2.5, ddm.DriftThreshold);
        Assert.Equal(200, ddm.WarningDelay);
    }

    #endregion

    #region EDDM Tests

    [Fact]
    public void EDDM_StableStream_NoDrift()
    {
        var eddm = new EDDM<double>();
        for (int i = 0; i < 200; i++)
        {
            eddm.AddObservation(0.0);
        }

        Assert.False(eddm.IsInDrift);
    }

    [Fact]
    public void EDDM_HighErrorStream_DetectsDrift()
    {
        var eddm = new EDDM<double>();

        // Stable period with sparse errors
        for (int i = 0; i < 100; i++)
        {
            eddm.AddObservation(i % 20 == 0 ? 1.0 : 0.0);
        }

        // Sudden high error rate
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            if (eddm.AddObservation(i % 2 == 0 ? 1.0 : 0.0)) // 50% error rate
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected || eddm.IsInDrift || eddm.IsInWarning);
    }

    [Fact]
    public void EDDM_Reset_ClearsState()
    {
        var eddm = new EDDM<double>();
        for (int i = 0; i < 50; i++)
            eddm.AddObservation(1.0);

        eddm.Reset();
        Assert.False(eddm.IsInDrift);
        Assert.Equal(0, eddm.ObservationCount);
    }

    #endregion

    #region ADWIN Tests

    [Fact]
    public void ADWIN_StableStream_NoDrift()
    {
        var adwin = new ADWIN<double>();
        for (int i = 0; i < 200; i++)
        {
            adwin.AddObservation(0.0);
        }

        Assert.False(adwin.IsInDrift);
    }

    [Fact]
    public void ADWIN_MeanShift_DetectsDrift()
    {
        var adwin = new ADWIN<double>();

        // Stable period with mean ~0
        for (int i = 0; i < 200; i++)
        {
            adwin.AddObservation(0.0);
        }

        // Shift to mean ~1
        bool driftDetected = false;
        for (int i = 0; i < 200; i++)
        {
            if (adwin.AddObservation(1.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected || adwin.IsInDrift);
    }

    [Fact]
    public void ADWIN_Reset_ClearsState()
    {
        var adwin = new ADWIN<double>();
        for (int i = 0; i < 50; i++)
            adwin.AddObservation(1.0);

        adwin.Reset();
        Assert.False(adwin.IsInDrift);
        Assert.Equal(0, adwin.ObservationCount);
    }

    #endregion

    #region PageHinkley Tests

    [Fact]
    public void PageHinkley_StableStream_NoDrift()
    {
        var ph = new PageHinkley<double>();
        for (int i = 0; i < 200; i++)
        {
            ph.AddObservation(0.0);
        }

        Assert.False(ph.IsInDrift);
    }

    [Fact]
    public void PageHinkley_MeanShift_DetectsDrift()
    {
        var ph = new PageHinkley<double>();

        // Stable period
        for (int i = 0; i < 100; i++)
            ph.AddObservation(0.0);

        // Shift
        bool driftDetected = false;
        for (int i = 0; i < 300; i++)
        {
            if (ph.AddObservation(5.0))
            {
                driftDetected = true;
                break;
            }
        }

        Assert.True(driftDetected || ph.IsInDrift);
    }

    [Fact]
    public void PageHinkley_Reset_ClearsState()
    {
        var ph = new PageHinkley<double>();
        for (int i = 0; i < 50; i++)
            ph.AddObservation(1.0);

        ph.Reset();
        Assert.False(ph.IsInDrift);
        Assert.Equal(0, ph.ObservationCount);
    }

    #endregion

    #region Wrapper Tests

    [Fact]
    public void DDMDriftDetector_BehavesLikeDDM()
    {
        var detector = new DDMDriftDetector<double>();
        for (int i = 0; i < 50; i++)
            detector.AddObservation(0.0);

        Assert.False(detector.IsInDrift);
        detector.Reset();
        Assert.Equal(0, detector.ObservationCount);
    }

    [Fact]
    public void EDDMDriftDetector_BehavesLikeEDDM()
    {
        var detector = new EDDMDriftDetector<double>();
        for (int i = 0; i < 50; i++)
            detector.AddObservation(0.0);

        Assert.False(detector.IsInDrift);
        detector.Reset();
        Assert.Equal(0, detector.ObservationCount);
    }

    [Fact]
    public void PageHinkleyDriftDetector_BehavesLikePageHinkley()
    {
        var detector = new PageHinkleyDriftDetector<double>();
        for (int i = 0; i < 50; i++)
            detector.AddObservation(0.0);

        Assert.False(detector.IsInDrift);
        detector.Reset();
        Assert.Equal(0, detector.ObservationCount);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllDetectors_ObservationCountIncreases()
    {
        IDriftDetector<double>[] detectors =
        [
            new DDM<double>(),
            new EDDM<double>(),
            new ADWIN<double>(),
            new PageHinkley<double>(),
        ];

        foreach (var detector in detectors)
        {
            for (int i = 0; i < 10; i++)
                detector.AddObservation(0.0);

            Assert.Equal(10, detector.ObservationCount);
        }
    }

    #endregion
}
