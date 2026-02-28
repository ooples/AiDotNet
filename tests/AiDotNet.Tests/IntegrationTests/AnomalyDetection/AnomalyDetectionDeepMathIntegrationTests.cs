using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Deep math-correctness integration tests for statistical anomaly detectors.
/// Every expected value is hand-calculated from the source code formulas.
/// </summary>
public class AnomalyDetectionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double MediumTolerance = 1e-4;

    private static Matrix<double> ToMatrix1D(double[] values)
    {
        var data = new double[values.Length, 1];
        for (int i = 0; i < values.Length; i++)
            data[i, 0] = values[i];
        return new Matrix<double>(data);
    }

    private static Matrix<double> ToMatrix(double[,] data) => new Matrix<double>(data);

    #region ZScoreDetector Tests

    [Fact]
    public void ZScore_HandCalculated_SymmetricData_ScoresMatchFormula()
    {
        // Data: [1, 3, 5, 7, 9] symmetric around 5
        // mean = 25/5 = 5.0
        // sum_sq = 1+9+25+49+81 = 165
        // pop_var = 165/5 - 25 = 33 - 25 = 8
        // pop_std = sqrt(8)
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new ZScoreDetector<double>(zThreshold: 3.0, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double std = Math.Sqrt(8.0);

        Assert.Equal(4.0 / std, scores[0], Tolerance);  // |1-5|/sqrt(8)
        Assert.Equal(2.0 / std, scores[1], Tolerance);  // |3-5|/sqrt(8)
        Assert.Equal(0.0, scores[2], Tolerance);          // |5-5|/sqrt(8)
        Assert.Equal(2.0 / std, scores[3], Tolerance);  // |7-5|/sqrt(8)
        Assert.Equal(4.0 / std, scores[4], Tolerance);  // |9-5|/sqrt(8)
    }

    [Fact]
    public void ZScore_UsesPopulationStd_NotSampleStd()
    {
        // Data: [2, 4, 6]
        // mean = 4, pop_var = (4+16+36)/3 - 16 = 8/3, pop_std = sqrt(8/3)
        // sample_std would be 2.0 (different value)
        var data = ToMatrix1D([2, 4, 6]);
        var detector = new ZScoreDetector<double>(zThreshold: 3.0, contamination: 0.4);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double popStd = Math.Sqrt(8.0 / 3.0);

        // pop_std: z(2) = 2/1.633 ≈ 1.2247
        // sample_std: z(2) = 2/2.0 = 1.0 (different!)
        Assert.Equal(2.0 / popStd, scores[0], Tolerance);
        Assert.Equal(0.0, scores[1], Tolerance);
        Assert.Equal(2.0 / popStd, scores[2], Tolerance);
    }

    [Fact]
    public void ZScore_FittedStatistics_MatchExpected()
    {
        var data = ToMatrix1D([10, 20, 30, 40, 50]);
        var detector = new ZScoreDetector<double>(contamination: 0.3);
        detector.Fit(data);

        // mean = 150/5 = 30
        Assert.Equal(30.0, detector.Means![0], Tolerance);

        // pop_var = (100+400+900+1600+2500)/5 - 900 = 1100 - 900 = 200
        Assert.Equal(Math.Sqrt(200.0), detector.StandardDeviations![0], Tolerance);
    }

    [Fact]
    public void ZScore_ConstantFeature_SkippedInScoring()
    {
        var data = ToMatrix(new double[,]
        {
            { 1, 5 },
            { 3, 5 },
            { 5, 5 },
            { 7, 5 },
            { 9, 5 }
        });
        var detector = new ZScoreDetector<double>(zThreshold: 3.0, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double std1 = Math.Sqrt(8.0);

        // Feature 2 std=0, skipped. Scores only reflect Feature 1
        Assert.Equal(4.0 / std1, scores[0], Tolerance);
        Assert.Equal(0.0, scores[2], Tolerance);
    }

    [Fact]
    public void ZScore_MultiFeature_UsesMaxAbsZAcrossFeatures()
    {
        // Feature 1: constant (skipped), Feature 2: [0,0,0,0,100]
        // Feature 2: mean=20, pop_var=(0+0+0+0+10000)/5-400=1600, std=40
        var data = ToMatrix(new double[,]
        {
            { 1, 0 },
            { 1, 0 },
            { 1, 0 },
            { 1, 0 },
            { 1, 100 }
        });
        var detector = new ZScoreDetector<double>(zThreshold: 3.0, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        // Feature 2: z(0) = |0-20|/40 = 0.5, z(100) = |100-20|/40 = 2.0
        Assert.Equal(0.5, scores[0], Tolerance);
        Assert.Equal(2.0, scores[4], Tolerance);
    }

    [Fact]
    public void ZScore_Predict_UsesZThreshold_NotContamination()
    {
        // Data: [0,0,0,0,0,0,0,0,0,100]
        // mean=10, pop_var=(10000)/10-100=900, std=30
        // z(100) = 90/30 = 3.0, z(0) = 10/30 = 0.333
        var data = ToMatrix1D([0, 0, 0, 0, 0, 0, 0, 0, 0, 100]);
        var detector = new ZScoreDetector<double>(zThreshold: 2.5, contamination: 0.3);
        detector.Fit(data);

        var predictions = detector.Predict(data);

        // z(100)=3.0 > 2.5 → anomaly (-1)
        Assert.Equal(-1.0, predictions[9], Tolerance);
        // z(0)=0.333 < 2.5 → inlier (1)
        Assert.Equal(1.0, predictions[0], Tolerance);
    }

    [Fact]
    public void ZScore_SymmetricScores_ForSymmetricData()
    {
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new ZScoreDetector<double>(contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        Assert.Equal(scores[0], scores[4], Tolerance);
        Assert.Equal(scores[1], scores[3], Tolerance);
        Assert.True(scores[0] > scores[1]);
    }

    #endregion

    #region MADDetector Tests

    [Fact]
    public void MAD_HandCalculated_EvenN_MatchesFormula()
    {
        // Data: [1..10], Median=(5+6)/2=5.5
        // |x-5.5|: [4.5,3.5,2.5,1.5,0.5,0.5,1.5,2.5,3.5,4.5]
        // Sorted: [0.5,0.5,1.5,1.5,2.5,2.5,3.5,3.5,4.5,4.5]
        // rawMAD = (2.5+2.5)/2 = 2.5, scaledMAD = 1.4826*2.5 = 3.7065
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        var detector = new MADDetector<double>(madThreshold: 3.5, scaleFactor: 1.4826, contamination: 0.2);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double scaledMAD = 1.4826 * 2.5;

        Assert.Equal(4.5 / scaledMAD, scores[0], Tolerance);
        Assert.Equal(0.5 / scaledMAD, scores[4], Tolerance);
        Assert.Equal(4.5 / scaledMAD, scores[9], Tolerance);
    }

    [Fact]
    public void MAD_HandCalculated_OddN_MatchesFormula()
    {
        // Data: [1,3,5,7,9], Median=5
        // |x-5|: [4,2,0,2,4], sorted: [0,2,2,4,4], rawMAD=2
        // scaledMAD = 1.4826*2 = 2.9652
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new MADDetector<double>(madThreshold: 3.5, scaleFactor: 1.4826, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double scaledMAD = 1.4826 * 2.0;

        Assert.Equal(4.0 / scaledMAD, scores[0], Tolerance);
        Assert.Equal(2.0 / scaledMAD, scores[1], Tolerance);
        Assert.Equal(0.0 / scaledMAD, scores[2], Tolerance);
        Assert.Equal(2.0 / scaledMAD, scores[3], Tolerance);
        Assert.Equal(4.0 / scaledMAD, scores[4], Tolerance);
    }

    [Fact]
    public void MAD_CustomScaleFactor_UsedCorrectly()
    {
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new MADDetector<double>(madThreshold: 3.5, scaleFactor: 1.0, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        // scaledMAD = 1.0 * 2.0 = 2.0
        Assert.Equal(4.0 / 2.0, scores[0], Tolerance);
        Assert.Equal(2.0 / 2.0, scores[1], Tolerance);
        Assert.Equal(0.0, scores[2], Tolerance);
    }

    [Fact]
    public void MAD_ConstantData_ScoresZero()
    {
        // All values=5 → MAD=0 clamped to 1e-10, but deviation=0 so score=0
        var data = ToMatrix1D([5, 5, 5, 5, 5]);
        var detector = new MADDetector<double>(contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, scores[i], Tolerance);
    }

    [Fact]
    public void MAD_RobustToOutliers_MedianAndMADUnchanged()
    {
        // Clean: [1..9], Median=5, rawMAD=2
        var dataClean = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var detClean = new MADDetector<double>(contamination: 0.2);
        detClean.Fit(dataClean);
        var scoresClean = detClean.ScoreAnomalies(dataClean);

        // Outlier: [1,2,3,4,5,6,7,8,1000], Median=5, rawMAD=2 (unchanged!)
        var dataOutlier = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 1000]);
        var detOutlier = new MADDetector<double>(contamination: 0.2);
        detOutlier.Fit(dataOutlier);
        var scoresOutlier = detOutlier.ScoreAnomalies(dataOutlier);

        // Scores for points 1-8 identical (median and MAD unchanged by outlier)
        for (int i = 0; i < 8; i++)
            Assert.Equal(scoresClean[i], scoresOutlier[i], Tolerance);
    }

    [Fact]
    public void MAD_SymmetricScores_ForSymmetricData()
    {
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new MADDetector<double>(contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(scores[0], scores[4], Tolerance);
        Assert.Equal(scores[1], scores[3], Tolerance);
    }

    #endregion

    #region ModifiedZScoreDetector Tests

    [Fact]
    public void ModifiedZScore_HandCalculated_MatchesFormula()
    {
        // Data: [1,3,5,7,9], Median=5, rawMAD=2
        // ModifiedZ(x) = 0.6745 * |x-5| / 2
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new ModifiedZScoreDetector<double>(modifiedZThreshold: 3.5, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double k = 0.6745;
        double rawMAD = 2.0;

        Assert.Equal(k * 4.0 / rawMAD, scores[0], Tolerance);
        Assert.Equal(k * 2.0 / rawMAD, scores[1], Tolerance);
        Assert.Equal(0.0, scores[2], Tolerance);
        Assert.Equal(k * 2.0 / rawMAD, scores[3], Tolerance);
        Assert.Equal(k * 4.0 / rawMAD, scores[4], Tolerance);
    }

    [Fact]
    public void ModifiedZScore_FittedStatistics_MatchExpected()
    {
        var data = ToMatrix1D([1, 3, 5, 7, 9]);
        var detector = new ModifiedZScoreDetector<double>(contamination: 0.3);
        detector.Fit(data);

        Assert.Equal(5.0, detector.Medians![0], Tolerance);
        Assert.Equal(2.0, detector.MADs![0], Tolerance);
    }

    [Fact]
    public void ModifiedZScore_ConstantFeature_Skipped()
    {
        var data = ToMatrix(new double[,]
        {
            { 1, 5 },
            { 3, 5 },
            { 5, 5 },
            { 7, 5 },
            { 9, 5 }
        });
        var detector = new ModifiedZScoreDetector<double>(contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double k = 0.6745;
        double rawMAD = 2.0;

        Assert.Equal(k * 4.0 / rawMAD, scores[0], Tolerance);
        Assert.Equal(0.0, scores[2], Tolerance);
    }

    [Fact]
    public void ModifiedZScore_Predict_UsesModifiedZThreshold()
    {
        // Data: [1..9, 100], Median=5.5, rawMAD=2.5
        // ModifiedZ(100) = 0.6745*94.5/2.5 = 25.496 >> 3.5
        // ModifiedZ(5) = 0.6745*0.5/2.5 = 0.135 << 3.5
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
        var detector = new ModifiedZScoreDetector<double>(modifiedZThreshold: 3.5, contamination: 0.2);
        detector.Fit(data);

        var predictions = detector.Predict(data);

        Assert.Equal(-1.0, predictions[9], Tolerance); // anomaly
        Assert.Equal(1.0, predictions[4], Tolerance);   // inlier
    }

    #endregion

    #region IQRDetector Tests

    [Fact]
    public void IQR_HandCalculated_QuartilesAndBounds_MatchFormula()
    {
        // Data: [2,4,6,8,10,12,14,16,50] (sorted, n=9)
        // Q1: pos=(9-1)*0.25=2.0 → index=2, frac=0 → Q1=6
        // Q3: pos=(9-1)*0.75=6.0 → index=6, frac=0 → Q3=14
        // IQR=8, k=1.5: LB=6-12=-6, UB=14+12=26
        var data = ToMatrix1D([2, 4, 6, 8, 10, 12, 14, 16, 50]);
        var detector = new IQRDetector<double>(multiplier: 1.5, contamination: 0.2);
        detector.Fit(data);

        Assert.Equal(6.0, detector.Q1![0], Tolerance);
        Assert.Equal(14.0, detector.Q3![0], Tolerance);
        Assert.Equal(8.0, detector.IQR![0], Tolerance);
        Assert.Equal(-6.0, detector.LowerBounds![0], Tolerance);
        Assert.Equal(26.0, detector.UpperBounds![0], Tolerance);
    }

    [Fact]
    public void IQR_PointsWithinBounds_ScoreIsZero()
    {
        var data = ToMatrix1D([2, 4, 6, 8, 10, 12, 14, 16, 50]);
        var detector = new IQRDetector<double>(multiplier: 1.5, contamination: 0.2);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        // Values 2-16 within [-6, 26] → score=0
        for (int i = 0; i < 8; i++)
            Assert.Equal(0.0, scores[i], Tolerance);
    }

    [Fact]
    public void IQR_OutlierScore_MatchesNormalizedDeviation()
    {
        var data = ToMatrix1D([2, 4, 6, 8, 10, 12, 14, 16, 50]);
        var detector = new IQRDetector<double>(multiplier: 1.5, contamination: 0.2);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        // 50 > 26 → score = (50-26)/8 = 3.0
        Assert.Equal(3.0, scores[8], Tolerance);
    }

    [Fact]
    public void IQR_QuartileInterpolation_FractionalPosition()
    {
        // Data: [1,2,3,4,5,6,7,8] (n=8)
        // Q1: pos=(8-1)*0.25=1.75 → sorted[1]*0.25+sorted[2]*0.75 = 2*0.25+3*0.75 = 2.75
        // Q3: pos=(8-1)*0.75=5.25 → sorted[5]*0.75+sorted[6]*0.25 = 6*0.75+7*0.25 = 6.25
        // IQR = 3.5
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8]);
        var detector = new IQRDetector<double>(multiplier: 1.5, contamination: 0.2);
        detector.Fit(data);

        Assert.Equal(2.75, detector.Q1![0], Tolerance);
        Assert.Equal(6.25, detector.Q3![0], Tolerance);
        Assert.Equal(3.5, detector.IQR![0], Tolerance);
    }

    [Fact]
    public void IQR_Multiplier3_WiderBounds()
    {
        // Same data, k=3: LB=6-3*8=-18, UB=14+3*8=38
        // score(50) = (50-38)/8 = 1.5
        var data = ToMatrix1D([2, 4, 6, 8, 10, 12, 14, 16, 50]);
        var detector = new IQRDetector<double>(multiplier: 3.0, contamination: 0.2);
        detector.Fit(data);

        Assert.Equal(-18.0, detector.LowerBounds![0], Tolerance);
        Assert.Equal(38.0, detector.UpperBounds![0], Tolerance);

        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(1.5, scores[8], Tolerance);
    }

    [Fact]
    public void IQR_ConstantFeature_Skipped()
    {
        var data = ToMatrix(new double[,]
        {
            { 1, 5 },
            { 2, 5 },
            { 3, 5 },
            { 100, 5 }
        });
        var detector = new IQRDetector<double>(multiplier: 1.5, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        // Feature 2 IQR=0, skipped. Only Feature 1 contributes.
        // Feature 1 outlier (100) must have a positive score
        Assert.True(scores[3] > 0);
    }

    #endregion

    #region PercentileDetector Tests

    [Fact]
    public void Percentile_HandCalculated_ThresholdsMatchFormula()
    {
        // Data: [1..10], P5: idx=(5/100)*9=0.45 → 1+0.45*1=1.45
        // P95: idx=(95/100)*9=8.55 → 9+0.55*1=9.55, Range=8.1
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        var detector = new PercentileDetector<double>(lowPercentile: 5, highPercentile: 95, contamination: 0.2);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        // Values 2-9 within [1.45, 9.55] → score=0
        for (int i = 1; i < 9; i++)
            Assert.Equal(0.0, scores[i], Tolerance);

        // Value 1 < 1.45 → score = 0.45/8.1
        Assert.Equal(0.45 / 8.1, scores[0], Tolerance);
        // Value 10 > 9.55 → score = 0.45/8.1
        Assert.Equal(0.45 / 8.1, scores[9], Tolerance);
    }

    [Fact]
    public void Percentile_CustomPercentiles_MatchFormula()
    {
        // P10: idx=0.9 → 1*0.1+2*0.9=1.9
        // P90: idx=8.1 → 9*0.9+10*0.1=9.1, Range=7.2
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        var detector = new PercentileDetector<double>(lowPercentile: 10, highPercentile: 90, contamination: 0.3);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);

        Assert.Equal(0.9 / 7.2, scores[0], Tolerance);
        Assert.Equal(0.9 / 7.2, scores[9], Tolerance);
    }

    [Fact]
    public void Percentile_WithinBounds_ScoreIsZero()
    {
        // P0=1, P100=10 → all values within bounds
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        var detector = new PercentileDetector<double>(lowPercentile: 0, highPercentile: 100, contamination: 0.2);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        for (int i = 0; i < 10; i++)
            Assert.Equal(0.0, scores[i], Tolerance);
    }

    [Fact]
    public void Percentile_OutlierScore_ProportionalToDistance()
    {
        // Data: [1..9, 100]
        // P95: idx=8.55 → 9+0.55*91=59.05, P5: 1+0.45*1=1.45, Range=57.6
        // score(100) = (100-59.05)/57.6
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
        var detector = new PercentileDetector<double>(lowPercentile: 5, highPercentile: 95, contamination: 0.2);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(40.95 / 57.6, scores[9], MediumTolerance);
    }

    #endregion

    #region GrubbsTestDetector Tests

    [Fact]
    public void Grubbs_ScoresIdenticalToZScores()
    {
        // G = |x-mean|/std, same formula as Z-score
        var data = ToMatrix1D([1, 3, 5, 7, 9]);

        var grubbs = new GrubbsTestDetector<double>(contamination: 0.3);
        grubbs.Fit(data);
        var gScores = grubbs.ScoreAnomalies(data);

        var zscore = new ZScoreDetector<double>(contamination: 0.3);
        zscore.Fit(data);
        var zScores = zscore.ScoreAnomalies(data);

        for (int i = 0; i < 5; i++)
            Assert.Equal(gScores[i], zScores[i], Tolerance);
    }

    [Fact]
    public void Grubbs_HandCalculated_MatchesFormula()
    {
        // Data: [2,4,6], mean=4, pop_std=sqrt(8/3)
        var data = ToMatrix1D([2, 4, 6]);
        var detector = new GrubbsTestDetector<double>(contamination: 0.4);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double std = Math.Sqrt(8.0 / 3.0);

        Assert.Equal(2.0 / std, scores[0], Tolerance);
        Assert.Equal(0.0, scores[1], Tolerance);
        Assert.Equal(2.0 / std, scores[2], Tolerance);
    }

    [Fact]
    public void Grubbs_MinimumSamples_RequiresAtLeast3()
    {
        var data2 = ToMatrix1D([1, 2]);
        var detector = new GrubbsTestDetector<double>();
        Assert.Throws<ArgumentException>(() => detector.Fit(data2));
    }

    [Fact]
    public void Grubbs_ConstantFeature_Skipped()
    {
        var data = ToMatrix(new double[,]
        {
            { 1, 5 },
            { 3, 5 },
            { 5, 5 }
        });
        var detector = new GrubbsTestDetector<double>(contamination: 0.4);
        detector.Fit(data);

        var scores = detector.ScoreAnomalies(data);
        double std = Math.Sqrt(8.0 / 3.0);

        Assert.Equal(2.0 / std, scores[0], Tolerance);
    }

    #endregion

    #region Cross-Detector Consistency Tests

    [Fact]
    public void MAD_And_ModifiedZ_ProduceSameScores()
    {
        // MAD score = |x-med|/(1.4826*rawMAD)
        // ModifiedZ = 0.6745*|x-med|/rawMAD = |x-med|/(rawMAD*1.4826...)
        // Since 1/0.6745 ≈ 1.4826, scores are nearly identical
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        var madDet = new MADDetector<double>(madThreshold: 3.5, scaleFactor: 1.4826, contamination: 0.2);
        madDet.Fit(data);
        var madScores = madDet.ScoreAnomalies(data);

        var modZDet = new ModifiedZScoreDetector<double>(modifiedZThreshold: 3.5, contamination: 0.2);
        modZDet.Fit(data);
        var modZScores = modZDet.ScoreAnomalies(data);

        for (int i = 0; i < 10; i++)
            Assert.Equal(madScores[i], modZScores[i], MediumTolerance);
    }

    [Fact]
    public void ZScore_And_Grubbs_ProduceSameScores()
    {
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        var zDet = new ZScoreDetector<double>(contamination: 0.2);
        zDet.Fit(data);
        var zScores = zDet.ScoreAnomalies(data);

        var gDet = new GrubbsTestDetector<double>(contamination: 0.2);
        gDet.Fit(data);
        var gScores = gDet.ScoreAnomalies(data);

        for (int i = 0; i < 10; i++)
            Assert.Equal(zScores[i], gScores[i], Tolerance);
    }

    [Fact]
    public void AllDetectors_ConstantData_ScoresZero()
    {
        var data = ToMatrix1D([5, 5, 5, 5, 5]);

        var z = new ZScoreDetector<double>(contamination: 0.3);
        z.Fit(data);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, z.ScoreAnomalies(data)[i], Tolerance);

        var iqr = new IQRDetector<double>(contamination: 0.3);
        iqr.Fit(data);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, iqr.ScoreAnomalies(data)[i], Tolerance);

        var mad = new MADDetector<double>(contamination: 0.3);
        mad.Fit(data);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, mad.ScoreAnomalies(data)[i], Tolerance);

        var modZ = new ModifiedZScoreDetector<double>(contamination: 0.3);
        modZ.Fit(data);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, modZ.ScoreAnomalies(data)[i], Tolerance);

        var pct = new PercentileDetector<double>(contamination: 0.3);
        pct.Fit(data);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, pct.ScoreAnomalies(data)[i], Tolerance);
    }

    [Fact]
    public void AllDetectors_OutlierScoredHigherThanInliers()
    {
        var vals = new double[] { 0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15, 100 };
        var data = ToMatrix1D(vals);

        var detectors = new (string Name, AnomalyDetectorBase<double> Det)[]
        {
            ("ZScore", new ZScoreDetector<double>(contamination: 0.2)),
            ("IQR", new IQRDetector<double>(contamination: 0.2)),
            ("MAD", new MADDetector<double>(contamination: 0.2)),
            ("ModifiedZ", new ModifiedZScoreDetector<double>(contamination: 0.2)),
            ("Percentile", new PercentileDetector<double>(contamination: 0.2)),
        };

        foreach (var (name, det) in detectors)
        {
            det.Fit(data);
            var scores = det.ScoreAnomalies(data);

            double outlierScore = scores[9];
            for (int i = 0; i < 9; i++)
            {
                Assert.True(outlierScore > scores[i],
                    $"{name}: outlier score ({outlierScore:F4}) should be > inlier score ({scores[i]:F4}) at index {i}");
            }
        }
    }

    [Fact]
    public void AllDetectors_RepeatedFit_SameResults()
    {
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        var z = new ZScoreDetector<double>(contamination: 0.2);
        z.Fit(data);
        var scores1 = z.ScoreAnomalies(data);
        z.Fit(data);
        var scores2 = z.ScoreAnomalies(data);

        for (int i = 0; i < 10; i++)
            Assert.Equal(scores1[i], scores2[i], Tolerance);
    }

    #endregion

    #region Edge Cases and Validation Tests

    [Fact]
    public void AllDetectors_NegativeValues_WorkCorrectly()
    {
        var data = ToMatrix1D([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8]);

        var z = new ZScoreDetector<double>(contamination: 0.2);
        z.Fit(data);

        // mean = -10/10 = -1
        Assert.Equal(-1.0, z.Means![0], Tolerance);

        var scores = z.ScoreAnomalies(data);
        // |-10-(-1)| = 9, |8-(-1)| = 9 → same z-score
        Assert.Equal(scores[0], scores[9], Tolerance);
    }

    [Fact]
    public void SetThreshold_FromContamination_CorrectPercentile()
    {
        // contamination=0.2, n=10: numOutliers=Ceiling(2)=2, thresholdIndex=8
        var data = ToMatrix1D([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        var detector = new ZScoreDetector<double>(contamination: 0.2);
        detector.Fit(data);

        double mean = 5.5;
        double popVar = 0;
        for (int i = 1; i <= 10; i++)
            popVar += (i - mean) * (i - mean);
        popVar /= 10;
        double std = Math.Sqrt(popVar);

        // Sorted scores: 0.5/std x2, 1.5/std x2, 2.5/std x2, 3.5/std x2, 4.5/std x2
        // threshold = sortedScores[8] = 4.5/std
        double expectedThreshold = 4.5 / std;
        Assert.Equal(expectedThreshold, detector.Threshold, Tolerance);
    }

    [Fact]
    public void Contamination_Validation_ThrowsForInvalidValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ZScoreDetector<double>(contamination: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ZScoreDetector<double>(contamination: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ZScoreDetector<double>(contamination: 0.6));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IQRDetector<double>(contamination: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MADDetector<double>(contamination: -0.1));
    }

    [Fact]
    public void Threshold_Validation_ThrowsForInvalidValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ZScoreDetector<double>(zThreshold: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ZScoreDetector<double>(zThreshold: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IQRDetector<double>(multiplier: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IQRDetector<double>(multiplier: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MADDetector<double>(madThreshold: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MADDetector<double>(scaleFactor: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ModifiedZScoreDetector<double>(modifiedZThreshold: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GrubbsTestDetector<double>(alpha: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GrubbsTestDetector<double>(alpha: 1));
    }

    [Fact]
    public void Percentile_Validation_ThrowsForInvalidValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PercentileDetector<double>(lowPercentile: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PercentileDetector<double>(highPercentile: 101));
        Assert.Throws<ArgumentException>(() => new PercentileDetector<double>(lowPercentile: 90, highPercentile: 10));
    }

    [Fact]
    public void AllDetectors_ScoreBeforeFit_Throws()
    {
        var data = ToMatrix1D([1, 2, 3]);

        Assert.Throws<InvalidOperationException>(() => new ZScoreDetector<double>().ScoreAnomalies(data));
        Assert.Throws<InvalidOperationException>(() => new IQRDetector<double>().ScoreAnomalies(data));
        Assert.Throws<InvalidOperationException>(() => new MADDetector<double>().ScoreAnomalies(data));
        Assert.Throws<InvalidOperationException>(() => new ModifiedZScoreDetector<double>().ScoreAnomalies(data));
        Assert.Throws<InvalidOperationException>(() => new PercentileDetector<double>().ScoreAnomalies(data));
    }

    [Fact]
    public void AllDetectors_PredictBeforeFit_Throws()
    {
        var data = ToMatrix1D([1, 2, 3]);

        Assert.Throws<InvalidOperationException>(() => new ZScoreDetector<double>().Predict(data));
        Assert.Throws<InvalidOperationException>(() => new IQRDetector<double>().Predict(data));
        Assert.Throws<InvalidOperationException>(() => new MADDetector<double>().Predict(data));
        Assert.Throws<InvalidOperationException>(() => new ModifiedZScoreDetector<double>().Predict(data));
        Assert.Throws<InvalidOperationException>(() => new PercentileDetector<double>().Predict(data));
    }

    [Fact]
    public void AllDetectors_IsFitted_TrueAfterFit()
    {
        var data = ToMatrix1D([1, 2, 3, 4, 5]);

        var z = new ZScoreDetector<double>(contamination: 0.3);
        Assert.False(z.IsFitted);
        z.Fit(data);
        Assert.True(z.IsFitted);

        var iqr = new IQRDetector<double>(contamination: 0.3);
        Assert.False(iqr.IsFitted);
        iqr.Fit(data);
        Assert.True(iqr.IsFitted);

        var mad = new MADDetector<double>(contamination: 0.3);
        Assert.False(mad.IsFitted);
        mad.Fit(data);
        Assert.True(mad.IsFitted);
    }

    [Fact]
    public void ZScore_ScoresScaleWithOutlierMagnitude()
    {
        var data1 = ToMatrix1D([0, 0, 0, 0, 0, 0, 0, 0, 0, 10]);
        var data2 = ToMatrix1D([0, 0, 0, 0, 0, 0, 0, 0, 0, 100]);

        var d1 = new ZScoreDetector<double>(contamination: 0.2);
        d1.Fit(data1);
        var s1 = d1.ScoreAnomalies(data1);

        var d2 = new ZScoreDetector<double>(contamination: 0.2);
        d2.Fit(data2);
        var s2 = d2.ScoreAnomalies(data2);

        // Outlier always has highest score
        Assert.True(s1[9] > s1[0]);
        Assert.True(s2[9] > s2[0]);
    }

    #endregion
}
