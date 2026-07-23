using AiDotNet.Tensors.Engines.DirectGpu.Profiling;
using Xunit;

namespace AiDotNet.Tests.Profiling;

#if NET8_0_OR_GREATER

/// <summary>
/// Tests for GPU profiling infrastructure.
/// </summary>
/// <remarks>
/// These tests validate the profiling classes that don't require GPU hardware.
/// GPU-specific tests are in GpuRecoveryTests with [Trait("Category", "GPU")].
/// </remarks>
public class GpuProfilingTests
{
    #region RooflineAnalyzer Tests

    [Fact(DisplayName = "RooflineAnalyzer: Ridge Point Calculation")]
    public void RooflineAnalyzer_CalculatesRidgePoint_Correctly()
    {
        // Arrange: RX 5500 XT specs
        double peakGflops = 5196.0;
        double peakBandwidth = 224.0;

        // Act
        var analyzer = new RooflineAnalyzer(peakGflops, peakBandwidth);

        // Assert
        double expectedRidgePoint = peakGflops / peakBandwidth;
        Assert.Equal(expectedRidgePoint, analyzer.RidgePoint, precision: 2);
        Assert.True(analyzer.RidgePoint > 20 && analyzer.RidgePoint < 25,
            $"Ridge point {analyzer.RidgePoint} should be ~23 for RX 5500 XT");
    }

    [Theory(DisplayName = "RooflineAnalyzer: Arithmetic Intensity for GEMM")]
    [InlineData(256, 256, 256, 4)]
    [InlineData(512, 512, 512, 4)]
    [InlineData(1024, 1024, 1024, 4)]
    [InlineData(2048, 2048, 2048, 4)]
    public void RooflineAnalyzer_CalculatesArithmeticIntensity_ForGemm(int m, int n, int k, int bytesPerElement)
    {
        // Act
        double ai = RooflineAnalyzer.CalculateGemmArithmeticIntensity(m, n, k, bytesPerElement);

        // Assert
        // For square GEMM: AI = 2*M*N*K / (4*(M*K + K*N + M*N))
        // Simplified for square: AI = 2*N^3 / (4*3*N^2) = N/6
        double expectedApprox = m / 6.0;
        Assert.True(ai > 0, "Arithmetic intensity must be positive");
        Assert.True(Math.Abs(ai - expectedApprox) < expectedApprox * 0.1,
            $"AI {ai} should be approximately {expectedApprox} for square matrix");
    }

    [Theory(DisplayName = "RooflineAnalyzer: Memory Bound Detection")]
    [InlineData(64, true)]    // Small: AI ~10.7 < ridge ~23, memory bound
    [InlineData(256, false)]  // Medium: AI ~42.7 > ridge ~23, compute bound
    [InlineData(1024, false)] // Large: AI ~170.7 > ridge ~23, compute bound
    public void RooflineAnalyzer_DetectsMemoryBound_BasedOnAI(int size, bool expectedMemoryBound)
    {
        // Arrange
        var analyzer = new RooflineAnalyzer(5196.0, 224.0);
        double ai = RooflineAnalyzer.CalculateGemmArithmeticIntensity(size, size, size);

        // Act
        bool isMemoryBound = analyzer.IsMemoryBound(ai);

        // Assert
        Assert.Equal(expectedMemoryBound, isMemoryBound);
    }

    [Fact(DisplayName = "RooflineAnalyzer: Roofline Limit Calculation")]
    public void RooflineAnalyzer_CalculatesRooflineLimit_Correctly()
    {
        // Arrange
        var analyzer = new RooflineAnalyzer(5196.0, 224.0);

        // Act & Assert
        // Low AI (memory bound): limit = bandwidth * AI
        double lowAI = 5.0;
        double lowLimit = analyzer.GetRooflineLimitGflops(lowAI);
        Assert.Equal(224.0 * 5.0, lowLimit, precision: 1);

        // High AI (compute bound): limit = peak
        double highAI = 100.0;
        double highLimit = analyzer.GetRooflineLimitGflops(highAI);
        Assert.Equal(5196.0, highLimit, precision: 1);
    }

    [Fact(DisplayName = "RooflineAnalyzer: Roofline Curve Generation")]
    public void RooflineAnalyzer_GeneratesRooflineCurve_WithCorrectShape()
    {
        // Arrange
        var analyzer = new RooflineAnalyzer(5196.0, 224.0);

        // Act
        var curve = analyzer.GenerateRooflineCurve(0.1, 100, 20);

        // Assert
        Assert.Equal(20, curve.Length);

        // Verify curve is monotonically non-decreasing
        for (int i = 1; i < curve.Length; i++)
        {
            Assert.True(curve[i].Gflops >= curve[i - 1].Gflops,
                "Roofline curve must be monotonically non-decreasing");
        }

        // Verify it eventually hits peak
        Assert.True(curve[^1].Gflops >= 5196.0 * 0.99,
            "Roofline curve should reach peak at high AI");
    }

    #endregion

    #region OccupancyCalculator Tests

    [Fact(DisplayName = "OccupancyCalculator: RDNA1 Architecture Specs")]
    public void OccupancyCalculator_Rdna1Specs_AreCorrect()
    {
        // Arrange & Act
        var rdna1 = GpuArchitectureSpec.Amd.Rdna1;

        // Assert
        Assert.Equal(GpuArchitectureFamily.AmdRdna1, rdna1.Family);
        Assert.Equal(1024, rdna1.VgprsPerSimd);
        Assert.Equal(65536, rdna1.LdsPerCuBytes);
        Assert.Equal(20, rdna1.MaxWavesPerSimd);
        Assert.Equal(32, rdna1.WavefrontSize);
        Assert.True(rdna1.SupportsWave32);
    }

    [Theory(DisplayName = "OccupancyCalculator: VGPR Limited Occupancy")]
    [InlineData(8, 64, 4)]     // 8 VGPRs/thread: 8*32=256 per wave, 1024/256=4 waves
    [InlineData(16, 64, 2)]    // 16 VGPRs/thread: 16*32=512 per wave, 1024/512=2 waves
    [InlineData(32, 64, 1)]    // 32 VGPRs/thread: 32*32=1024 per wave, 1024/1024=1 wave
    public void OccupancyCalculator_CalculatesVgprLimit_Correctly(int vgprs, int workgroupSize, int expectedWaves)
    {
        // Arrange
        // RDNA1: 1024 VGPRs per SIMD, 32 wavefront size
        // VGPRs per wave = vgprsPerThread * wavefrontSize
        // Max waves = VGPRsPerSimd / VGPRsPerWave
        var arch = GpuArchitectureSpec.Amd.Rdna1;

        // Act
        var result = OccupancyCalculator.Calculate(arch, vgprs, 0, workgroupSize);

        // Assert
        Assert.True(result.VgprLimitedWaves >= expectedWaves - 1 && result.VgprLimitedWaves <= expectedWaves + 1,
            $"Expected ~{expectedWaves} VGPR-limited waves, got {result.VgprLimitedWaves}");
    }

    [Fact(DisplayName = "OccupancyCalculator: LDS Limited Occupancy")]
    public void OccupancyCalculator_CalculatesLdsLimit_Correctly()
    {
        // Arrange
        var arch = GpuArchitectureSpec.Amd.Rdna1;

        // Act - Use 32KB LDS per workgroup (half of 64KB)
        var result = OccupancyCalculator.Calculate(arch, 32, 32768, 64);

        // Assert
        Assert.True(result.LdsLimitedWaves <= arch.MaxWavesPerSimd,
            "LDS-limited waves should not exceed max");
        Assert.True(result.LdsLimitedWaves >= 1,
            "Should fit at least 1 wavefront");
    }

    [Fact(DisplayName = "OccupancyCalculator: Identifies Limiting Factor")]
    public void OccupancyCalculator_IdentifiesLimitingFactor_Correctly()
    {
        // Arrange
        var arch = GpuArchitectureSpec.Amd.Rdna1;

        // Act - High VGPR usage should be VGPR limited
        var highVgpr = OccupancyCalculator.Calculate(arch, 256, 0, 64);

        // Assert
        Assert.Equal(OccupancyLimitingFactor.Vgpr, highVgpr.LimitingFactor);
    }

    [Fact(DisplayName = "OccupancyCalculator: Provides Recommendations")]
    public void OccupancyCalculator_ProvidesRecommendations_ForLowOccupancy()
    {
        // Arrange
        var arch = GpuArchitectureSpec.Amd.Rdna1;

        // Act - Very high VGPR usage for low occupancy
        var result = OccupancyCalculator.Calculate(arch, 256, 0, 64);
        var recommendations = result.GetRecommendations();

        // Assert
        Assert.NotEmpty(recommendations);
        Assert.Contains(recommendations, r => r.Contains("VGPR"));
    }

    [Fact(DisplayName = "OccupancyCalculator: GEMM VGPR Estimation")]
    public void OccupancyCalculator_EstimatesGemmVgprs_Correctly()
    {
        // Arrange: CLBlast typical parameters
        int mwg = 64, nwg = 64, kwg = 16;
        int mdimc = 8, ndimc = 8;

        // Act
        int vgprs = OccupancyCalculator.EstimateGemmVgprs(mwg, nwg, kwg, mdimc, ndimc);

        // Assert
        // Thread tile: (64/8) x (64/8) = 8x8 = 64 accumulators
        // Plus A tile (8) + B tile (8) + misc (8) = ~88 VGPRs
        Assert.True(vgprs >= 70 && vgprs <= 100,
            $"Expected 70-100 VGPRs for 8x8 thread tile, got {vgprs}");
    }

    [Fact(DisplayName = "OccupancyCalculator: GEMM LDS Estimation")]
    public void OccupancyCalculator_EstimatesGemmLds_Correctly()
    {
        // Arrange
        int mwg = 64, nwg = 64, kwg = 16;
        int ldsPad = 1;

        // Act
        int ldsBytes = OccupancyCalculator.EstimateGemmLds(mwg, nwg, kwg, ldsPad);

        // Assert
        // A tile: 64 * (16+1) * 4 = 4352 bytes
        // B tile: 16 * (64+1) * 4 = 4160 bytes
        // Total: ~8512 bytes
        Assert.True(ldsBytes >= 8000 && ldsBytes <= 10000,
            $"Expected 8-10KB LDS for 64x64x16 tiles, got {ldsBytes}");
    }

    #endregion

    #region GpuArchitecture Tests

    [Theory(DisplayName = "GpuArchitecture: Device Name Detection")]
    [InlineData("gfx1012", GpuArchitectureFamily.AmdRdna1)]
    [InlineData("Navi 14 [Radeon RX 5500]", GpuArchitectureFamily.AmdRdna1)]
    [InlineData("gfx1030", GpuArchitectureFamily.AmdRdna2)]
    [InlineData("gfx1100", GpuArchitectureFamily.AmdRdna3)]
    [InlineData("MI100", GpuArchitectureFamily.AmdCdna)]
    [InlineData("Vega 64", GpuArchitectureFamily.AmdGcn)]
    public void GpuArchitecture_DetectsFromDeviceName_Correctly(string deviceName, GpuArchitectureFamily expectedFamily)
    {
        // Act
        var spec = GpuArchitectureSpec.DetectFromDeviceName(deviceName);

        // Assert
        Assert.Equal(expectedFamily, spec.Family);
    }

    [Fact(DisplayName = "GpuArchitecture: MaxWavesPerCu Calculation")]
    public void GpuArchitecture_CalculatesMaxWavesPerCu_Correctly()
    {
        // Arrange
        var rdna1 = GpuArchitectureSpec.Amd.Rdna1;

        // Act
        int maxWavesPerCu = rdna1.MaxWavesPerCu;

        // Assert: 20 waves/SIMD * 2 SIMDs = 40 waves/CU
        Assert.Equal(40, maxWavesPerCu);
    }

    #endregion

    #region ProfileResult Tests

    [Fact(DisplayName = "ProfileResult: Calculates Best Stats")]
    public void ProfileResult_CalculatesBestStats_Correctly()
    {
        // Arrange
        var result = new ProfileResult
        {
            DeviceName = "Test GPU",
            PeakGflops = 5000,
            PeakBandwidthGBs = 200,
            RidgePoint = 25,
            ProfileStartTime = DateTime.Now,
            Entries =
            [
                new GemmProfileEntry { M = 256, N = 256, K = 256, Gflops = 1000, EfficiencyPercent = 20 },
                new GemmProfileEntry { M = 1024, N = 1024, K = 1024, Gflops = 3000, EfficiencyPercent = 60 },
                new GemmProfileEntry { M = 2048, N = 2048, K = 2048, Gflops = 2500, EfficiencyPercent = 50 }
            ]
        };

        // Act & Assert
        Assert.Equal(3000, result.BestGflops);
        Assert.Equal(60, result.BestEfficiencyPercent);
        Assert.Equal((1024, 1024, 1024), result.BestSize);
    }

    [Fact(DisplayName = "ProfileResult: Gets Bottleneck Groups")]
    public void ProfileResult_GetsBottleneckGroups_Correctly()
    {
        // Arrange
        var result = new ProfileResult
        {
            Entries =
            [
                new GemmProfileEntry { M = 128, Bottleneck = BottleneckType.LaunchOverhead },
                new GemmProfileEntry { M = 256, Bottleneck = BottleneckType.LaunchOverhead },
                new GemmProfileEntry { M = 512, Bottleneck = BottleneckType.MemoryBandwidth },
                new GemmProfileEntry { M = 1024, Bottleneck = BottleneckType.Compute }
            ]
        };

        // Act
        var groups = result.GetBottleneckGroups();

        // Assert
        Assert.Equal(3, groups.Count);
        Assert.Equal(2, groups[BottleneckType.LaunchOverhead].Count);
        Assert.Single(groups[BottleneckType.MemoryBandwidth]);
        Assert.Single(groups[BottleneckType.Compute]);
    }

    [Fact(DisplayName = "ProfileResult: Generates Console Report")]
    public void ProfileResult_GeneratesConsoleReport_WithAllSections()
    {
        // Arrange
        var result = new ProfileResult
        {
            DeviceName = "Test GPU",
            Architecture = GpuArchitectureSpec.Amd.Rdna1,
            PeakGflops = 5000,
            PeakBandwidthGBs = 200,
            RidgePoint = 25,
            ProfileStartTime = DateTime.Now,
            ProfileDurationSeconds = 10.5,
            Entries =
            [
                new GemmProfileEntry
                {
                    M = 1024, N = 1024, K = 1024,
                    Gflops = 2500, EfficiencyPercent = 50,
                    MemoryBandwidthGBs = 150, ArithmeticIntensity = 170,
                    Bottleneck = BottleneckType.Compute,
                    RecommendedAction = OptimizationAction.None
                }
            ]
        };

        // Act
        string report = result.GenerateConsoleReport();

        // Assert
        Assert.Contains("GPU GEMM PROFILING REPORT", report);
        Assert.Contains("Test GPU", report);
        Assert.Contains("RDNA1", report);
        Assert.Contains("1024", report);
        Assert.Contains("BOTTLENECK", report);
    }

    #endregion

    #region ProfileExporter Tests

    [Fact(DisplayName = "ProfileExporter: Generates Valid JSON")]
    public void ProfileExporter_GeneratesValidJson()
    {
        // Arrange
        var result = new ProfileResult
        {
            DeviceName = "Test GPU",
            PeakGflops = 5000,
            PeakBandwidthGBs = 200,
            RidgePoint = 25,
            ProfileStartTime = DateTime.Now,
            Entries =
            [
                new GemmProfileEntry { M = 1024, N = 1024, K = 1024, Gflops = 2500 }
            ]
        };

        // Act
        string json = ProfileExporter.ToJson(result);

        // Assert
        Assert.Contains("DeviceName", json);
        Assert.Contains("Test GPU", json);
        Assert.Contains("PeakGflops", json);
        Assert.Contains("5000", json);
    }

    [Fact(DisplayName = "ProfileExporter: Generates Valid CSV")]
    public void ProfileExporter_GeneratesValidCsv()
    {
        // Arrange
        var result = new ProfileResult
        {
            Entries =
            [
                new GemmProfileEntry { M = 256, N = 256, K = 256, Gflops = 1000 },
                new GemmProfileEntry { M = 512, N = 512, K = 512, Gflops = 2000 }
            ]
        };

        // Act
        string csv = ProfileExporter.ToCsv(result);
        var lines = csv.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        // Assert
        Assert.Equal(3, lines.Length); // Header + 2 entries
        Assert.Contains("M,N,K,GFLOPS", lines[0]);
        Assert.Contains("256,256,256", lines[1]);
        Assert.Contains("512,512,512", lines[2]);
    }

    [Fact(DisplayName = "ProfileExporter: Generates Valid Markdown")]
    public void ProfileExporter_GeneratesValidMarkdown()
    {
        // Arrange
        var result = new ProfileResult
        {
            DeviceName = "Test GPU",
            Architecture = GpuArchitectureSpec.Amd.Rdna1,
            PeakGflops = 5000,
            Entries =
            [
                new GemmProfileEntry
                {
                    M = 1024, N = 1024, K = 1024,
                    Gflops = 2500, EfficiencyPercent = 50,
                    Bottleneck = BottleneckType.Compute
                }
            ]
        };

        // Act
        string md = ProfileExporter.ToMarkdown(result);

        // Assert
        Assert.Contains("# GPU GEMM Profiling Report", md);
        Assert.Contains("## Device Information", md);
        Assert.Contains("## Performance Results", md);
        Assert.Contains("| Size |", md);
        Assert.Contains("| 1024 |", md);
    }

    [Fact(DisplayName = "ProfileExporter: Compact Summary")]
    public void ProfileExporter_GeneratesCompactSummary()
    {
        // Arrange
        var result = new ProfileResult
        {
            DeviceName = "Test GPU",
            PeakGflops = 5000,
            PeakBandwidthGBs = 200,
            Entries =
            [
                new GemmProfileEntry { M = 256, Bottleneck = BottleneckType.LaunchOverhead, Gflops = 500 },
                new GemmProfileEntry { M = 1024, Bottleneck = BottleneckType.Compute, Gflops = 3000 },
                new GemmProfileEntry { M = 4096, Bottleneck = BottleneckType.Compute, Gflops = 2500 }
            ]
        };

        // Act
        string summary = ProfileExporter.ToCompactSummary(result);

        // Assert
        Assert.Contains("GPU: Test GPU", summary);
        Assert.Contains("Peak:", summary);
        Assert.Contains("Best:", summary);
    }

    #endregion
}

#endif
