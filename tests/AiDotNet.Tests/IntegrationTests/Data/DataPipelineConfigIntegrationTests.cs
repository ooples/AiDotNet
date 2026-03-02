using AiDotNet.Data.Pipeline;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

/// <summary>
/// Integration tests for data pipeline configuration classes:
/// CacheInfo, DiskCacheOptions, CacheEvictionPolicy.
/// </summary>
public class DataPipelineConfigIntegrationTests
{
    #region CacheInfo

    [Fact]
    public void CacheInfo_DefaultValues()
    {
        var info = new CacheInfo();

        Assert.False(info.IsValid);
        Assert.Equal(0, info.EntryCount);
        Assert.Equal(0L, info.TotalSizeBytes);
        Assert.Equal(string.Empty, info.CacheDirectory);
    }

    [Fact]
    public void CacheInfo_SetProperties()
    {
        var info = new CacheInfo
        {
            IsValid = true,
            EntryCount = 100,
            TotalSizeBytes = 1024 * 1024 * 50, // 50 MB
            CacheDirectory = "/tmp/cache"
        };

        Assert.True(info.IsValid);
        Assert.Equal(100, info.EntryCount);
        Assert.Equal("/tmp/cache", info.CacheDirectory);
    }

    [Fact]
    public void CacheInfo_FormattedSize_Bytes()
    {
        var info = new CacheInfo { TotalSizeBytes = 512 };
        Assert.Equal("512 B", info.FormattedSize);
    }

    [Fact]
    public void CacheInfo_FormattedSize_KB()
    {
        var info = new CacheInfo { TotalSizeBytes = 2048 };
        Assert.Equal("2.0 KB", info.FormattedSize);
    }

    [Fact]
    public void CacheInfo_FormattedSize_MB()
    {
        var info = new CacheInfo { TotalSizeBytes = 5 * 1024 * 1024 };
        Assert.Equal("5.0 MB", info.FormattedSize);
    }

    [Fact]
    public void CacheInfo_FormattedSize_GB()
    {
        var info = new CacheInfo { TotalSizeBytes = 2L * 1024 * 1024 * 1024 };
        Assert.Equal("2.00 GB", info.FormattedSize);
    }

    #endregion

    #region DiskCacheOptions

    [Fact]
    public void DiskCacheOptions_DefaultValues()
    {
        var options = new DiskCacheOptions();

        Assert.Contains("pipeline_cache", options.CacheDirectory);
        Assert.Equal(10L * 1024 * 1024 * 1024, options.MaxCacheSizeBytes); // 10 GB
        Assert.Equal(CacheEvictionPolicy.LeastRecentlyUsed, options.EvictionPolicy);
        Assert.True(options.VerifyIntegrity);
        Assert.True(options.AutoInvalidateOnSourceChange);
        Assert.Null(options.MaxAge);
        Assert.False(options.CompressData);
    }

    [Fact]
    public void DiskCacheOptions_CanSetAllProperties()
    {
        var options = new DiskCacheOptions
        {
            CacheDirectory = "/custom/path",
            MaxCacheSizeBytes = 1024 * 1024, // 1 MB
            EvictionPolicy = CacheEvictionPolicy.LargestFirst,
            VerifyIntegrity = false,
            AutoInvalidateOnSourceChange = false,
            MaxAge = TimeSpan.FromHours(2),
            CompressData = true
        };

        Assert.Equal("/custom/path", options.CacheDirectory);
        Assert.Equal(1024 * 1024, options.MaxCacheSizeBytes);
        Assert.Equal(CacheEvictionPolicy.LargestFirst, options.EvictionPolicy);
        Assert.False(options.VerifyIntegrity);
        Assert.False(options.AutoInvalidateOnSourceChange);
        Assert.Equal(TimeSpan.FromHours(2), options.MaxAge);
        Assert.True(options.CompressData);
    }

    #endregion

    #region CacheEvictionPolicy

    [Fact]
    public void CacheEvictionPolicy_HasExpectedValues()
    {
        Assert.Equal(0, (int)CacheEvictionPolicy.LeastRecentlyUsed);
        Assert.Equal(1, (int)CacheEvictionPolicy.OldestFirst);
        Assert.Equal(2, (int)CacheEvictionPolicy.LargestFirst);
    }

    #endregion
}
