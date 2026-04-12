using AiDotNet.Data;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNetTests.UnitTests.Data;

/// <summary>
/// Tests for the ImageTensorLayout option on vision data loaders.
/// Uses AutoDownload=false and SkippableFact — reports SKIP when
/// datasets aren't cached so CI output clearly shows missing coverage.
/// </summary>
public class ImageTensorLayoutTests
{
    private static bool MnistCacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("mnist");
        return File.Exists(Path.Combine(cachePath, "train-images-idx3-ubyte"))
            && File.Exists(Path.Combine(cachePath, "train-labels-idx1-ubyte"));
    }

    private static bool Cifar10CacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("cifar-10");
        return Directory.Exists(cachePath)
            && Directory.EnumerateFiles(cachePath, "data_batch*", SearchOption.AllDirectories).Any();
    }

    private static bool FashionMnistCacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("fashion-mnist");
        return File.Exists(Path.Combine(cachePath, "train-images-idx3-ubyte"))
            && File.Exists(Path.Combine(cachePath, "train-labels-idx1-ubyte"));
    }

    private static void AssertShape(Tensor<float> tensor, params int[] expected)
    {
        var shape = tensor.Shape;
        Assert.Equal(expected.Length, shape.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], shape[i]);
    }

    // --- MNIST ---

    [SkippableFact]
    public async Task MnistNHWC_DefaultShape()
    {
        Skip.IfNot(MnistCacheExists(), "MNIST not cached locally");
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 28, 28, 1);
    }

    [SkippableFact]
    public async Task MnistNCHW_Shape()
    {
        Skip.IfNot(MnistCacheExists(), "MNIST not cached locally");
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 1, 28, 28);
    }

    [SkippableFact]
    public async Task MnistFlatten_IgnoresLayout()
    {
        Skip.IfNot(MnistCacheExists(), "MNIST not cached locally");
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Flatten = true, Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 784);
    }

    // --- CIFAR-10 ---

    [SkippableFact]
    public async Task Cifar10NCHW_Shape()
    {
        Skip.IfNot(Cifar10CacheExists(), "CIFAR-10 not cached locally");
        var loader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 1,
            Normalize = false, Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 1).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 1, 3, 32, 32);
    }

    // --- FashionMNIST ---

    [SkippableFact]
    public async Task FashionMnistNCHW_Shape()
    {
        Skip.IfNot(FashionMnistCacheExists(), "FashionMNIST not cached locally");
        var loader = new FashionMnistDataLoader<float>(new FashionMnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 1, 28, 28);
    }
}
