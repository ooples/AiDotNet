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

    private static bool Cifar100CacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("cifar-100");
        return Directory.Exists(cachePath)
            && Directory.EnumerateFiles(cachePath, "train.bin", SearchOption.AllDirectories).Any();
    }

    private static bool EuroSatCacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("eurosat");
        if (!Directory.Exists(cachePath)) return false;
        // EuroSat is a class-folder dataset; require at least one
        // loader-supported image file. Match the loader's own extension
        // set (see EuroSatDataLoader.cs): .jpg / .jpeg / .png / .tif,
        // case-insensitive. A narrower probe causes false SKIPs on
        // valid caches and hides regressions.
        return Directory.EnumerateFiles(cachePath, "*.*", SearchOption.AllDirectories)
            .Any(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                   || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                   || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
                   || f.EndsWith(".tif", StringComparison.OrdinalIgnoreCase));
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

    // --- CIFAR-10 ---
    // Reintroduced alongside the Tensor<T>.CopyTo-based loader fix. The
    // NHWC default path previously threw "Cannot get a contiguous span
    // from a non-contiguous tensor view" on AsSpan() of a TensorPermute
    // result. The loader now calls Tensor<T>.CopyTo (AiDotNet.Tensors
    // 0.46.0) which handles strided layouts in a single pass.

    /// <summary>
    /// Regression for the default NHWC path — previously threw on
    /// AsSpan() of a TensorPermute view.
    /// </summary>
    [SkippableFact]
    public async Task Cifar10NHWC_DefaultShape_LoadsWithoutThrowing()
    {
        Skip.IfNot(Cifar10CacheExists(), "CIFAR-10 not cached locally");
        var loader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = false,
            MaxSamples = 2,
            Normalize = false,
            // Layout defaults to NHWC — the regression point.
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 32, 32, 3);
    }

    /// <summary>
    /// Locks the permute direction: <c>NHWC[n,y,x,c] == NCHW[n,c,y,x]</c>
    /// across five probe positions spanning all three channels and
    /// corners. Catches silent axis-swap regressions in either the loader
    /// or Tensor<T>.CopyTo's strided walk.
    /// </summary>
    [SkippableFact]
    public async Task Cifar10NHWC_PixelValues_MatchNCHW()
    {
        Skip.IfNot(Cifar10CacheExists(), "CIFAR-10 not cached locally");

        var nchwLoader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 1,
            Normalize = false, Layout = ImageTensorLayout.NCHW,
        });
        var nhwcLoader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 1,
            Normalize = false, Layout = ImageTensorLayout.NHWC,
        });
        await nchwLoader.LoadAsync();
        await nhwcLoader.LoadAsync();

        using var ncEnum = nchwLoader.GetBatches(batchSize: 1).GetEnumerator();
        using var nhEnum = nhwcLoader.GetBatches(batchSize: 1).GetEnumerator();
        Assert.True(ncEnum.MoveNext() && nhEnum.MoveNext(), "Both loaders must yield one batch");

        var nchw = ncEnum.Current.Features;
        var nhwc = nhEnum.Current.Features;
        AssertShape(nchw, 1, 3, 32, 32);
        AssertShape(nhwc, 1, 32, 32, 3);

        (int c, int y, int x)[] probes = [(0, 0, 0), (1, 7, 11), (2, 15, 15), (0, 31, 31), (2, 0, 31)];
        foreach (var (c, y, x) in probes)
        {
            float ncValue = nchw[0, c, y, x];
            float nhValue = nhwc[0, y, x, c];
            Assert.Equal(ncValue, nhValue);
        }
    }

    // --- CIFAR-100 ---

    /// <summary>
    /// Regression: same AsSpan-on-permuted-view pattern as CIFAR-10.
    /// SkippableFact because CIFAR-100 is a separate download.
    /// </summary>
    [SkippableFact]
    public async Task Cifar100NHWC_DefaultShape_LoadsWithoutThrowing()
    {
        Skip.IfNot(Cifar100CacheExists(), "CIFAR-100 not cached locally");
        var loader = new Cifar100DataLoader<float>(new Cifar100DataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = false,
            MaxSamples = 2,
            Normalize = false,
            // Layout defaults to NHWC — the regression point.
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 32, 32, 3);
    }

    // --- EuroSat ---
    // Mirror of the CIFAR bug: EuroSat returns HWC from the disk loader
    // and permuted to CHW for the NCHW layout; the NCHW branch is the
    // one that hit AsSpan()-on-permuted-view.

    /// <summary>
    /// Regression for the NCHW permute path. EuroSat images are 64x64 RGB
    /// per <c>EuroSatDataLoader.ImageSize</c>.
    /// </summary>
    [SkippableFact]
    public async Task EuroSatNCHW_Shape_LoadsWithoutThrowing()
    {
        Skip.IfNot(EuroSatCacheExists(), "EuroSat not cached locally");
        var loader = new EuroSatDataLoader<float>(new EuroSatDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Normalize = false, Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 3, 64, 64);
    }

    /// <summary>
    /// Value-correctness probe: EuroSat NHWC and NCHW layouts must agree
    /// on <c>NHWC[y,x,c] == NCHW[c,y,x]</c>. Asserts both shapes
    /// explicitly before probing so a wrong 64x64 layout cannot pass.
    /// </summary>
    [SkippableFact]
    public async Task EuroSatNHWC_PixelValues_MatchNCHW()
    {
        Skip.IfNot(EuroSatCacheExists(), "EuroSat not cached locally");

        var nhwcLoader = new EuroSatDataLoader<float>(new EuroSatDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 1,
            Normalize = false, Layout = ImageTensorLayout.NHWC,
        });
        var nchwLoader = new EuroSatDataLoader<float>(new EuroSatDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 1,
            Normalize = false, Layout = ImageTensorLayout.NCHW,
        });
        await nhwcLoader.LoadAsync();
        await nchwLoader.LoadAsync();

        using var nhEnum = nhwcLoader.GetBatches(batchSize: 1).GetEnumerator();
        using var ncEnum = nchwLoader.GetBatches(batchSize: 1).GetEnumerator();
        Assert.True(nhEnum.MoveNext() && ncEnum.MoveNext(), "Both loaders must yield one batch");

        var nhwc = nhEnum.Current.Features;
        var nchw = ncEnum.Current.Features;
        AssertShape(nhwc, 1, 64, 64, 3);
        AssertShape(nchw, 1, 3, 64, 64);

        (int c, int y, int x)[] probes =
        [
            (0, 0, 0), (1, 32, 32), (2, 63, 63), (0, 1, 63), (2, 63, 0),
        ];
        foreach (var (c, y, x) in probes)
        {
            float ncValue = nchw[0, c, y, x];
            float nhValue = nhwc[0, y, x, c];
            Assert.Equal(ncValue, nhValue);
        }
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
