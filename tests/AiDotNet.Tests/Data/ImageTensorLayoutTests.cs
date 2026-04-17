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

    private static bool EuroSatCacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("eurosat");
        // EuroSat is a class-folder structured dataset; require at least one
        // known class directory with at least one image file.
        return Directory.Exists(cachePath)
            && Directory.EnumerateDirectories(cachePath, "*", SearchOption.AllDirectories)
                .Any(d => Directory.EnumerateFiles(d, "*.jpg").Any()
                       || Directory.EnumerateFiles(d, "*.png").Any());
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

    /// <summary>
    /// Regression: Cifar10DataLoader with the default NHWC layout used to
    /// throw "Cannot get a contiguous span from a non-contiguous tensor
    /// view" because TensorPermute returns a strided view and the copy path
    /// immediately called AsSpan() on it. The fix materializes the permuted
    /// tensor via .Contiguous() before the copy.
    /// </summary>
    [SkippableFact]
    public async Task Cifar10NHWC_DefaultShape_LoadsWithoutThrowing()
    {
        Skip.IfNot(Cifar10CacheExists(), "CIFAR-10 not cached locally");
        var loader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Normalize = false,
            // Layout defaults to NHWC — no explicit override. The regression
            // point is that the default path must not throw.
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 32, 32, 3);
    }

    /// <summary>
    /// Sanity check that NHWC and NCHW layouts produce the same pixel values
    /// under the expected index mapping. A regression in the permute+copy
    /// path could silently scramble channels; this locks the mapping:
    ///   NCHW[c, y, x]  ==  NHWC[y, x, c]
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

        var nchw = ncEnum.Current.Features;          // [1, 3, 32, 32]
        var nhwc = nhEnum.Current.Features;          // [1, 32, 32, 3]
        AssertShape(nchw, 1, 3, 32, 32);
        AssertShape(nhwc, 1, 32, 32, 3);

        // Spot-check a handful of (c, y, x) triples spanning all three
        // channels and a few spatial positions. If the permute dropped a
        // channel or flipped an axis, this will catch it.
        (int c, int y, int x)[] probes = [(0, 0, 0), (1, 7, 11), (2, 15, 15), (0, 31, 31), (2, 0, 31)];
        foreach (var (c, y, x) in probes)
        {
            float ncValue = nchw[0, c, y, x];
            float nhValue = nhwc[0, y, x, c];
            Assert.Equal(ncValue, nhValue);
        }
    }

    // --- CIFAR-100 ---

    // --- EuroSat ---

    /// <summary>
    /// Regression: EuroSat is the mirror of the CIFAR bug. The NHWC default
    /// path writes directly into the HWC buffer (fine), but the NCHW path
    /// previously threw "Cannot get a contiguous span from a non-contiguous
    /// tensor view" because it called AsSpan() on a TensorPermute result.
    /// Fix materializes via .Contiguous() before the copy.
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
        // EuroSat images are 64x64 RGB per EuroSatDataLoader.ImageSize.
        AssertShape(enumerator.Current.Features, 2, 3, 64, 64);
    }

    /// <summary>
    /// Value-correctness probe: EuroSat NHWC and NCHW layouts must agree on
    /// <c>NHWC[y,x,c] == NCHW[c,y,x]</c> across a handful of probe positions.
    /// Locks the permute direction so a future bad refactor silently
    /// scrambling axes still fails the test.
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
        var hwcShape = nhwc.Shape;   // [1, H, W, 3]
        int height = hwcShape[1];
        int width = hwcShape[2];
        Assert.Equal(3, hwcShape[3]);

        (int c, int y, int x)[] probes =
        [
            (0, 0, 0), (1, height / 2, width / 2), (2, height - 1, width - 1),
            (0, 1, width - 1), (2, height - 1, 0),
        ];
        foreach (var (c, y, x) in probes)
        {
            float ncValue = nchw[0, c, y, x];
            float nhValue = nhwc[0, y, x, c];
            Assert.Equal(ncValue, nhValue);
        }
    }

    // --- CIFAR-100 ---

    /// <summary>
    /// Regression: identical pattern to Cifar10DataLoader. If CIFAR-100 is not
    /// cached locally the test is skipped; otherwise the default NHWC path
    /// must not throw on AsSpan() of a permuted tensor view.
    /// </summary>
    [SkippableFact]
    public async Task Cifar100NHWC_DefaultShape_LoadsWithoutThrowing()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("cifar-100");
        bool exists = Directory.Exists(cachePath)
            && Directory.EnumerateFiles(cachePath, "train.bin", SearchOption.AllDirectories).Any();
        Skip.IfNot(exists, "CIFAR-100 not cached locally");

        var loader = new Cifar100DataLoader<float>(new Cifar100DataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Normalize = false,
            // Default NHWC layout; the regression point.
        });
        await loader.LoadAsync();
        Assert.True(loader.TotalCount > 0, "Loader returned no samples");
        using var enumerator = loader.GetBatches(batchSize: 2).GetEnumerator();
        Assert.True(enumerator.MoveNext(), "No batches returned");
        AssertShape(enumerator.Current.Features, 2, 32, 32, 3);
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
