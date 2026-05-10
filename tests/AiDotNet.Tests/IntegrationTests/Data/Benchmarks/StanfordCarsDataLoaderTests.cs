using System.Threading.Tasks;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data.Benchmarks;

public class StanfordCarsDataLoaderTests
{
    [Fact]
    public async Task AutoDownloadAttempt_Throws()
    {
        // Stanford URLs are unstable; AutoDownload must be off and the loader
        // refuses to attempt download.
        var loader = new StanfordCarsDataLoader<float>(new StanfordCarsDataLoaderOptions
        { AutoDownload = true });
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await loader.LoadAsync());
    }

    [Fact]
    public async Task Fixture_MissingDir_ThrowsDirectoryNotFound()
    {
        string root = DatasetLoaderTestHelpers.CreateTempDir("stanford-cars-miss");
        try
        {
            var loader = new StanfordCarsDataLoader<float>(new StanfordCarsDataLoaderOptions
            { DataPath = root, AutoDownload = false });
            await Assert.ThrowsAsync<DirectoryNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(root); }
    }

    // Stanford Cars uses a struct-array .mat layout that's complex to
    // synthesize as a fixture. End-to-end .mat parsing is verified via
    // SVHN/Flowers-102 fixtures (which exercise the same MatFileHandler reader path).
    // No network test — Stanford URLs have been intermittent for years.
}
