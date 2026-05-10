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
        // Use a child path that doesn't exist so the throw is unambiguously about
        // the missing root rather than missing internal subdirectories.
        string tempRoot = DatasetLoaderTestHelpers.CreateTempDir("stanford-cars-miss");
        string missingDataPath = System.IO.Path.Combine(tempRoot, "does-not-exist");
        try
        {
            var loader = new StanfordCarsDataLoader<float>(new StanfordCarsDataLoaderOptions
            { DataPath = missingDataPath, AutoDownload = false });
            await Assert.ThrowsAsync<DirectoryNotFoundException>(async () => await loader.LoadAsync());
        }
        finally { DatasetLoaderTestHelpers.TryCleanup(tempRoot); }
    }

    // Stanford Cars uses a struct-array .mat layout that's complex to
    // synthesize as a fixture. End-to-end .mat parsing is verified via
    // SVHN/Flowers-102 fixtures (which exercise the same MatFileHandler reader path).
    // No network test — Stanford URLs have been intermittent for years.
}
