using System;
using System.Globalization;
using System.IO;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// Files and replies the library parses are always written with a '.' decimal separator, so the parse must not depend
/// on the current culture (#2359). Under de-DE, where '.' is the group separator, a culture-dependent parse reads
/// "0.25" as 25.
/// </summary>
/// <remarks>CurrentCulture is per thread (and flows with the async context), so setting it here affects no other test.</remarks>
public class InvariantNumberParsingTests
{
    [Fact]
    public void WordVectorFile_LoadsTheSameValues_UnderACommaDecimalCulture()
    {
        string path = Path.Combine(Path.GetTempPath(), $"aidotnet-vectors-{Guid.NewGuid():N}.txt");
        var original = CultureInfo.CurrentCulture;
        try
        {
            File.WriteAllText(path, "king 0.25 1.5\nqueen 0.5 0.75\n");
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");

            var model = StaticWordEmbeddingModel<double>.LoadFromTextFile(path);
            var king = model.Embed("king");

            // The ratio survives any normalization the embedder applies. A de-DE parse read 25 and 15, a ratio of
            // 5/3 instead of 1/6.
            Assert.Equal(0.25 / 1.5, king[0] / king[1], 12);
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
