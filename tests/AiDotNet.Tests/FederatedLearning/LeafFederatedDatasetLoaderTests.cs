using System.IO;
using AiDotNet.FederatedLearning.Benchmarks.Leaf;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class LeafFederatedDatasetLoaderTests
{
    private const string TinyLeafJson = @"
{
  ""users"": [""u1"", ""u2""],
  ""num_samples"": [2, 1],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [[1, 2], [3]],
        [[4, 5], [6]]
      ],
      ""y"": [0, 1]
    },
    ""u2"": {
      ""x"": [
        [7, 8, 9]
      ],
      ""y"": [1]
    }
  }
}";

    [Fact]
    public void LoadSplitFromJson_ValidTinyFixture_ParsesClientsAndFlattensFeatures()
    {
        var loader = new LeafFederatedDatasetLoader<double>();
        var split = loader.LoadSplitFromJson(TinyLeafJson);

        Assert.Equal(2, split.ClientCount);
        Assert.Equal(new[] { "u1", "u2" }, split.UserIds);

        var u1 = split.UserData["u1"];
        Assert.Equal(2, u1.SampleCount);
        Assert.Equal(2, u1.Features.Rows);
        Assert.Equal(3, u1.Features.Columns);
        Assert.Equal(1.0, u1.Features[0, 0]);
        Assert.Equal(2.0, u1.Features[0, 1]);
        Assert.Equal(3.0, u1.Features[0, 2]);
        Assert.Equal(4.0, u1.Features[1, 0]);
        Assert.Equal(5.0, u1.Features[1, 1]);
        Assert.Equal(6.0, u1.Features[1, 2]);
        Assert.Equal(0.0, u1.Labels[0]);
        Assert.Equal(1.0, u1.Labels[1]);

        var u2 = split.UserData["u2"];
        Assert.Equal(1, u2.SampleCount);
        Assert.Equal(1, u2.Features.Rows);
        Assert.Equal(3, u2.Features.Columns);
        Assert.Equal(7.0, u2.Features[0, 0]);
        Assert.Equal(8.0, u2.Features[0, 1]);
        Assert.Equal(9.0, u2.Features[0, 2]);
        Assert.Equal(1.0, u2.Labels[0]);

        var clients = split.ToClientIdDictionary(out var clientIdToUserId);
        Assert.Equal("u1", clientIdToUserId[0]);
        Assert.Equal("u2", clientIdToUserId[1]);
        Assert.Equal(2, clients[0].SampleCount);
        Assert.Equal(1, clients[1].SampleCount);
    }

    [Fact]
    public void LoadSplitFromJson_WhenDeclaredNumSamplesMismatch_ThrowsInvalidDataException()
    {
        const string badJson = @"
{
  ""users"": [""u1""],
  ""num_samples"": [1],
  ""user_data"": {
    ""u1"": { ""x"": [[1, 2]], ""y"": [0, 1] }
  }
}";

        var loader = new LeafFederatedDatasetLoader<double>();
        Assert.Throws<InvalidDataException>(() => loader.LoadSplitFromJson(badJson));
    }

    [Fact]
    public void LoadSplitFromJson_WhenMaxUsersSpecified_LoadsSubset()
    {
        var loader = new LeafFederatedDatasetLoader<double>();
        var split = loader.LoadSplitFromJson(
            TinyLeafJson,
            new LeafFederatedDatasetLoadOptions { MaxUsers = 1 });

        Assert.Single(split.UserIds);
        Assert.Equal("u1", split.UserIds[0]);
        Assert.True(split.UserData.ContainsKey("u1"));
        Assert.False(split.UserData.ContainsKey("u2"));
    }

    [Fact]
    public void LoadSplitFromFile_WhenFileExists_LoadsSplit()
    {
        var path = Path.GetTempFileName();

        try
        {
            File.WriteAllText(path, TinyLeafJson);

            var loader = new LeafFederatedDatasetLoader<double>();
            var split = loader.LoadSplitFromFile(path);

            Assert.Equal(2, split.ClientCount);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LoadDatasetFromFiles_WithTrainAndTest_LoadsBothSplits()
    {
        var trainPath = Path.GetTempFileName();
        var testPath = Path.GetTempFileName();

        try
        {
            File.WriteAllText(trainPath, TinyLeafJson);
            File.WriteAllText(testPath, TinyLeafJson);

            var loader = new LeafFederatedDatasetLoader<double>();
            var dataset = loader.LoadDatasetFromFiles(trainPath, testPath);

            Assert.NotNull(dataset.Train);
            Assert.NotNull(dataset.Test);
            Assert.Equal(2, dataset.Train.ClientCount);
            Assert.Equal(2, dataset.Test!.ClientCount);
        }
        finally
        {
            File.Delete(trainPath);
            File.Delete(testPath);
        }
    }

    [Fact]
    public void LoadSplitFromJson_WhenMaxUsersIsNonPositive_ThrowsArgumentOutOfRangeException()
    {
        var loader = new LeafFederatedDatasetLoader<double>();
        Assert.Throws<ArgumentOutOfRangeException>(() => loader.LoadSplitFromJson(
            TinyLeafJson,
            new LeafFederatedDatasetLoadOptions { MaxUsers = 0 }));
    }
}
