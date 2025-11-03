using Xunit;
using AiDotNet.Config;

namespace UnitTests.Config
{
    public class ConfigLoaderTests
    {
        private sealed class Sample
        {
            public string Name { get; set; } = string.Empty;
            public int Epochs { get; set; }
        }

        [Fact]
        public void LoadJson_Loads_Poco()
        {
            var path = System.IO.Path.GetTempFileName();
            System.IO.File.WriteAllText(path, "{ \"name\": \"demo\", \"epochs\": 5 }");
            var cfg = ConfigLoader.LoadJson<Sample>(path);
            Assert.Equal("demo", cfg.Name);
            Assert.Equal(5, cfg.Epochs);
        }

        [Fact]
        public void LoadYaml_Throws_NotSupported()
        {
            var path = System.IO.Path.GetTempFileName();
            System.IO.File.WriteAllText(path, "name: demo\nepochs: 5\n");
            Assert.Throws<System.NotSupportedException>(() => ConfigLoader.LoadYaml<Sample>(path));
        }
    }
}

