using System.Text;
using AiDotNet.Configuration;
using AiDotNet.ProgramSynthesis.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunConfigurationTemplateTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TemplateRetainsSearchSettingsAndMarksEveryRemovedBinding(bool includeSource)
    {
        var config = JsonConvert.DeserializeObject<YamlModelConfig>("""
            {"Evolution":{"RunId":"private-run-name","Seed":77,"MaxEvaluationAttempts":13,"EnableEvaluationCache":false},
             "ProgramEvolution":{"Language":"Python","SeedPrograms":["print(7)"],"EvaluatorScript":"def evaluate(): pass"},
             "ChatClient":{"Type":"ManualChatClient","Params":{"apiKey":"configured-private-token","modelId":"private-model"}}}
            """)!;
        config.ProgramEvolution!.TestCases.Add(new ProgramInputOutputExample { Input = "private-example", ExpectedOutput = "7" });
        var exported = RunConfigurationTemplate.Create(config, includeSource);
        RunEvidenceBundle.ValidateMetadata(exported.Json, exported.ProhibitedValues);
        string text = Encoding.UTF8.GetString(exported.Json);
        var document = JObject.Parse(text);
        Assert.Equal(77, (int?)document.SelectToken("Configuration.Evolution.Seed"));
        Assert.Equal(13, (int?)document.SelectToken("Configuration.Evolution.MaxEvaluationAttempts"));
        Assert.False((bool?)document.SelectToken("Configuration.Evolution.EnableEvaluationCache"));
        Assert.True((bool?)document.SelectToken("Configuration.ChatClient.Params.RequiresBinding"));
        Assert.Null(document.SelectToken("Configuration.ChatClient.Params.ValueSha256"));
        Assert.DoesNotContain("configured-private-token", text);
        Assert.DoesNotContain("private-run-name", text);
        Assert.DoesNotContain("private-model", text);
        Assert.Equal(includeSource, text.Contains("print(7)", StringComparison.Ordinal));
        Assert.Equal(includeSource, text.Contains("private-example", StringComparison.Ordinal));
        Assert.Contains(document["RequiredBindings"]!, item => item.Value<string>() == "/ChatClient/Params");
    }

    [Fact]
    public void KnownCredentialsEmbeddedInAnIncludedScriptCannotEscapeMetadataValidation()
    {
        var config = JsonConvert.DeserializeObject<YamlModelConfig>("""
            {"Evolution":{},"ProgramEvolution":{"SeedPrograms":["print('configured-private-token')"]},
             "ChatClient":{"Type":"ManualChatClient","Params":{"apiKey":"configured-private-token"}}}
            """)!;
        var exported = RunConfigurationTemplate.Create(config, true);
        Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.ValidateMetadata(exported.Json, exported.ProhibitedValues));
    }
}
