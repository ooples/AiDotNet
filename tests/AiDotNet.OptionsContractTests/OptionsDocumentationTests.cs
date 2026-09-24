using System.Xml.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.OptionsContractTests;

public class OptionsDocumentationTests
{
    [Theory]
    [InlineData(typeof(JambaOptions), "2403.19887")]
    [InlineData(typeof(Mamba2Options), "2405.21060")]
    [InlineData(typeof(SambaOptions), "2406.07522")]
    [InlineData(typeof(XLSTMOptions), "2405.04517")]
    [InlineData(typeof(ZambaOptions), "2405.16712")]
    public void ClassDocumentation_IsEmittedWithBeginnerContextAndOriginalPaper(Type optionsType, string arxivId)
    {
        var member = DocumentationMember("T:" + optionsType.FullName);
        Assert.False(string.IsNullOrWhiteSpace(member.Element("summary")?.Value));
        var remarks = member.Element("remarks") ?? throw new InvalidOperationException("Missing class remarks.");
        Assert.Contains("For Beginners:", remarks.Value);
        Assert.Contains("library", remarks.Value);
        Assert.Contains(member.Elements("seealso"), seeAlso =>
            (string?)seeAlso.Attribute("href") == "https://arxiv.org/abs/" + arxivId);
    }

    [Fact]
    public void SharedFamilyDocumentation_DoesNotAdvertiseAnUnusedGradientAlias()
    {
        var removedMemberName = "P:" + typeof(ModelHyperparameterOptions).FullName + ".MaxGradNorm";
        Assert.DoesNotContain(Documentation().Descendants("member"),
            member => (string?)member.Attribute("name") == removedMemberName);
    }

    private static XElement DocumentationMember(string memberName)
        => Assert.Single(Documentation().Descendants("member"), member => (string?)member.Attribute("name") == memberName);

    private static XDocument Documentation()
    {
        // .NET Framework shadow-copies assemblies, but not their adjacent XML docs.
        // The application base remains the actual test output directory on every TFM.
        var assemblyName = typeof(OptionsDocumentationTests).Assembly.GetName().Name
            ?? throw new InvalidOperationException("The test assembly has no name.");
        return XDocument.Load(Path.Combine(AppContext.BaseDirectory, assemblyName + ".xml"));
    }
}
