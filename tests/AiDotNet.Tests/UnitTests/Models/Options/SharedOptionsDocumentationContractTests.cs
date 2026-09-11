using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Xml.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public sealed class SharedOptionsDocumentationContractTests
{
    [Fact]
    public void EveryDeclaredAudioOptionDocumentsValueAndBeginnerEffect()
    {
        Type type = typeof(AudioHyperparameterOptions);
        // Framework test hosts may shadow-copy the assembly without its adjacent XML file.
        string path = Path.Combine(AppContext.BaseDirectory, type.Assembly.GetName().Name + ".xml");
        XDocument documentation = XDocument.Load(path);
        PropertyInfo[] properties = type.GetProperties(BindingFlags.DeclaredOnly | BindingFlags.Public | BindingFlags.Instance);
        Assert.Equal(10, properties.Length);
        foreach (PropertyInfo property in properties)
        {
            string memberName = $"P:{type.FullName}.{property.Name}";
            XElement member = Assert.Single(documentation.Descendants("member"),
                element => (string?)element.Attribute("name") == memberName);
            Assert.False(string.IsNullOrWhiteSpace(member.Element("value")?.Value), memberName);
            Assert.Contains("For Beginners:", member.Element("remarks")?.Value ?? string.Empty);
        }
    }

    [Fact]
    public void VisionAndDocumentBasesUseTheSamePublicTowerNames()
    {
        Type vision = typeof(VisionLanguageModelOptions);
        foreach (PropertyInfo property in typeof(DocumentNeuralNetworkOptions)
                     .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                     .Where(property => property.Name == nameof(DocumentNeuralNetworkOptions.VisionDim)
                         || property.Name == nameof(DocumentNeuralNetworkOptions.VisionLayers)))
        {
            PropertyInfo? corresponding = vision.GetProperty(property.Name);
            Assert.NotNull(corresponding);
            Assert.Equal(property.PropertyType, corresponding.PropertyType);
        }
        Assert.NotNull(vision.GetProperty(nameof(DocumentNeuralNetworkOptions.VisionDim)));
        Assert.NotNull(vision.GetProperty(nameof(DocumentNeuralNetworkOptions.VisionLayers)));
        Assert.Null(vision.GetProperty("VisionHiddenDim"));
        Assert.Null(vision.GetProperty("NumVisionLayers"));
    }
}
