using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Xml.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public sealed class VisionLanguageCopyContractTests
{
    [Theory]
    [InlineData(typeof(VisionLanguageInputOptions))]
    [InlineData(typeof(VisionLanguageModelOptions))]
    [InlineData(typeof(Blip2Options))]
    [InlineData(typeof(AudioVisualEventLocalizationOptions))]
    public void EveryDeclaredOptionDocumentsItsValueAndBeginnerEffect(Type optionsType)
    {
        var documentation = XDocument.Load(Path.Combine(AppContext.BaseDirectory, optionsType.Assembly.GetName().Name + ".xml"));
        var properties = optionsType.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly);
        Assert.NotEmpty(properties);
        foreach (var property in properties)
        {
            string name = $"P:{optionsType.FullName}.{property.Name}";
            var member = Assert.Single(documentation.Descendants("member"), element => (string?)element.Attribute("name") == name);
            Assert.False(string.IsNullOrWhiteSpace(member.Element("summary")?.Value), name);
            Assert.False(string.IsNullOrWhiteSpace(member.Element("value")?.Value), name);
            Assert.Contains("For Beginners:", member.Element("remarks")?.Value ?? string.Empty);
        }
    }

    [Theory]
    [InlineData(typeof(VisionLanguageInputOptions))]
    [InlineData(typeof(VisionLanguageModelOptions))]
    public void ValidationHelpersAreAssemblyOnlyRatherThanExternalSubclassApi(Type optionsType)
    {
        var methods = optionsType.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
            .Where(method => method.Name.StartsWith("Validate", StringComparison.Ordinal));
        Assert.NotEmpty(methods);
        Assert.All(methods, method => Assert.True(method.IsAssembly, method.Name));
        var flags = optionsType.GetNestedTypes(BindingFlags.Public | BindingFlags.NonPublic).Where(type => type.IsEnum);
        Assert.NotEmpty(flags);
        Assert.All(flags, type => Assert.True(type.IsNestedAssembly, type.Name));
    }

    [Theory]
    [InlineData(typeof(ClipOptions))]
    [InlineData(typeof(Blip2Options))]
    [InlineData(typeof(AudioVisualCorrespondenceOptions))]
    [InlineData(typeof(AudioVisualEventLocalizationOptions))]
    public void PublicCopyConstructorPreservesEveryDeclaredAndInheritedSetting(Type optionsType)
    {
        var constructor = optionsType.GetConstructor(new[] { optionsType });
        Assert.NotNull(constructor);
        var source = Assert.IsAssignableFrom<ModelOptions>(Activator.CreateInstance(optionsType));
        var properties = optionsType.GetProperties(BindingFlags.Instance | BindingFlags.Public)
            .Where(property => property.CanRead && property.CanWrite && property.GetIndexParameters().Length == 0).ToArray();
        foreach (var property in properties)
        {
            var valueType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
            object value;
            if (valueType == typeof(int)) value = 37;
            else if (valueType == typeof(double)) value = 0.375;
            else if (valueType == typeof(bool)) value = true;
            else if (valueType.IsEnum) value = Enum.GetValues(valueType).Cast<object>().Last();
            else throw new InvalidOperationException($"Add an independent copy oracle for {optionsType.Name}.{property.Name} ({valueType}).");
            property.SetValue(source, value);
        }
        var copy = Assert.IsAssignableFrom<ModelOptions>(constructor.Invoke(new object[] { source }));
        Assert.NotSame(source, copy);
        foreach (var property in properties)
            Assert.Equal(property.GetValue(source), property.GetValue(copy));
        source.Seed = 999;
        Assert.NotEqual(source.Seed, copy.Seed);
    }

    [Theory]
    [InlineData(typeof(ClipOptions))]
    [InlineData(typeof(Blip2Options))]
    [InlineData(typeof(AudioVisualCorrespondenceOptions))]
    [InlineData(typeof(AudioVisualEventLocalizationOptions))]
    public void PublicCopyConstructorRejectsNullAtTheSharedBaseBoundary(Type optionsType)
    {
        var constructor = optionsType.GetConstructor(new[] { optionsType });
        Assert.NotNull(constructor);
        var invocation = Assert.Throws<TargetInvocationException>(() => constructor.Invoke(new object?[] { null }));
        Assert.Equal("other", Assert.IsType<ArgumentNullException>(invocation.InnerException).ParamName);
    }
}
