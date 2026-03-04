namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

public class ModelTypeRegistryTests
{
    [Fact]
    public void RegisteredTypeCount_IsPositive()
    {
        // The static constructor auto-discovers IModelSerializer types
        Assert.True(ModelTypeRegistry.RegisteredTypeCount > 0,
            "Registry should auto-discover model types from the AiDotNet assembly.");
    }

    [Fact]
    public void RegisteredTypeNames_IsNotEmpty()
    {
        var names = ModelTypeRegistry.RegisteredTypeNames;
        Assert.NotEmpty(names);
    }

    [Fact]
    public void Register_AddsCustomType()
    {
        int countBefore = ModelTypeRegistry.RegisteredTypeCount;

        // Register a custom type with a unique name to avoid colliding with real types
        string uniqueName = $"TestModel_{Guid.NewGuid():N}";
        ModelTypeRegistry.Register(uniqueName, typeof(StubModelSerializer));

        // Resolve should find it
        var resolved = ModelTypeRegistry.Resolve(uniqueName);
        Assert.NotNull(resolved);
        Assert.Equal(typeof(StubModelSerializer), resolved);
    }

    [Fact]
    public void Register_ThrowsOnNullName()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelTypeRegistry.Register(null, typeof(StubModelSerializer)));
    }

    [Fact]
    public void Register_ThrowsOnEmptyName()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelTypeRegistry.Register("", typeof(StubModelSerializer)));
    }

    [Fact]
    public void Register_ThrowsOnNullType()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.Register("SomeName", null));
    }

    [Fact]
    public void Resolve_ReturnsNullForUnknownType()
    {
        var result = ModelTypeRegistry.Resolve("CompletelyFakeModelType_12345");
        Assert.Null(result);
    }

    [Fact]
    public void Resolve_ReturnsNullForNullName()
    {
        var result = ModelTypeRegistry.Resolve(null);
        Assert.Null(result);
    }

    [Fact]
    public void Resolve_ReturnsNullForEmptyName()
    {
        var result = ModelTypeRegistry.Resolve("");
        Assert.Null(result);
    }

    [Fact]
    public void Resolve_IsCaseInsensitive()
    {
        string uniqueName = $"CaseTestModel_{Guid.NewGuid():N}";
        ModelTypeRegistry.Register(uniqueName, typeof(StubModelSerializer));

        var resolved = ModelTypeRegistry.Resolve(uniqueName.ToUpperInvariant());
        Assert.NotNull(resolved);
    }

    [Fact]
    public void RegisterFactory_ThrowsOnNullName()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelTypeRegistry.RegisterFactory(null, t => null));
    }

    [Fact]
    public void RegisterFactory_ThrowsOnNullFactory()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.RegisterFactory("SomeName", null));
    }

    [Fact]
    public void RegisterAssembly_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.RegisterAssembly(null));
    }

    [Fact]
    public void CreateInstance_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.CreateInstance<double>(null));
    }

    [Fact]
    public void CreateInstance_NonGenericType_ReturnsInstance()
    {
        // StubModelSerializer is non-generic and has a parameterless constructor
        var instance = ModelTypeRegistry.CreateInstance<double>(typeof(StubModelSerializer));
        Assert.NotNull(instance);
        Assert.IsType<StubModelSerializer>(instance);
    }

    [Fact]
    public void RegisterFactory_IsUsedByCreateInstance()
    {
        // Use a unique type name derived from the real type to avoid global state leaks
        string typeName = typeof(StubModelSerializer).Name;
        var stubInstance = new StubModelSerializer { Payload = new byte[] { 99 } };

        // Register both the type and a factory under the same key
        ModelTypeRegistry.Register(typeName, typeof(StubModelSerializer));
        ModelTypeRegistry.RegisterFactory(typeName, t => stubInstance);

        var result = ModelTypeRegistry.CreateInstance<double>(typeof(StubModelSerializer));
        Assert.Same(stubInstance, result);

        // Clean up: remove the factory so other tests aren't affected
        ModelTypeRegistry.RegisterFactory(typeName, null);
    }
}
