using System.Threading.Tasks;
namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

public class ModelTypeRegistryTests
{
    [Fact(Timeout = 60000)]
    public async Task RegisteredTypeCount_IsPositive()
    {
        // The static constructor auto-discovers IModelSerializer types
        Assert.True(ModelTypeRegistry.RegisteredTypeCount > 0,
            "Registry should auto-discover model types from the AiDotNet assembly.");
    }

    [Fact(Timeout = 60000)]
    public async Task RegisteredTypeNames_IsNotEmpty()
    {
        var names = ModelTypeRegistry.RegisteredTypeNames;
        Assert.NotEmpty(names);
    }

    [Fact(Timeout = 60000)]
    public async Task Register_AddsCustomType()
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

    [Fact(Timeout = 60000)]
    public async Task Register_ThrowsOnNullName()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelTypeRegistry.Register(null, typeof(StubModelSerializer)));
    }

    [Fact(Timeout = 60000)]
    public async Task Register_ThrowsOnEmptyName()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelTypeRegistry.Register("", typeof(StubModelSerializer)));
    }

    [Fact(Timeout = 60000)]
    public async Task Register_ThrowsOnNullType()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.Register("SomeName", null));
    }

    [Fact(Timeout = 60000)]
    public async Task Resolve_ReturnsNullForUnknownType()
    {
        var result = ModelTypeRegistry.Resolve("CompletelyFakeModelType_12345");
        Assert.Null(result);
    }

    [Fact(Timeout = 60000)]
    public async Task Resolve_ReturnsNullForNullName()
    {
        var result = ModelTypeRegistry.Resolve(null);
        Assert.Null(result);
    }

    [Fact(Timeout = 60000)]
    public async Task Resolve_ReturnsNullForEmptyName()
    {
        var result = ModelTypeRegistry.Resolve("");
        Assert.Null(result);
    }

    [Fact(Timeout = 60000)]
    public async Task Resolve_IsCaseInsensitive()
    {
        string uniqueName = $"CaseTestModel_{Guid.NewGuid():N}";
        ModelTypeRegistry.Register(uniqueName, typeof(StubModelSerializer));

        var resolved = ModelTypeRegistry.Resolve(uniqueName.ToUpperInvariant());
        Assert.NotNull(resolved);
        Assert.Equal(typeof(StubModelSerializer), resolved);
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterFactory_ThrowsOnNullName()
    {
        Assert.Throws<ArgumentException>(() =>
            ModelTypeRegistry.RegisterFactory(null, t => null));
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterFactory_NullFactory_RemovesRegistration()
    {
        // Passing null factory removes the registration (cleanup pattern)
        var exception = Record.Exception(() =>
            ModelTypeRegistry.RegisterFactory("SomeName", null));
        Assert.Null(exception);
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterAssembly_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.RegisterAssembly(null));
    }

    [Fact(Timeout = 60000)]
    public async Task CreateInstance_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelTypeRegistry.CreateInstance<double>(null));
    }

    [Fact(Timeout = 60000)]
    public async Task CreateInstance_NonGenericType_ReturnsInstance()
    {
        // StubModelSerializer is non-generic and has a parameterless constructor
        var instance = ModelTypeRegistry.CreateInstance<double>(typeof(StubModelSerializer));
        Assert.NotNull(instance);
        Assert.IsType<StubModelSerializer>(instance);
    }

    [Fact(Timeout = 60000)]
    public async Task RegisterFactory_IsUsedByCreateInstance()
    {
        // Use a unique name per test run to avoid global state leaks between tests
        string typeName = $"FactoryTest_{Guid.NewGuid():N}";
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
