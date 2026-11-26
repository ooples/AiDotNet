using System.Reflection;
using Xunit.Sdk;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Reflection over private state for tests, failing with a named message instead of a null dereference.</summary>
internal static class PrivateState
{
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    internal static FieldInfo Field(Type type, string name) =>
        type.GetField(name, Flags) ?? throw new XunitException(type.Name + " has no field '" + name + "'.");

    internal static object? Value(Type type, string name, object owner) => Field(type, name).GetValue(owner);

    internal static object Required(Type type, string name, object owner) =>
        Value(type, name, owner) ?? throw new XunitException(type.Name + "." + name + " is null.");

    internal static object? Property(object owner, string name)
    {
        var property = owner.GetType().GetProperty(name, Flags)
            ?? throw new XunitException(owner.GetType().Name + " has no property '" + name + "'.");
        return property.GetValue(owner);
    }

    internal static T NotNull<T>(T? value, string what) where T : class =>
        value ?? throw new XunitException("Expected " + what + " to be present.");
}
