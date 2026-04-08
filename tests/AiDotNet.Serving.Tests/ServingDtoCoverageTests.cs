using System.Linq;
using System.Reflection;
using Xunit;

namespace AiDotNet.Serving.Tests;

public sealed class ServingDtoCoverageTests
{
    [Fact]
    public void Serving_PublicDtos_AreConstructible_AndPropertiesAreAccessible()
    {
        var assembly = typeof(Program).Assembly;

        var dtoTypes = assembly
            .GetExportedTypes()
            .Where(t => t.IsClass)
            .Where(t => !t.IsAbstract)
            .Where(t => !t.ContainsGenericParameters)
            .Where(t => t.Namespace is string ns && (
                ns.StartsWith("AiDotNet.Serving.Configuration", StringComparison.Ordinal) ||
                ns.StartsWith("AiDotNet.Serving.Models", StringComparison.Ordinal) ||
                ns.StartsWith("AiDotNet.Serving.Persistence.Entities", StringComparison.Ordinal) ||
                ns.StartsWith("AiDotNet.Serving.ProgramSynthesis", StringComparison.Ordinal) ||
                ns.StartsWith("AiDotNet.Serving.Sandboxing", StringComparison.Ordinal) ||
                ns.StartsWith("AiDotNet.Serving.Security", StringComparison.Ordinal)))
            .Where(t => t.GetConstructor(Type.EmptyTypes) is not null)
            .ToList();

        Assert.NotEmpty(dtoTypes);

        foreach (var dtoType in dtoTypes)
        {
            var instance = Activator.CreateInstance(dtoType);
            Assert.NotNull(instance);

            foreach (var property in dtoType.GetProperties(BindingFlags.Instance | BindingFlags.Public))
            {
                if (property.GetIndexParameters().Length != 0)
                {
                    continue;
                }

                if (property.CanWrite && property.SetMethod is not null && property.SetMethod.IsPublic)
                {
                    var value = CreateSimpleValue(property.PropertyType);
                    if (value is not null || !property.PropertyType.IsValueType)
                    {
                        property.SetValue(instance, value);
                    }
                }

                if (property.CanRead && property.GetMethod is not null && property.GetMethod.IsPublic)
                {
                    _ = property.GetValue(instance);
                }
            }
        }
    }

    private static object? CreateSimpleValue(Type type)
    {
        if (type == typeof(string))
        {
            return "value";
        }

        if (type == typeof(bool))
        {
            return true;
        }

        if (type == typeof(int))
        {
            return 1;
        }

        if (type == typeof(long))
        {
            return 1L;
        }

        if (type == typeof(double))
        {
            return 1.0;
        }

        if (type == typeof(float))
        {
            return 1.0f;
        }

        if (type == typeof(Guid))
        {
            return Guid.Empty;
        }

        if (type == typeof(DateTime))
        {
            return DateTime.UnixEpoch;
        }

        if (type == typeof(DateTimeOffset))
        {
            return DateTimeOffset.UnixEpoch;
        }

        if (type.IsEnum)
        {
            var values = Enum.GetValues(type);
            return values.Length == 0 ? Activator.CreateInstance(type) : values.GetValue(0);
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Nullable<>))
        {
            type = Nullable.GetUnderlyingType(type)!;
            return CreateSimpleValue(type);
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>))
        {
            return Activator.CreateInstance(type);
        }

        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Dictionary<,>))
        {
            return Activator.CreateInstance(type);
        }

        if (type.IsArray)
        {
            return Array.CreateInstance(type.GetElementType()!, 0);
        }

        if (!type.IsAbstract && type.GetConstructor(Type.EmptyTypes) is not null)
        {
            return Activator.CreateInstance(type);
        }

        return null;
    }
}
