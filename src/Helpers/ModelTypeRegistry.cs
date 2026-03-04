using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
#if NETFRAMEWORK
using System.Runtime.Serialization;
#endif
using AiDotNet.Interfaces;

namespace AiDotNet.Helpers;

/// <summary>
/// A registry that maps model type names to their .NET types, enabling automatic model discovery and instantiation.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When you save a model to a file, the file header includes the model's type name
/// (e.g., "ConvolutionalNeuralNetwork`1"). When you later want to load that file, the registry
/// looks up this name to find the correct .NET type so it can create an instance and deserialize into it.
///
/// The registry automatically discovers all model types in the AiDotNet assembly at startup.
/// External plugins can register additional types using the <see cref="Register"/> method.
///
/// This follows the same pattern as <see cref="DeserializationHelper"/> which auto-discovers layer types.
/// </remarks>
internal static class ModelTypeRegistry
{
    private static readonly ConcurrentDictionary<string, Type> TypesByName = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, Type> TypesByQualifiedName = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, Func<Type, IModelSerializer>> Factories = new(StringComparer.OrdinalIgnoreCase);

    static ModelTypeRegistry()
    {
        DiscoverModelTypes(Assembly.GetExecutingAssembly());
    }

    /// <summary>
    /// Discovers and registers all non-abstract types implementing IModelSerializer from the given assembly.
    /// </summary>
    private static void DiscoverModelTypes(Assembly assembly)
    {
        try
        {
            var modelTypes = assembly.GetTypes()
                .Where(t => !t.IsAbstract && !t.IsInterface &&
                            t.GetInterfaces().Any(i =>
                                i == typeof(IModelSerializer) ||
                                (i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IModelSerializer))));

            foreach (var type in modelTypes)
            {
                string name = type.Name;
                TypesByName.TryAdd(name, type);

                string qualifiedName = type.AssemblyQualifiedName ?? type.FullName ?? name;
                TypesByQualifiedName.TryAdd(qualifiedName, type);

                // Also register the FullName for more robust matching
                if (type.FullName is not null && type.FullName != qualifiedName)
                {
                    TypesByQualifiedName.TryAdd(type.FullName, type);
                }
            }
        }
        catch (ReflectionTypeLoadException ex)
        {
            // Process the types that DID load successfully
            var loadedTypes = ex.Types
                .Where(t => t is not null &&
                            !t.IsAbstract && !t.IsInterface &&
                            t.GetInterfaces().Any(i =>
                                i == typeof(IModelSerializer) ||
                                (i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IModelSerializer))));

            foreach (var type in loadedTypes)
            {
                if (type is null) continue;
                string name = type.Name;
                TypesByName.TryAdd(name, type);

                string qualifiedName = type.AssemblyQualifiedName ?? type.FullName ?? name;
                TypesByQualifiedName.TryAdd(qualifiedName, type);

                if (type.FullName is not null && type.FullName != qualifiedName)
                {
                    TypesByQualifiedName.TryAdd(type.FullName, type);
                }
            }
        }
    }

    /// <summary>
    /// Registers an external model type by name.
    /// </summary>
    /// <param name="name">The short type name to register (e.g., "MyCustomModel`1").</param>
    /// <param name="type">The .NET type of the model. Must implement IModelSerializer.</param>
    public static void Register(string name, Type type)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty.", nameof(name));
        }

        if (type is null)
        {
            throw new ArgumentNullException(nameof(type));
        }

        bool implementsSerializer = type.GetInterfaces().Any(i =>
            i == typeof(IModelSerializer) ||
            (i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IModelSerializer)));

        if (!implementsSerializer && !type.IsGenericTypeDefinition)
        {
            throw new ArgumentException(
                $"Type '{type.FullName}' does not implement IModelSerializer.", nameof(type));
        }

        TypesByName[name] = type;

        string qualifiedName = type.AssemblyQualifiedName ?? type.FullName ?? name;
        TypesByQualifiedName[qualifiedName] = type;

        // Also register by FullName for consistency with DiscoverModelTypes
        if (type.FullName is not null && type.FullName != qualifiedName)
        {
            TypesByQualifiedName[type.FullName] = type;
        }
    }

    /// <summary>
    /// Registers a factory delegate for creating model instances that require complex construction.
    /// </summary>
    /// <param name="name">The short type name to associate with this factory.</param>
    /// <param name="factory">
    /// A factory that takes the closed generic type (e.g., ConvolutionalNeuralNetwork&lt;double&gt;)
    /// and returns an IModelSerializer instance.
    /// </param>
    public static void RegisterFactory(string name, Func<Type, IModelSerializer>? factory)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty.", nameof(name));
        }

        if (factory is null)
        {
            // Passing null removes the factory registration (useful for test cleanup)
            Factories.TryRemove(name, out _);
            return;
        }

        Factories[name] = factory;
    }

    /// <summary>
    /// Discovers model types from an external assembly, useful for plugin scenarios.
    /// </summary>
    /// <param name="assembly">The assembly to scan for model types.</param>
    public static void RegisterAssembly(Assembly assembly)
    {
        if (assembly is null)
        {
            throw new ArgumentNullException(nameof(assembly));
        }

        DiscoverModelTypes(assembly);
    }

    /// <summary>
    /// Resolves a model type by its short name or assembly-qualified name.
    /// </summary>
    /// <param name="typeName">The short type name (e.g., "ConvolutionalNeuralNetwork`1").</param>
    /// <param name="assemblyQualifiedName">Optional assembly-qualified name for fallback resolution.</param>
    /// <returns>The resolved Type, or null if no matching type was found.</returns>
    public static Type? Resolve(string typeName, string? assemblyQualifiedName = null)
    {
        if (string.IsNullOrWhiteSpace(typeName))
        {
            return null;
        }

        // Try short name first (most common case)
        if (TypesByName.TryGetValue(typeName, out var type))
        {
            return type;
        }

        // Try assembly-qualified name as fallback
        if (assemblyQualifiedName is not null && !string.IsNullOrWhiteSpace(assemblyQualifiedName))
        {
            string qualifiedKey = assemblyQualifiedName;
            if (TypesByQualifiedName.TryGetValue(qualifiedKey, out type))
            {
                return type;
            }

            // Last resort: try Type.GetType with the assembly-qualified name
            type = Type.GetType(qualifiedKey, throwOnError: false);
            if (type is not null)
            {
                // Only cache by qualified name to prevent a malformed header from
                // permanently aliasing an unrelated short name in the process-wide registry.
                TypesByQualifiedName.TryAdd(qualifiedKey, type);

                // Only cache by short name if the resolved type's Name actually matches
                if (string.Equals(type.Name, typeName, StringComparison.OrdinalIgnoreCase))
                {
                    TypesByName.TryAdd(typeName, type);
                }

                return type;
            }
        }

        return null;
    }

    /// <summary>
    /// Creates an instance of the given model type, closing the generic with the specified numeric type T.
    /// </summary>
    /// <typeparam name="T">The numeric type to use (e.g., double, float).</typeparam>
    /// <param name="openGenericType">
    /// The open generic type from the registry (e.g., typeof(ConvolutionalNeuralNetwork&lt;&gt;)).
    /// If the type is already closed, it is used directly.
    /// </param>
    /// <returns>An IModelSerializer instance ready for deserialization.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the instance cannot be created.</exception>
    public static IModelSerializer CreateInstance<T>(Type openGenericType)
    {
        if (openGenericType is null)
        {
            throw new ArgumentNullException(nameof(openGenericType));
        }

        // Close the generic type if needed
        Type closedType;
        if (openGenericType.IsGenericTypeDefinition)
        {
            try
            {
                closedType = openGenericType.MakeGenericType(typeof(T));
            }
            catch (ArgumentException ex)
            {
                throw new InvalidOperationException(
                    $"Cannot close generic type '{openGenericType.Name}' with type argument '{typeof(T).Name}'. " +
                    "The type parameter may have constraints that are not satisfied.", ex);
            }
        }
        else
        {
            closedType = openGenericType;
        }

        // Strategy 1: Try registered factory
        string typeName = openGenericType.Name;
        if (Factories.TryGetValue(typeName, out var factory))
        {
            var factoryResult = factory(closedType);
            if (factoryResult is not null)
            {
                return factoryResult;
            }
        }

        // Strategy 2: Try parameterless constructor
        var ctor = closedType.GetConstructor(
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
            null, Type.EmptyTypes, null);

        if (ctor is not null)
        {
            var instance = ctor.Invoke(null);
            if (instance is IModelSerializer serializer)
            {
                return serializer;
            }
        }

        // Strategy 3: Create uninitialized object (no constructor called)
        // Deserialize() will populate all state. This works because Deserialize()
        // is designed to fully reconstruct the model from serialized bytes.
        try
        {
#if NETFRAMEWORK
            var uninitializedObj = FormatterServices.GetUninitializedObject(closedType);
#else
            var uninitializedObj = RuntimeHelpers.GetUninitializedObject(closedType);
#endif
            if (uninitializedObj is IModelSerializer uninitializedSerializer)
            {
                return uninitializedSerializer;
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to create an instance of '{closedType.FullName}'. " +
                "The type must have either a parameterless constructor, a registered factory, " +
                "or support uninitialized object creation.", ex);
        }

        throw new InvalidOperationException(
            $"Type '{closedType.FullName}' does not implement IModelSerializer. " +
            "Ensure the type is a valid AiDotNet model.");
    }

    /// <summary>
    /// Gets the number of registered model types.
    /// </summary>
    public static int RegisteredTypeCount => TypesByName.Count;

    /// <summary>
    /// Gets all registered type names for diagnostic purposes.
    /// </summary>
    public static IReadOnlyCollection<string> RegisteredTypeNames => TypesByName.Keys.ToList().AsReadOnly();
}
