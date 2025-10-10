namespace AiDotNet.Compression.KnowledgeDistillation;

using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;

/// <summary>
/// Static registry for distilled model factory methods.
/// </summary>
/// <remarks>
/// <para>
/// This class maintains a registry of factory methods for deserializing distilled models
/// of different types.
/// </para>
/// <para><b>For Beginners:</b> This keeps track of ways to create different distilled models.
/// 
/// Since different model types need different deserialization approaches, this registry:
/// - Maps model types to their appropriate factory methods
/// - Allows the deserializer to work with any supported model type
/// - Can be extended to support new model types
/// </para>
/// </remarks>
public static class DistilledModelFactoryRegistry
{
    private static readonly Dictionary<Type, object> _factories = 
        new Dictionary<Type, object>();
    
    /// <summary>
    /// Registers a factory for a specific model type.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <typeparam name="TModel">The type of model.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <param name="factory">The factory to register.</param>
    /// <remarks>
    /// <para>
    /// This method registers a factory that can create distilled models of the specified type.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new model type to the registry.
    ///
    /// When adding support for a new model type:
    /// 1. Create a factory that knows how to deserialize that distilled model type
    /// 2. Register it using this method
    /// 3. The distillation system can now work with that model type
    /// </para>
    /// </remarks>
    public static void RegisterFactory<T, TModel, TInput, TOutput>(
        IDistilledModelFactory<T, TModel, TInput, TOutput> factory)
        where T : unmanaged
        where TModel : class, IFullModel<T, TInput, TOutput>
    {
        _factories[typeof(TModel)] = factory;
    }
    
    /// <summary>
    /// Gets a factory for a specific model type.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <typeparam name="TModel">The type of model.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <returns>The registered factory, or null if no factory is registered for the type.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves a factory that can create distilled models of the specified type.
    /// </para>
    /// <para><b>For Beginners:</b> This finds the right factory for a model type.
    ///
    /// When deserializing a model:
    /// 1. We need to know how to create that specific distilled model type
    /// 2. This method finds the factory that knows how to do that
    /// 3. The factory then handles the model-specific deserialization
    /// </para>
    /// </remarks>
    public static IDistilledModelFactory<T, TModel, TInput, TOutput>? GetFactory<T, TModel, TInput, TOutput>()
        where T : unmanaged
        where TModel : class, IFullModel<T, TInput, TOutput>
    {
        if (_factories.TryGetValue(typeof(TModel), out var factory))
        {
            return (IDistilledModelFactory<T, TModel, TInput, TOutput>)factory;
        }

        return null;
    }
}