using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;

namespace AiDotNet.Compression.Pruning
{
    /// <summary>
    /// Static registry for pruned model factory methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class maintains a registry of factory methods for deserializing pruned models
    /// of different types.
    /// </para>
    /// <para><b>For Beginners:</b> This keeps track of ways to create different pruned models.
    /// 
    /// Since different model types need different deserialization approaches, this registry:
    /// - Maps model types to their appropriate factory methods
    /// - Allows the deserializer to work with any supported model type
    /// - Can be extended to support new model types
    /// </para>
    /// </remarks>
    public static class PrunedModelFactoryRegistry
    {
        private static readonly Dictionary<Type, object> _factories = 
            new Dictionary<Type, object>();
        private static readonly object _lock = new object();
        
        /// <summary>
        /// Registers a factory for a specific model type.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <typeparam name="TModel">The type of model.</typeparam>
        /// <typeparam name="TInput">The input type for the model.</typeparam>
        /// <typeparam name="TOutput">The output type for the model.</typeparam>
        /// <param name="factory">The factory to register.</param>
        /// <remarks>
        /// <para>
        /// This method registers a factory that can create pruned models of the specified type.
        /// </para>
        /// <para><b>For Beginners:</b> This adds a new model type to the registry.
        /// 
        /// When adding support for a new model type:
        /// 1. Create a factory that knows how to deserialize that pruned model type
        /// 2. Register it using this method
        /// 3. The pruning system can now work with that model type
        /// </para>
        /// </remarks>
        public static void RegisterFactory<T, TModel, TInput, TOutput>(
            IPrunedModelFactory<T, TModel, TInput, TOutput> factory)
            where T : unmanaged
            where TModel : class, IFullModel<T, TInput, TOutput>
        {
            lock (_lock)
            {
                _factories[typeof(TModel)] = factory;
            }
        }
        
        /// <summary>
        /// Gets a factory for a specific model type.
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <typeparam name="TModel">The type of model.</typeparam>
        /// <typeparam name="TInput">The input type for the model.</typeparam>
        /// <typeparam name="TOutput">The output type for the model.</typeparam>
        /// <returns>The registered factory, or null if no factory is registered for the type.</returns>
        /// <remarks>
        /// <para>
        /// This method retrieves a factory that can create pruned models of the specified type.
        /// </para>
        /// <para><b>For Beginners:</b> This finds the right factory for a model type.
        /// 
        /// When deserializing a model:
        /// 1. We need to know how to create that specific pruned model type
        /// 2. This method finds the factory that knows how to do that
        /// 3. The factory then handles the model-specific deserialization
        /// </para>
        /// </remarks>
        public static IPrunedModelFactory<T, TModel, TInput, TOutput> GetFactory<T, TModel, TInput, TOutput>()
            where T : unmanaged
            where TModel : class, IFullModel<T, TInput, TOutput>
        {
            lock (_lock)
            {
                if (_factories.TryGetValue(typeof(TModel), out var factory))
                {
                    return (IPrunedModelFactory<T, TModel, TInput, TOutput>)factory;
                }
                
                return null;
            }
        }
        
        /// <summary>
        /// Checks if a factory is registered for a specific model type.
        /// </summary>
        /// <typeparam name="TModel">The type of model.</typeparam>
        /// <returns>True if a factory is registered for the model type, false otherwise.</returns>
        /// <remarks>
        /// <para>
        /// This method checks if a factory exists for the specified model type.
        /// </para>
        /// <para><b>For Beginners:</b> This checks if we know how to create a specific pruned model type.
        /// </para>
        /// </remarks>
        public static bool IsFactoryRegistered<TModel>()
        {
            lock (_lock)
            {
                return _factories.ContainsKey(typeof(TModel));
            }
        }
        
        /// <summary>
        /// Unregisters a factory for a specific model type.
        /// </summary>
        /// <typeparam name="TModel">The type of model.</typeparam>
        /// <returns>True if the factory was successfully removed, false if it wasn't registered.</returns>
        /// <remarks>
        /// <para>
        /// This method removes a factory from the registry.
        /// </para>
        /// <para><b>For Beginners:</b> This removes support for a model type.
        /// </para>
        /// </remarks>
        public static bool UnregisterFactory<TModel>()
        {
            lock (_lock)
            {
                return _factories.Remove(typeof(TModel));
            }
        }
        
        /// <summary>
        /// Clears all registered factories.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method removes all factories from the registry.
        /// </para>
        /// <para><b>For Beginners:</b> This removes support for all model types.
        /// 
        /// Use this carefully as it will remove all registered factories,
        /// making deserialization impossible until factories are re-registered.
        /// </para>
        /// </remarks>
        public static void Clear()
        {
            lock (_lock)
            {
                _factories.Clear();
            }
        }
    }
}