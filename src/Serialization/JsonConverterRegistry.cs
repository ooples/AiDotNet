namespace AiDotNet.Serialization;

/// <summary>
/// Provides global registration of custom JSON converters for serialization of library-specific types.
/// </summary>
/// <remarks>
/// <para>
/// This static class handles the registration of custom JSON converters that are needed for proper
/// serialization and deserialization of complex types like Matrix&lt;T&gt;, Vector&lt;T&gt;, and Tensor&lt;T&gt;.
/// It should be called during application startup to ensure converters are available globally.
/// </para>
/// <para><b>For Beginners:</b> This class sets up special handlers that teach JSON how to save and load our custom types.
/// 
/// Think of this like setting up translators for different languages:
/// - Our Matrix, Vector, and Tensor types are like foreign languages that JSON doesn't understand natively
/// - These converters act as translators that help JSON work with these special types
/// - By registering them once at startup, we ensure they're available throughout the application
/// 
/// This prevents serialization errors when:
/// - Saving models to files
/// - Loading models from files
/// - Sending models between different parts of your application
/// </para>
/// </remarks>
public static class JsonConverterRegistry
{
    private static bool _isInitialized = false;
    private static readonly object _lockObject = new object();
    private static readonly Dictionary<Type, List<JsonConverter>> _cachedConverters = [];

    /// <summary>
    /// Registers all custom JSON converters for the library.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method registers all custom JSON converters needed for the library's serialization.
    /// It's thread-safe and ensures converters are only registered once, even if called from multiple threads.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up all the special handlers our library needs for JSON serialization.
    /// 
    /// Call this method once when your application starts to make sure all the necessary JSON converters are registered.
    /// It's safe to call this method multiple times - it will only do the setup work once.
    /// 
    /// Example usage in your application startup code:
    /// ```csharp
    /// JsonConverterRegistry.RegisterAllConverters();
    /// ```
    /// </para>
    /// </remarks>
    public static void RegisterAllConverters()
    {
        if (_isInitialized)
            return;

        lock (_lockObject)
        {
            if (_isInitialized)
                return;

            // Register converters for common numeric types
            RegisterConverters<float>();
            RegisterConverters<double>();
            RegisterConverters<decimal>();
            RegisterConverters<int>();
            RegisterConverters<long>();

            // Set up the global Newtonsoft.Json settings
            SetupGlobalJsonSettings();

            _isInitialized = true;
        }
    }

    /// <summary>
    /// Registers JSON converters for a specific numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type for which to register converters.</typeparam>
    /// <remarks>
    /// <para>
    /// This method registers Matrix&lt;T&gt;, Vector&lt;T&gt;, and Tensor&lt;T&gt; converters for a specific numeric type.
    /// It stores these converters in a cache for later use with global JSON settings.
    /// </para>
    /// <para><b>For Beginners:</b> This method registers converters for a specific number type like float or double.
    /// 
    /// Since our library works with different numeric types (float, double, decimal), we need
    /// separate converters for each type combination. This method sets up converters for one specific type.
    /// </para>
    /// </remarks>
    private static void RegisterConverters<T>()
    {
        var matrixConverter = new MatrixJsonConverter<T>();
        var vectorConverter = new VectorJsonConverter<T>();
        var tensorConverter = new TensorJsonConverter<T>();

        var converters = new List<JsonConverter>
        {
            matrixConverter,
            vectorConverter,
            tensorConverter
        };

        _cachedConverters[typeof(T)] = converters;
    }

    /// <summary>
    /// Sets up global JSON settings to include our custom converters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method configures the global JsonConvert.DefaultSettings to include all our custom converters.
    /// It merges any existing converters with our custom ones to avoid overriding other settings.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the global JSON settings to use our special handlers.
    /// 
    /// It's like setting default translation rules for all JSON operations in the application.
    /// </para>
    /// </remarks>
    private static void SetupGlobalJsonSettings()
    {
        // Preserve any existing DefaultSettings
        var existingSettingsFactory = JsonConvert.DefaultSettings;

        // Create a new factory that incorporates our converters
        JsonConvert.DefaultSettings = () =>
        {
            // Start with existing settings or create new ones
            var settings = existingSettingsFactory?.Invoke() ?? new JsonSerializerSettings();

            // Create a list for all converters
            var allConverters = new List<JsonConverter>();

            // Add existing converters if any
            if (settings.Converters != null)
            {
                allConverters.AddRange(settings.Converters);
            }

            // Add our cached converters
            foreach (var converterList in _cachedConverters.Values)
            {
                allConverters.AddRange(converterList);
            }

            // Update the settings
            settings.Converters = allConverters;

            return settings;
        };
    }

    /// <summary>
    /// Gets JSON converters for a specific numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type for which to get converters.</typeparam>
    /// <returns>A list of converters for the specified type.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the list of JSON converters for a specific numeric type.
    /// If converters for this type haven't been registered yet, it registers them first.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you just the converters needed for a specific number type.
    /// 
    /// This is useful when you only need converters for one particular type, rather than all types.
    /// </para>
    /// </remarks>
    public static List<JsonConverter> GetConvertersForType<T>()
    {
        // Ensure we're initialized
        RegisterAllConverters();

        // Get or create converters for this type
        if (!_cachedConverters.TryGetValue(typeof(T), out var converters))
        {
            RegisterConverters<T>();
            converters = _cachedConverters[typeof(T)];
        }

        return converters;
    }
}