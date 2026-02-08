using System.Globalization;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.Agents;

/// <summary>
/// Applies agent-recommended hyperparameters to a model's options using reflection.
/// </summary>
/// <remarks>
/// <para>
/// This service takes parsed hyperparameter recommendations (from HyperparameterResponseParser)
/// and applies them to a model's options object (accessed via IConfigurableModel). It uses the
/// HyperparameterRegistry to map LLM parameter names to C# property names and validates values
/// against known ranges.
/// </para>
/// <para><b>For Beginners:</b> This is the "hands" of the hyperparameter auto-apply feature.
/// Once the AI agent recommends settings and the parser extracts them, this class actually
/// sets those values on your model's configuration. It reports what was applied, what was
/// skipped, and what failed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
internal class AgentHyperparameterApplicator<T>
{
    private readonly HyperparameterRegistry _registry;

    /// <summary>
    /// Creates a new applicator with the specified registry.
    /// </summary>
    /// <param name="registry">The registry that maps LLM parameter names to C# property names.</param>
    public AgentHyperparameterApplicator(HyperparameterRegistry registry)
    {
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
    }

    /// <summary>
    /// Applies the recommended hyperparameters to the model's options.
    /// </summary>
    /// <param name="model">The model implementing IConfigurableModel.</param>
    /// <param name="modelType">The ModelType for registry lookups.</param>
    /// <param name="hyperparameters">The parsed hyperparameters dictionary.</param>
    /// <returns>A result object describing what was applied, skipped, and failed.</returns>
    public HyperparameterApplicationResult Apply(
        IConfigurableModel<T> model,
        ModelType modelType,
        Dictionary<string, object> hyperparameters)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (hyperparameters is null) throw new ArgumentNullException(nameof(hyperparameters));

        var result = new HyperparameterApplicationResult();
        var options = model.GetOptions()
            ?? throw new InvalidOperationException("Model.GetOptions() returned null. Ensure the model is properly configured.");

        foreach (var kvp in hyperparameters)
        {
            var paramName = kvp.Key;
            var paramValue = kvp.Value;

            try
            {
                ApplyParameter(options, modelType, paramName, paramValue, result);
            }
            catch (TargetInvocationException ex)
            {
                result.Failed[paramName] = $"Unexpected error: {ex.InnerException?.Message ?? ex.Message}";
            }
            catch (InvalidOperationException ex)
            {
                result.Failed[paramName] = $"Unexpected error: {ex.Message}";
            }
        }

        return result;
    }

    private void ApplyParameter(
        ModelOptions options,
        ModelType modelType,
        string paramName,
        object paramValue,
        HyperparameterApplicationResult result)
    {
        // Step 1: Look up the C# property name via registry
        var propertyName = _registry.GetPropertyName(modelType, paramName);

        // Step 2: If not found in registry, try the raw name as a property name
        if (propertyName == null)
        {
            propertyName = paramName;
        }

        // Step 3: Find the property via reflection (case-insensitive)
        var property = FindProperty(options, propertyName);

        if (property == null)
        {
            result.Skipped[paramName] = paramValue;
            return;
        }

        // Step 4: Validate against registry ranges
        var validation = _registry.Validate(modelType, paramName, paramValue);
        if (!validation.IsValid)
        {
            result.Failed[paramName] = validation.Warning ?? $"Validation failed for {paramName}";
            return;
        }
        if (validation.HasWarning)
        {
            result.Warnings.Add(validation.Warning ?? $"Warning for {paramName}");
        }

        // Step 5: Convert type and set value
        var convertedValue = ConvertValue(paramValue, property.PropertyType);
        if (convertedValue == null)
        {
            result.Failed[paramName] = $"Cannot convert value '{paramValue}' to type '{property.PropertyType.Name}'";
            return;
        }

        // Step 6: Skip if user already set a non-default value (don't clobber user config)
        var currentValue = property.GetValue(options);
        if (currentValue != null && !currentValue.Equals(GetDefaultValue(property.PropertyType)))
        {
            result.Skipped[paramName] = paramValue;
            result.Warnings.Add($"Skipping '{paramName}': user-configured value '{currentValue}' preserved.");
            return;
        }

        try
        {
            property.SetValue(options, convertedValue);
            result.Applied[paramName] = convertedValue;
        }
        catch (TargetInvocationException ex)
        {
            result.Failed[paramName] = $"Failed to set property: {ex.InnerException?.Message ?? ex.Message}";
        }
        catch (ArgumentException ex)
        {
            result.Failed[paramName] = $"Failed to set property: {ex.Message}";
        }
    }

    private static PropertyInfo? FindProperty(object target, string propertyName)
    {
        var type = target.GetType();

        // Try exact match first
        var property = type.GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
        if (property != null && property.CanWrite)
        {
            return property;
        }

        // Try normalized match (remove underscores, case-insensitive)
        var normalized = HyperparameterDefinition.NormalizeName(propertyName);
        return type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .FirstOrDefault(p => p.CanWrite && HyperparameterDefinition.NormalizeName(p.Name) == normalized);
    }

    /// <summary>
    /// Converts a hyperparameter value to the target property type.
    /// Reuses the conversion logic pattern from AutoMLHyperparameterApplicator.
    /// </summary>
    internal static object? ConvertValue(object value, Type targetType)
    {
        if (value is null)
        {
            return null;
        }

        var valueType = value.GetType();

        // If already the correct type, return as-is
        if (targetType.IsAssignableFrom(valueType))
        {
            return value;
        }

        // Handle nullable types
        var underlyingType = Nullable.GetUnderlyingType(targetType);
        if (underlyingType != null)
        {
            return ConvertValue(value, underlyingType);
        }

        // Handle enum types
        if (targetType.IsEnum)
        {
            if (value is string strVal)
            {
                try
                {
                    return Enum.Parse(targetType, strVal, ignoreCase: true);
                }
                catch (ArgumentException)
                {
                    return null;
                }
            }

            return null;
        }

        try
        {
            // Handle numeric conversions with InvariantCulture
            if (targetType == typeof(double))
            {
                return Convert.ToDouble(value, CultureInfo.InvariantCulture);
            }
            if (targetType == typeof(float))
            {
                return Convert.ToSingle(value, CultureInfo.InvariantCulture);
            }
            if (targetType == typeof(int))
            {
                return Convert.ToInt32(value, CultureInfo.InvariantCulture);
            }
            if (targetType == typeof(long))
            {
                return Convert.ToInt64(value, CultureInfo.InvariantCulture);
            }
            if (targetType == typeof(bool))
            {
                return Convert.ToBoolean(value, CultureInfo.InvariantCulture);
            }
            if (targetType == typeof(string))
            {
                return value.ToString();
            }

            // Try using Convert.ChangeType as a last resort
            return Convert.ChangeType(value, targetType, CultureInfo.InvariantCulture);
        }
        catch (InvalidCastException)
        {
            return null;
        }
        catch (FormatException)
        {
            return null;
        }
        catch (OverflowException)
        {
            return null;
        }
    }

    private static object? GetDefaultValue(Type type)
    {
        return type.IsValueType ? Activator.CreateInstance(type) : null;
    }
}
