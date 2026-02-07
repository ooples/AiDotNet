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
public class AgentHyperparameterApplicator<T>
{
    private readonly HyperparameterRegistry _registry;

    /// <summary>
    /// Creates a new applicator with the specified registry.
    /// </summary>
    /// <param name="registry">The registry that maps LLM parameter names to C# property names.</param>
    public AgentHyperparameterApplicator(HyperparameterRegistry registry)
    {
        _registry = registry;
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
        var result = new HyperparameterApplicationResult();
        var options = model.GetOptions();

        foreach (var kvp in hyperparameters)
        {
            var paramName = kvp.Key;
            var paramValue = kvp.Value;

            try
            {
                ApplyParameter(options, modelType, paramName, paramValue, result);
            }
            catch (Exception ex)
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

        try
        {
            property.SetValue(options, convertedValue);
            result.Applied[paramName] = paramValue;
        }
        catch (Exception ex)
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
        foreach (var prop in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (prop.CanWrite && HyperparameterDefinition.NormalizeName(prop.Name) == normalized)
            {
                return prop;
            }
        }

        return null;
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

        try
        {
            // Handle numeric conversions
            if (targetType == typeof(double))
            {
                return Convert.ToDouble(value);
            }
            if (targetType == typeof(float))
            {
                return Convert.ToSingle(value);
            }
            if (targetType == typeof(int))
            {
                return Convert.ToInt32(value);
            }
            if (targetType == typeof(long))
            {
                return Convert.ToInt64(value);
            }
            if (targetType == typeof(bool))
            {
                return Convert.ToBoolean(value);
            }
            if (targetType == typeof(string))
            {
                return value.ToString();
            }

            // Try using Convert.ChangeType as a last resort
            return Convert.ChangeType(value, targetType);
        }
        catch
        {
            return null;
        }
    }
}
