using System.Globalization;
using System.Reflection;

namespace AiDotNet.AutoML;

internal static class AutoMLHyperparameterApplicator
{
    public static void ApplyToOptions(object options, IReadOnlyDictionary<string, object> parameters)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        var optionsType = options.GetType();

        foreach (var (key, value) in parameters)
        {
            if (string.Equals(key, "ModelType", StringComparison.Ordinal))
            {
                continue;
            }

            var property = optionsType.GetProperty(
                key,
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);

            if (property is null || !property.CanWrite)
            {
                continue;
            }

            object? converted = ConvertValue(value, property.PropertyType);
            if (converted is null && property.PropertyType.IsValueType && Nullable.GetUnderlyingType(property.PropertyType) is null)
            {
                continue;
            }

            property.SetValue(options, converted, null);
        }
    }

    private static object? ConvertValue(object value, Type targetType)
    {
        if (value is null)
        {
            return null;
        }

        var underlyingNullable = Nullable.GetUnderlyingType(targetType);
        if (underlyingNullable is not null)
        {
            return ConvertValue(value, underlyingNullable);
        }

        if (targetType.IsInstanceOfType(value))
        {
            return value;
        }

        if (targetType.IsEnum)
        {
            if (value is string s)
            {
                try
                {
                    return Enum.Parse(targetType, s, ignoreCase: true);
                }
                catch (ArgumentException)
                {
                    return null;
                }
                catch (OverflowException)
                {
                    return null;
                }
            }

            try
            {
                return Enum.ToObject(targetType, Convert.ToInt32(value, CultureInfo.InvariantCulture));
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

        if (targetType == typeof(bool))
        {
            try
            {
                return Convert.ToBoolean(value, CultureInfo.InvariantCulture);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException)
            {
                return null;
            }
        }

        if (targetType == typeof(int))
        {
            try
            {
                return Convert.ToInt32(value, CultureInfo.InvariantCulture);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
            {
                return null;
            }
        }

        if (targetType == typeof(double))
        {
            try
            {
                return Convert.ToDouble(value, CultureInfo.InvariantCulture);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
            {
                return null;
            }
        }

        if (targetType == typeof(float))
        {
            try
            {
                return Convert.ToSingle(value, CultureInfo.InvariantCulture);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
            {
                return null;
            }
        }

        if (targetType == typeof(long))
        {
            try
            {
                return Convert.ToInt64(value, CultureInfo.InvariantCulture);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
            {
                return null;
            }
        }

        if (targetType == typeof(string))
        {
            try
            {
                return Convert.ToString(value, CultureInfo.InvariantCulture);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException)
            {
                return null;
            }
        }

        try
        {
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
        catch (ArgumentException)
        {
            return null;
        }
        catch (NotSupportedException)
        {
            return null;
        }
    }
}
