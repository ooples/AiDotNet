using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;

namespace AiDotNet.Configuration;

/// <summary>Reads and writes an <see cref="EvolutionDescriptorDefinition"/> in a YAML configuration file.</summary>
/// <remarks>
/// <para>
/// A descriptor definition is immutable and takes its five values through a constructor that validates them
/// together, which is what stops a half-built axis from ever reaching an archive. The general YAML object mapper can
/// only fill settable properties, so without this converter the one section that decides what an archive measures
/// would be the one section a configuration file could not express. The converter keeps the type immutable and
/// teaches the serializer its shape instead.
/// </para>
/// <para>
/// The YAML form is a mapping with <c>name</c>, <c>minimum</c>, <c>maximum</c>, <c>binCount</c>, and an optional
/// <c>outOfRangePolicy</c> naming one of the <see cref="EvolutionOutOfRangePolicy"/> values. Numbers are read with
/// the invariant culture so a file means the same thing on every machine, an unknown key is refused rather than
/// ignored because a misspelled bound would silently change the grid, and the definition's own constructor performs
/// the validation, so a configuration file and a code path reject exactly the same mistakes.
/// </para>
/// <para><b>For Beginners:</b> This is what lets you write the archive axes in your YAML file:
/// <code>
/// evolution:
///   descriptors:
///     - name: length
///       minimum: 0
///       maximum: 500
///       binCount: 20
///       outOfRangePolicy: Clamp
/// </code>
/// </para>
/// </remarks>
public sealed class EvolutionDescriptorYamlConverter : IYamlTypeConverter
{
    /// <inheritdoc/>
    public bool Accepts(Type type) => type == typeof(EvolutionDescriptorDefinition);

    /// <inheritdoc/>
    /// <exception cref="YamlException">A required key is missing, a value is malformed, or a key is unknown.</exception>
    public object ReadYaml(IParser parser, Type type, ObjectDeserializer rootDeserializer)
    {
        Mark start = parser.Current?.Start ?? Mark.Empty;
        parser.Consume<MappingStart>();

        string? name = null;
        double? minimum = null;
        double? maximum = null;
        int? binCount = null;
        EvolutionOutOfRangePolicy policy = EvolutionOutOfRangePolicy.Reject;

        while (!parser.TryConsume<MappingEnd>(out _))
        {
            Scalar key = parser.Consume<Scalar>();
            Scalar value = parser.Consume<Scalar>();
            switch (key.Value.ToLowerInvariant())
            {
                case "name":
                    name = value.Value;
                    break;
                case "minimum":
                    minimum = ParseDouble(value, key.Value);
                    break;
                case "maximum":
                    maximum = ParseDouble(value, key.Value);
                    break;
                case "bincount":
                    binCount = ParseInt(value, key.Value);
                    break;
                case "outofrangepolicy":
                    policy = ParsePolicy(value);
                    break;
                default:
                    throw new YamlException(key.Start, key.End,
                        $"'{key.Value}' is not a descriptor setting. Use name, minimum, maximum, binCount, or " +
                        "outOfRangePolicy.");
            }
        }

        if (name is null || minimum is null || maximum is null || binCount is null)
            throw new YamlException(start, start,
                "A descriptor needs name, minimum, maximum, and binCount; outOfRangePolicy is optional.");

        try
        {
            return new EvolutionDescriptorDefinition(name, minimum.Value, maximum.Value, binCount.Value, policy);
        }
        catch (Exception exception) when (exception is ArgumentException or ArgumentOutOfRangeException)
        {
            throw new YamlException(start, start, "The descriptor '" + name + "' is not valid: " + exception.Message,
                exception);
        }
    }

    /// <inheritdoc/>
    public void WriteYaml(IEmitter emitter, object? value, Type type, ObjectSerializer serializer)
    {
        if (value is not EvolutionDescriptorDefinition descriptor)
        {
            emitter.Emit(new Scalar(string.Empty));
            return;
        }

        emitter.Emit(new MappingStart());
        EmitPair(emitter, "name", descriptor.Name);
        EmitPair(emitter, "minimum", descriptor.Minimum.ToString("R", CultureInfo.InvariantCulture));
        EmitPair(emitter, "maximum", descriptor.Maximum.ToString("R", CultureInfo.InvariantCulture));
        EmitPair(emitter, "binCount", descriptor.BinCount.ToString(CultureInfo.InvariantCulture));
        EmitPair(emitter, "outOfRangePolicy", descriptor.OutOfRangePolicy.ToString());
        emitter.Emit(new MappingEnd());
    }

    private static void EmitPair(IEmitter emitter, string key, string value)
    {
        emitter.Emit(new Scalar(key));
        emitter.Emit(new Scalar(value));
    }

    private static double ParseDouble(Scalar value, string key) =>
        double.TryParse(value.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed)
            ? parsed
            : throw new YamlException(value.Start, value.End,
                $"'{value.Value}' is not a number, so descriptor setting '{key}' cannot be read.");

    private static int ParseInt(Scalar value, string key) =>
        int.TryParse(value.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
            ? parsed
            : throw new YamlException(value.Start, value.End,
                $"'{value.Value}' is not a whole number, so descriptor setting '{key}' cannot be read.");

    private static EvolutionOutOfRangePolicy ParsePolicy(Scalar value) =>
        Enum.TryParse(value.Value, ignoreCase: true, out EvolutionOutOfRangePolicy parsed) &&
        Enum.IsDefined(typeof(EvolutionOutOfRangePolicy), parsed)
            ? parsed
            : throw new YamlException(value.Start, value.End,
                $"'{value.Value}' is not an out-of-range policy. Use one of: " +
                string.Join(", ", Enum.GetNames(typeof(EvolutionOutOfRangePolicy))) + ".");
}
