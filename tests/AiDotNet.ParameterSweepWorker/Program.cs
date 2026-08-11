using System.Collections;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Models.Parameters;

return ParameterSweepWorker.Run(args);

internal static class ParameterSweepWorker
{
    public static int Run(string[] args)
    {
        SweepMeasurement result;
        try
        {
            if ((args.Length is not 3 and not 4) || !long.TryParse(args[2], out long maximum))
                throw new ArgumentException("Usage: <type> <includeChunks> <maximum> [includeLayerBreakdown].");

            bool includeLayerBreakdown = args.Length == 4 && bool.Parse(args[3]);
            result = Measure(args[0], bool.Parse(args[1]), maximum, includeLayerBreakdown);
        }
        catch (Exception ex)
        {
            var cause = ex.GetBaseException();
            result = new SweepMeasurement("error", -1, -1, -1, 0, "Unknown",
                $"{cause.GetType().Name}: {cause.Message}");
        }

        Console.WriteLine(JsonSerializer.Serialize(result));
        return result.Status == "error" ? 2 : 0;
    }

    private static SweepMeasurement Measure(
        string assemblyQualifiedType,
        bool includeChunks,
        long maximum,
        bool includeLayerBreakdown)
    {
        var type = Type.GetType(assemblyQualifiedType, throwOnError: true)!;
        var ctor = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Where(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue))
            .OrderBy(c => c.GetParameters().Length)
            .FirstOrDefault();
        if (ctor is null)
            return new SweepMeasurement("unconstructable", -1, -1, -1, 0, "Unknown", null);

        var ctorArgs = ctor.GetParameters()
            .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue)
            .ToArray();
        object? instance = null;
        try
        {
            instance = ctor.Invoke(ctorArgs);
            string readiness = "Unknown";
            long manifestCount = -1;
            if (instance is IParameterManifestProvider provider)
            {
                var layout = provider.ParameterLayout;
                readiness = layout.Readiness.ToString();
                manifestCount = layout.ParameterCount ?? -1;
                if (layout.Readiness == ParameterReadiness.ShapeDeferred)
                    return new SweepMeasurement("deferred", manifestCount, -1, -1, 0, readiness, null);
                if (layout.Readiness == ParameterReadiness.ShapeResolvedUnmaterialized)
                    return new SweepMeasurement("unmaterialized", manifestCount, -1, -1, 0, readiness, null);
            }

            long declared = ReadLong(instance, "ParameterCount");
            if (declared < 0)
                return new SweepMeasurement("unsupported", declared, -1, -1, 0, readiness, null);
            if (declared > maximum)
                return new SweepMeasurement("too-large", declared, -1, -1, 0, readiness, null);
            if (ReadBool(instance, "HasUninitializedParameters"))
                return new SweepMeasurement("deferred", declared, -1, -1, 0, readiness, null);

            long flat = ReadVectorLength(instance);
            if (!includeChunks)
                return new SweepMeasurement("ok", declared, flat, -1, 0, readiness, null);

            var chunksMethod = type.GetMethod("GetParameterChunks",
                BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
            // Default interface methods are real public API but Type.GetMethod on the concrete
            // class does not return them. Most non-neural hierarchies intentionally inherit the
            // universal IParameterizable flat fallback, so inspect the closed interface before
            // classifying a model as having no chunk API.
            if (chunksMethod is null)
            {
                var parameterizable = type.GetInterfaces().FirstOrDefault(i =>
                    i.IsGenericType &&
                    i.GetGenericTypeDefinition() == typeof(AiDotNet.Interfaces.IParameterizable<,,>));
                chunksMethod = parameterizable?.GetMethod("GetParameterChunks",
                    BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
            }
            if (chunksMethod is null)
                return new SweepMeasurement("no-chunks", declared, flat, -1, 0, readiness, null);

            long chunkSum = 0;
            int chunkCount = 0;
            if (chunksMethod.Invoke(instance, null) is IEnumerable chunks)
            {
                foreach (var chunk in chunks)
                {
                    if (chunk is null) continue;
                    var length = chunk.GetType().GetProperty("Length");
                    if (length is not null) chunkSum = checked(chunkSum + Convert.ToInt64(length.GetValue(chunk)));
                    chunkCount++;
                    if (chunkSum > maximum)
                        return new SweepMeasurement("too-large", declared, flat, chunkSum, chunkCount, readiness, null);
                }
            }

            var layers = includeLayerBreakdown ? ReadLayerBreakdown(instance) : null;
            return new SweepMeasurement("ok", declared, flat, chunkSum, chunkCount, readiness, null, layers);
        }
        finally
        {
            (instance as IDisposable)?.Dispose();
        }
    }

    private static bool ReadBool(object instance, string name)
    {
        try
        {
            return instance.GetType().GetProperty(name,
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy)?.GetValue(instance) is true;
        }
        catch { return false; }
    }

    private static long ReadLong(object instance, string name)
    {
        try
        {
            var value = instance.GetType().GetProperty(name,
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy)?.GetValue(instance);
            return value is null ? -1 : Convert.ToInt64(value);
        }
        catch { return -1; }
    }

    private static long ReadVectorLength(object instance)
    {
        var method = instance.GetType().GetMethod("GetParameters",
            BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
        if (method is null) return -1;
        var vector = method.Invoke(instance, null);
        if (vector is null) return 0;
        var length = vector.GetType().GetProperty("Length");
        return length is null ? -1 : Convert.ToInt64(length.GetValue(vector));
    }

    private static IReadOnlyList<LayerMeasurement>? ReadLayerBreakdown(object instance)
    {
        PropertyInfo? layersProperty = null;
        for (var current = instance.GetType(); current is not null && layersProperty is null; current = current.BaseType)
        {
            layersProperty = current.GetProperty("Layers",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly);
        }

        if (layersProperty?.GetValue(instance) is not IEnumerable layers) return null;

        var result = new List<LayerMeasurement>();
        int index = 0;
        foreach (var layer in layers)
        {
            if (layer is null) continue;
            long declared = ReadLong(layer, "ParameterCount");
            long flat = ReadVectorLength(layer);
            bool supportsTraining = ReadBool(layer, "SupportsTraining");
            long trainable = 0;
            int trainableCount = 0;
            var trainableMethod = layer.GetType().GetMethod("GetTrainableParameters",
                BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
            if (trainableMethod?.Invoke(layer, null) is IEnumerable tensors)
            {
                foreach (var tensor in tensors)
                {
                    if (tensor is null) continue;
                    var length = tensor.GetType().GetProperty("Length");
                    if (length is not null)
                        trainable = checked(trainable + Convert.ToInt64(length.GetValue(tensor)));
                    trainableCount++;
                }
            }

            result.Add(new LayerMeasurement(index++, layer.GetType().FullName ?? layer.GetType().Name,
                declared, flat, supportsTraining, trainable, trainableCount));
        }

        return result;
    }
}

internal sealed record SweepMeasurement(
    string Status,
    long Declared,
    long Flat,
    long ChunkSum,
    int ChunkCount,
    string Readiness,
    string? Error,
    IReadOnlyList<LayerMeasurement>? Layers = null);

internal sealed record LayerMeasurement(
    int Index,
    string Type,
    long Declared,
    long Flat,
    bool SupportsTraining,
    long Trainable,
    int TrainableTensorCount);
