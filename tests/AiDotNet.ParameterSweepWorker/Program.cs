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
            if (args.Length != 3 || !long.TryParse(args[2], out long maximum))
                throw new ArgumentException("Usage: <type> <includeChunks> <maximum>.");

            result = Measure(args[0], bool.Parse(args[1]), maximum);
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

    private static SweepMeasurement Measure(string assemblyQualifiedType, bool includeChunks, long maximum)
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

            return new SweepMeasurement("ok", declared, flat, chunkSum, chunkCount, readiness, null);
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
}

internal sealed record SweepMeasurement(
    string Status,
    long Declared,
    long Flat,
    long ChunkSum,
    int ChunkCount,
    string Readiness,
    string? Error);
