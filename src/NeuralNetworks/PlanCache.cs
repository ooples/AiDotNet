using System.Security.Cryptography;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Disk-backed store for compiled inference plans. Persists the traced plan after the
/// first compilation so subsequent process starts load the pre-compiled plan instead
/// of re-tracing + re-compiling. Directly wraps Tensors' <see cref="CompiledPlanLoader"/>
/// and <see cref="ICompiledPlan{T}.SaveAsync"/>.
/// </summary>
/// <remarks>
/// <para>
/// PyTorch-parity equivalent: <c>torch.jit.save(traced_module, path)</c> +
/// <c>torch.jit.load(path)</c>. The facade integration is opt-in via
/// <c>AiModelBuilder.ConfigurePlanCaching(directory)</c>; once configured, save/load
/// is transparent to the caller.
/// </para>
/// <para>
/// Plans are keyed by (modelTypeName, T, structureVersion, inputShapeHash,
/// hardwareFingerprint). Plans compiled on one host cannot be loaded on a host with
/// a different <see cref="PlanCompatibilityInfo"/> — Tensors rejects the load and we
/// fall through to a fresh compile.
/// </para>
/// </remarks>
public sealed class PlanCache
{
    private static PlanCache? _current;

    /// <summary>
    /// The currently-active plan cache, or null if caching is disabled. Set via
    /// <see cref="SetCurrent(PlanCache?)"/>. <see cref="CompiledModelHost{T}"/>
    /// consults this during Predict to decide whether to attempt disk load/save.
    /// </summary>
    public static PlanCache? Current => _current;

    public static void SetCurrent(PlanCache? cache)
    {
        _current = cache;
    }

    public string Directory { get; }

    public PlanCache(string directory)
    {
        Directory = directory ?? throw new ArgumentNullException(nameof(directory));
        System.IO.Directory.CreateDirectory(directory);
    }

    /// <summary>
    /// Computes a stable filename for a plan identified by model type, element type,
    /// structure version, and input shape. Hardware fingerprint is not part of the
    /// key — <see cref="PlanCompatibilityInfo"/> checks that at load time.
    /// </summary>
    public string GetPlanPath(string modelTypeName, Type elementType, int structureVersion, int[] inputShape)
    {
        var hash = ComputeShapeHash(inputShape);
        var safeModelName = SanitizeFilename(modelTypeName);
        return Path.Combine(
            Directory,
            $"{safeModelName}_{elementType.Name}_v{structureVersion}_s{hash}.plan");
    }

    /// <summary>
    /// Attempts to load a pre-compiled inference plan from disk. Returns null if the
    /// file doesn't exist, is incompatible with the current host, or fails to
    /// deserialize. A null return cleanly triggers fresh compilation upstream.
    /// </summary>
    public async Task<ICompiledPlan<T>?> TryLoadInferenceAsync<T>(
        string modelTypeName,
        int structureVersion,
        int[] inputShape,
        IEngine engine,
        CancellationToken cancellationToken = default)
    {
        var path = GetPlanPath(modelTypeName, typeof(T), structureVersion, inputShape);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            return await CompiledPlanLoader.LoadInferenceAsync<T>(path, engine, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // Incompatible plan, corrupted file, or stream error — fall through to
            // recompile. Trace the failure so perf regressions don't go silent.
            System.Diagnostics.Trace.TraceWarning(
                $"PlanCache: load failed for '{path}': {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Persists a compiled plan to disk via atomic write (tmp-file + rename) so a
    /// crash partway through doesn't leave a corrupt cache entry.
    /// </summary>
    public async Task SaveInferenceAsync<T>(
        ICompiledPlan<T> plan,
        string modelTypeName,
        int structureVersion,
        int[] inputShape,
        CancellationToken cancellationToken = default)
    {
        var path = GetPlanPath(modelTypeName, typeof(T), structureVersion, inputShape);
        var tmpPath = path + ".tmp";

        try
        {
            using (var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await plan.SaveAsync(fs, cancellationToken).ConfigureAwait(false);
            }

            if (File.Exists(path))
            {
                File.Delete(path);
            }
            File.Move(tmpPath, path);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"PlanCache: save failed for '{path}': {ex.GetType().Name}: {ex.Message}");
            try { if (File.Exists(tmpPath)) File.Delete(tmpPath); } catch { }
        }
    }

    private static string SanitizeFilename(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(name.Length);
        foreach (var c in name)
        {
            sb.Append(Array.IndexOf(invalid, c) >= 0 ? '_' : c);
        }
        return sb.ToString();
    }

    private static string ComputeShapeHash(int[] shape)
    {
        using var sha = SHA256.Create();
        var bytes = new byte[shape.Length * 4];
        Buffer.BlockCopy(shape, 0, bytes, 0, bytes.Length);
        var hash = sha.ComputeHash(bytes);
        var sb = new StringBuilder(16);
        for (int i = 0; i < 8; i++) sb.Append(hash[i].ToString("x2"));
        return sb.ToString();
    }
}
