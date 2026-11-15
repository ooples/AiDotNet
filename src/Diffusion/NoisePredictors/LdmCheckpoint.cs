using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// One module of an LDM checkpoint: the official names it reads and how to load them. A unit loads all of its
/// tensors or, when one is missing and partial loading is allowed, none of them.
/// </summary>
internal sealed class LdmCheckpointUnit
{
    public LdmCheckpointUnit(IReadOnlyList<string> names, Func<TorchTensorSource, IReadOnlyList<string>> load)
    {
        Names = names;
        Load = load;
    }

    /// <summary>The official tensor names, prefix included.</summary>
    public IReadOnlyList<string> Names { get; }

    /// <summary>Loads the unit and returns the names it read.</summary>
    public Func<TorchTensorSource, IReadOnlyList<string>> Load { get; }
}

/// <summary>
/// Loading helpers for official LDM / Stable Diffusion U-Net checkpoints (<c>openaimodel.py</c> names), shared by the
/// U-Net's modules.
/// </summary>
internal static class LdmCheckpoint
{
    /// <summary>Loads <c>{name}.weight [out, in, k, k]</c> and <c>{name}.bias [out]</c> into a convolution.</summary>
    public static IReadOnlyList<string> Conv<T>(TorchTensorSource source, ConvolutionalLayer<T> conv, string name,
        int inputs, int outputs, int kernel)
    {
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Conv2d(conv, source, name + ".weight",
            new[] { outputs, inputs, kernel, kernel }, name + ".bias");
        return Pair(name);
    }

    /// <summary>Loads <c>{name}.weight</c> and <c>{name}.bias</c>, both <c>[C]</c>, into a GroupNorm.</summary>
    public static IReadOnlyList<string> GroupNorm<T>(TorchTensorSource source, GroupNormalizationLayer<T> norm, string name)
    {
        var engine = AiDotNetEngine.Current;
        var gamma = norm.GetGammaTensor();
        var beta = norm.GetBetaTensor();
        source.CopyTo(name + ".weight", new[] { gamma.Length }, gamma);
        source.CopyTo(name + ".bias", new[] { beta.Length }, beta);
        engine.InvalidatePersistentTensor(gamma);
        engine.InvalidatePersistentTensor(beta);
        return Pair(name);
    }

    /// <summary>Loads <c>{name}.weight [out, in]</c> and <c>{name}.bias [out]</c> into a dense layer.</summary>
    public static IReadOnlyList<string> Linear<T>(TorchTensorSource source, DenseLayer<T> layer, string name, int inputs, int outputs)
    {
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(AiDotNetEngine.Current, layer, inputs, outputs, source,
            name + ".weight", name + ".bias");
        return Pair(name);
    }

    /// <summary>The <c>.weight</c> and <c>.bias</c> names of a module.</summary>
    public static string[] Pair(string name) => new[] { name + ".weight", name + ".bias" };

    /// <summary>
    /// Loads every unit. A unit with a missing tensor throws, or is skipped (keeping its current weights) when
    /// <paramref name="allowPartial"/> is set. A checkpoint tensor under <paramref name="prefix"/> that no unit reads
    /// throws too, unless partial loading is allowed. Both checks run before anything is written, and every name is
    /// read once.
    /// </summary>
    /// <returns>The names loaded, in order.</returns>
    public static IReadOnlyList<string> LoadUnits(TorchTensorSource source, IEnumerable<LdmCheckpointUnit> units,
        IEnumerable<string>? checkpointNames, string prefix, bool allowPartial)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (units is null) throw new ArgumentNullException(nameof(units));
        var all = units.ToList();
        var declared = new HashSet<string>(StringComparer.Ordinal);
        foreach (var unit in all)
        {
            foreach (var name in unit.Names)
            {
                if (!declared.Add(name))
                    throw new InvalidOperationException($"Two U-Net modules declare the checkpoint tensor '{name}'.");
            }
        }

        var present = new List<LdmCheckpointUnit>(all.Count);
        var missing = new List<string>();
        foreach (var unit in all)
        {
            var absent = unit.Names.Where(name => !source.Contains(name)).ToList();
            if (absent.Count == 0) present.Add(unit);
            else missing.AddRange(absent);
        }

        if (!allowPartial)
        {
            if (missing.Count > 0)
                throw new InvalidOperationException(
                    $"The checkpoint is missing {missing.Count} tensor(s) this U-Net needs, first: {string.Join(", ", missing.Take(5))}. " +
                    "Pass allowPartial: true to load the modules that are present.");
            if (checkpointNames is not null)
            {
                var unknown = checkpointNames
                    .Where(name => name.StartsWith(prefix, StringComparison.Ordinal) && !declared.Contains(name))
                    .ToList();
                if (unknown.Count > 0)
                    throw new InvalidOperationException(
                        $"The checkpoint has {unknown.Count} tensor(s) this U-Net has no module for, first: {string.Join(", ", unknown.Take(5))}. " +
                        "Pass allowPartial: true to ignore them.");
            }
        }

        var loaded = new List<string>();
        foreach (var unit in present)
        {
            var read = unit.Load(source);
            if (!read.SequenceEqual(unit.Names, StringComparer.Ordinal))
                throw new InvalidOperationException(
                    $"A U-Net module read different tensors than it declares (first declared: {unit.Names.FirstOrDefault()}).");
            loaded.AddRange(read);
        }

        return loaded;
    }
}
