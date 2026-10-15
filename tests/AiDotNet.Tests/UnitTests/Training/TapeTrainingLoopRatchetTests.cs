using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// No new hand-written tape-and-optimizer training loop may appear outside the shared training step.
/// </summary>
/// <remarks>
/// <para>
/// Every base class that trains on the gradient tape steps through <c>TapeTrainingStepper</c>: the fused compiled plan
/// (forward, backward and the optimizer update as one replay, on CPU or GPU) when it applies, one shared eager tape step
/// otherwise. A model that opens its own <c>GradientTape</c> and applies its own update gets neither the fused path nor
/// the fixes made to the shared step (the update running inside the tape's arena scope, the committed-plan rule, the
/// #1822 persistence probe), and each copy drifts on its own. That is how the base classes ended up with a dozen
/// divergent loops before #1804.
/// </para>
/// <para>
/// A file offends when its code (comments excluded) both opens a tape (<c>new GradientTape&lt;</c>) and applies an
/// update itself: builds a <c>TapeStepContext</c> for an optimizer, subtracts gradients in place, or calls
/// <c>UpdateParameters</c> / <c>ApplyGradients</c>. Opening a tape only to read gradients (attribution, adversarial
/// attacks, a gradient API) is not a training loop and does not match.
/// </para>
/// <para>
/// <see cref="KnownLoops"/> lists the loops that existed when the ratchet was introduced. It only shrinks: a migrated
/// file must be removed from it (the second test fails until it is), and a file not on it fails the first test.
/// </para>
/// </remarks>
public class TapeTrainingLoopRatchetTests
{
    private static readonly Regex OpensTape = new(
        @"\bnew\s+(?:[\w.]+\.)?GradientTape<", RegexOptions.Compiled);

    // The definition "void ApplyGradients(" is an API, not a call; only calls count.
    private static readonly Regex AppliesUpdate = new(
        @"\bnew\s+(?:[\w.]+\.)?TapeStepContext<|\bTensorSubtractInPlace\(|\.UpdateParameters\(|(?<!\bvoid )(?<!\.)\bApplyGradients\(|\.ApplyGradients\(",
        RegexOptions.Compiled);

    private static readonly Regex CommentLine = new(@"^\s*(//|\*|/\*)", RegexOptions.Compiled);

    /// <summary>The shared training infrastructure itself, which is where the loop is supposed to live.</summary>
    private static readonly string[] SharedInfrastructurePrefixes =
    {
        "Training/",
    };

    /// <summary>
    /// Files allowed to keep their loops for a stated structural reason, not as debt.
    /// </summary>
    private static readonly Dictionary<string, string> Exempt = new(StringComparer.Ordinal)
    {
        ["NeuralNetworks/NeuralNetworkBase.cs"] =
            "the base the shared session was extracted from; its remaining tape loops are the gradient-accumulation, "
            + "streaming and custom-loss variants of that same step, which own the NN-specific parameter selection",
    };

    /// <summary>
    /// Hand-written loops that predate the ratchet. Remove an entry when its file moves onto the shared step; never add one.
    /// </summary>
    private static readonly HashSet<string> KnownLoops = new(StringComparer.Ordinal)
    {
        "Diffusion/NoisePredictors/NoisePredictorBase.cs",
        "MetaLearning/Algorithms/ATAMLAlgorithm.cs",
        "MetaLearning/Algorithms/LEOAlgorithm.cs",
        "MetaLearning/Algorithms/MCLAlgorithm.cs",
        "MetaLearning/Algorithms/MatchingNetworksAlgorithm.cs",
        "MetaLearning/Algorithms/MetaOptNetAlgorithm.cs",
        "MetaLearning/Algorithms/ProtoNetsAlgorithm.cs",
        "MetaLearning/Algorithms/RelationNetworkAlgorithm.cs",
        "NeuralNetworks/GraphGenerationModel.cs",
        "NeuralNetworks/SyntheticData/AutoDiffTabGenerator.cs",
        "NeuralNetworks/SyntheticData/MisGANGenerator.cs",
        "NeuralNetworks/SyntheticData/TabSynGenerator.cs",
        "NeuralNetworks/SyntheticData/TimeGANGenerator.cs",
        "ReinforcementLearning/Agents/MuZeroAgent.cs",
        "ReinforcementLearning/Agents/QMIXAgent.cs",
    };

    [Fact]
    public void NoNewHandWrittenTapeTrainingLoop()
    {
        var offenders = ScanOffenders(out int scanned);
        var added = offenders
            .Where(file => !KnownLoops.Contains(file) && !Exempt.ContainsKey(file))
            .OrderBy(file => file, StringComparer.Ordinal)
            .ToList();

        Assert.True(
            added.Count == 0,
            $"{added.Count} file(s) of {scanned} scanned open a GradientTape and apply their own optimizer update. "
            + "Train through the shared step instead: TapeTrainingStepper.Step (forward + loss; fused plan on any "
            + "engine, eager fallback), TapeTrainingStepper.EagerObjectiveStep (an objective that is not a forward of "
            + "one input), or FusedTrainingStep.Step for a model without a base-class stepper. Offenders: "
            + string.Join(", ", added));
    }

    [Fact]
    public void KnownLoopsListShrinksWhenALoopIsMigrated()
    {
        var offenders = new HashSet<string>(ScanOffenders(out _), StringComparer.Ordinal);
        var stale = KnownLoops
            .Where(file => !offenders.Contains(file))
            .OrderBy(file => file, StringComparer.Ordinal)
            .ToList();

        Assert.True(
            stale.Count == 0,
            "These files no longer contain a hand-written tape training loop; remove them from "
            + $"{nameof(TapeTrainingLoopRatchetTests)}.{nameof(KnownLoops)} so the ratchet keeps them migrated: "
            + string.Join(", ", stale));
    }

    [Fact]
    public void ScanDetectsTheSharedStepItself()
    {
        // Non-vacuity: the shared step opens a tape and applies the update, so a scan that cannot see it is
        // reading the wrong tree or the patterns have drifted from the code.
        var all = ScanOffenders(out int scanned, includeShared: true);
        Assert.True(scanned > 500, $"Only {scanned} source files scanned; the guard is not reaching the library.");
        Assert.Contains("Training/TapeTrainingStepper.cs", all);
    }

    private static List<string> ScanOffenders(out int scanned, bool includeShared = false)
    {
        string sourceRoot = LocateSourceRoot();
        var offenders = new List<string>();
        scanned = 0;
        foreach (var path in EnumerateSources(sourceRoot))
        {
            scanned++;
            string relative = Relative(sourceRoot, path);
            if (!includeShared && SharedInfrastructurePrefixes.Any(prefix => relative.StartsWith(prefix, StringComparison.Ordinal)))
                continue;

            var code = string.Join("\n", File.ReadAllLines(path).Where(line => !CommentLine.IsMatch(line)));
            if (OpensTape.IsMatch(code) && AppliesUpdate.IsMatch(code))
                offenders.Add(relative);
        }

        return offenders;
    }

    private static IEnumerable<string> EnumerateSources(string root)
    {
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            string directory = pending.Pop();
            foreach (var child in Directory.GetDirectories(directory))
            {
                string name = Path.GetFileName(child);
                if (!string.Equals(name, "bin", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(name, "obj", StringComparison.OrdinalIgnoreCase))
                {
                    pending.Push(child);
                }
            }

            foreach (var file in Directory.GetFiles(directory, "*.cs"))
                yield return file;
        }
    }

    private static string Relative(string root, string path)
    {
        string full = Path.GetFullPath(path);
        string prefix = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        string relative = full.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ? full.Substring(prefix.Length) : full;
        return relative.Replace('\\', '/');
    }

    private static string LocateSourceRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine(current.FullName, "src");
            if (File.Exists(Path.Combine(current.FullName, "AiDotNet.sln")) && Directory.Exists(candidate))
                return candidate;
            current = current.Parent;
        }

        throw new DirectoryNotFoundException(
            $"Could not find AiDotNet.sln walking up from '{AppContext.BaseDirectory}', so the source tree this "
            + "guard reads is unavailable.");
    }
}