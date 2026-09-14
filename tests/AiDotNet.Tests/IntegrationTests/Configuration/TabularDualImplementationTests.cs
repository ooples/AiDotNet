using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Guards the tabular models that exist TWICE against the two copies honouring different options.
/// </summary>
/// <remarks>
/// <para>
/// Six tabular models ship as two independent implementations over one shared options class:
/// <c>XNetwork</c>, which builds its layers through <c>LayerHelper</c>, and <c>XBase</c> (with
/// <c>XClassifier</c> / <c>XRegression</c> derived from it), which builds them inline. Nothing
/// connects them, so a property wired into one is silently ignored by the other — setting it then
/// changes the model or not depending on which class the caller happened to pick, with no error.
/// </para>
/// <para>
/// That is exactly how <c>HiddenVectorActivation</c> behaved after it was first wired: the
/// LayerHelper path honoured it and the estimator path did not. The instance is fixed; this test
/// exists so the CLASS cannot recur silently. It is deliberately a structural check rather than a
/// numeric one — it asks whether both implementations mention each shared knob at all, which is
/// cheap, needs no training run, and fails loudly the moment one side gains a property the other
/// never learns about.
/// </para>
/// </remarks>
public class TabularDualImplementationTests
{
    private readonly ITestOutputHelper _output;

    public TabularDualImplementationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Models that exist as both an <c>XNetwork</c> and an <c>XBase</c> over one options class.
    /// </summary>
    private static readonly string[] DualImplementationModels =
    {
        "AutoInt", "Mambular", "NODE", "SAINT", "TabDPT", "TabPFN",
    };

    /// <summary>
    /// Options one implementation reads and the other does not.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Lower this as the two sides converge; never raise it.</b> A rise means a property was
    /// wired into one implementation and not the other -- the defect this guard exists for.
    /// </para>
    /// <para>
    /// Established at 49 when the guard was written. That number is the finding, not a formality:
    /// the divergence was assumed to be one property (HiddenVectorActivation, the one being wired
    /// at the time) and is in fact pervasive -- AutoInt reads DropoutRate only in the Network path
    /// and HiddenActivation only in the Base path, and every one of the six models has a similar
    /// split. Some are structural rather than defects (MaxGradNorm drives the tape-training loop,
    /// which only the Network path has), so the honest move is to record the count and forbid
    /// growth rather than to curate a long exclusion list that quietly grows instead.
    /// </para>
    /// </remarks>
    private const int DivergenceBaseline = 49;

    /// <summary>
    /// Options properties whose absence from one side is legitimate rather than a divergence.
    /// </summary>
    /// <remarks>
    /// Kept deliberately small, and each entry names why. A blanket exclusion list would let the
    /// defect back in one name at a time.
    /// </remarks>
    private static readonly HashSet<string> StructurallyDivergent = new(StringComparer.Ordinal)
    {
        // The estimator path owns its own fit loop, so training-schedule knobs do not apply to the
        // layer-building path at all.
        "MaxEpochs", "BatchSize", "LearningRate", "EarlyStoppingPatience", "ValidationFraction",
        "Seed", "RandomSeed", "Verbose",
    };

    [Fact(Timeout = 120000)]
    public async Task BothImplementationsReadTheSameOptions()
    {
        await Task.Yield();

        var assembly = typeof(AiDotNet.Models.Options.ModelOptions).Assembly;
        var divergences = new List<string>();
        int compared = 0;

        foreach (string model in DualImplementationModels)
        {
            var network = assembly.GetTypes().FirstOrDefault(t => t.Name == model + "Network`1");
            var estimator = assembly.GetTypes().FirstOrDefault(t => t.Name == model + "Base`1");

            if (network is null || estimator is null)
            {
                divergences.Add($"{model}: expected both {model}Network and {model}Base to exist; "
                    + $"found network={network is not null}, estimator={estimator is not null}.");
                continue;
            }

            compared++;

            var networkReads = OptionPropertyNamesRead(network);
            var estimatorReads = OptionPropertyNamesRead(estimator);

            // Only knobs BOTH sides could plausibly use: a property neither reads is the unread
            // defect, which UnreadOptionsRatchetTests already counts.
            var shared = networkReads.Union(estimatorReads)
                .Where(name => !StructurallyDivergent.Contains(name))
                .OrderBy(name => name, StringComparer.Ordinal);

            foreach (string name in shared)
            {
                bool inNetwork = networkReads.Contains(name);
                bool inEstimator = estimatorReads.Contains(name);
                if (inNetwork == inEstimator) continue;

                divergences.Add($"{model}.{name}: read by {(inNetwork ? model + "Network" : model + "Base")} "
                    + $"but not by {(inNetwork ? model + "Base" : model + "Network")}.");
            }
        }

        _output.WriteLine($"Compared {compared} dual-implementation models.");
        foreach (string divergence in divergences) _output.WriteLine("  " + divergence);

        Assert.True(
            divergences.Count <= DivergenceBaseline,
            $"Divergences rose from {DivergenceBaseline} to {divergences.Count}. A tabular model's "
            + "two implementations honour different options, so the same setting changes the model "
            + "or not depending on which class the caller picked:"
            + Environment.NewLine + string.Join(Environment.NewLine, divergences)
            + Environment.NewLine
            + "If this instead shows a DROP, lower DivergenceBaseline so the progress is recorded "
            + "in the diff rather than silently absorbed.");
    }

    /// <summary>
    /// Names of options properties whose getters this type's IL calls.
    /// </summary>
    private static HashSet<string> OptionPropertyNamesRead(Type type)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);

        IEnumerable<MethodBase> methods;
        try
        {
            methods = type
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                    | BindingFlags.Static | BindingFlags.DeclaredOnly)
                .Cast<MethodBase>()
                .Concat(type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic
                    | BindingFlags.Instance | BindingFlags.DeclaredOnly));
        }
        catch
        {
            return names;
        }

        foreach (var method in methods)
        {
            byte[]? il;
            try
            {
                il = method.GetMethodBody()?.GetILAsByteArray();
            }
            catch
            {
                continue;
            }

            if (il is null) continue;

            Type[]? typeArgs = null;
            try
            {
                typeArgs = method.DeclaringType?.IsGenericType == true
                    ? method.DeclaringType.GetGenericArguments()
                    : null;
            }
            catch
            {
                // A type whose arguments cannot be resolved contributes nothing; skipping it can
                // only under-report, and the assertion below treats absence as a divergence only
                // when the OTHER side reports the name.
            }

            foreach (int token in CallTokens(il))
            {
                MethodBase? target;
                try
                {
                    target = method.Module.ResolveMethod(token, typeArgs, null);
                }
                catch
                {
                    continue;
                }

                if (target?.DeclaringType is null) continue;
                if (!target.Name.StartsWith("get_", StringComparison.Ordinal)) continue;
                if (!typeof(AiDotNet.Models.Options.ModelOptions).IsAssignableFrom(target.DeclaringType))
                {
                    continue;
                }

                names.Add(target.Name.Substring(4));
            }
        }

        return names;
    }

    /// <summary>
    /// Operand tokens of call / callvirt instructions, stepping the IL so operand bytes are never
    /// mistaken for opcodes.
    /// </summary>
    private static IEnumerable<int> CallTokens(byte[] il)
    {
        var tokens = new List<int>();
        int i = 0;

        while (i < il.Length)
        {
            int opcodeStart = i;
            int opcode = il[i++];
            if (opcode == 0xFE)
            {
                if (i >= il.Length) break;
                opcode = 0xFE00 | il[i++];
            }

            int operandSize = OperandSize(opcode, il, i, out bool isSwitch);
            if (isSwitch)
            {
                if (i + 4 > il.Length) break;
                int count = BitConverter.ToInt32(il, i);
                i += 4 + (count * 4);
                continue;
            }

            if (operandSize < 0) break;

            // call (0x28), callvirt (0x6F)
            if ((opcode == 0x28 || opcode == 0x6F) && i + 4 <= il.Length)
            {
                tokens.Add(BitConverter.ToInt32(il, i));
            }

            i += operandSize;
            if (i <= opcodeStart) break;
        }

        return tokens;
    }

    private static int OperandSize(int opcode, byte[] il, int position, out bool isSwitch)
    {
        isSwitch = false;

        switch (opcode)
        {
            case 0x45: // switch
                isSwitch = true;
                return 0;

            // inline none
            case 0x00: case 0x01: case 0x02: case 0x03: case 0x04: case 0x05: case 0x06:
            case 0x07: case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D:
            case 0x14: case 0x15: case 0x16: case 0x17: case 0x18: case 0x19: case 0x1A:
            case 0x1B: case 0x1C: case 0x1D: case 0x1E: case 0x25: case 0x26: case 0x2A:
            case 0x58: case 0x59: case 0x5A: case 0x5B: case 0x5C: case 0x5D: case 0x5E:
            case 0x5F: case 0x60: case 0x61: case 0x62: case 0x63: case 0x64: case 0x65:
            case 0x66: case 0x67: case 0x68: case 0x69: case 0x6A: case 0x6B: case 0x6C:
            case 0x6D: case 0x6E: case 0x82: case 0x83: case 0x84: case 0x85: case 0x86:
            case 0x87: case 0x88: case 0x89: case 0x8A: case 0x90: case 0x91: case 0x92:
            case 0x93: case 0x94: case 0x95: case 0x96: case 0x97: case 0x98: case 0x99:
            case 0x9A: case 0x9B: case 0x9C: case 0x9D: case 0x9E: case 0x9F: case 0xA0:
            case 0xA1: case 0xA2: case 0xA3: case 0xA4: case 0xB3: case 0xB4: case 0xB5:
            case 0xB6: case 0xB7: case 0xB8: case 0xB9: case 0xBA: case 0xBB: case 0xBC:
            case 0xBD: case 0xBE: case 0xBF: case 0xC0: case 0xC1: case 0xC2: case 0xC3:
            case 0xCE: case 0xCF: case 0xD0: case 0xD1: case 0xD2: case 0xD3: case 0xD4:
            case 0xD5: case 0xD6: case 0xD7: case 0xDA: case 0xDB: case 0xDC: case 0xDD:
                return 0;

            // inline i1 / var / short branch
            case 0x0E: case 0x0F: case 0x10: case 0x11: case 0x12: case 0x13: case 0x1F:
            case 0x2B: case 0x2C: case 0x2D: case 0x2E: case 0x2F: case 0x30: case 0x31:
            case 0x32: case 0x33: case 0x34: case 0x35: case 0x36: case 0x37: case 0xDE:
                return 1;

            // inline var (2 bytes)
            case 0xFE0C: case 0xFE0D: case 0xFE0E: case 0xFE0F:
                return 2;

            // inline i8 / r8
            case 0x21: case 0x23:
                return 8;

            default:
                // Everything else in use here is a 4-byte operand: tokens, i4, r4, branches.
                return position + 4 <= il.Length ? 4 : -1;
        }
    }
}
