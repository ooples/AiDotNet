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
                if (count < 0 || i + 4 + ((long)count * 4) > il.Length) break;
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

    // Operand sizes from the runtime's own opcode table. A hand-written table missed the 0xFE-prefixed opcodes with no
    // operand (ceq, cgt, clt, readonly.) and unaligned.'s 1-byte operand, treating them as 4-byte tokens: the walk
    // then desynchronized, read a garbage switch count and indexed off the array (IndexOutOfRangeException).
    private static readonly Dictionary<int, System.Reflection.Emit.OperandType> OpcodeOperands = BuildOpcodeTable();

    private static Dictionary<int, System.Reflection.Emit.OperandType> BuildOpcodeTable()
    {
        var table = new Dictionary<int, System.Reflection.Emit.OperandType>();
        foreach (var field in typeof(System.Reflection.Emit.OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetValue(null) is System.Reflection.Emit.OpCode op)
                table[(ushort)op.Value] = op.OperandType;
        }
        return table;
    }

    private static int OperandSize(int opcode, byte[] il, int position, out bool isSwitch)
    {
        isSwitch = false;
        if (!OpcodeOperands.TryGetValue(opcode, out var operand)) return -1;   // unknown opcode: stop, never guess
        switch (operand)
        {
            case System.Reflection.Emit.OperandType.InlineSwitch:
                isSwitch = true;
                return 0;
            case System.Reflection.Emit.OperandType.InlineNone:
                return 0;
            case System.Reflection.Emit.OperandType.ShortInlineBrTarget:
            case System.Reflection.Emit.OperandType.ShortInlineI:
            case System.Reflection.Emit.OperandType.ShortInlineVar:
                return 1;
            case System.Reflection.Emit.OperandType.InlineVar:
                return 2;
            case System.Reflection.Emit.OperandType.InlineI8:
            case System.Reflection.Emit.OperandType.InlineR:
                return 8;
            default:
                return position + 4 <= il.Length ? 4 : -1;
        }
    }
}
