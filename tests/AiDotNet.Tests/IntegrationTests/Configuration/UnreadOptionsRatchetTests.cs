using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Ratchets the options surface that <see cref="OptionsSurfaceRatchetTests"/> cannot see.
/// </summary>
/// <remarks>
/// <para>
/// Both metrics in <see cref="OptionsSurfaceRatchetTests"/> count DEFAULTED CONSTRUCTOR
/// PARAMETERS. Driving them to their floor would leave the larger half of #2090 untouched,
/// because a property that is declared and then read by nobody is invisible to a parameter
/// count. That is the worse defect of the two: a constructor parameter at least does something,
/// whereas an unread property advertises configurability that does not exist.
/// </para>
/// <para>
/// <b>Why IL and not a text search.</b> Every previous count in this issue was produced by a
/// regular expression over C# source, and every one of them was an undercount -- 205 to 470, 806
/// to 1067, 97 to 408, 63 to 110 -- each time because the pattern was narrower than the defect.
/// Generic type arguments alone defeated three separate tools. Reading the compiled IL asks the
/// question the source cannot be trusted to answer: does any method in the assembly actually
/// CALL this property's getter?
/// </para>
/// </remarks>
public class UnreadOptionsRatchetTests
{
    /// <summary>
    /// Options properties whose getter is called from nowhere in the product assembly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Lower this deliberately as the surface shrinks, exactly like the constructor ratchets.
    /// It must never rise: a new unread property is a new instance of the defect.
    /// </para>
    /// <para>
    /// The three standing at the time of writing, and why each is here rather than fixed:
    /// </para>
    /// <list type="bullet">
    /// <item><description>
    /// <c>MatryoshkaEmbeddingOptions.MaxEmbeddingDimension</c> — the model bounds requested
    /// dimensions against an inherited <c>EmbeddingDimension</c> instead. One of the two is
    /// redundant and deciding which needs the embedding family looked at as a whole.
    /// </description></item>
    /// <item><description>
    /// <c>AudioVisualEventLocalizationOptions.LearningRate</c> — the unread-rate defect, one
    /// instance of which survived the sweep across 127 models.
    /// </description></item>
    /// <item><description>
    /// <c>ProgressiveGANOptions.LearningRateDecay</c> — set to 0.9999 in the constructor and
    /// consulted by nothing; the GAN training loop needs checking before it is wired or dropped.
    /// </description></item>
    /// </list>
    /// <para>
    /// Three more were deleted outright rather than counted: <c>InputType</c> on
    /// ProgressiveGANOptions, BigGANOptions and SAGANOptions duplicated
    /// <see cref="AiDotNet.NeuralNetworks.NeuralNetworkArchitecture{T}"/>'s own property, which is
    /// the authority on a model's input shape.
    /// </para>
    /// </remarks>
    /// <para>
    /// Raised 3 -> 35 when the scan stopped counting getter calls made from inside the options
    /// hierarchy itself. That is not a regression: nothing became unread, 32 properties that were
    /// ALREADY unread stopped being hidden by their own class's copy constructor. The old number
    /// was measuring the wrong thing, and a metric that flatters itself is worse than a high one.
    /// </para>
    /// <para>
    /// What it exposed, in descending size: ConcertoOptions 10 (a self-supervised pretraining
    /// configuration -- teacher momentum, loss weights, upcast levels -- that the model never
    /// consults), FinchOptions 6 (Beta1, Beta2, WeightDecay and the gradient-clipping pair, none
    /// of which reach an optimizer), WhisperOptions 5 and AudioGenOptions 4 (ONNX component paths
    /// that nothing loads), TtsOptions 2, then eight singles. Each is the defect #2090 calls the
    /// worse of the two: a property that advertises configurability which does not exist.
    /// </para>
    /// <para>
    /// Every one of these is real work, not an accounting artefact. Lower this as they are wired
    /// in or deleted.
    /// </para>
    private const int UnreadBaseline = 35;

    /// <summary>
    /// Zero. A ratchet with headroom is a ratchet that drifts; the constructor ratchets carry
    /// slack only because their counts are in the hundreds and move in bulk. This one is small
    /// enough to be exact, so any movement in either direction should be recorded deliberately.
    /// </summary>
    private const int Slack = 0;

    private readonly ITestOutputHelper _output;

    public UnreadOptionsRatchetTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Property names that are read by infrastructure rather than by a model, or whose getter is
    /// invoked reflectively, so an IL scan cannot see the call.
    /// </summary>
    private static readonly HashSet<string> ReflectivelyUsed = new(StringComparer.Ordinal)
    {
        // Serialization, cloning and the AutoML search space reach these by name.
        "Seed",
        "RandomSeed",
        "Item",
    };

    [Fact(Timeout = 600000)]
    public async Task OptionsPropertiesAreRead_DoesNotRegress()
    {
        await Task.Yield();

        var assembly = typeof(ModelOptions).Assembly;
        var optionsTypes = assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && IsOptionsType(t))
            .ToList();

        // Every property getter declared by an options type, keyed by metadata token so an IL
        // call site can be matched back to it without resolving generic context.
        var getters = new Dictionary<int, PropertyInfo>();
        foreach (var type in optionsTypes)
        {
            foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                if (property.GetIndexParameters().Length > 0) continue;
                if (ReflectivelyUsed.Contains(property.Name)) continue;

                var getter = property.GetGetMethod();
                if (getter == null) continue;

                getters[getter.MetadataToken] = property;
            }
        }

        var called = ScanCalledMethodTokens(assembly);

        var unread = getters
            .Where(pair => !called.Contains(pair.Key))
            .Select(pair => pair.Value)
            .ToList();

        int count = unread.Count;

        var byType = unread
            .GroupBy(p => p.DeclaringType?.Name ?? "?")
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key, StringComparer.Ordinal)
            .ToList();

        _output.WriteLine($"Options property getters declared: {getters.Count}");
        _output.WriteLine($"Never called from anywhere in {assembly.GetName().Name}: {count}");
        _output.WriteLine(string.Empty);
        _output.WriteLine("Largest offenders:");
        foreach (var group in byType.Take(30))
        {
            _output.WriteLine($"  {StripArity(group.Key)} ({group.Count()}): "
                + string.Join(", ", group.Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal).Take(10)));
        }

        Assert.True(count <= UnreadBaseline + Slack,
            $"Unread options properties rose from {UnreadBaseline} to {count}. "
            + "A declared property that nothing reads advertises configurability that does not "
            + "exist -- either wire it into the model or delete it. If this run instead shows a "
            + $"DROP, lower the UnreadBaseline constant to {count} so the progress is recorded in "
            + "the diff rather than silently absorbed.");

        Assert.True(count >= UnreadBaseline - Slack,
            $"Unread options properties fell from {UnreadBaseline} to {count}. That is the goal -- "
            + $"now lower the UnreadBaseline constant to {count}.");
    }

    /// <summary>
    /// Collects the metadata token of every method targeted by a call/callvirt instruction
    /// anywhere in the assembly.
    /// </summary>
    /// <param name="assembly">The product assembly.</param>
    /// <returns>The set of called method tokens.</returns>
    /// <remarks>
    /// <para>
    /// Tokens are compared directly rather than resolved. Resolving a token from a generic method
    /// body needs the declaring type's generic arguments and throws without them, which would make
    /// the scan silently skip exactly the generic models this codebase is made of. A raw token
    /// comparison is both cheaper and immune to that.
    /// </para>
    /// </remarks>
    private static HashSet<int> ScanCalledMethodTokens(Assembly assembly)
    {
        var (single, multi) = BuildOpCodeTables();
        var called = new HashSet<int>();

        foreach (var type in assembly.GetTypes())
        {
            // An options class reading its OWN properties proves nothing about whether a model
            // reads them, and the copy constructor every one of these classes carries reads every
            // property it declares -- `TeacherMomentum = other.TeacherMomentum;` is a real getter
            // call in IL. Counting those made the whole hierarchy look consumed: ConcertoOptions'
            // seven never-read properties were invisible purely because it has a copy
            // constructor, while MatryoshkaEmbeddingOptions.MaxEmbeddingDimension was caught only
            // because that class has none. Validate() has the same effect on a smaller scale.
            //
            // Skipping the hierarchy asks the question that matters: does anything OUTSIDE the
            // options classes consume this value?
            if (IsOptionsType(type)) continue;

            IEnumerable<MethodBase> methods;
            try
            {
                methods = type
                    .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                        | BindingFlags.Static | BindingFlags.DeclaredOnly)
                    .Cast<MethodBase>()
                    .Concat(type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic
                        | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly));
            }
            catch
            {
                continue;
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

                if (il == null) continue;

                Walk(il, single, multi, called);
            }
        }

        return called;
    }

    /// <summary>
    /// Decodes one method body, stepping over each operand by its declared size.
    /// </summary>
    /// <param name="il">The method's IL.</param>
    /// <param name="single">Single-byte opcodes.</param>
    /// <param name="multi">Two-byte (0xFE-prefixed) opcodes.</param>
    /// <param name="called">Set collecting every called method token.</param>
    /// <remarks>
    /// <para>
    /// A naive scan for the call/callvirt opcode BYTES would also match those values occurring
    /// inside an operand, marking properties as read that never are -- an undercount, which is the
    /// failure mode this whole issue keeps repeating. Stepping the instruction stream properly is
    /// the only way the answer means anything.
    /// </para>
    /// </remarks>
    private static void Walk(
        byte[] il,
        IReadOnlyDictionary<byte, System.Reflection.Emit.OpCode> single,
        IReadOnlyDictionary<byte, System.Reflection.Emit.OpCode> multi,
        HashSet<int> called)
    {
        int i = 0;
        while (i < il.Length)
        {
            System.Reflection.Emit.OpCode opCode;

            if (il[i] == 0xFE)
            {
                if (i + 1 >= il.Length || !multi.TryGetValue(il[i + 1], out opCode)) return;
                i += 2;
            }
            else
            {
                if (!single.TryGetValue(il[i], out opCode)) return;
                i += 1;
            }

            int operandSize;
            switch (opCode.OperandType)
            {
                case System.Reflection.Emit.OperandType.InlineNone:
                    operandSize = 0;
                    break;
                case System.Reflection.Emit.OperandType.ShortInlineBrTarget:
                case System.Reflection.Emit.OperandType.ShortInlineI:
                case System.Reflection.Emit.OperandType.ShortInlineVar:
                    operandSize = 1;
                    break;
                case System.Reflection.Emit.OperandType.InlineVar:
                    operandSize = 2;
                    break;
                case System.Reflection.Emit.OperandType.InlineI8:
                case System.Reflection.Emit.OperandType.InlineR:
                    operandSize = 8;
                    break;
                case System.Reflection.Emit.OperandType.InlineSwitch:
                    if (i + 4 > il.Length) return;
                    operandSize = 4 + (4 * BitConverter.ToInt32(il, i));
                    break;
                default:
                    operandSize = 4;
                    break;
            }

            if (i + operandSize > il.Length) return;

            bool isCall = opCode.OperandType == System.Reflection.Emit.OperandType.InlineMethod;
            if (isCall && operandSize == 4)
            {
                called.Add(BitConverter.ToInt32(il, i));
            }

            i += operandSize;
        }
    }

    /// <summary>Builds the opcode lookup tables used to step the instruction stream.</summary>
    /// <returns>The single-byte and 0xFE-prefixed opcode tables.</returns>
    private static (Dictionary<byte, System.Reflection.Emit.OpCode> Single,
        Dictionary<byte, System.Reflection.Emit.OpCode> Multi) BuildOpCodeTables()
    {
        var single = new Dictionary<byte, System.Reflection.Emit.OpCode>();
        var multi = new Dictionary<byte, System.Reflection.Emit.OpCode>();

        foreach (var field in typeof(System.Reflection.Emit.OpCodes)
            .GetFields(BindingFlags.Public | BindingFlags.Static))
        {
            if (field.GetValue(null) is not System.Reflection.Emit.OpCode opCode) continue;

            if (opCode.Size == 1)
            {
                single[(byte)(opCode.Value & 0xFF)] = opCode;
            }
            else
            {
                multi[(byte)(opCode.Value & 0xFF)] = opCode;
            }
        }

        return (single, multi);
    }

    /// <summary>
    /// Restricted to the MODEL HYPERPARAMETER options, which are the only ones carrying the
    /// contract this ratchet enforces.
    /// </summary>
    /// <param name="type">A candidate type.</param>
    /// <returns>True when the type is a model hyperparameter options class.</returns>
    /// <remarks>
    /// <para>
    /// <see cref="ModelHyperparameterOptions"/> documents the rule in its own summary: "Every
    /// property here must be read by the model that owns it." Nothing else in the codebase makes
    /// that promise.
    /// </para>
    /// <para>
    /// Widening this to every type whose name ends in "Options" measures something else entirely
    /// and reports around 6,665 properties as unread. Infrastructure configuration --
    /// ReportOptions, RLTrainingOptions, AiModelResultOptions -- is a PUBLIC API surface consumed
    /// by AiDotNet.Serving, AiDotNet.Dashboard and user code, none of which is in the assembly
    /// being scanned, so an absent call proves nothing there. Counting those would make the
    /// ratchet a number that cannot be driven to zero and does not describe a defect.
    /// </para>
    /// </remarks>
    private static bool IsOptionsType(Type type)
        => typeof(ModelHyperparameterOptions).IsAssignableFrom(type);

    private static string StripArity(string typeName)
    {
        int tick = typeName.IndexOf('`');
        return tick < 0 ? typeName : typeName.Substring(0, tick);
    }
}
