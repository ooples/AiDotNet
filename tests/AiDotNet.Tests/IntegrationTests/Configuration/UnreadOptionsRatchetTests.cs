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
    /// <para>
    /// This briefly read 517, and that number was an artefact of this test, not a fact about the
    /// codebase. Re-parenting <c>FinancialNeuralNetworkOptions</c> brought the tabular and
    /// synthetic-data options into scope, and because those classes are GENERIC the raw-token
    /// comparison then in use could not match a single read of them — see the remarks on
    /// <c>ScanCalledGetters</c>. They were reported wholly unconsumed while their models read them
    /// constantly: <c>CTGANGenerator</c> reads <c>EmbeddingDimension</c> nine times.
    /// </para>
    /// <para>
    /// 112 of 1236 declared getters is the measured figure once call sites resolve properly, and
    /// it agrees with the source property by property — <c>CTGANOptions</c> now reports exactly
    /// one unread member, <c>Epochs</c>, which is the one a grep of <c>CTGANGenerator</c> also
    /// finds zero reads of.
    /// </para>
    /// <para>
    /// Worth recording about how the wrong number survived as long as it did: a conservation check
    /// confirmed the rise (482) could not exceed the newly-visible getters (488), and it did not.
    /// That check passes whether the properties are genuinely unread or systematically mis-scanned
    /// — it bounds magnitude, never correctness. Reading one model and comparing it against the
    /// claim is what exposed it.
    /// </para>
    /// <para>
    /// 112 -> 97 with the duplicate-clipping cluster. <c>EnableGradientClipping</c> and
    /// <c>MaxGradientNorm</c> were deleted from TabROptions, FinchOptions, FTTransformerOptions,
    /// TabNetOptions and TabMOptions: they duplicated the <c>MaxGradNorm</c> inherited from
    /// <c>ModelHyperparameterOptions</c>, which already documents zero-or-negative as "off", so
    /// the separate boolean expressed no state the single double could not. TabNet's published
    /// bound of 2.0 moved onto <c>MaxGradNorm</c> in a new parameterless constructor rather than
    /// being dropped. <c>WeightDecay</c> was wired instead of deleted — TabR, TabM and GANDALF
    /// each built a BARE <c>AdamOptimizer</c>, so it reached nothing; they now build AdamW, whose
    /// learning-rate default is identical, and Finch built no optimizer at all.
    /// </para>
    /// <para>
    /// Found while doing it, and fixed with it: none of those Clone()/copy constructors carried
    /// the INHERITED MaxGradNorm, so a clone silently reset it. That was survivable while each
    /// class had its own duplicate shadowing it, and became a live defect the moment the
    /// duplicate was removed — the same copy-constructor hazard this file already documents,
    /// reached from the opposite direction.
    /// </para>
    /// <para>
    /// 97 -> 66 across two clusters and one detector repair.
    /// </para>
    /// <para>
    /// <b>Epochs, 21 synthetic-data generators.</b> Each declared a per-model published value
    /// (TimeGAN 2000, MedGAN and TabDDPM 1000, TabFlow 500, AutoDiffTab 200, three at 100, the rest
    /// 300) while <c>Fit</c> and <c>FitAsync</c> took <c>epochs</c> as a REQUIRED argument, so none
    /// of them could ever apply -- the doc examples showed both at once. The parameter is now
    /// <c>int? epochs = null</c> resolving to the options value. Deliberately not a sentinel
    /// (<c>int epochs = 0</c> plus <c>epochs &gt; 0 ? epochs : _options.Epochs</c>): that is the
    /// shape removed from CSDI and its siblings in the previous commit, where it made the options
    /// value reachable only by passing zero. TabSyn held the same defect in MIRROR form --
    /// <c>_options.VAEEpochs &gt; 0 ? _options.VAEEpochs : epochs</c>, always true because
    /// VAEEpochs defaults to 100, so the caller's required argument was dead.
    /// </para>
    /// <para>
    /// <b>The tabular cluster.</b> <c>HiddenVectorActivation</c> now reaches the hidden dense
    /// layers of SAINT, TabDPT, TabPFN, Mambular and AutoInt, and <c>FeedForwardDimension</c>
    /// replaces a hardcoded <c>* 4</c> in four builders -- the literal that had been shadowing the
    /// declared multiplier. Fixing that exposed a second defect: <c>FeedForwardDimension</c> was
    /// computed from <c>EmbeddingDimension</c> on SAINT and TabTransformer, which those models
    /// never pass to layer construction, so the computed 128 bore no relation to the 512 actually
    /// built. It now derives from <c>HiddenDimension</c> and every value is unchanged at runtime.
    /// Three properties were DELETED rather than wired, having nothing to wire to: NODE's
    /// <c>HiddenVectorActivation</c> (a tree ensemble plus an output projection has no hidden
    /// activation) and TabR's feed-forward pair (its builder has no transformer feed-forward).
    /// </para>
    /// <para>
    /// <b>The detector under-reported its own fix.</b> Following computed properties required
    /// walking options-class getters, which are otherwise skipped. The first attempt kept the
    /// existing <c>seen</c> set -- keyed on the CALL TARGET and used to skip work -- and the count
    /// rose to 156: whichever walker reached a getter token first claimed it, so a read from
    /// inside an options class recorded its edge and every later read of that property BY A MODEL
    /// was skipped before it could be counted. 83 genuinely-read properties reported as unread.
    /// The repair caches the resolution instead of the decision. Worth keeping in mind here: this
    /// is the second dedup-shaped under-report in this file, after raw metadata-token comparison
    /// once made every generic options class read as 100% unread.
    /// </para>
    /// <para>
    /// The drop from 73 to 66 is exactly 3 deletions plus 4 false positives removed
    /// (FeedForwardMultiplier on SAINT, TabDPT, TabPFN and TabTransformer, each consumed through
    /// the computed property). Matching the prediction to the property is what shows the
    /// propagation reaches what it was built for and nothing else.
    /// </para>
    /// <para>
    /// OPEN: <c>NODEOptions.HiddenActivation</c> is as unwired as the vector sibling just deleted
    /// -- NODENetwork reads no activation at all -- yet it is not reported. It is not in
    /// <see cref="ReflectivelyUsed"/>, the property collection applies no type filter, and the
    /// generated clone registry mentions the name only as a string literal. Something marks it
    /// read and the path was not identified; until it is, this count may UNDER-report.
    /// </para>
    /// </remarks>
    private const int UnreadBaseline = 66;

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

        // Keyed by (declaring type definition, property name) rather than by metadata token.
        //
        // The token version was silently blind to every GENERIC options class. A call to a member
        // of a constructed generic type -- `_options.EmbeddingDimension` where `_options` is
        // `CTGANOptions<T>` -- emits a MemberRef token, never the MethodDef token of the generic
        // definition's getter, so the two integers could not match however many times the model
        // read the property. CTGANGenerator reads EmbeddingDimension nine times and the ratchet
        // called it unread. Since the tabular and synthetic-data options classes are all generic,
        // they appeared ~100% unread, which is how this test once reported 517.
        //
        // Resolving each call site to its declaring type DEFINITION and comparing on names is
        // immune to that: MethodDef, MemberRef and MethodSpec all normalise to the same key.
        var getters = new Dictionary<(string Type, string Name), PropertyInfo>();
        foreach (var type in optionsTypes)
        {
            foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                if (property.GetIndexParameters().Length > 0) continue;
                if (ReflectivelyUsed.Contains(property.Name)) continue;

                var getter = property.GetGetMethod();
                if (getter == null) continue;

                var declaring = property.DeclaringType;
                if (declaring == null) continue;
                if (declaring.IsGenericType) declaring = declaring.GetGenericTypeDefinition();

                getters[(declaring.FullName ?? declaring.Name, property.Name)] = property;
            }
        }

        var called = ScanCalledGetters(assembly, out int unresolvedTokens, out int totalTokens);

        // The defect this test just recovered from was silent blindness, so a scan that cannot
        // resolve a meaningful share of its call sites must fail rather than under-report. A few
        // failures are expected and harmless (tokens from types that fail to load).
        Assert.True(
            totalTokens > 0 && unresolvedTokens < totalTokens / 10,
            $"{unresolvedTokens} of {totalTokens} call tokens could not be resolved. Above a tenth "
                + "the scan is guessing, and an unread property would go unreported the way every "
                + "generic options class did before the token comparison was replaced.");

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
        // Listed in full rather than truncated. This goes to test output, not to an assertion
        // message, so there is no readability budget to protect -- and a truncated report is
        // exactly what makes the remaining work unactionable: the tail below the cut is invisible,
        // so nobody can tell whether it is one more property or forty.
        _output.WriteLine("Unread, by declaring type:");
        foreach (var group in byType)
        {
            _output.WriteLine($"  {StripArity(group.Key)} ({group.Count()}): "
                + string.Join(", ", group.Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal)));
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
    /// This once compared raw tokens, on the reasoning that resolving them needs the declaring
    /// type's generic arguments and throws without them. That was exactly backwards: a raw
    /// comparison is not immune to generics, it is BLIND to them. A call to a member of a
    /// constructed generic type emits a MemberRef token, which never equals the MethodDef token of
    /// the generic definition's getter, so every generic options class read as wholly unconsumed.
    /// </para>
    /// <para>
    /// Each token is now resolved with the declaring method's generic context supplied, then
    /// normalised to its generic type DEFINITION and recorded by (type, name) — the key MethodDef,
    /// MemberRef and MethodSpec all agree on. Tokens that still fail to resolve are COUNTED and
    /// reported to the caller, because the failure mode being repaired here was a scan that
    /// quietly answered "no" when it meant "I could not tell".
    /// </para>
    /// </remarks>
    private static HashSet<(string Type, string Name)> ScanCalledGetters(
        Assembly assembly, out int unresolved, out int total)
    {
        var (single, multi) = BuildOpCodeTables();
        var called = new HashSet<(string Type, string Name)>();

        // computed options property -> the options properties its getter reads
        var derivedFrom = new Dictionary<(string Type, string Name),
            HashSet<(string Type, string Name)>>();
        // Resolution cache. Deliberately NOT a skip-set: the same getter token is reached from
        // many call sites, and what matters is what the CALLER is (a model, or a computed property
        // inside an options class). Skipping a token after the first sighting silently discards
        // every later call site -- which is exactly how this scan once under-reported by 83.
        var resolvedTokens = new Dictionary<(int Module, int Token), (string Type, string Name)?>();
        unresolved = 0;
        total = 0;

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
            // Options types are not scanned for direct reads -- see above -- but their property
            // GETTERS are collected separately, so a computed property can pass consumption on to
            // the properties it derives from.
            bool isOptions = IsOptionsType(type);

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

                // Resolution needs the generic context of the method the IL belongs to, which is
                // the argument the old raw-token comparison was written to avoid needing.
                Type[]? typeArgs = null;
                Type[]? methodArgs = null;
                try
                {
                    typeArgs = method.DeclaringType?.IsGenericType == true
                        ? method.DeclaringType.GetGenericArguments()
                        : null;
                    methodArgs = method.IsGenericMethodDefinition ? method.GetGenericArguments() : null;
                }
                catch
                {
                    // leave both null; resolution will simply fail and be counted
                }

                var tokens = new HashSet<int>();
                Walk(il, single, multi, tokens);

                foreach (int token in tokens)
                {
                    // Successes are deduplicated: the same token recurs at thousands of call sites
                    // and resolves to the same member every time. FAILURES deliberately are not --
                    // the same token can fail under one method's generic context and resolve under
                    // another's, and skipping the retry would reproduce exactly the silent "no"
                    // this scan is being repaired for.
                    var key = (method.Module.MetadataToken, token);
                    if (!resolvedTokens.TryGetValue(key, out var cached))
                    {
                        total++;
                        MethodBase? target = null;
                        try
                        {
                            target = method.Module.ResolveMethod(token, typeArgs, methodArgs);
                        }
                        catch
                        {
                            target = null;
                        }

                        if (target?.DeclaringType == null)
                        {
                            unresolved++;
                            cached = null;
                        }
                        else
                        {
                            var declaringType = target.DeclaringType;
                            if (declaringType.IsGenericType)
                            {
                                declaringType = declaringType.GetGenericTypeDefinition();
                            }

                            cached = (declaringType.FullName ?? declaringType.Name,
                                target.Name.StartsWith("get_", StringComparison.Ordinal)
                                    ? target.Name.Substring(4)
                                    : target.Name);
                        }

                        resolvedTokens[key] = cached;
                    }

                    if (cached == null) continue;

                    // Property getters are named get_X; store the property name so the key matches
                    // the one built from PropertyInfo on the other side.
                    var targetKey = cached.Value;
                    if (isOptions)
                    {
                        // Only a property getter propagates. A copy constructor or Validate()
                        // reading the same property proves nothing about whether a MODEL reads it.
                        if (method.IsSpecialName
                            && method.Name.StartsWith("get_", StringComparison.Ordinal)
                            && method.DeclaringType != null)
                        {
                            var ownerType = method.DeclaringType.IsGenericType
                                ? method.DeclaringType.GetGenericTypeDefinition()
                                : method.DeclaringType;
                            var ownerKey = (ownerType.FullName ?? ownerType.Name,
                                method.Name.Substring(4));
                            if (!derivedFrom.TryGetValue(ownerKey, out var set))
                            {
                                set = new HashSet<(string Type, string Name)>();
                                derivedFrom[ownerKey] = set;
                            }

                            set.Add(targetKey);
                        }

                        continue;
                    }

                    called.Add(targetKey);
                }
            }
        }

        // A computed property that is read from outside consumes whatever its getter reads, so
        // walk those edges transitively. `FeedForwardDimension` being read is what makes
        // `FeedForwardMultiplier` read; without this the multiplier reports as unread while
        // changing it demonstrably changes the model.
        var queue = new Queue<(string Type, string Name)>(called);
        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            if (!derivedFrom.TryGetValue(current, out var sources)) continue;

            foreach (var source in sources)
            {
                if (called.Add(source)) queue.Enqueue(source);
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
