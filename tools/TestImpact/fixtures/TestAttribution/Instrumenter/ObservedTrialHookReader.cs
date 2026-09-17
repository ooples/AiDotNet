using AiDotNet.TestImpact;
using AttributionRuntime;
using Mono.Cecil;

internal enum ObservedTrialHookProof { Unresolved, ReviewedHookAndCompletedScope }
internal sealed record ObservedTrialHookAssessment(string Owner, ObservedTrialHookProof Proof, string Slot,
    TrialHookRequirement[] Requirements);
internal sealed record OwnerBodyWindow(string Owner, YieldBodyWindow Frame,
    OwnerConcurrencyContract Concurrency = OwnerConcurrencyContract.Unresolved);
internal sealed record ObservedLifecycleReview(OwnerCompletionObservation[] Owners, ObservedTrialHookAssessment[] Hooks, OwnerBodyWindow[] Bodies);

// Joins the actual lifecycle roots with verified owner/scope observations.
// Body isolation and execution-context requirements are NOT discharged here;
// callers must not use this partial assessment as a reuse authorization.
internal static class ObservedTrialHookReader
{
    internal static ObservedTrialHookAssessment[] ReadAll(string bundle, string assemblyFile, string[] owners,
        VerifiedObservedExecution observed) => Review(bundle, assemblyFile, owners, observed).Hooks;

    internal static ObservedLifecycleReview Review(string bundle, string assemblyFile, string[] owners,
        VerifiedObservedExecution observed)
    {
        ObservedTrialHookAssessment Unknown(string owner) => new(owner, ObservedTrialHookProof.Unresolved, "", []);
        ObservedLifecycleReview UnknownAll() => new(owners.Select(owner => new OwnerCompletionObservation(owner, OwnerCompletionProof.Unresolved)).ToArray(),
            owners.Select(Unknown).ToArray(), owners.Select(owner => new OwnerBodyWindow(owner, new(YieldBodyContract.Unresolved, 0, 0))).ToArray());
        try
        {
            if (owners.Length == 0) return new([], [], []);
            if (owners.Any(string.IsNullOrWhiteSpace) || owners.Distinct(StringComparer.Ordinal).Count() != owners.Length ||
                Path.GetFileName(assemblyFile) != assemblyFile || Path.GetExtension(assemblyFile) != ".dll") return UnknownAll();
            OwnerCompletionObservation[] reviewedOwners = ReviewedOwnerCompletion.ReadAll(bundle, assemblyFile, owners, observed);
            var completion = reviewedOwners
                .ToDictionary(item => item.Owner, item => item.Proof, StringComparer.Ordinal);
            if (completion.Values.All(proof => proof == OwnerCompletionProof.Unresolved)) return UnknownAll();
            using var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(Path.GetFullPath(bundle));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            using var assembly = AssemblyDefinition.ReadAssembly(Path.Combine(bundle, assemblyFile),
                new ReaderParameters { AssemblyResolver = resolver });
            SourceTestLifecycle[] lifecycles = XunitLifecycleReader.ReadObservedOwners(assembly, owners);
            var methods = Types(assembly.MainModule.Types).SelectMany(type => type.Methods)
                .ToDictionary(DependencyGraph.Stable, StringComparer.Ordinal);
            var tests = lifecycles.ToLookup(test => test.Owner, StringComparer.Ordinal);
            var scopes = observed.TrialScopes.ToLookup(scope => scope.Owner, StringComparer.Ordinal);
            var checkedHooks = new Dictionary<MethodDefinition, TrialHookAssessment>();
            var result = new List<ObservedTrialHookAssessment>();
            var classConcurrency = new Dictionary<string, OwnerConcurrencyContract>(StringComparer.Ordinal);
            var bodies = owners.Select(owner =>
            {
                if (completion[owner] != OwnerCompletionProof.ReviewedStandardTaskObserved)
                    return new OwnerBodyWindow(owner, new(YieldBodyContract.Unresolved, 0, 0));
                string classId = owner[..owner.LastIndexOf('.')];
                if (!classConcurrency.TryGetValue(classId, out OwnerConcurrencyContract concurrency))
                {
                    concurrency = SerializedOwnerReader.Read(assembly, owner);
                    classConcurrency.Add(classId, concurrency);
                }
                return new OwnerBodyWindow(owner, YieldBodyReader.ReadOwner(assembly, owner), concurrency);
            }).ToArray();
            foreach (string owner in owners)
            {
                SourceTestLifecycle[] matched = tests[owner].ToArray();
                TrialScopeObservation[] samples = scopes[owner].ToArray();
                if (completion[owner] != OwnerCompletionProof.ReviewedStandardTaskObserved || matched.Length != 1 || !matched[0].Complete ||
                    samples.Length != 1 || samples[0].State != TrialScopeState.Complete)
                { result.Add(Unknown(owner)); continue; }
                MethodDefinition[] roots = matched[0].Roots.Where(methods.ContainsKey).Select(root => methods[root]).ToArray();
                MethodDefinition[] before = roots.Where(method => method.Name == "Before").ToArray();
                MethodDefinition[] after = roots.Where(method => method.Name == "After").ToArray();
                if (before.Length != 1 || after.Length != 1 || before[0].DeclaringType != after[0].DeclaringType)
                { result.Add(Unknown(owner)); continue; }
                if (!checkedHooks.TryGetValue(before[0], out TrialHookAssessment? hook))
                {
                    hook = TrialHookReader.Read(before[0], after[0]);
                    checkedHooks.Add(before[0], hook);
                }
                result.Add(hook.Contract == TrialHookContract.ObservedSaveRestore
                    ? new(owner, ObservedTrialHookProof.ReviewedHookAndCompletedScope, hook.Slot,
                        hook.Requirements.Where(requirement => requirement != TrialHookRequirement.ObservedOwner).ToArray())
                    : Unknown(owner));
            }
            return RunnerBinding.HashBundle(bundle) == observed.Execution.Context.BuildFingerprint ? new(reviewedOwners, result.ToArray(), bodies) : UnknownAll();
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return UnknownAll();
        }
    }

    private static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types)
    {
        foreach (TypeDefinition type in types)
        {
            yield return type;
            foreach (TypeDefinition nested in Types(type.NestedTypes)) yield return nested;
        }
    }
}
