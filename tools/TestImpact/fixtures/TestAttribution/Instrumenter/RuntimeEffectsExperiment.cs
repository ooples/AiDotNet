using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Deliberately separate from SourceImpact and reuse certificates. Open dispatch
// remains open: these candidates must be checked against an independent full run.
internal static class RuntimeEffectsExperiment
{
    internal static object Read(string beforeDirectory, string afterDirectory, DiscoveryManifest inventory, string typeName, string methodName)
    {
        string before = Path.GetFullPath(beforeDirectory), after = Path.GetFullPath(afterDirectory);
        if (RunnerBinding.HashBundle(after) != inventory.Context.BuildFingerprint)
            throw new InvalidDataException("Experiment inventory does not bind the after bundle.");
        var oldFiles = Files(before);
        var newFiles = Files(after);
        if (!oldFiles.Keys.Order(StringComparer.Ordinal).SequenceEqual(newFiles.Keys.Order(StringComparer.Ordinal)))
            throw new InvalidDataException("Experiment bundle membership changed.");
        string[] changedFiles = oldFiles.Keys.Where(file => oldFiles[file] != newFiles[file]).Order(StringComparer.Ordinal).ToArray();
        string[] binaries = changedFiles.Where(file => file.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)).ToArray();
        if (binaries.Length != 1 || changedFiles.Any(file => file != binaries[0] && file != Path.ChangeExtension(binaries[0], ".pdb")))
            throw new InvalidDataException("Experiment requires one changed managed binary and its optional symbols only.");
        using var oldResolver = Resolver(before);
        using var newResolver = Resolver(after);
        using var oldAssembly = AssemblyDefinition.ReadAssembly(Path.Combine(before, binaries[0]), new ReaderParameters { AssemblyResolver = oldResolver });
        using var newAssembly = AssemblyDefinition.ReadAssembly(Path.Combine(after, binaries[0]), new ReaderParameters { AssemblyResolver = newResolver });
        MethodDefinition oldMethod = Types(oldAssembly.MainModule.Types).Single(type => type.FullName == typeName).Methods.Single(method => method.Name == methodName);
        MethodDefinition newMethod = Types(newAssembly.MainModule.Types).Single(type => type.FullName == typeName).Methods.Single(method => method.Name == methodName);
        OwnedResultEffect oldEffect = new OwnedResultEffects().Read(oldMethod) ?? throw new InvalidDataException("Unsupported before ownership.");
        OwnedResultEffect newEffect = new OwnedResultEffects().Read(newMethod) ?? throw new InvalidDataException("Unsupported after ownership.");
        if (oldEffect.ShapeHash != newEffect.ShapeHash || !oldEffect.OwnershipDependencies.SequenceEqual(newEffect.OwnershipDependencies) ||
            oldEffect.Writes.SequenceEqual(newEffect.Writes)) throw new InvalidDataException("Change is not confined to owned boolean values.");
        var oldBodies = Bodies(oldAssembly);
        var newBodies = Bodies(newAssembly);
        string changedMethod = DependencyGraph.Stable(newMethod);
        if (!oldBodies.Keys.Order(StringComparer.Ordinal).SequenceEqual(newBodies.Keys.Order(StringComparer.Ordinal)) ||
            !oldBodies.Keys.Where(key => oldBodies[key] != newBodies[key]).SequenceEqual([changedMethod]))
            throw new InvalidDataException("Additional method bodies changed.");
        // Read fresh definitions: metadata normalization destroys method bodies.
        if (Metadata(Path.Combine(before, binaries[0])) != Metadata(Path.Combine(after, binaries[0])))
            throw new InvalidDataException("Metadata, resources or declarations changed.");
        string[] owners = inventory.Cases.Select(test => test.MethodId).Distinct(StringComparer.Ordinal).ToArray();
        string[] testFiles = owners.Select(owner => owner.Split(':')[0] + ".dll").Distinct(StringComparer.Ordinal).ToArray();
        if (testFiles.Length != 1 || !newFiles.ContainsKey(testFiles[0])) throw new InvalidDataException("One real test assembly required.");
        using var tests = AssemblyDefinition.ReadAssembly(Path.Combine(after, testFiles[0]), new ReaderParameters { AssemblyResolver = newResolver });
        var linked = new HashSet<string>([Path.Combine(after, binaries[0]), Path.Combine(after, testFiles[0])], StringComparer.OrdinalIgnoreCase);
        var calls = new Dictionary<string, string[]>(StringComparer.Ordinal);
        var sourceMethods = new Dictionary<string, MethodDefinition>(StringComparer.Ordinal);
        foreach (AssemblyDefinition assembly in new[] { tests, newAssembly })
        {
            string hash = newFiles[Path.GetFileName(assembly.MainModule.FileName)];
            var ids = Types(assembly.MainModule.Types).SelectMany(type => type.Methods).ToDictionary(
                method => $"{hash}:{method.MetadataToken.ToInt32():X8}", DependencyGraph.Stable, StringComparer.Ordinal);
            foreach (MethodDefinition method in Types(assembly.MainModule.Types).SelectMany(type => type.Methods))
                sourceMethods.Add(DependencyGraph.Stable(method), method);
            foreach (MethodDependencyNode node in DependencyGraph.Read(assembly, hash, linked).Methods)
                calls.Add(ids[node.Key], node.LocalCalls.Select(call => ids.TryGetValue(call, out string? id) ? id : call).ToArray());
        }
        SourceLifecycleMap lifecycle = XunitLifecycleReader.Read(tests).Map;
        HashSet<string> Reachable(IEnumerable<string> roots)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            var pending = new Stack<string>(roots);
            while (pending.TryPop(out string? current))
            {
                if (!seen.Add(current)) continue;
                if (calls.TryGetValue(current, out string[]? children)) foreach (string child in children) pending.Push(child);
            }
            return seen;
        }
        string[] candidates = owners.Where(owner =>
        {
            SourceTestLifecycle? test = lifecycle.Tests.SingleOrDefault(test => test.Owner == owner);
            return test is null || !test.Complete || Reachable(test.Roots.Concat(lifecycle.GroupRoots)).Contains(changedMethod);
        }).Order(StringComparer.Ordinal).ToArray();
        var consumerUses = candidates.Select(owner =>
        {
            SourceTestLifecycle? test = lifecycle.Tests.SingleOrDefault(test => test.Owner == owner);
            HashSet<string> reachable = Reachable((test?.Roots ?? []).Concat(lifecycle.GroupRoots));
            var uses = new List<object>();
            foreach (MethodDefinition caller in reachable.Where(sourceMethods.ContainsKey).Select(id => sourceMethods[id]).Where(method => method.HasBody))
            for (int index = 0; index < caller.Body.Instructions.Count; index++)
            {
                Instruction instruction = caller.Body.Instructions[index];
                if (instruction.OpCode.Code is not (Code.Call or Code.Callvirt) || instruction.Operand is not MethodReference target ||
                    target.Name != newMethod.Name || target.DeclaringType.GetElementType().FullName != newMethod.DeclaringType.FullName)
                    continue;
                RuntimeContractAssessment[] contracts = caller.Body.Instructions.Skip(index + 1)
                    .Where(item => item.OpCode.Code is Code.Call or Code.Callvirt)
                    .Select(item => item.Operand).OfType<MethodReference>()
                    .Select(ReviewedRuntimeContracts.Assess)
                    .Where(assessment => assessment.Status == RuntimeContractStatus.ReviewedConditional)
                    .DistinctBy(assessment => assessment.Method).ToArray();
                var failureExits = caller.Body.Instructions.Select((item, offset) => new { item, offset })
                    .Where(site => site.offset > index && site.item.OpCode.Code == Code.Call && site.item.Operand is MethodReference reference &&
                        contracts.Any(contract => contract.Method == reference.FullName))
                    .Select(site => new { Instruction = site.offset, Propagation = AssertionFailureReader.Read(caller, site.offset) }).ToArray();
                uses.Add(new { Caller = DependencyGraph.Stable(caller), Use = OwnedReturnUseReader.Read(caller, index),
                    ReviewedContracts = contracts, FailureExits = failureExits, RequirementsProven = false });
            }
            return new { Owner = owner, Calls = uses.ToArray(), MissingCallsiteProof = uses.Count == 0 };
        }).ToArray();
        // Recheck immutable inputs after reading; this is an experiment, not a
        // provenance or reuse certificate, even if every control later passes.
        if (!oldFiles.OrderBy(pair => pair.Key).SequenceEqual(Files(before).OrderBy(pair => pair.Key)) ||
            !newFiles.OrderBy(pair => pair.Key).SequenceEqual(Files(after).OrderBy(pair => pair.Key)))
            throw new InvalidDataException("Experiment inputs changed during analysis.");
        return new { ChangedMethod = changedMethod, Before = oldEffect, After = newEffect,
            Candidates = candidates, ConsumerUses = consumerUses, DiscoveredMethods = owners.Length, RequiresFullControl = true,
            ProductionSelectionEnabled = false, CanAuthorizeReuse = false };
    }

    private static DefaultAssemblyResolver Resolver(string directory)
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(directory);
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }
    private static Dictionary<string, string> Files(string directory)
    {
        // HashBundle rejects links before this traversal.
        RunnerBinding.HashBundle(directory);
        return Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories).ToDictionary(
            file => Path.GetRelativePath(directory, file), file => { using var stream = File.OpenRead(file); return Convert.ToHexStringLower(SHA256.HashData(stream)); }, StringComparer.Ordinal);
    }
    private static Dictionary<string, string> Bodies(AssemblyDefinition assembly)
    {
        using var stream = File.OpenRead(assembly.MainModule.FileName);
        using var pe = new PEReader(stream);
        return Types(assembly.MainModule.Types).SelectMany(type => type.Methods).ToDictionary(DependencyGraph.Stable,
            method => SourceSnapshotReader.BodyHash(pe, method), StringComparer.Ordinal);
    }
    private static string Metadata(string file)
    {
        using var resolver = Resolver(Path.GetDirectoryName(file) ?? throw new InvalidDataException("Missing binary directory."));
        using var assembly = AssemblyDefinition.ReadAssembly(file, new ReaderParameters { AssemblyResolver = resolver });
        assembly.MainModule.Mvid = Guid.Empty;
        foreach (MethodDefinition method in Types(assembly.MainModule.Types).SelectMany(type => type.Methods).Where(method => method.HasBody))
        {
            method.Body = new MethodBody(method);
            method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
        }
        using var normalized = new MemoryStream();
        assembly.Write(normalized, new WriterParameters { WriteSymbols = false, Timestamp = 0 });
        return Convert.ToHexStringLower(SHA256.HashData(normalized.ToArray()));
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
