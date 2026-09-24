using Mono.Cecil;

internal enum WorkloadBodyKind { Unresolved, OwnedConfiguration, ExpectedNullBackend }
internal enum WorkloadBodyRequirement { ReviewedStartup, ExclusiveHostLifetime, NoExceptionObservers, SuccessfulOwners }
internal sealed record WorkloadBodyAssessment(string Owner, WorkloadBodyKind Kind, string[] SharedMaps);
internal sealed record WorkloadBodyReview(WorkloadBodyAssessment[] Bodies, WorkloadBodyRequirement[] RemainingRequirements)
{
    internal bool AllBodiesRecognized => Bodies.Length > 0 && Bodies.All(body => body.Kind != WorkloadBodyKind.Unresolved);
}

// An exact-inventory join, not an existential "some bodies were recognized"
// check. Every body must use one exhaustive protocol before body-originating
// file/slot/cache/map escape can be excluded. Startup and host callbacks remain
// explicit obligations, so this object is deliberately not a reuse certificate.
internal static class WorkloadBodyReader
{
    internal static WorkloadBodyReview Read(AssemblyDefinition assembly, string[] owners)
    {
        WorkloadBodyAssessment Unknown(string owner) => new(owner, WorkloadBodyKind.Unresolved, []);
        if (owners.Length == 0 || owners.Any(string.IsNullOrWhiteSpace) || owners.Distinct(StringComparer.Ordinal).Count() != owners.Length)
            throw new ArgumentException("Expected an exact nonempty owner inventory.", nameof(owners));
        var result = new List<WorkloadBodyAssessment>();
        foreach (string owner in owners)
        {
            try
            {
                string prefix = assembly.Name.Name + ":";
                int separator = owner.LastIndexOf('.');
                if (!owner.StartsWith(prefix, StringComparison.Ordinal) || separator <= prefix.Length)
                { result.Add(Unknown(owner)); continue; }
                TypeDefinition? type = assembly.MainModule.GetType(owner[prefix.Length..separator].Replace('+', '/'));
                MethodDefinition[] entries = type?.Methods.Where(method => method.Name == owner[(separator + 1)..]).ToArray() ?? [];
                CustomAttribute[] markers = entries.Length == 1 ? entries[0].CustomAttributes.Where(attribute =>
                    attribute.AttributeType.FullName == "System.Runtime.CompilerServices.AsyncStateMachineAttribute").ToArray() : [];
                MethodDefinition[] bodies = markers.Length == 1 && markers[0].ConstructorArguments.Count == 1 &&
                    markers[0].ConstructorArguments[0].Value is TypeReference state && state.Resolve() is TypeDefinition machine
                    ? machine.Methods.Where(method => method.Name == "MoveNext").ToArray() : [];
                if (bodies.Length != 1) { result.Add(Unknown(owner)); continue; }
                MethodDefinition body = bodies[0];
                if (NullBackendBodyReader.ReadShape(entries[0], body) is not null)
                { result.Add(new(owner, WorkloadBodyKind.ExpectedNullBackend, [])); continue; }
                OwnedConfigurationBodyShape? shape = OwnedConfigurationBodyReader.ReadShape(entries[0], body);
                if (shape is null || shape.Backend.LockedTail.Contract != LockedInitializationContract.ConstantInsertIfAbsent ||
                    shape.Backend.NumericBase.Contract != NumericBaseConstructorContract.OwnedFieldsAndDoubleProvider ||
                    shape.Backend.NumericBase.Provider?.CacheInitialization?.Contract != RuntimeCacheInitializationContract.PrivateTypeAccelerationCache ||
                    shape.Configurations.Any(allocation => ConfigurationConstructorReader.Read(allocation.Constructor, true, allocation.LearningRate) is not
                        { Contract: ConfigurationConstructorContract.OwnedReceiverWithDoubleLearningRate,
                          Provider.CacheInitialization.Contract: RuntimeCacheInitializationContract.PrivateTypeAccelerationCache }) ||
                    body.Body.Instructions[shape.Backend.Instruction].Operand is not MethodReference backend ||
                    backend.DeclaringType.Resolve() is not TypeDefinition backendType ||
                    PrivateMapInitializerReader.Read(backendType) is not { Contract: PrivateMapInitializerContract.FreshPrivateDefaultStringMaps } initializer ||
                    !initializer.Fields.Contains(shape.Backend.LockedTail.Map, StringComparer.Ordinal) ||
                    !initializer.Fields.Contains(shape.Backend.LockedTail.Lock, StringComparer.Ordinal))
                { result.Add(Unknown(owner)); continue; }
                result.Add(new(owner, WorkloadBodyKind.OwnedConfiguration, [shape.Backend.LockedTail.Map]));
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
                AssemblyResolutionException or ResolutionException or InvalidOperationException)
            { result.Add(Unknown(owner)); }
        }
        return new(result.ToArray(), Enum.GetValues<WorkloadBodyRequirement>());
    }
}
