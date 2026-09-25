using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class ReviewedFactDiscoveryTests
{
    public enum Mutation { Package, Core, Execution, Framework, NamedProperty, ConstructorValue }

    [Fact]
    public void ReviewedDiscoveryDoesNotClaimStandardCaseExecution()
    {
        using var resolver = Resolver(Path.GetDirectoryName(typeof(SkippableFactAttribute).Assembly.Location)
            ?? throw new InvalidOperationException("Missing bundle."));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(ReviewedFactDiscoveryTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        XunitLifecycleResult result = XunitLifecycleReader.Read(assembly);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        SourceMethod[] reviewed = result.SyntheticMethods.Where(method => method.Dependency.Id.StartsWith("reviewed-discovery:", StringComparison.Ordinal)).ToArray();
        if (supported)
        {
            Assert.Single(reviewed);
            Assert.DoesNotContain(result.Map.GroupRoots, root => root == "unresolved:xunit:custom-discoverer:Xunit.SkippableFactAttribute");
        }
        else Assert.Empty(reviewed);
        // The fixture includes skippable cases: discovery is known, their
        // execution semantics are still custom and cannot justify task reuse.
        Assert.Contains(result.Map.Tests, test => !test.Complete);
    }

    [Theory]
    [InlineData(Mutation.Package)]
    [InlineData(Mutation.Core)]
    [InlineData(Mutation.Execution)]
    [InlineData(Mutation.Framework)]
    [InlineData(Mutation.NamedProperty)]
    [InlineData(Mutation.ConstructorValue)]
    public void PackageOrMetadataChangesLoseDiscoveryContract(Mutation mutation)
    {
        string root = Directory.CreateTempSubdirectory("discovery-contract-").FullName;
        try
        {
            string original = Path.GetDirectoryName(typeof(SkippableFactAttribute).Assembly.Location)
                ?? throw new InvalidOperationException("Missing bundle.");
            foreach (string name in new[] { "Xunit.SkippableFact.dll", "xunit.core.dll", "xunit.execution.dotnet.dll", "Attribution.Xunit.dll" })
                File.Copy(Path.Combine(original, name), Path.Combine(root, name));
            string? changed = mutation switch
            {
                Mutation.Package => "Xunit.SkippableFact.dll", Mutation.Core => "xunit.core.dll",
                Mutation.Execution => "xunit.execution.dotnet.dll", Mutation.Framework => "Attribution.Xunit.dll", _ => null
            };
            if (changed is not null)
            {
                using var stream = new FileStream(Path.Combine(root, changed), FileMode.Append);
                stream.WriteByte(0);
            }
            using var resolver = Resolver(root);
            using var assembly = AssemblyDefinition.ReadAssembly(Path.Combine(root, "Xunit.SkippableFact.dll"),
                new ReaderParameters { AssemblyResolver = resolver });
            MethodDefinition constructor = assembly.MainModule.Types.Single(type => type.FullName == "Xunit.SkippableFactAttribute")
                .Methods.Single(method => method.IsConstructor);
            var attribute = new CustomAttribute(constructor);
            attribute.ConstructorArguments.Add(new(constructor.Parameters[0].ParameterType, Array.Empty<CustomAttributeArgument>()));
            if (mutation == Mutation.NamedProperty)
                attribute.Properties.Add(new("UnknownCallback", new(assembly.MainModule.TypeSystem.String, "value")));
            if (mutation == Mutation.ConstructorValue)
                attribute.ConstructorArguments[0] = new(constructor.Parameters[0].ParameterType, "not-type-metadata");
            Assert.Null(new ReviewedFactDiscovery().Read(attribute));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CustomTraitDiscoveryRemainsAGroupBoundary(bool custom)
    {
        using var resolver = Resolver(Path.GetDirectoryName(typeof(ReviewedFactDiscoveryTests).Assembly.Location)
            ?? throw new InvalidOperationException("Missing bundle."));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(ReviewedFactDiscoveryTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        TypeDefinition target = assembly.MainModule.Types.Single(type => type.FullName == typeof(ReviewedFactDiscoveryTests).FullName);
        if (custom)
        {
            TypeDefinition trait = assembly.MainModule.Types.Single(type => type.FullName == typeof(OpaqueTraitAttribute).FullName);
            target.CustomAttributes.Add(new(trait.Methods.Single(method => method.IsConstructor)));
        }
        SourceLifecycleMap map = XunitLifecycleReader.Read(assembly).Map;
        Assert.Equal(custom, map.GroupRoots.Contains("unresolved:xunit:trait-discoverer:" + typeof(OpaqueTraitAttribute).FullName));
        Assert.DoesNotContain(map.GroupRoots, root => root == "unresolved:xunit:trait-discoverer:Xunit.TraitAttribute");
    }

    private static DefaultAssemblyResolver Resolver(string bundle)
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(bundle);
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }
}

public sealed class OpaqueTraitAttribute : Attribute, Xunit.Sdk.ITraitAttribute { }
