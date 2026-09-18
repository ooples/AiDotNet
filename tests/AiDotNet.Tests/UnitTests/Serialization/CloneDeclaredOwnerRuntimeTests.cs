using System.Reflection;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serialization;

public sealed class CloneDeclaredOwnerRuntimeTests
{
    public CloneDeclaredOwnerRuntimeTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void ExplicitOpenGenericOwners_PreserveDifferingRuntimeTypesAndIndependentOptions()
    {
        RegisterTyped(typeof(AssignedOwner<int>), Field(typeof(AssignedOwner<>), "_options"),
            Field(typeof(GenericOptionsOwner<>), "_options"));
        var source = new AssignedOwner<int>(new OwnSettings { Width = 23 }, new BaseSettings { Width = 47 });
        var clone = Assert.IsType<AssignedOwner<int>>(CloneEngine.CopyConfiguration(source));
        Assert.Equal(23, clone.Own.Width);
        Assert.Equal(47, clone.Inherited.Width);
        Assert.NotSame(source.Own, clone.Own);
        Assert.NotSame(source.Inherited, clone.Inherited);
        clone.Own.Width = 11;
        clone.Inherited.Width = 13;
        Assert.Equal(23, source.Own.Width);
        Assert.Equal(47, source.Inherited.Width);

        CloneEngine.RestoreMutableConstructorConfiguration(source, clone);
        Assert.Equal(23, clone.Own.Width);
        Assert.Equal(47, clone.Inherited.Width);
        Assert.NotSame(source.Inherited, clone.Inherited);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExplicitMissingOrWrongClosedOwner_ThrowsWithoutDefaultReconstruction(bool wrongClosedGeneric)
    {
        var owner = wrongClosedGeneric ? typeof(GenericOptionsOwner<string>) : typeof(UnrelatedOwner);
        RegisterTyped(typeof(AssignedOwner<long>), Field(typeof(AssignedOwner<>), "_options"), Field(owner, "_options"));
        var source = new AssignedOwner<long>(new OwnSettings { Width = 23 }, new BaseSettings { Width = 47 });
        AssignedOwner<long>.Constructions = 0;
        var error = Assert.Throws<InvalidOperationException>(() => CloneEngine.CopyConfiguration(source));
        Assert.Contains("explicit clone owner", error.Message);
        Assert.Equal(0, AssignedOwner<long>.Constructions);
        Assert.Equal(47, source.Inherited.Width);
    }

    [Fact]
    public void OriginalFourArgumentPlanConstructor_RemainsCallable()
    {
        var signature = new[] { typeof(Type), typeof(IReadOnlyList<ClonePlanEntry>),
            typeof(IReadOnlyList<string>), typeof(IReadOnlyList<IReadOnlyList<string>>) };
        var constructor = typeof(ClonePlan).GetConstructor(signature);
        Assert.NotNull(constructor);
        var plan = Assert.IsType<ClonePlan>(constructor.Invoke(new object?[]
            { typeof(OwnSettings), Array.Empty<ClonePlanEntry>(), null, null }));
        Assert.Empty(plan.ConstructorBindings);
    }

    [Fact]
    public void BindingDefensivelyCopiesMemberPath_AndDistinguishesDefaultKind()
    {
        var original = Field(typeof(GenericOptionsOwner<>), "_options");
        var path = new MemberInfo[] { original };
        var binding = new CloneConstructorArgumentBinding("inheritedOptions", path);
        path[0] = Field(typeof(UnrelatedOwner), "_options");
        Assert.Same(original, Assert.Single(binding.Members));
        Assert.Equal(CloneConstructorArgumentKind.MemberPath, binding.Kind);
        Assert.Equal(CloneConstructorArgumentKind.ParameterDefault,
            CloneConstructorArgumentBinding.UseDefault("seed").Kind);
        Assert.Throws<ArgumentException>(() => new CloneConstructorArgumentBinding("seed", Array.Empty<MemberInfo>()));
        Assert.Throws<ArgumentException>(() => new CloneConstructorArgumentBinding("seed", new MemberInfo[1]));
    }

    [Fact]
    public void PlanSnapshotsBothBindingLists_AndRejectsNullEntries()
    {
        var argument = new CloneConstructorArgumentBinding("options", new[] { Field(typeof(AssignedOwner<>), "_options") });
        var inner = new List<CloneConstructorArgumentBinding> { argument };
        var outer = new List<IReadOnlyList<CloneConstructorArgumentBinding>> { inner };
        var plan = new ClonePlan(typeof(AssignedOwner<int>), Array.Empty<ClonePlanEntry>(), null, null, outer);
        inner.Clear();
        outer.Clear();
        Assert.Same(argument, Assert.Single(Assert.Single(plan.ConstructorBindings)));
        Assert.Throws<ArgumentException>(() => new ClonePlan(typeof(AssignedOwner<int>), Array.Empty<ClonePlanEntry>(),
            null, null, new IReadOnlyList<CloneConstructorArgumentBinding>[1]));
        Assert.Throws<ArgumentException>(() => new ClonePlan(typeof(AssignedOwner<int>), Array.Empty<ClonePlanEntry>(),
            null, null, new[] { new CloneConstructorArgumentBinding[1] }));
    }

    [Fact]
    public void NullNestedAlternative_SkipsCandidateAndRestorationWithoutLosingNativeState()
    {
        var modelType = typeof(AlternativeOwner<int>);
        var own = new CloneConstructorArgumentBinding("options", new[] { Field(modelType, "_options") });
        var inherited = new CloneConstructorArgumentBinding("inheritedOptions", new[] { Field(typeof(GenericOptionsOwner<>), "_options") });
        var alternate = new CloneConstructorArgumentBinding("alternateOptions", new MemberInfo[]
            { Field(modelType, "_alternate"), Field(typeof(OptionalOwner), "Options") });
        CloneRegistry.Register(new ClonePlan(modelType, Array.Empty<ClonePlanEntry>(), null, null,
            new[] { new[] { own, inherited, alternate }, new[] { own, inherited } }));
        var source = new AlternativeOwner<int>(new OwnSettings { Width = 23 }, new BaseSettings { Width = 47 });
        var clone = Assert.IsType<AlternativeOwner<int>>(CloneEngine.CopyConfiguration(source));
        Assert.False(clone.HasAlternate);
        Assert.Equal(23, clone.Own.Width);
        Assert.Equal(47, clone.Inherited.Width);
        clone.Own.Width = 11;
        clone.Inherited.Width = 13;
        CloneEngine.RestoreMutableConstructorConfiguration(source, clone);
        Assert.Equal(23, clone.Own.Width);
        Assert.Equal(47, clone.Inherited.Width);
    }

    [Fact]
    public void AliasedSourceOptions_RestoreEveryIndependentDestination()
    {
        var modelType = typeof(AliasedOptionsOwner);
        CloneRegistry.Register(new ClonePlan(modelType, Array.Empty<ClonePlanEntry>(), null, null,
            new[] { new[] {
                new CloneConstructorArgumentBinding("left", new[] { Field(modelType, "_left") }),
                new CloneConstructorArgumentBinding("right", new[] { Field(modelType, "_right") }) } }));
        var options = new OwnSettings { Width = 23 };
        var source = new AliasedOptionsOwner(options, options);
        var clone = Assert.IsType<AliasedOptionsOwner>(CloneEngine.CopyConfiguration(source));
        Assert.NotSame(clone.Left, clone.Right);
        clone.Left.Width = 5;
        clone.Right.Width = 7;
        CloneEngine.RestoreMutableConstructorConfiguration(source, clone);
        Assert.Equal(23, clone.Left.Width);
        Assert.Equal(23, clone.Right.Width);
    }

    private static void RegisterTyped(Type modelType, MemberInfo own, MemberInfo inherited)
        => CloneRegistry.Register(new ClonePlan(modelType, Array.Empty<ClonePlanEntry>(),
            new[] { "_options", "_options" }, new[] { new[] { "_options", "_options" } },
            new[] { new[] { new CloneConstructorArgumentBinding("options", new[] { own }),
                new CloneConstructorArgumentBinding("inheritedOptions", new[] { inherited }) } }));

    private static FieldInfo Field(Type owner, string name)
        => owner.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            ?? throw new InvalidOperationException("Missing actual fixture field.");

    public sealed class OwnSettings : ModelOptions { public int Width { get; set; } = 1; }
    public sealed class BaseSettings : ModelOptions { public int Width { get; set; } = 2; }

    public abstract class GenericOptionsOwner<T>
    {
        // Both owners deliberately declare object, so a runtime-type-only lookup cannot replace
        // the recorded declaring identity. The stored values have distinct concrete option types.
        protected readonly object _options;
        protected GenericOptionsOwner(BaseSettings? options) => _options = options ?? new BaseSettings();
        public BaseSettings Inherited => (BaseSettings)_options;
    }

    public sealed class AssignedOwner<T> : GenericOptionsOwner<T>
    {
        private new readonly object _options;
        public static int Constructions;
        public OwnSettings Own => (OwnSettings)_options;
        public AssignedOwner(OwnSettings? options = null, BaseSettings? inheritedOptions = null) : base(inheritedOptions)
        {
            Constructions++;
            _options = options ?? new OwnSettings();
        }
    }

    public sealed class UnrelatedOwner
    {
        private readonly object _options = new BaseSettings();
        public object StoredOptions => _options;
    }

    public sealed class OptionalOwner
    {
        public readonly OwnSettings Options;
        public OptionalOwner(OwnSettings options) => Options = options;
    }

    public sealed class AlternativeOwner<T> : GenericOptionsOwner<T>
    {
        private new readonly object _options;
        private readonly OptionalOwner? _alternate;
        public OwnSettings Own => (OwnSettings)_options;
        public bool HasAlternate => _alternate is not null;
        public AlternativeOwner(OwnSettings? options = null, BaseSettings? inheritedOptions = null) : base(inheritedOptions)
            => _options = options ?? new OwnSettings();
        public AlternativeOwner(OwnSettings options, BaseSettings inheritedOptions, OwnSettings alternateOptions) : base(inheritedOptions)
        {
            _options = options;
            _alternate = new OptionalOwner(alternateOptions);
        }
    }

    public sealed class AliasedOptionsOwner
    {
        private readonly OwnSettings _left;
        private readonly OwnSettings _right;
        public OwnSettings Left => _left;
        public OwnSettings Right => _right;
        public AliasedOptionsOwner(OwnSettings left, OwnSettings right)
        {
            _left = left;
            _right = right;
        }
    }
}
