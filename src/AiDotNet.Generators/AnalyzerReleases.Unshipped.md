<!--
DIAGNOSTIC ID PREFIX -- SETTLED, and deliberately NOT by renaming.

This table carries five prefixes: AIDN (shipped), and ADN / ADNTEST / ADNSHAPE /
ADNGEN (unshipped). Unshipped IDs are the cheap moment to unify, and that was
weighed rather than skipped.

DECISION: the unshipped prefixes STAY, and no new prefix is introduced.

Why not fold them into AIDN. The shipped range is three-digit (AIDN001..AIDN062)
and already occupies 050-052 for ComponentMetadata. Renaming ADN0050..ADN0054
into that range gives AIDN0050 sitting beside AIDN050 -- two distinct rules one
character apart, which is worse for grep, .editorconfig and suppression comments
than the split it was meant to cure. A fresh non-colliding range would avoid that
but renumbers rules referenced by sibling PRs in this 18-part split, and a rename
landing in one part while another still emits the old ID is a build break nobody
owns.

What the prefixes now mean, so the split is a scheme rather than an accident:
  AIDN      shipped rules, frozen
  ADN00xx   LayerStateGenerator (serialization round-trip)
  ADNSHAPE  ShapeDeclarationValidationGenerator
  ADNTEST   TestScaffoldGenerator, scaffold correctness
  ADNGEN    TestScaffoldGenerator, coverage gaps

Revisit in the PR that lands last in the split, where a rename can be atomic.
-->

### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|------
AIDN001 | AiDotNet.ModelMetadata | Error | ModelMetadataValidationGenerator, Missing required model metadata attribute
AIDN010 | AiDotNet.ModelMetadata | Error | ModelMetadataValidationGenerator, Missing XML doc summary
AIDN011 | AiDotNet.ModelMetadata | Error | ModelMetadataValidationGenerator, Missing beginner-friendly remarks
AIDN012 | AiDotNet.ModelMetadata | Error | ModelMetadataValidationGenerator, Missing usage example
AIDN020 | AiDotNet.ModelMetadata | Error | ModelMetadataValidationGenerator, Invalid ModelPaper URL
AIDN030 | AiDotNet.Compatibility | Error | CompatibilityMatrixGenerator, Conflicting optimizer requirements across model categories
AIDN031 | AiDotNet.Compatibility | Warning | CompatibilityMatrixGenerator, Compatibility matrix issue
AIDN040 | AiDotNet.TestCoverage | Error | TestScaffoldGenerator, Model has no test coverage
AIDN041 | AiDotNet.TestCoverage | Warning | TestScaffoldGenerator, Model test coverage summary
AIDN042 | AiDotNet.TestCoverage | Warning | TestScaffoldGenerator, Activation function test coverage summary
AIDN043 | AiDotNet.TestCoverage | Warning | TestScaffoldGenerator, Loss function test coverage summary
AIDN044 | AiDotNet.TestCoverage | Warning | TestScaffoldGenerator, Layer test coverage summary
AIDN050 | AiDotNet.ComponentMetadata | Error | ComponentMetadataValidationGenerator, Activation function missing required metadata
AIDN051 | AiDotNet.ComponentMetadata | Error | ComponentMetadataValidationGenerator, Loss function missing required metadata
AIDN052 | AiDotNet.ComponentMetadata | Error | ComponentMetadataValidationGenerator, Layer missing required metadata
AIDN060 | AiDotNet.TypeSafety | Info | HardcodedDoubleFieldGenerator, Hardcoded double field in generic <T> class
AIDN061 | AiDotNet.TypeSafety | Info | HardcodedDoubleFieldGenerator, Hardcoded double[] field in generic <T> class
AIDN062 | AiDotNet.TypeSafety | Info | HardcodedDoubleFieldGenerator, Hardcoded double[,]/double[][] field in generic <T> class
ADN0050 | AiDotNet.Serialization | Error | LayerStateGenerator, Layer with [LayerState] must be partial
ADN0051 | AiDotNet.Serialization | Error | LayerStateGenerator, [LayerState] parameter has no readable backing member
ADN0052 | AiDotNet.Serialization | Error | LayerStateGenerator, [LayerState] parameter type cannot be serialized
ADN0053 | AiDotNet.Serialization | Error | LayerStateGenerator, Required constructor parameter cannot be restored
ADN0054 | AiDotNet.Serialization | Warning | LayerStateGenerator, Hand-written GetMetadata may drift from [LayerState]
ADN0055 | AiDotNet.Serialization | Warning | LayerStateGenerator, [LayerState] layer cannot be registered in the generated factory
ADN0056 | AiDotNet.Serialization | Error | LayerStateGenerator, [LayerState] is only supported on a class deriving from LayerBase
ADNTEST001 | AiDotNet.TestScaffold | Warning | TestScaffoldGenerator, Float test scaffold rewrite was a no-op
ADNTEST002 | AiDotNet.TestScaffold | Disabled | TestScaffoldGenerator, Generated scaffold architecture size disagrees with its InputShape
ADNTEST003 | AiDotNet.TestScaffold | Error | TestScaffoldGenerator, Two models share a simple name with no registered owner
ADNSHAPE001 | AiDotNet.Shapes | Error | ShapeDeclarationValidationGenerator, Two tensor layouts accept the same rank with different axis names
ADNSHAPE002 | AiDotNet.Shapes | Error | ShapeDeclarationValidationGenerator, A tensor layout repeats an axis role
ADNSHAPE003 | AiDotNet.Shapes | Warning | ShapeDeclarationValidationGenerator, Type implements IShapeContract but declares no input layout
ADNSHAPE004 | AiDotNet.Shapes | Error | ShapeDeclarationValidationGenerator, Layer overrides Forward instead of ForwardTraced and is invisible to graph tracing
ADNGEN001 | AiDotNet.TestScaffold | Warning | TestScaffoldGenerator, Model cannot be auto-generated a test and therefore has NO coverage
