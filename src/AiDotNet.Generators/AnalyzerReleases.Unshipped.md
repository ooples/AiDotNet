; Unshipped analyzer release
; Diagnostic prefixes remain split by generator until the final generator-refactor PR can
; renumber them atomically: AIDN (shipped rules), ADN00xx (layer state), ADNSHAPE
; (shape contracts), ADNTEST (scaffold correctness), and ADNGEN (coverage gaps).

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
ADNSHAPE004 | AiDotNet.Shapes | Warning | ShapeDeclarationValidationGenerator, Layer overrides Forward instead of ForwardTraced and is invisible to graph tracing (Warning until the #1789 conversion completes; the final slice raises it to Error)
ADNPORT001 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Integer-index port does not declare its legal upper bound
ADNPORT002 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Tensor-port contract references a missing member
ADNPORT003 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Duplicate input or output port name
ADNPORT004 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Adjacent sequential layers have incompatible value domains
ADNPORT005 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Model input-shape constraint references a missing member
ADNPORT006 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Type using a generated tensor/model-input contract is not partial
ADNPORT007 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Generated rank-routing or model-input geometry contains impossible values
ADNPORT008 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Tensor-contract member has an incompatible signature
ADNPORT009 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Generated forward contract is ambiguous or uses unsupported parameters
ADNPORT010 | AiDotNet.TensorPorts | Error | TensorPortContractGenerator, Stable input port identity collides across inherited/local declarations
ADN0058 | AiDotNet.Serialization | Error | CloneAutomationAnalyzer, Clone override duplicates what the base class already does
ADN0059 | AiDotNet.Serialization | Info | CloneAutomationAnalyzer, Model cannot be rebuilt from its own state
ADN0060 | AiDotNet.Serialization | Error | CloneAutomationAnalyzer, Serialization is hand-written instead of declared
ADNGEN001 | AiDotNet.TestScaffold | Warning | TestScaffoldGenerator, Model cannot be auto-generated a test and therefore has NO coverage
AIDN085 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Model owns weights outside Layers but is not partial, so the generator cannot register them
AIDN086 | AiDotNet.ComponentMetadata | Error | ComponentMetadataValidationGenerator, Layer declares a contradictory gradient contract
AIDN087 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, ParameterCount compared against zero as a readiness test (Warning while the backlog is non-zero; promote to Error at zero per the ADNSHAPE006/007 ladder)
AIDN088 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Numeric state requires an explicit semantic classification
AIDN089 | AiDotNet.ParameterAutomation | Error | ParameterAutomationAnalyzer, Numeric state has conflicting semantic classifications
AIDN090 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Nullable persistent state requires an explicit availability lifecycle
AIDN091 | AiDotNet.ParameterAutomation | Error | ParameterAutomationAnalyzer, Parameter alias target is invalid
