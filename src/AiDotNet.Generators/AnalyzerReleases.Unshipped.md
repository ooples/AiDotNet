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
AIDN070 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, Options copy constructor does not copy every property
AIDN071 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, Null-forgiving operator is not permitted
AIDN072 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, Use RandomHelper instead of new Random()
AIDN073 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, Regex without a timeout (ReDoS)
AIDN074 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, NotImplementedException in production code
AIDN075 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, Console output used instead of a logging abstraction
AIDN076 | AiDotNet.GoldenPattern | Warning | GoldenPatternValidationGenerator, Catch block swallows the exception
AIDN060 | AiDotNet.TypeSafety | Info | HardcodedDoubleFieldGenerator, Hardcoded double field in generic <T> class
AIDN061 | AiDotNet.TypeSafety | Info | HardcodedDoubleFieldGenerator, Hardcoded double[] field in generic <T> class
AIDN062 | AiDotNet.TypeSafety | Info | HardcodedDoubleFieldGenerator, Hardcoded double[,]/double[][] field in generic <T> class
ADN0050 | AiDotNet.Serialization | Error | LayerStateGenerator
ADN0051 | AiDotNet.Serialization | Error | LayerStateGenerator
ADN0052 | AiDotNet.Serialization | Error | LayerStateGenerator
ADN0053 | AiDotNet.Serialization | Error | LayerStateGenerator
ADN0054 | AiDotNet.Serialization | Warning | LayerStateGenerator
ADNTEST001 | AiDotNet.TestScaffold | Warning | TestScaffoldGenerator, Float test scaffold rewrite was a no-op
ADNTEST002 | AiDotNet.TestScaffold | Disabled | TestScaffoldGenerator, Generated scaffold architecture size disagrees with its InputShape
ADNTEST003 | AiDotNet.TestScaffold | Error | TestScaffoldGenerator, Two models share a simple name with no registered owner
ADNSHAPE001 | AiDotNet.Shapes | Error | ShapeDeclarationValidationGenerator, Two tensor layouts accept the same rank with different axis names
ADNSHAPE002 | AiDotNet.Shapes | Error | ShapeDeclarationValidationGenerator, A tensor layout repeats an axis role
ADNSHAPE003 | AiDotNet.Shapes | Warning | ShapeDeclarationValidationGenerator, Type implements IShapeContract but declares no input layout
ADNSHAPE004 | AiDotNet.Shapes | Error | ShapeDeclarationValidationGenerator, Layer overrides Forward instead of ForwardTraced and is invisible to graph tracing
ADNGEN001 | AiDotNet.TestScaffold | Warning | TestScaffoldGenerator, Model cannot be auto-generated a test and therefore has NO coverage
AIDN070 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Layer does not use automatic parameter discovery
AIDN071 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Parameter surface is derived and should not be overridden
AIDN072 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Model parameter surface is derived and should not be overridden
AIDN073 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Field holds weights the parameter generator cannot see
AIDN080 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Layer does not use automatic parameter discovery
AIDN081 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Parameter surface is derived and should not be overridden
AIDN082 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Model parameter surface is derived and should not be overridden
AIDN083 | AiDotNet.ParameterAutomation | Warning | ParameterAutomationAnalyzer, Field holds weights the parameter generator cannot see
