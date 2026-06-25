---
title: "Program Synthesis"
description: "All 103 public types in the AiDotNet.programsynthesis namespace, organized by kind."
section: "API Reference"
---

**103** public types in this namespace, organized by kind.

## Models & Types (72)

| Type | Summary |
|:-----|:--------|
| [`CodeAstEdge`](/docs/reference/wiki/programsynthesis/codeastedge/) | Represents a relationship between two AST nodes (typically parent/child). |
| [`CodeAstNodePath`](/docs/reference/wiki/programsynthesis/codeastnodepath/) | Represents a stable, structured path to an AST node. |
| [`CodeAstPathSegment`](/docs/reference/wiki/programsynthesis/codeastpathsegment/) | A structured segment in an AST node path. |
| [`CodeBERT<T>`](/docs/reference/wiki/programsynthesis/codebert/) | CodeBERT is a bimodal pre-trained model for programming and natural languages. |
| [`CodeBugDetectionRequest`](/docs/reference/wiki/programsynthesis/codebugdetectionrequest/) | Request for identifying potential bugs and issues in code. |
| [`CodeBugDetectionResult`](/docs/reference/wiki/programsynthesis/codebugdetectionresult/) |  |
| [`CodeBugFixingRequest`](/docs/reference/wiki/programsynthesis/codebugfixingrequest/) | Request for repairing code issues. |
| [`CodeBugFixingResult`](/docs/reference/wiki/programsynthesis/codebugfixingresult/) |  |
| [`CodeCallGraphEdge`](/docs/reference/wiki/programsynthesis/codecallgraphedge/) |  |
| [`CodeCloneDetectionRequest`](/docs/reference/wiki/programsynthesis/codeclonedetectionrequest/) | Request for clone detection over a corpus. |
| [`CodeCloneDetectionResult`](/docs/reference/wiki/programsynthesis/codeclonedetectionresult/) |  |
| [`CodeCloneGroup`](/docs/reference/wiki/programsynthesis/codeclonegroup/) |  |
| [`CodeCloneInstance`](/docs/reference/wiki/programsynthesis/codecloneinstance/) |  |
| [`CodeCompletionCandidate`](/docs/reference/wiki/programsynthesis/codecompletioncandidate/) |  |
| [`CodeCompletionRequest`](/docs/reference/wiki/programsynthesis/codecompletionrequest/) | Request for code completion. |
| [`CodeCompletionResult`](/docs/reference/wiki/programsynthesis/codecompletionresult/) |  |
| [`CodeComplexityMetrics`](/docs/reference/wiki/programsynthesis/codecomplexitymetrics/) |  |
| [`CodeCorpusDocument`](/docs/reference/wiki/programsynthesis/codecorpusdocument/) | A code document that may be searched or used for clone detection. |
| [`CodeCorpusReference`](/docs/reference/wiki/programsynthesis/codecorpusreference/) | Defines a corpus either by embedding the documents in the request, or by referencing an indexed corpus in Serving. |
| [`CodeDependency`](/docs/reference/wiki/programsynthesis/codedependency/) |  |
| [`CodeDocumentationRequest`](/docs/reference/wiki/programsynthesis/codedocumentationrequest/) | Request for generating or improving code documentation. |
| [`CodeDocumentationResult`](/docs/reference/wiki/programsynthesis/codedocumentationresult/) |  |
| [`CodeEditOperation`](/docs/reference/wiki/programsynthesis/codeeditoperation/) | A machine-applicable edit operation for code transforms. |
| [`CodeExecutionTelemetry`](/docs/reference/wiki/programsynthesis/codeexecutiontelemetry/) | Minimal execution telemetry for sandboxed runs. |
| [`CodeFixSuggestion`](/docs/reference/wiki/programsynthesis/codefixsuggestion/) |  |
| [`CodeGenerationRequest`](/docs/reference/wiki/programsynthesis/codegenerationrequest/) | Request for code generation from a description and/or examples. |
| [`CodeGenerationResult`](/docs/reference/wiki/programsynthesis/codegenerationresult/) |  |
| [`CodeHotspot`](/docs/reference/wiki/programsynthesis/codehotspot/) |  |
| [`CodeIssue`](/docs/reference/wiki/programsynthesis/codeissue/) | Represents a structured issue found in code. |
| [`CodeLocation`](/docs/reference/wiki/programsynthesis/codelocation/) | Location information for an item within code. |
| [`CodePosition`](/docs/reference/wiki/programsynthesis/codeposition/) | Represents a position in source text. |
| [`CodeProvenance`](/docs/reference/wiki/programsynthesis/codeprovenance/) | Provenance metadata for a result hit when an indexed corpus is used. |
| [`CodeRefactoringRequest`](/docs/reference/wiki/programsynthesis/coderefactoringrequest/) | Request for refactoring code without changing behavior. |
| [`CodeRefactoringResult`](/docs/reference/wiki/programsynthesis/coderefactoringresult/) |  |
| [`CodeReviewRequest`](/docs/reference/wiki/programsynthesis/codereviewrequest/) | Request for structured code review output. |
| [`CodeReviewResult`](/docs/reference/wiki/programsynthesis/codereviewresult/) |  |
| [`CodeSearchHit`](/docs/reference/wiki/programsynthesis/codesearchhit/) |  |
| [`CodeSearchRequest`](/docs/reference/wiki/programsynthesis/codesearchrequest/) | Request for searching code using a query against a corpus. |
| [`CodeSearchResult`](/docs/reference/wiki/programsynthesis/codesearchresult/) |  |
| [`CodeSecurityHotspot`](/docs/reference/wiki/programsynthesis/codesecurityhotspot/) |  |
| [`CodeSpan`](/docs/reference/wiki/programsynthesis/codespan/) | Represents a span in source text. |
| [`CodeSummarizationRequest`](/docs/reference/wiki/programsynthesis/codesummarizationrequest/) | Request for summarizing code into natural language. |
| [`CodeSummarizationResult`](/docs/reference/wiki/programsynthesis/codesummarizationresult/) |  |
| [`CodeSymbol`](/docs/reference/wiki/programsynthesis/codesymbol/) |  |
| [`CodeSynthesisArchitecture<T>`](/docs/reference/wiki/programsynthesis/codesynthesisarchitecture/) | Defines the architecture configuration for code synthesis and understanding models. |
| [`CodeT5<T>`](/docs/reference/wiki/programsynthesis/codet5/) | CodeT5 is an encoder-decoder model for code understanding and generation. |
| [`CodeTaskTelemetry`](/docs/reference/wiki/programsynthesis/codetasktelemetry/) | Telemetry captured during a code task execution. |
| [`CodeTestGenerationRequest`](/docs/reference/wiki/programsynthesis/codetestgenerationrequest/) | Request for generating tests for code. |
| [`CodeTestGenerationResult`](/docs/reference/wiki/programsynthesis/codetestgenerationresult/) |  |
| [`CodeTokenizationPipeline`](/docs/reference/wiki/programsynthesis/codetokenizationpipeline/) | Default implementation of `ICodeTokenizationPipeline`. |
| [`CodeTokenizationResult`](/docs/reference/wiki/programsynthesis/codetokenizationresult/) | Represents the result of code-aware tokenization, including token IDs and token-to-source spans. |
| [`CodeTransformDiff`](/docs/reference/wiki/programsynthesis/codetransformdiff/) | Structured diff for code transforms. |
| [`CodeTranslationRequest`](/docs/reference/wiki/programsynthesis/codetranslationrequest/) | Request for translating code from one language to another. |
| [`CodeTranslationResult`](/docs/reference/wiki/programsynthesis/codetranslationresult/) |  |
| [`CodeUnderstandingRequest`](/docs/reference/wiki/programsynthesis/codeunderstandingrequest/) | Request for structured code understanding output. |
| [`CodeUnderstandingResult`](/docs/reference/wiki/programsynthesis/codeunderstandingresult/) |  |
| [`CompilationDiagnostic`](/docs/reference/wiki/programsynthesis/compilationdiagnostic/) |  |
| [`GraphCodeBERT<T>`](/docs/reference/wiki/programsynthesis/graphcodebert/) | GraphCodeBERT extends CodeBERT by incorporating data flow analysis. |
| [`NeuralProgramSynthesizer<T>`](/docs/reference/wiki/programsynthesis/neuralprogramsynthesizer/) | Neural network-based program synthesizer that generates programs from specifications. |
| [`ProgramEvaluateIoRequest`](/docs/reference/wiki/programsynthesis/programevaluateiorequest/) |  |
| [`ProgramEvaluateIoResponse`](/docs/reference/wiki/programsynthesis/programevaluateioresponse/) |  |
| [`ProgramEvaluateIoTestResult`](/docs/reference/wiki/programsynthesis/programevaluateiotestresult/) |  |
| [`ProgramExecuteRequest`](/docs/reference/wiki/programsynthesis/programexecuterequest/) |  |
| [`ProgramExecuteResponse`](/docs/reference/wiki/programsynthesis/programexecuteresponse/) |  |
| [`ProgramInputOutputExample`](/docs/reference/wiki/programsynthesis/programinputoutputexample/) | Represents a single input-output example for program synthesis. |
| [`ProgramInput<T>`](/docs/reference/wiki/programsynthesis/programinput/) | Represents the input specification for program synthesis. |
| [`ProgramSynthesisServingClient`](/docs/reference/wiki/programsynthesis/programsynthesisservingclient/) |  |
| [`Program<T>`](/docs/reference/wiki/programsynthesis/program/) | Represents a synthesized program with its source code and metadata. |
| [`ServingProgramExecutionEngine`](/docs/reference/wiki/programsynthesis/servingprogramexecutionengine/) | Program execution engine that delegates sandboxed execution to an AiDotNet.Serving instance. |
| [`SqlExecuteRequest`](/docs/reference/wiki/programsynthesis/sqlexecuterequest/) |  |
| [`SqlExecuteResponse`](/docs/reference/wiki/programsynthesis/sqlexecuteresponse/) |  |
| [`SqlValue`](/docs/reference/wiki/programsynthesis/sqlvalue/) |  |

## Base Classes (3)

| Type | Summary |
|:-----|:--------|
| [`CodeModelBase<T>`](/docs/reference/wiki/programsynthesis/codemodelbase/) | Base class for code models that provides shared tokenization, task dispatch, and structured outputs. |
| [`CodeTaskRequestBase`](/docs/reference/wiki/programsynthesis/codetaskrequestbase/) | Base type for all code task execution requests. |
| [`CodeTaskResultBase`](/docs/reference/wiki/programsynthesis/codetaskresultbase/) | Base type for structured results returned from code tasks. |

## Interfaces (6)

| Type | Summary |
|:-----|:--------|
| [`ICodeModel<T>`](/docs/reference/wiki/programsynthesis/icodemodel/) | Represents a code understanding model capable of processing and analyzing source code. |
| [`ICodeTokenizationPipeline`](/docs/reference/wiki/programsynthesis/icodetokenizationpipeline/) | Defines a code tokenization pipeline that builds on the core tokenizer stack and adds code-oriented metadata. |
| [`IProgramExecutionEngine`](/docs/reference/wiki/programsynthesis/iprogramexecutionengine/) | Defines an execution boundary for running synthesized programs against inputs. |
| [`IProgramSynthesisServingClient`](/docs/reference/wiki/programsynthesis/iprogramsynthesisservingclient/) |  |
| [`IProgramSynthesizer<T>`](/docs/reference/wiki/programsynthesis/iprogramsynthesizer/) | Represents a program synthesis engine capable of automatically generating programs. |
| [`ISqlSyntaxValidator`](/docs/reference/wiki/programsynthesis/isqlsyntaxvalidator/) | Validates whether a string is syntactically valid SQL, used by the program synthesizer to reject malformed candidate SQL programs. |

## Enums (15)

| Type | Summary |
|:-----|:--------|
| [`CodeCloneType`](/docs/reference/wiki/programsynthesis/codeclonetype/) | Standard clone taxonomy (Type-1..Type-4). |
| [`CodeEditOperationType`](/docs/reference/wiki/programsynthesis/codeeditoperationtype/) |  |
| [`CodeIssueCategory`](/docs/reference/wiki/programsynthesis/codeissuecategory/) |  |
| [`CodeIssueSeverity`](/docs/reference/wiki/programsynthesis/codeissueseverity/) |  |
| [`CodeMatchType`](/docs/reference/wiki/programsynthesis/codematchtype/) |  |
| [`CodeSymbolKind`](/docs/reference/wiki/programsynthesis/codesymbolkind/) |  |
| [`CodeTask`](/docs/reference/wiki/programsynthesis/codetask/) | Defines the different types of code-related tasks that can be performed. |
| [`CompilationDiagnosticSeverity`](/docs/reference/wiki/programsynthesis/compilationdiagnosticseverity/) |  |
| [`ProgramExecuteErrorCode`](/docs/reference/wiki/programsynthesis/programexecuteerrorcode/) | Classifies program execution failures in a structured, machine-readable way. |
| [`ProgramLanguage`](/docs/reference/wiki/programsynthesis/programlanguage/) | Defines the programming languages that can be synthesized or processed. |
| [`ProgramSynthesisModelKind`](/docs/reference/wiki/programsynthesis/programsynthesismodelkind/) | Defines the built-in program synthesis model implementations that can be configured via the primary builder/result APIs. |
| [`SqlDialect`](/docs/reference/wiki/programsynthesis/sqldialect/) | Supported SQL dialects for ProgramLanguage.SQL execution and evaluation. |
| [`SqlExecuteErrorCode`](/docs/reference/wiki/programsynthesis/sqlexecuteerrorcode/) | Classifies SQL execution failures in a structured, machine-readable way. |
| [`SqlValueKind`](/docs/reference/wiki/programsynthesis/sqlvaluekind/) |  |
| [`SynthesisType`](/docs/reference/wiki/programsynthesis/synthesistype/) | Defines the different types of program synthesis approaches available. |

## Options & Configuration (4)

| Type | Summary |
|:-----|:--------|
| [`CodeTokenizationPipelineOptions`](/docs/reference/wiki/programsynthesis/codetokenizationpipelineoptions/) | Options for `CodeTokenizationPipeline` that control optional structural extraction. |
| [`NeuralProgramSynthesizerOptions`](/docs/reference/wiki/programsynthesis/neuralprogramsynthesizeroptions/) | Configuration options for the NeuralProgramSynthesizer. |
| [`ProgramSynthesisOptions`](/docs/reference/wiki/programsynthesis/programsynthesisoptions/) | Configuration options for enabling Program Synthesis / Code Tasks via the primary builder/result APIs. |
| [`ProgramSynthesisServingClientOptions`](/docs/reference/wiki/programsynthesis/programsynthesisservingclientoptions/) | Configuration for calling an AiDotNet.Serving instance for Program Synthesis operations. |

## Helpers & Utilities (3)

| Type | Summary |
|:-----|:--------|
| [`CodeAstNode`](/docs/reference/wiki/programsynthesis/codeastnode/) | Represents a node in an abstract syntax tree (AST) for a piece of source code. |
| [`ProgramSynthesisTokenizerFactory`](/docs/reference/wiki/programsynthesis/programsynthesistokenizerfactory/) | Creates safe, code-aware default tokenizers for `ProgramLanguage` values. |
| [`SqlSyntaxValidation`](/docs/reference/wiki/programsynthesis/sqlsyntaxvalidation/) | Global registration point for the optional precise SQL syntax validator used by `NeuralProgramSynthesizer`. |

