---
title: "Prompt Engineering"
description: "All 55 public types in the AiDotNet.promptengineering namespace, organized by kind."
section: "API Reference"
---

**55** public types in this namespace, organized by kind.

## Models & Types (39)

| Type | Summary |
|:-----|:--------|
| [`BeamSearchOptimizer<T>`](/docs/reference/wiki/promptengineering/beamsearchoptimizer/) | Optimizer that uses beam search to explore multiple promising prompt variations. |
| [`CachingCompressor`](/docs/reference/wiki/promptengineering/cachingcompressor/) | Wrapper compressor that caches compression results for frequently used prompts. |
| [`ChainOfThoughtExample`](/docs/reference/wiki/promptengineering/chainofthoughtexample/) | Represents an example for few-shot chain-of-thought prompting. |
| [`ChainOfThoughtTemplate`](/docs/reference/wiki/promptengineering/chainofthoughttemplate/) | Template that structures prompts for chain-of-thought reasoning. |
| [`ChatMessage`](/docs/reference/wiki/promptengineering/chatmessage/) | Represents a single message in a chat conversation. |
| [`ChatPromptTemplate`](/docs/reference/wiki/promptengineering/chatprompttemplate/) |  |
| [`ClusterBasedExampleSelector<T>`](/docs/reference/wiki/promptengineering/clusterbasedexampleselector/) | Selects examples using a clustering approach to ensure broad coverage. |
| [`ComplexityAnalyzer`](/docs/reference/wiki/promptengineering/complexityanalyzer/) | Analyzer that focuses on measuring prompt complexity and structure. |
| [`CompositeCompressor`](/docs/reference/wiki/promptengineering/compositecompressor/) | Compressor that chains multiple compressors together in sequence. |
| [`CompositePromptTemplate`](/docs/reference/wiki/promptengineering/compositeprompttemplate/) | Template that combines multiple prompt templates in sequence. |
| [`CompressionResult`](/docs/reference/wiki/promptengineering/compressionresult/) | Contains the result of a prompt compression operation including metrics. |
| [`ConditionalPromptTemplate`](/docs/reference/wiki/promptengineering/conditionalprompttemplate/) | Template that supports conditional sections based on variable presence or values. |
| [`ContextWindowManager`](/docs/reference/wiki/promptengineering/contextwindowmanager/) | Manages context window limits for LLM prompts, providing token estimation and text truncation utilities. |
| [`DiscreteSearchOptimizer<T>`](/docs/reference/wiki/promptengineering/discretesearchoptimizer/) | Optimizer that uses discrete search to find better prompts by testing variations. |
| [`DiversityExampleSelector<T>`](/docs/reference/wiki/promptengineering/diversityexampleselector/) | Selects diverse examples to maximize coverage of different patterns. |
| [`EnsembleOptimizer<T>`](/docs/reference/wiki/promptengineering/ensembleoptimizer/) | Optimizer that combines multiple optimization strategies for better results. |
| [`FewShotPromptTemplate<T>`](/docs/reference/wiki/promptengineering/fewshotprompttemplate/) |  |
| [`FixedExampleSelector<T>`](/docs/reference/wiki/promptengineering/fixedexampleselector/) |  |
| [`GeneticOptimizer<T>`](/docs/reference/wiki/promptengineering/geneticoptimizer/) | Optimizer that uses genetic algorithms to evolve better prompts. |
| [`GreedyHillClimbingOptimizer<T>`](/docs/reference/wiki/promptengineering/greedyhillclimbingoptimizer/) | Simple greedy optimizer that always moves toward better solutions. |
| [`InstructionFollowingTemplate`](/docs/reference/wiki/promptengineering/instructionfollowingtemplate/) | Template optimized for clear, structured instruction-following tasks. |
| [`LLMSummarizationCompressor`](/docs/reference/wiki/promptengineering/llmsummarizationcompressor/) | Compressor that uses an LLM to intelligently summarize and compress prompts. |
| [`MMRExampleSelector<T>`](/docs/reference/wiki/promptengineering/mmrexampleselector/) | Selects examples using Maximum Marginal Relevance (MMR) to balance relevance and diversity. |
| [`PatternDetectionAnalyzer`](/docs/reference/wiki/promptengineering/patterndetectionanalyzer/) | Analyzer that specializes in detecting prompt patterns and categorizing prompts. |
| [`PromptChain`](/docs/reference/wiki/promptengineering/promptchain/) | Composes multiple prompt templates into a single formatted prompt. |
| [`PromptIssue`](/docs/reference/wiki/promptengineering/promptissue/) | Represents an issue or warning detected during prompt validation. |
| [`PromptMetrics`](/docs/reference/wiki/promptengineering/promptmetrics/) | Contains metrics and analysis results for a prompt. |
| [`PromptValidator`](/docs/reference/wiki/promptengineering/promptvalidator/) | Specialized prompt validator with comprehensive validation rules. |
| [`RandomExampleSelector<T>`](/docs/reference/wiki/promptengineering/randomexampleselector/) |  |
| [`RedundancyRemovalCompressor`](/docs/reference/wiki/promptengineering/redundancyremovalcompressor/) | Compressor that removes redundant phrases and verbose language from prompts. |
| [`RolePlayingTemplate`](/docs/reference/wiki/promptengineering/roleplayingtemplate/) | Template that creates persona-based prompts for role-playing scenarios. |
| [`SemanticSimilarityExampleSelector<T>`](/docs/reference/wiki/promptengineering/semanticsimilarityexampleselector/) | Selects examples based on semantic similarity to the query. |
| [`SentenceCompressor`](/docs/reference/wiki/promptengineering/sentencecompressor/) | Compressor that shortens sentences while preserving their core meaning. |
| [`SimplePromptTemplate`](/docs/reference/wiki/promptengineering/simpleprompttemplate/) |  |
| [`SimulatedAnnealingOptimizer<T>`](/docs/reference/wiki/promptengineering/simulatedannealingoptimizer/) | Optimizer that uses simulated annealing to escape local optima. |
| [`StopWordRemovalCompressor`](/docs/reference/wiki/promptengineering/stopwordremovalcompressor/) | Compressor that removes common stop words to reduce prompt length. |
| [`StructuredOutputTemplate`](/docs/reference/wiki/promptengineering/structuredoutputtemplate/) | Template that guides models to produce structured output in specific formats. |
| [`TokenCountAnalyzer`](/docs/reference/wiki/promptengineering/tokencountanalyzer/) | Analyzer that provides accurate token counting and cost estimation for prompts. |
| [`ValidationSummary`](/docs/reference/wiki/promptengineering/validationsummary/) | Summary of validation results. |

## Base Classes (5)

| Type | Summary |
|:-----|:--------|
| [`FewShotExampleSelectorBase<T>`](/docs/reference/wiki/promptengineering/fewshotexampleselectorbase/) | Base class for few-shot example selector implementations. |
| [`PromptAnalyzerBase`](/docs/reference/wiki/promptengineering/promptanalyzerbase/) | Provides a base implementation for prompt analyzers with common functionality. |
| [`PromptCompressorBase`](/docs/reference/wiki/promptengineering/promptcompressorbase/) | Provides a base implementation for prompt compressors with common functionality. |
| [`PromptOptimizerBase<T>`](/docs/reference/wiki/promptengineering/promptoptimizerbase/) | Base class for prompt optimizer implementations. |
| [`PromptTemplateBase`](/docs/reference/wiki/promptengineering/prompttemplatebase/) | Base class for prompt template implementations providing common functionality and validation. |

## Enums (4)

| Type | Summary |
|:-----|:--------|
| [`AggressivenessLevel`](/docs/reference/wiki/promptengineering/aggressivenesslevel/) | Defines the aggressiveness level of stop word removal. |
| [`EnsembleStrategy<T>`](/docs/reference/wiki/promptengineering/ensemblestrategy/) | Strategy for combining ensemble results. |
| [`IssueSeverity`](/docs/reference/wiki/promptengineering/issueseverity/) | Severity levels for prompt validation issues. |
| [`OutputFormat`](/docs/reference/wiki/promptengineering/outputformat/) | Supported output formats. |

## Options & Configuration (2)

| Type | Summary |
|:-----|:--------|
| [`CompressionOptions`](/docs/reference/wiki/promptengineering/compressionoptions/) | Options for controlling prompt compression behavior. |
| [`ValidationOptions`](/docs/reference/wiki/promptengineering/validationoptions/) | Options for controlling prompt validation behavior. |

## Helpers & Utilities (5)

| Type | Summary |
|:-----|:--------|
| [`ChainOfThoughtBuilder`](/docs/reference/wiki/promptengineering/chainofthoughtbuilder/) | Builder for constructing chain-of-thought templates fluently. |
| [`CompositeTemplateBuilder`](/docs/reference/wiki/promptengineering/compositetemplatebuilder/) | Builder for constructing composite templates fluently. |
| [`InstructionFollowingBuilder`](/docs/reference/wiki/promptengineering/instructionfollowingbuilder/) | Builder for constructing instruction-following templates fluently. |
| [`RolePlayingBuilder`](/docs/reference/wiki/promptengineering/roleplayingbuilder/) | Builder for constructing role-playing templates fluently. |
| [`StructuredOutputBuilder`](/docs/reference/wiki/promptengineering/structuredoutputbuilder/) | Builder for constructing structured output templates fluently. |

