---
title: "Safety"
description: "All 177 public types in the AiDotNet.safety namespace, organized by kind."
section: "API Reference"
---

**177** public types in this namespace, organized by kind.

## Models & Types (102)

| Type | Summary |
|:-----|:--------|
| [`AcousticToxicityDetector<T>`](/docs/reference/wiki/safety/acoustictoxicitydetector/) | Detects toxic/aggressive speech patterns directly from acoustic features without transcription. |
| [`AdversarialBenchmark<T>`](/docs/reference/wiki/safety/adversarialbenchmark/) | Benchmark for evaluating adversarial robustness of the safety pipeline against evasion attacks. |
| [`AdversarialImageEvaluator<T>`](/docs/reference/wiki/safety/adversarialimageevaluator/) | Detects adversarial perturbations in images via the learnable feature-squeezing ensemble described by Xu et al. |
| [`AdversarialRobustnessEvaluator<T>`](/docs/reference/wiki/safety/adversarialrobustnessevaluator/) | Evaluates text inputs for adversarial perturbations designed to evade safety filters. |
| [`AudioSafetyResult`](/docs/reference/wiki/safety/audiosafetyresult/) | Detailed result from audio safety evaluation. |
| [`AudioSealWatermarker<T>`](/docs/reference/wiki/safety/audiosealwatermarker/) | Audio watermarker using AudioSeal-style localized watermarking for voice cloning detection. |
| [`AudioWatermarker<T>`](/docs/reference/wiki/safety/audiowatermarker/) | Embeds and detects invisible watermarks in audio content using spread-spectrum techniques. |
| [`BiasBenchmark<T>`](/docs/reference/wiki/safety/biasbenchmark/) | Benchmark for evaluating bias and fairness detection accuracy across demographic groups. |
| [`BiasReport`](/docs/reference/wiki/safety/biasreport/) | Detailed report from bias and fairness evaluation. |
| [`ClaimVerdict`](/docs/reference/wiki/safety/claimverdict/) | Verdict for a single claim in a hallucination check. |
| [`ClassifierToxicityDetector<T>`](/docs/reference/wiki/safety/classifiertoxicitydetector/) | Detects toxic text using a trained linear classifier over character n-gram features. |
| [`ComplianceReport`](/docs/reference/wiki/safety/compliancereport/) | Detailed report from regulatory compliance evaluation. |
| [`ComplianceViolation`](/docs/reference/wiki/safety/complianceviolation/) | A specific compliance violation found during evaluation. |
| [`CompositeGuardrail<T>`](/docs/reference/wiki/safety/compositeguardrail/) | Chains multiple guardrails into a single composite guardrail that runs all child guardrails in sequence and aggregates their findings. |
| [`CompositePIIDetector<T>`](/docs/reference/wiki/safety/compositepiidetector/) | Combines multiple PII detection strategies into a unified detector with deduplication. |
| [`ComprehensiveSafetyBenchmark<T>`](/docs/reference/wiki/safety/comprehensivesafetybenchmark/) | Comprehensive safety benchmark that aggregates all individual benchmarks into a single suite. |
| [`ConsistencyDeepfakeDetector<T>`](/docs/reference/wiki/safety/consistencydeepfakedetector/) | Detects deepfake/AI-generated images by checking spatial consistency and natural image statistics violations. |
| [`ContextAwarePIIDetector<T>`](/docs/reference/wiki/safety/contextawarepiidetector/) | Context-aware PII detector that reduces false positives by analyzing surrounding text. |
| [`CopyrightMatch`](/docs/reference/wiki/safety/copyrightmatch/) | A match between generated text and potentially copyrighted content. |
| [`CopyrightResult`](/docs/reference/wiki/safety/copyrightresult/) | Detailed result from copyright and memorization detection. |
| [`CrossModalConsistencyChecker<T>`](/docs/reference/wiki/safety/crossmodalconsistencychecker/) | Checks consistency between different modalities (text, image, audio) to detect misaligned or manipulated multimodal content. |
| [`CustomRule`](/docs/reference/wiki/safety/customrule/) | A custom safety rule that evaluates text and optionally returns a finding. |
| [`CustomRuleGuardrail<T>`](/docs/reference/wiki/safety/customruleguardrail/) | User-defined guardrail that applies custom validation rules to text content. |
| [`DeepfakeResult`](/docs/reference/wiki/safety/deepfakeresult/) | Detailed result from deepfake and AI-generated image detection. |
| [`DemographicParityChecker<T>`](/docs/reference/wiki/safety/demographicparitychecker/) | Checks for demographic parity violations by detecting differential treatment of demographic groups in model outputs. |
| [`EUAIActComplianceChecker<T>`](/docs/reference/wiki/safety/euaiactcompliancechecker/) | Checks compliance with the EU AI Act requirements for AI systems. |
| [`EmbeddingCopyrightDetector<T>`](/docs/reference/wiki/safety/embeddingcopyrightdetector/) | Detects potential copyright violations using embedding-based semantic similarity to known copyrighted works. |
| [`EmbeddingToxicityDetector<T>`](/docs/reference/wiki/safety/embeddingtoxicitydetector/) | Detects toxic text using embedding-based cosine similarity to known toxic concept vectors. |
| [`EnsembleImageSafetyClassifier<T>`](/docs/reference/wiki/safety/ensembleimagesafetyclassifier/) | Combines multiple image safety classifiers into a weighted ensemble for robust detection. |
| [`EnsembleJailbreakDetector<T>`](/docs/reference/wiki/safety/ensemblejailbreakdetector/) | Combines multiple jailbreak detection strategies into a robust ensemble. |
| [`EnsembleToxicityDetector<T>`](/docs/reference/wiki/safety/ensembletoxicitydetector/) | Combines multiple toxicity detectors into a weighted ensemble for improved accuracy. |
| [`EntailmentHallucinationDetector<T>`](/docs/reference/wiki/safety/entailmenthallucinationdetector/) | Detects hallucinations using textual entailment (NLI) principles: checking whether reference documents entail (support) each claim in the model output. |
| [`EqualizedOddsChecker<T>`](/docs/reference/wiki/safety/equalizedoddschecker/) | Checks for equalized odds violations by analyzing whether model quality or effort varies across demographic groups mentioned in the text. |
| [`FrameSamplingVideoModerator<T>`](/docs/reference/wiki/safety/framesamplingvideomoderator/) | Video content moderator that samples frames and applies image safety classification. |
| [`FrequencyDeepfakeDetector<T>`](/docs/reference/wiki/safety/frequencydeepfakedetector/) | Detects AI-generated/deepfake images by analyzing frequency domain artifacts. |
| [`FrequencyImageWatermarker<T>`](/docs/reference/wiki/safety/frequencyimagewatermarker/) | Image watermarker that embeds watermarks in the frequency domain using DCT/DWT coefficients. |
| [`GDPRComplianceChecker<T>`](/docs/reference/wiki/safety/gdprcompliancechecker/) | Checks compliance with GDPR requirements related to AI and personal data processing. |
| [`GradientJailbreakDetector<T>`](/docs/reference/wiki/safety/gradientjailbreakdetector/) | Detects gradient-based adversarial jailbreak attacks by analyzing token-level anomalies in the input text. |
| [`GroupBiasResult`](/docs/reference/wiki/safety/groupbiasresult/) | Bias analysis result for a specific demographic group. |
| [`GuardrailResult`](/docs/reference/wiki/safety/guardrailresult/) | Result from guardrail evaluation. |
| [`GuardrailRule`](/docs/reference/wiki/safety/guardrailrule/) | Defines a declarative guardrail rule with a condition expression and corresponding action. |
| [`HallucinationBenchmark<T>`](/docs/reference/wiki/safety/hallucinationbenchmark/) | Benchmark for evaluating hallucination and factual grounding detection accuracy. |
| [`HallucinationResult`](/docs/reference/wiki/safety/hallucinationresult/) | Detailed result from hallucination detection with per-claim verdicts. |
| [`ImageSafetyResult`](/docs/reference/wiki/safety/imagesafetyresult/) | Detailed result from image safety classification. |
| [`ImageWatermarker<T>`](/docs/reference/wiki/safety/imagewatermarker/) | Embeds and detects invisible watermarks in images using frequency-domain techniques. |
| [`InputGuardrail<T>`](/docs/reference/wiki/safety/inputguardrail/) | Input guardrail that validates user input before it reaches the model. |
| [`IntersectionalBiasDetector<T>`](/docs/reference/wiki/safety/intersectionalbiasdetector/) | Detects intersectional bias — bias that uniquely affects individuals at the intersection of multiple demographic identities (e.g., Black women, elderly Asian men). |
| [`InvisibleImageWatermarker<T>`](/docs/reference/wiki/safety/invisibleimagewatermarker/) | Image watermarker that embeds imperceptible spatial-domain watermarks. |
| [`JailbreakBenchmark<T>`](/docs/reference/wiki/safety/jailbreakbenchmark/) | Benchmark for evaluating jailbreak and prompt injection detection accuracy. |
| [`KnowledgeTripletHallucinationDetector<T>`](/docs/reference/wiki/safety/knowledgetriplethallucinationdetector/) | Detects hallucinations by extracting (subject, predicate, object) knowledge triplets and verifying them against reference documents. |
| [`LexicalWatermarker<T>`](/docs/reference/wiki/safety/lexicalwatermarker/) | Text watermarker that embeds watermarks via synonym substitution patterns. |
| [`MaskingVoiceProtector<T>`](/docs/reference/wiki/safety/maskingvoiceprotector/) | Protects voice recordings against cloning using psychoacoustic masking — adding noise that is hidden beneath audible content but disrupts speaker embedding extraction. |
| [`MultimodalGuardrail<T>`](/docs/reference/wiki/safety/multimodalguardrail/) | Unified guardrail for vision-language models (VLMs) and multimodal AI systems that validates both text and image content together. |
| [`MultimodalSafetyResult`](/docs/reference/wiki/safety/multimodalsafetyresult/) | Detailed result from multimodal safety evaluation. |
| [`MultimodalVideoModerator<T>`](/docs/reference/wiki/safety/multimodalvideomoderator/) | Comprehensive video moderator that combines frame-level content classification, temporal deepfake detection, and optional audio track analysis. |
| [`NERPIIDetector<T>`](/docs/reference/wiki/safety/nerpiidetector/) | Detects PII using Named Entity Recognition (NER) heuristics based on contextual patterns. |
| [`NeuralImageWatermarker<T>`](/docs/reference/wiki/safety/neuralimagewatermarker/) | Image watermarker that uses an encoder-decoder neural network approach for embedding. |
| [`NgramCopyrightDetector<T>`](/docs/reference/wiki/safety/ngramcopyrightdetector/) | Detects potential copyright violations by measuring n-gram overlap with known copyrighted works. |
| [`OutputGuardrail<T>`](/docs/reference/wiki/safety/outputguardrail/) | Output guardrail that validates model output before it reaches the user. |
| [`PIIBenchmark<T>`](/docs/reference/wiki/safety/piibenchmark/) | Benchmark for evaluating PII (Personally Identifiable Information) detection accuracy. |
| [`PIIEntity`](/docs/reference/wiki/safety/piientity/) | Represents a detected PII entity in text. |
| [`PIIResult`](/docs/reference/wiki/safety/piiresult/) | Detailed result from PII detection with detected entities and redacted text. |
| [`PatternJailbreakDetector<T>`](/docs/reference/wiki/safety/patternjailbreakdetector/) | Pattern-based jailbreak and prompt injection detector. |
| [`PerplexityMemorizationDetector<T>`](/docs/reference/wiki/safety/perplexitymemorizationdetector/) | Detects potential training data memorization by estimating text perplexity. |
| [`PerturbationVoiceProtector<T>`](/docs/reference/wiki/safety/perturbationvoiceprotector/) | Protects voice recordings against cloning by adding imperceptible adversarial perturbations. |
| [`ProvenanceDeepfakeDetector<T>`](/docs/reference/wiki/safety/provenancedeepfakedetector/) | Detects deepfake/AI-generated images by analyzing provenance signals: compression artifacts, statistical fingerprints, and embedded watermark traces. |
| [`ReferenceBasedHallucinationDetector<T>`](/docs/reference/wiki/safety/referencebasedhallucinationdetector/) | Detects hallucinations by comparing model output against provided reference documents. |
| [`RegexPIIDetector<T>`](/docs/reference/wiki/safety/regexpiidetector/) | Regex-based PII (Personally Identifiable Information) detector for common PII patterns. |
| [`RepresentationalBiasDetector<T>`](/docs/reference/wiki/safety/representationalbiasdetector/) | Detects representational bias by analyzing whether demographic groups are underrepresented, overrepresented, or systematically associated with specific roles/contexts in text. |
| [`RuleBasedToxicityDetector<T>`](/docs/reference/wiki/safety/rulebasedtoxicitydetector/) | Rule-based toxicity detector using pattern matching for harmful content detection. |
| [`SOC2ComplianceChecker<T>`](/docs/reference/wiki/safety/soc2compliancechecker/) | Checks compliance with SOC 2 requirements for AI system security and availability. |
| [`SafetyBenchmarkCase`](/docs/reference/wiki/safety/safetybenchmarkcase/) | A single test case for the safety benchmark. |
| [`SafetyBenchmarkCategoryResult`](/docs/reference/wiki/safety/safetybenchmarkcategoryresult/) | Per-category benchmark results. |
| [`SafetyBenchmarkReport`](/docs/reference/wiki/safety/safetybenchmarkreport/) | Comprehensive report from running all safety benchmarks. |
| [`SafetyBenchmarkResult`](/docs/reference/wiki/safety/safetybenchmarkresult/) | Results from running a safety benchmark. |
| [`SafetyBenchmarkRunner<T>`](/docs/reference/wiki/safety/safetybenchmarkrunner/) | Runs safety benchmarks against a configured safety pipeline to measure detection performance. |
| [`SafetyFinding`](/docs/reference/wiki/safety/safetyfinding/) | Represents a single safety finding from a safety module evaluation. |
| [`SafetyPipeline<T>`](/docs/reference/wiki/safety/safetypipeline/) | Composable safety pipeline that runs multiple safety modules and aggregates their findings. |
| [`SafetyReport`](/docs/reference/wiki/safety/safetyreport/) | Unified safety report aggregating findings from all safety modules in the pipeline. |
| [`SamplingWatermarker<T>`](/docs/reference/wiki/safety/samplingwatermarker/) | Text watermarker that modifies token sampling distributions to embed watermarks (SynthID-style). |
| [`SceneGraphSafetyClassifier<T>`](/docs/reference/wiki/safety/scenegraphsafetyclassifier/) | Scene graph-based image safety classifier that analyzes spatial relationships between detected entities to identify unsafe content configurations. |
| [`SelfConsistencyHallucinationDetector<T>`](/docs/reference/wiki/safety/selfconsistencyhallucinationdetector/) | Detects hallucinations by checking internal consistency of claims within the text. |
| [`SemanticJailbreakDetector<T>`](/docs/reference/wiki/safety/semanticjailbreakdetector/) | Detects jailbreak attempts using semantic embedding similarity to known attack patterns. |
| [`SpectralAudioWatermarker<T>`](/docs/reference/wiki/safety/spectralaudiowatermarker/) | Audio watermarker that embeds watermarks in the frequency domain using spectral modification. |
| [`StereotypeDetector<T>`](/docs/reference/wiki/safety/stereotypedetector/) | Detects stereotypical associations between demographic groups and attributes in text. |
| [`SyntacticWatermarker<T>`](/docs/reference/wiki/safety/syntacticwatermarker/) | Text watermarker that embeds watermarks through syntactic structure rearrangement. |
| [`TemporalConsistencyDetector<T>`](/docs/reference/wiki/safety/temporalconsistencydetector/) | Detects deepfake videos by analyzing temporal consistency between consecutive frames. |
| [`TextImageAlignmentChecker<T>`](/docs/reference/wiki/safety/textimagealignmentchecker/) | Checks semantic alignment between text descriptions and associated images to detect mismatched or deceptive text-image pairs. |
| [`TextWatermarker<T>`](/docs/reference/wiki/safety/textwatermarker/) | Embeds and detects invisible watermarks in AI-generated text using token distribution biasing. |
| [`TopicRestrictionGuardrail<T>`](/docs/reference/wiki/safety/topicrestrictionguardrail/) | Guardrail that restricts conversation to approved topics using keyword and pattern matching. |
| [`ToxicSpan`](/docs/reference/wiki/safety/toxicspan/) | A span of text identified as toxic. |
| [`ToxicityBenchmark<T>`](/docs/reference/wiki/safety/toxicitybenchmark/) | Benchmark for evaluating toxicity detection accuracy across hate speech, threats, and harassment. |
| [`ToxicityResult`](/docs/reference/wiki/safety/toxicityresult/) | Detailed result from toxicity detection with per-category scores and spans. |
| [`TranscriptionToxicityDetector<T>`](/docs/reference/wiki/safety/transcriptiontoxicitydetector/) | Detects toxic content in audio by analyzing acoustic features indicative of aggressive speech. |
| [`ViTImageSafetyClassifier<T>`](/docs/reference/wiki/safety/vitimagesafetyclassifier/) | Vision Transformer (ViT)-inspired image safety classifier using patch-based feature extraction and multi-head attention pooling for multi-label safety classification. |
| [`VideoSafetyResult`](/docs/reference/wiki/safety/videosafetyresult/) | Detailed result from video safety evaluation with per-frame annotations. |
| [`VoiceprintDeepfakeDetector<T>`](/docs/reference/wiki/safety/voiceprintdeepfakedetector/) | Detects deepfake audio by analyzing speaker voiceprint consistency throughout the recording. |
| [`WatermarkBenchmark<T>`](/docs/reference/wiki/safety/watermarkbenchmark/) | Benchmark for evaluating watermark detection accuracy across text watermarking techniques. |
| [`WatermarkDeepfakeDetector<T>`](/docs/reference/wiki/safety/watermarkdeepfakedetector/) | Detects AI-generated audio by looking for the presence or absence of known watermark patterns (e.g., AudioSeal-style localized watermarks). |
| [`WatermarkDetector<T>`](/docs/reference/wiki/safety/watermarkdetector/) | Unified watermark detector that combines multiple watermark detection strategies. |
| [`WatermarkResult`](/docs/reference/wiki/safety/watermarkresult/) | Detailed result from watermark detection across any modality. |
| [`WatermarkVoiceProtector<T>`](/docs/reference/wiki/safety/watermarkvoiceprotector/) | Protects voice recordings by embedding imperceptible watermarks that survive voice cloning and can be detected in cloned output. |

## Base Classes (21)

| Type | Summary |
|:-----|:--------|
| [`AudioDeepfakeDetectorBase<T>`](/docs/reference/wiki/safety/audiodeepfakedetectorbase/) | Abstract base class for audio deepfake detection modules. |
| [`AudioSafetyModuleBase<T>`](/docs/reference/wiki/safety/audiosafetymodulebase/) | Abstract base class for audio safety modules. |
| [`AudioWatermarkerBase<T>`](/docs/reference/wiki/safety/audiowatermarkerbase/) | Abstract base class for audio watermarking modules. |
| [`ComplianceModuleBase<T>`](/docs/reference/wiki/safety/compliancemodulebase/) | Abstract base class for regulatory compliance checking modules. |
| [`CopyrightDetectorBase<T>`](/docs/reference/wiki/safety/copyrightdetectorbase/) | Abstract base class for copyright and memorization detection modules. |
| [`DeepfakeDetectorBase<T>`](/docs/reference/wiki/safety/deepfakedetectorbase/) | Abstract base class for image deepfake detection modules. |
| [`GuardrailBase<T>`](/docs/reference/wiki/safety/guardrailbase/) | Abstract base class for guardrail modules. |
| [`HallucinationDetectorBase<T>`](/docs/reference/wiki/safety/hallucinationdetectorbase/) | Abstract base class for hallucination detection modules. |
| [`ImageSafetyClassifierBase<T>`](/docs/reference/wiki/safety/imagesafetyclassifierbase/) | Abstract base class for image safety classifiers. |
| [`ImageSafetyModuleBase<T>`](/docs/reference/wiki/safety/imagesafetymodulebase/) | Abstract base class for image safety modules. |
| [`ImageWatermarkerBase<T>`](/docs/reference/wiki/safety/imagewatermarkerbase/) | Abstract base class for image watermarking modules. |
| [`JailbreakDetectorBase<T>`](/docs/reference/wiki/safety/jailbreakdetectorbase/) | Abstract base class for jailbreak and prompt injection detection modules. |
| [`MultimodalSafetyModuleBase<T>`](/docs/reference/wiki/safety/multimodalsafetymodulebase/) | Abstract base class for multimodal safety modules that analyze cross-modal content. |
| [`PIIDetectorBase<T>`](/docs/reference/wiki/safety/piidetectorbase/) | Abstract base class for PII detection modules. |
| [`SafetyBenchmarkBase<T>`](/docs/reference/wiki/safety/safetybenchmarkbase/) | Abstract base class for safety benchmark modules. |
| [`SafetyModuleBase<T>`](/docs/reference/wiki/safety/safetymodulebase/) | Abstract base class for all safety modules, providing common infrastructure. |
| [`TextSafetyModuleBase<T>`](/docs/reference/wiki/safety/textsafetymodulebase/) | Abstract base class for text safety modules providing common text processing utilities. |
| [`TextToxicityDetectorBase<T>`](/docs/reference/wiki/safety/texttoxicitydetectorbase/) | Abstract base class for toxicity detection modules. |
| [`TextWatermarkerBase<T>`](/docs/reference/wiki/safety/textwatermarkerbase/) | Abstract base class for text watermarking modules. |
| [`VideoSafetyModuleBase<T>`](/docs/reference/wiki/safety/videosafetymodulebase/) | Abstract base class for video safety modules. |
| [`VoiceProtectorBase<T>`](/docs/reference/wiki/safety/voiceprotectorbase/) | Abstract base class for voice protection modules. |

## Interfaces (17)

| Type | Summary |
|:-----|:--------|
| [`IAudioDeepfakeDetector<T>`](/docs/reference/wiki/safety/iaudiodeepfakedetector/) | Interface for audio deepfake and voice cloning detection modules. |
| [`IAudioToxicityDetector<T>`](/docs/reference/wiki/safety/iaudiotoxicitydetector/) | Interface for audio toxicity detection modules that identify harmful speech content. |
| [`IAudioWatermarker<T>`](/docs/reference/wiki/safety/iaudiowatermarker/) | Interface for audio watermarking modules that embed and detect watermarks in audio. |
| [`IComplianceModule<T>`](/docs/reference/wiki/safety/icompliancemodule/) | Interface for regulatory compliance checking modules. |
| [`ICopyrightDetector<T>`](/docs/reference/wiki/safety/icopyrightdetector/) | Interface for copyright and memorization detection modules. |
| [`IDeepfakeDetector<T>`](/docs/reference/wiki/safety/ideepfakedetector/) | Interface for deepfake and AI-generated image detection modules. |
| [`IGuardrail<T>`](/docs/reference/wiki/safety/iguardrail/) | Interface for guardrail modules that validate and filter input/output content. |
| [`IHallucinationDetector<T>`](/docs/reference/wiki/safety/ihallucinationdetector/) | Interface for hallucination detection modules that identify fabricated or unfaithful content. |
| [`IImageSafetyClassifier<T>`](/docs/reference/wiki/safety/iimagesafetyclassifier/) | Interface for image safety classifiers that detect NSFW, violent, or otherwise harmful images. |
| [`IImageWatermarker<T>`](/docs/reference/wiki/safety/iimagewatermarker/) | Interface for image watermarking modules that embed and detect watermarks in images. |
| [`IJailbreakDetector<T>`](/docs/reference/wiki/safety/ijailbreakdetector/) | Interface for jailbreak and prompt injection detection modules. |
| [`IMultimodalSafetyModule<T>`](/docs/reference/wiki/safety/imultimodalsafetymodule/) | Interface for multimodal safety modules that analyze cross-modal content interactions. |
| [`IPIIDetector<T>`](/docs/reference/wiki/safety/ipiidetector/) | Interface for PII (Personally Identifiable Information) detection modules. |
| [`ISafetyBenchmark<T>`](/docs/reference/wiki/safety/isafetybenchmark/) | Interface for safety benchmark modules that evaluate safety module accuracy. |
| [`ITextToxicityDetector<T>`](/docs/reference/wiki/safety/itexttoxicitydetector/) | Interface for toxicity detection modules that identify harmful, abusive, or toxic text content. |
| [`ITextWatermarker<T>`](/docs/reference/wiki/safety/itextwatermarker/) | Interface for text watermarking modules that embed and detect watermarks in text. |
| [`IVoiceProtector<T>`](/docs/reference/wiki/safety/ivoiceprotector/) | Interface for voice protection modules that defend against voice cloning and deepfake attacks. |

## Enums (8)

| Type | Summary |
|:-----|:--------|
| [`AudioWatermarkType`](/docs/reference/wiki/safety/audiowatermarktype/) | The type of audio watermarking technique to use. |
| [`GuardrailConditionType`](/docs/reference/wiki/safety/guardrailconditiontype/) | Specifies the type of condition a guardrail rule checks. |
| [`GuardrailDirection`](/docs/reference/wiki/safety/guardraildirection/) | Specifies whether a guardrail checks input, output, or both directions. |
| [`ImageClassifierType`](/docs/reference/wiki/safety/imageclassifiertype/) | The type of image safety classifier to use. |
| [`ImageWatermarkType`](/docs/reference/wiki/safety/imagewatermarktype/) | The type of image watermarking technique to use. |
| [`RiskLevel`](/docs/reference/wiki/safety/risklevel/) | EU AI Act risk classification levels (Articles 5, 6, 50, 52). |
| [`TextWatermarkType`](/docs/reference/wiki/safety/textwatermarktype/) | The type of text watermarking technique to use. |
| [`VoiceProtectionType`](/docs/reference/wiki/safety/voiceprotectiontype/) | The type of voice protection technique to use. |

## Options & Configuration (26)

| Type | Summary |
|:-----|:--------|
| [`AudioSafetyConfig`](/docs/reference/wiki/safety/audiosafetyconfig/) | Configuration for audio safety detection modules. |
| [`AudioSafetyConfig`](/docs/reference/wiki/safety/audiosafetyconfig-2/) | Configuration for audio safety modules. |
| [`AudioWatermarkConfig`](/docs/reference/wiki/safety/audiowatermarkconfig/) | Configuration for audio watermarking modules. |
| [`BiasConfig`](/docs/reference/wiki/safety/biasconfig/) | Configuration for bias and fairness detection modules. |
| [`ComplianceConfig`](/docs/reference/wiki/safety/complianceconfig/) | Configuration for regulatory compliance checking modules. |
| [`ComplianceConfig`](/docs/reference/wiki/safety/complianceconfig-2/) | Configuration for regulatory compliance. |
| [`CopyrightDetectorConfig`](/docs/reference/wiki/safety/copyrightdetectorconfig/) | Configuration for copyright and memorization detection modules. |
| [`DeepfakeDetectorConfig`](/docs/reference/wiki/safety/deepfakedetectorconfig/) | Configuration for deepfake and AI-generated image detection modules. |
| [`FairnessConfig`](/docs/reference/wiki/safety/fairnessconfig/) | Configuration for fairness and bias detection. |
| [`GuardrailConfig`](/docs/reference/wiki/safety/guardrailconfig/) | Configuration for input/output guardrails. |
| [`GuardrailConfig`](/docs/reference/wiki/safety/guardrailconfig-2/) | Configuration for guardrail modules. |
| [`HallucinationDetectorConfig`](/docs/reference/wiki/safety/hallucinationdetectorconfig/) | Configuration for hallucination detection modules. |
| [`ImageSafetyConfig`](/docs/reference/wiki/safety/imagesafetyconfig/) | Configuration for image safety classification modules. |
| [`ImageSafetyConfig`](/docs/reference/wiki/safety/imagesafetyconfig-2/) | Configuration for image safety modules. |
| [`ImageWatermarkConfig`](/docs/reference/wiki/safety/imagewatermarkconfig/) | Configuration for image watermarking modules. |
| [`JailbreakDetectorConfig`](/docs/reference/wiki/safety/jailbreakdetectorconfig/) | Configuration for jailbreak and prompt injection detection modules. |
| [`MultimodalSafetyConfig`](/docs/reference/wiki/safety/multimodalsafetyconfig/) | Configuration for multimodal safety modules. |
| [`PIIDetectorConfig`](/docs/reference/wiki/safety/piidetectorconfig/) | Configuration for PII detection modules. |
| [`SafetyConfig`](/docs/reference/wiki/safety/safetyconfig/) | Master configuration for the comprehensive safety pipeline. |
| [`TextSafetyConfig`](/docs/reference/wiki/safety/textsafetyconfig/) | Configuration for text safety modules. |
| [`TextWatermarkConfig`](/docs/reference/wiki/safety/textwatermarkconfig/) | Configuration for text watermarking modules. |
| [`ToxicityDetectorConfig`](/docs/reference/wiki/safety/toxicitydetectorconfig/) | Configuration for toxicity detection modules. |
| [`VideoSafetyConfig`](/docs/reference/wiki/safety/videosafetyconfig/) | Configuration for video safety detection modules. |
| [`VideoSafetyConfig`](/docs/reference/wiki/safety/videosafetyconfig-2/) | Configuration for video safety modules. |
| [`VoiceProtectorConfig`](/docs/reference/wiki/safety/voiceprotectorconfig/) | Configuration for voice protection (anti-cloning) modules. |
| [`WatermarkConfig`](/docs/reference/wiki/safety/watermarkconfig/) | Configuration for watermarking modules. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`SafetyPipelineBuilder<T>`](/docs/reference/wiki/safety/safetypipelinebuilder/) | Fluent builder for constructing a `SafetyPipeline` with custom modules. |
| [`StandardSafetyBenchmarks`](/docs/reference/wiki/safety/standardsafetybenchmarks/) | Provides standard safety benchmark test suites inspired by published safety evaluation datasets. |

## Exceptions (1)

| Type | Summary |
|:-----|:--------|
| [`SafetyViolationException`](/docs/reference/wiki/safety/safetyviolationexception/) | Exception thrown when content fails a safety check and the configuration requires throwing. |

