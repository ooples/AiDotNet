---
title: "Adversarial Robustness"
description: "All 20 public types in the AiDotNet.adversarialrobustness namespace, organized by kind."
section: "API Reference"
---

**20** public types in this namespace, organized by kind.

## Models & Types (17)

| Type | Summary |
|:-----|:--------|
| [`AdaptiveRandomizedSmoothing<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/adaptiverandomizedsmoothing/) | Implements Adaptive Randomized Smoothing with f-Differential Privacy (f-DP) certified defense. |
| [`AdversarialPreferenceAlignment<T>`](/docs/reference/wiki/adversarialrobustness/adversarialpreferencealignment/) | Implements adversarial preference alignment that combines RLHF with adversarial robustness, ensuring the model maintains alignment properties even under adversarial perturbation. |
| [`AdversarialPromptDefense<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/adversarialpromptdefense/) | Implements adaptive visual prompt-based defense that prepends learned perturbation-resistant tokens/patches to inputs, improving adversarial robustness without model retraining. |
| [`AdversarialTraining<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/adversarialtraining/) | Implements adversarial training as a defense mechanism. |
| [`AutoAttack<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/autoattack/) | Implements the AutoAttack framework - an ensemble of diverse attacks. |
| [`CROWNVerification<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/crownverification/) | Implements CROWN (Convex Relaxation based perturbation analysis Of Neural networks) for computing certified robustness bounds. |
| [`CWAttack<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/cwattack/) | Implements the Carlini and Wagner (C and W) attack. |
| [`ContentClassificationResult<T>`](/docs/reference/wiki/adversarialrobustness/contentclassificationresult/) | Result of content classification by an ML model. |
| [`FGSMAttack<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/fgsmattack/) | Implements the Fast Gradient Sign Method (FGSM) attack. |
| [`IntervalBoundPropagation<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/intervalboundpropagation/) | Implements Interval Bound Propagation (IBP) for certifying neural network robustness. |
| [`ModelCard`](/docs/reference/wiki/adversarialrobustness/modelcard/) | Represents a Model Card for documenting AI model characteristics and performance. |
| [`PGDAttack<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/pgdattack/) | Implements the Projected Gradient Descent (PGD) attack. |
| [`RLHFAlignment<T>`](/docs/reference/wiki/adversarialrobustness/rlhfalignment/) | Implements Reinforcement Learning from Human Feedback (RLHF) for AI alignment. |
| [`RandomizedSmoothing<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/randomizedsmoothing/) | Implements Randomized Smoothing for certified robustness. |
| [`RuleBasedContentClassifier<T>`](/docs/reference/wiki/adversarialrobustness/rulebasedcontentclassifier/) | A rule-based content classifier that uses pattern matching for classification. |
| [`SafetyFilter<T>`](/docs/reference/wiki/adversarialrobustness/safetyfilter/) | Implements comprehensive safety filtering for AI model inputs and outputs. |
| [`ViTAdversarialAttack<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/vitadversarialattack/) | Implements Vision Transformer (ViT)-specific adversarial attacks that exploit the self-attention mechanism and patch-based architecture. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`AdversarialAttackBase<T, TInput, TOutput>`](/docs/reference/wiki/adversarialrobustness/adversarialattackbase/) | Base class for adversarial attack implementations. |
| [`ContentClassifierBase<T>`](/docs/reference/wiki/adversarialrobustness/contentclassifierbase/) | Base class for ML-based content classifiers. |

## Interfaces (1)

| Type | Summary |
|:-----|:--------|
| [`IContentClassifier<T>`](/docs/reference/wiki/adversarialrobustness/icontentclassifier/) | Defines the interface for ML-based content classification. |

