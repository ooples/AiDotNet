---
title: "Exceptions"
description: "All 13 public types in the AiDotNet.exceptions namespace, organized by kind."
section: "API Reference"
---

**13** public types in this namespace, organized by kind.

## Enums (1)

| Type | Summary |
|:-----|:--------|
| [`TrialExpirationReason`](/docs/reference/wiki/exceptions/trialexpirationreason/) | Specifies the reason a trial period expired or a license check failed. |

## Exceptions (12)

| Type | Summary |
|:-----|:--------|
| [`AiDotNetException`](/docs/reference/wiki/exceptions/aidotnetexception/) | Base exception for all AiDotNet-specific exceptions. |
| [`ForwardPassRequiredException`](/docs/reference/wiki/exceptions/forwardpassrequiredexception/) | Exception thrown when an operation is attempted before a required forward pass has been completed. |
| [`InvalidDataValueException`](/docs/reference/wiki/exceptions/invaliddatavalueexception/) | Exception thrown when input data contains invalid values such as NaN or infinity. |
| [`InvalidInputDimensionException`](/docs/reference/wiki/exceptions/invalidinputdimensionexception/) | Exception thrown when input data dimensions are invalid for a specific algorithm or operation. |
| [`InvalidInputTypeException`](/docs/reference/wiki/exceptions/invalidinputtypeexception/) | Exception thrown when a neural network receives an input type that doesn't match its requirements. |
| [`LicenseRequiredException`](/docs/reference/wiki/exceptions/licenserequiredexception/) | Exception thrown when a model persistence operation (save or load) is attempted after the free trial period has expired and no valid license key is configured. |
| [`ModelTrainingException`](/docs/reference/wiki/exceptions/modeltrainingexception/) | Exception thrown when model training operations fail. |
| [`SerializationException`](/docs/reference/wiki/exceptions/serializationexception/) | Exception thrown when serialization or deserialization operations fail. |
| [`TensorDimensionException`](/docs/reference/wiki/exceptions/tensordimensionexception/) | Exception thrown when a tensor's dimension doesn't match the expected value. |
| [`TensorRankException`](/docs/reference/wiki/exceptions/tensorrankexception/) | Exception thrown when a tensor's rank doesn't match the expected rank. |
| [`TensorShapeMismatchException`](/docs/reference/wiki/exceptions/tensorshapemismatchexception/) | Exception thrown when a tensor's shape doesn't match the expected shape. |
| [`VectorLengthMismatchException`](/docs/reference/wiki/exceptions/vectorlengthmismatchexception/) | Exception thrown when a vector's length doesn't match the expected value. |

