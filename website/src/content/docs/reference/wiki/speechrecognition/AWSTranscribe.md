---
title: "AWSTranscribe<T>"
description: "AWS Transcribe: Amazon's scalable speech recognition service"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

AWS Transcribe: Amazon's scalable speech recognition service

## For Beginners

AWS Transcribe provides automatic speech recognition optimized for production workloads. The service supports real-time streaming and batch transcription across 100+ languages. Key features include custom vocabulary for domain terms, content redac...

## How It Works

**References:**

- API: "Amazon Transcribe" (AWS, 2024)

AWS Transcribe provides automatic speech recognition optimized for production workloads. The service supports real-time streaming and batch transcription across 100+ languages. Key features include custom vocabulary for domain terms, content redaction for PII, automatic language identification, and toxicity detection. Transcribe Medical is a specialized variant for HIPAA-compliant clinical documentation. The service integrates with the AWS ecosystem (S3, Lambda, Connect) for scalable speech processing pipelines.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using AWS Transcribe's cloud ASR architecture. |

