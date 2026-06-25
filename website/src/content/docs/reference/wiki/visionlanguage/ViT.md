---
title: "ViT<T>"
description: "Vision Transformer (ViT) that splits images into patches and processes them with a standard Transformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Encoders`

Vision Transformer (ViT) that splits images into patches and processes them with a standard Transformer.

## For Beginners

ViT (Vision Transformer) applies the Transformer architecture
— originally designed for text — directly to images. It splits an image into a grid of
fixed-size patches (e.g., 16x16 pixels), treats each patch like a "word", and processes
the sequence of patches through standard Transformer encoder layers. A special [CLS]
token collects information from all patches for classification. Default values follow
the original paper settings.

## How It Works

ViT (Dosovitskiy et al., ICLR 2021) demonstrated that a pure Transformer applied directly to
sequences of image patches can perform very well on image classification. An image is split into
fixed-size patches (e.g., 16x16), each linearly embedded and processed by Transformer encoder
layers. A learnable [CLS] token aggregates information for the final representation.

**References:**

- Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021)

