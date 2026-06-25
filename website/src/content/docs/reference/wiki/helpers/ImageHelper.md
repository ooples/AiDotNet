---
title: "ImageHelper<T>"
description: "Helper class for loading and saving images as tensors."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Helper class for loading and saving images as tensors.

## For Beginners

This class converts image files into tensors for neural networks.
Images are loaded as [channels, height, width] or [batch, channels, height, width] tensors.

## How It Works

Supports common image formats without external dependencies:

- BMP: Windows Bitmap format (uncompressed)
- PPM/PGM: Portable Pixmap/Graymap (simple text or binary)
- RAW: Raw pixel data with specified dimensions

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadBmp(String,Boolean)` | Loads a BMP (Windows Bitmap) image file. |
| `LoadImage(String,Boolean)` | Loads an image from a file path and returns it as a tensor. |
| `LoadImageWithImageSharp(String,Boolean)` | Loads an image using SixLabors.ImageSharp for formats not natively supported (PNG, JPEG, GIF, TIFF, WebP, etc.). |
| `LoadPgm(String,Boolean)` | Loads a PGM (Portable Graymap) image file. |
| `LoadPpm(String,Boolean)` | Loads a PPM (Portable Pixmap) image file. |
| `LoadRaw(String,Int32,Int32,Int32,Int32,Boolean)` | Loads raw pixel data from a file. |
| `ReadPnmToken(BinaryReader)` | Reads a token from a PNM file (skipping comments and whitespace). |
| `SaveBmp(Tensor<>,String,Boolean)` | Saves a tensor as a BMP image file. |
| `SavePpm(Tensor<>,String,Boolean,Boolean)` | Saves a tensor as a PPM image file. |

