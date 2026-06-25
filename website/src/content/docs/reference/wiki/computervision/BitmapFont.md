---
title: "BitmapFont<T>"
description: "A simple 5x7 bitmap font for rendering text on images."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Visualization`

A simple 5x7 bitmap font for rendering text on images.

## For Beginners

This class provides a way to draw text on images
without requiring external font libraries. Each character is defined as a
5-wide by 7-tall grid of pixels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BitmapFont` | Creates a new bitmap font instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DrawChar(Tensor<>,Char,Int32,Int32,ValueTuple<Double,Double,Double>,Int32,Int32,Int32,Int32)` | Draws a single character at the specified position. |
| `DrawText(Tensor<>,String,Int32,Int32,ValueTuple<Double,Double,Double>,Int32)` | Draws text on an image at the specified position. |
| `DrawTextWithBackground(Tensor<>,String,Int32,Int32,ValueTuple<Double,Double,Double>,ValueTuple<Double,Double,Double>,Int32,Int32)` | Draws text with a background box. |
| `InitializeGlyphs` | Initializes the 5x7 font glyphs. |
| `MeasureWidth(String)` | Measures the width of a text string in pixels. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CharHeight` | Character height in pixels. |
| `CharSpacing` | Horizontal spacing between characters. |
| `CharWidth` | Character width in pixels. |
| `_glyphs` | Font glyphs stored as bit patterns. |

