---
title: "WeightLoader"
description: "Loads pre-trained model weights from various file formats."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Weights`

Loads pre-trained model weights from various file formats.

## For Beginners

This class reads saved neural network weights from files.
Deep learning models are trained on massive datasets, and the learned parameters
(weights) can be saved and reloaded. This allows you to use pre-trained models
without training from scratch.

## How It Works

Supported formats:

- PyTorch (.pt, .pth) - Python pickle with tensor data
- SafeTensors (.safetensors) - Safe tensor format
- NumPy (.npy, .npz) - NumPy array format
- ONNX (.onnx) - Open Neural Network Exchange format

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightLoader` | Creates a new weight loader. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BFloat16ToFloat(Byte[],Int32)` | Converts bfloat16 to single-precision (float32). |
| `GetDTypeSize(String)` | Gets the byte size of a data type. |
| `HalfToFloat(Byte[],Int32)` | Converts IEEE 754 half-precision (float16) to single-precision (float32). |
| `LoadBinaryWeights(String)` | Loads weights from raw binary format with shape metadata. |
| `LoadNpyTensor(Stream)` | Loads tensor from NumPy .npy format. |
| `LoadNumPyArchive(String)` | Loads multiple NumPy arrays from .npz archive. |
| `LoadNumPySingle(String)` | Loads a single NumPy array from .npy file. |
| `LoadPyTorchPickleFormat(Stream,BinaryReader)` | Loads PyTorch weights from legacy pickle format. |
| `LoadPyTorchWeights(String)` | Loads weights from a PyTorch .pt or .pth file. |
| `LoadPyTorchZipFormat(String)` | Loads PyTorch weights from modern ZIP-based format. |
| `LoadSafeTensors(String)` | Loads weights from SafeTensors format. |
| `LoadWeights(String)` | Loads weights from a file and returns them as a dictionary. |
| `ParseNpyHeader(String)` | Parses NumPy .npy header to extract shape and dtype. |
| `ParsePickleMetadata(Stream)` | Parses pickle metadata to extract tensor information. |
| `ParseSafeTensorsHeader(String)` | Parses SafeTensors header JSON. |
| `ParseTensorData(Byte[],Int32[],String)` | Parses raw tensor data bytes into a Tensor. |

