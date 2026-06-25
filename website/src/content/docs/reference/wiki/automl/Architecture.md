---
title: "Architecture<T>"
description: "Represents a neural network architecture discovered through NAS."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Represents a neural network architecture discovered through NAS.

## For Beginners

Think of this as a blueprint for a neural network. NAS algorithms explore
many possible blueprints and find the best one for your task. This class stores that blueprint
so you can:

- Save it to disk for later use
- Share it with others
- Load it to recreate the same network structure

## How It Works

This class captures the structure of a neural network discovered through Neural Architecture Search (NAS).
It includes the operations connecting nodes and optional channel information for cost estimation.

## Properties

| Property | Summary |
|:-----|:--------|
| `NodeChannels` | Optional per-node channel counts (typically output channels) used for cost estimation. |
| `NodeCount` | Number of nodes in the architecture |
| `Operations` | Operations in the architecture: (to_node, from_node, operation) |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddOperation(Int32,Int32,String)` | Adds an operation to the architecture |
| `FromBytes(Byte[])` | Deserializes an architecture from a binary byte array. |
| `FromJson(String)` | Deserializes an architecture from a JSON string. |
| `FromSerializable(ArchitectureDto)` | Creates an architecture from a serializable DTO. |
| `GetDescription` | Gets a description of the architecture. |
| `LoadFromFile(String)` | Loads an architecture from a JSON file. |
| `SaveToFile(String,Boolean)` | Saves the architecture to a JSON file. |
| `ToBytes` | Serializes the architecture to a binary byte array. |
| `ToJson(Boolean)` | Serializes the architecture to a JSON string. |
| `ToSerializable` | Converts the architecture to a serializable DTO. |

