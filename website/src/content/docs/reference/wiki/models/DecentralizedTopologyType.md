---
title: "DecentralizedTopologyType"
description: "Specifies the decentralized topology for peer-to-peer federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the decentralized topology for peer-to-peer federated learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `DFedAvgM` | DFedAvgM — decentralized averaging with momentum for faster convergence. |
| `DFedBCA` | DFedBCA — block coordinate ascent with partial model sharing per round. |
| `DeTAG` | DeTAG — gradient tracking for exact convergence in decentralized non-convex optimization. |
| `Gossip` | Gossip — randomized peer selection each round. |
| `RingAllReduce` | Ring AllReduce — bandwidth-optimal ring-based averaging. |
| `SegmentedGossip` | Segmented gossip — exchange only model segments per round for bandwidth efficiency. |
| `TimeVarying` | Time-varying topology — dynamic graph that changes each round for better mixing. |

