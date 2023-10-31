---
name: Bug report
about: Create a report to help us improve
title: "[BUG]: "
labels: bug
assignees: ooples
---

body:
- type: textarea
  id: repro
  attributes:
    label: Reproduction steps
    description: "How do you trigger this bug? Please walk us through it step by step."
    value: |
      1.
      2.
      3.
      ...
    render: bash
  validations:
    required: true
