# ComfyUI Image Analysis Tools — Handoff

**Last updated:** 2026-10-13
**Repository:** `ThatGlennD/ComfyUI-Image-Analysis-Tools`
**Working branch:** `security/ghostaction-hardening`
**Base commit:** `5a26d6eb7c828c44b40e29a71bfc0104361e38a6`

## GitHub Actions supply-chain hardening

`.github/workflows/publish_action.yml` now declares `contents: read`, pins
`actions/checkout` to a full commit SHA, and pins
`Comfy-Org/publish-node-action` to the full SHA currently referenced by its
`main` branch. The workflow trigger and publishing secret input are unchanged.
The pinned composite action itself currently refers to nested actions by version
tags and installs `comfy-cli` without a version pin; those transitive dependencies
remain a residual supply-chain risk. This audit did not rotate the registry token.

Changes are local and have not been pushed. No Actions run has occurred.
