# ACC-03 Pre-Edit Baseline & Artifact Manifest

**Timestamp**: 2026-10-01T10:43:00+07:00  
**Task ID**: `task_90b6735ee553`  
**Dispatch ID**: `ctx_88794eec8d79`  

---

## 1. Production Dataset and Model Hashes

These baseline artifacts must remain completely untouched throughout ACC-03:

| Name | File Path | Expected SHA-256 Checksum |
| :--- | :--- | :--- |
| `custom` | `data/custom_airpad.npz` | `498699e1f28a78031193476fbafff2a2cd307bb9d82a8baee6a57f3097019169` |
| `hagrid` | `data/hagrid_airpad.npz` | `ec38045d608aa6d30386796e4e0586ff619468d4abfd755619835bb8fa61c081` |
| `combined` | `data/combined_airpad.npz` | `c0c6d4e26c26f8a303077c6de971a54c2d4085444f9479cef9e021986cc9e8bf` |
| `model` | `models/airpad_svm.json` | `4cc3c55f5e4b571c5462de31afd6d5c4db64c803466d306cdefab34b31d7b977` |

---

## 2. Pre-Edit Source Code Backup

| Name | Source Path | Backup Path | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `pre_edit_collector` | `tools/collect_data.py` | `reports/acc03/before/tools/collect_data.py` | `1a1c43225045a979957aa65e183fc9f9f8b52d88924b7f55e75d776d0daeda17` |

Checksums recorded in `reports/acc03/before/manifest.sha256`.
