# Bonn University EEG Dataset (Mirror)

## Description

This is a **redistribution mirror** of the Bonn University EEG dataset, originally published by Andrzejak et al. (2001). The dataset is provided here for convenience of reproducibility in computational neuroscience and signal processing research.

**This is not an original dataset.** All credit belongs to the original authors.

## Original Source

**Original download page:**
http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3

**Direct download links (University of Bonn server):**
- Set Z: http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/Z.zip
- Set O: http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/O.zip
- Set N: http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/N.zip
- Set F: http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/F.zip
- Set S: http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/S.zip

**Alternative source (UPF Nonlinear Time Series Analysis group):**
https://www.upf.edu/web/ntsa/downloads

## Citation

If you use this dataset, please cite the **original publication**:

> Andrzejak, R.G., Lehnertz, K., Mormann, F., Rieke, C., David, P., & Elger, C.E. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. *Physical Review E*, 64(6), 061907. https://doi.org/10.1103/PhysRevE.64.061907

## Dataset Description

- **500 single-channel EEG recordings** (100 per class)
- **Sampling rate:** 173.61 Hz
- **Duration:** 23.6 seconds per recording (4097 samples)
- **Bandpass filter:** 0.53–40 Hz (12 dB/octave)
- **Format:** Plain text, one value per line (amplitude in microvolts)

## Classes

| Directory | Set | Description | Recording Type |
|-----------|-----|-------------|----------------|
| `Set_Z/` | Z | Healthy volunteers, eyes open | Surface (scalp) EEG |
| `Set_O/` | O | Healthy volunteers, eyes closed | Surface (scalp) EEG |
| `Set_N/` | N | Epilepsy patients, interictal, epileptogenic zone | Intracranial (depth) EEG |
| `Set_F/` | F | Epilepsy patients, interictal, hippocampal formation (contralateral) | Intracranial (depth) EEG |
| `Set_S/` | S | Epilepsy patients, ictal (seizure activity) | Intracranial (depth) EEG |

## File Naming

Each file is named `{CLASS}{NUMBER}.txt` (e.g., `Z001.txt`, `S100.txt`), where:
- `CLASS` = one of Z, O, N, F, S
- `NUMBER` = recording index (001–100)

## Important Notes

- Sets Z and O are **scalp** EEG from healthy volunteers (5 each, 100 recordings per set)
- Sets N, F, and S are **intracranial** EEG from epilepsy patients (5 each)
- Sets N and F are seizure-free (interictal); Set S contains seizure activity (ictal)
- The mixing of scalp and intracranial recordings across classes is a known limitation
- No per-subject labels are provided; recordings within a set may come from the same subject

## License

The original dataset is freely available for academic and research purposes from the University of Bonn. This mirror preserves the original data without modification. Please refer to the original source for licensing terms.

## Mirror Metadata

- **Mirror created:** 2026-03-01
- **Mirror author:** Kirill Vasilchenko, Holon Institute of Technology (HIT)
- **Purpose:** Convenience redistribution for reproducibility
- **Modifications:** File extensions normalized to lowercase `.txt`; files organized into subdirectories by class. No data values were altered.
