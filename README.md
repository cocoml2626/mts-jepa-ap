# MTS-JEPA: Multi-Resolution JEPA for Time-Series Anomaly Prediction

**Authors:** Anonymous for review  
**Contact:** anonymized for review

**Summary.**  
A JEPA-based pipeline for multi-scale time-series representation learning and anomaly prediction.

---

## Quick Start (Reviewer-Focused)

**Requirements**
- Python ≥ 3.9
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

**Representative run**

```bash
# Train MTS--JEPA model 
python train.py

# Evaluate downstream anomaly prediction 
python eval.py --use_probs
```



---

## Data

A sample PSM dataset is included in `data/Source Data/PSM/` and is ready to run.

For all datasets, experiments use publicly available benchmarks. Raw data should be obtained from the official sources below and converted into patch-level inputs matching the expected format.

**Official sources**

* **MSL & SMAP (NASA):** [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
* **SWaT:** [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
* **PSM:** [https://github.com/eBay/Pyraformer](https://github.com/eBay/Pyraformer)

**Expected input format**

* `x_patches`: shape `(N, P, L, V)`
* `y_label`: shape `(N,)` or `(N, P)`

**Data Setup**

The included PSM sample dataset is in `data/Source Data/PSM/` containing `train_norm.npz` and `test_norm.npz`. To use it, either:
- Update `DATA_DIR` filepath in `train.py` and `eval.py` to `"data/Source Data/PSM/"`
- Or copy files to project root: `cp data/Source\ Data/PSM/*.npz ./`

For other datasets, use `data/preprocess.py` to convert raw time series data into normalized patch-level files.

All experiments assume `P = 5` and `L = 20` (see constants in `train.py`).

---

## Repository Structure

* `train.py` — main training entrypoint (MTS-JEPA)
* `eval.py` — evaluation and inference
* `engine/` — training logic and losses
* `models/` — encoder, decoder, predictor, quantizer
* `data/` — dataset wrappers and utilities
* `downstream/` — downstream anomaly prediction


---

## License & Citation

Released under the MIT License.

```bibtex
@inproceedings{anonymous,
  title={Concise Title of the Contribution},
  author={Anonymous}
}
```
