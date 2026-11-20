# data/

Put the downloaded archives here **unextracted** (nothing in this folder except this README is committed).
`python scripts/prepare_data.py` finds them by name, checks SHA-256 where a reference exists, and writes
`data/processed/<dataset>/{images,masks,fov}/` plus `data/processed/MANIFEST.json`.

| File (as downloaded)                                              | Dataset   | Source |
|-------------------------------------------------------------------|-----------|--------|
| `datasets.zip` (contains `training.zip`, `test.zip`)              | DRIVE     | https://drive.grand-challenge.org/ (official; test labels are not included) |
| `drive_paddleseg_mirror.zip`                                      | DRIVE test labels | https://bj.bcebos.com/paddleseg/dataset/drive/drive.zip (third-party mirror, SHA-256 pinned) |
| `all-images.zip` (or `stare-images.tar`)                          | STARE images | https://cecas.clemson.edu/~ahoover/stare/ |
| `labels-ah.tar`, `labels-vk.tar`                                  | STARE labels | https://cecas.clemson.edu/~ahoover/stare/probing/ |
| *(downloaded by the script)*                                      | CHASE_DB1 | Hugging Face `MedOtter/CHASE_DB1`, pinned revision (Kingston server offline) |
| `all.zip`                                                         | HRF       | https://www5.cs.fau.de/research/data/fundus-images/ |
| `FIVES A Fundus Image Dataset for AI-based Vessel Segmentation.rar` | FIVES   | https://doi.org/10.6084/m9.figshare.19688169 |

Generated sub-folders: `extracted/` (temporary), `processed/` (standardised PNGs), `cache/` (scale-normalised arrays), `hf_cache/`.
