# ComPipeline
Event identification for the ComPair experiment

Reads a MEGAlib `.sim` or `.sim.gz` simulation file, classifies each event through a two-layer cascade pipeline, and writes the results to a `.etp` file.

The first layer separates signal photon events from background (ACD veto, optionally backed by a Random Forest or PCA lookup table). The second layer classifies surviving events as Compton (CO), Pair Production (PA), or Photoelectric (PH) using PointNet.

---

## Requirements

- MEGAlib (with `$MEGALIB` set)
- PyTorch
- scikit-learn
- skops
- tqdm

---

## Basic usage

```bash
python3 EventType.py -i <input> -o <output_dir> -g <geometry> -m <model>
```

By default, background rejection uses only the ACD veto (fastest). To also run the Random Forest classifier, add `--disable-onlyacd`.

---

## Options

### `-i / --input`
Path to a `.sim` or `.sim.gz` file, or a directory containing multiple such files. If a directory is given, all `.sim` and `.sim.gz` files in it will be processed.

Default: `./mini_test.sim.gz`

---

### `-o / --output-dir`
Directory where the output `.etp` files will be saved. The output filename matches the input (`mini_test.sim` → `mini_test.etp`). The directory is created automatically if it doesn't exist.

Default: `./output_etp`

---

### `-g / --geometry`
Path to the MEGAlib geometry setup file (`.geo.setup`). Required for reading the simulation correctly.

Default: `../../simuComPair/Geometry/ComPair_23/ComPair23.geo.setup`

---

### `-m / --model`
Path to the PointNet model weights file (`.pth`). This model handles the Compton vs Pair Production classification in the second layer.

Default: `./PointNetModels/test_torch_model_params_final_26-06.pth`

---

### `--disable-onlyacd`
By default the background rejection uses only the ACD veto: any event with at least one hit in the ACD (detector type 4) is tagged as `MU` and discarded. Passing this flag enables the second-stage background classifier (Random Forest or PCA lookup table) on events that survive the ACD veto.

---

### `-rf / --random-forest`
Path to the Random Forest model file (`.skops`). Only used when `--disable-onlyacd` is set. If both `-rf` and `-pca` are provided, `-rf` takes priority.

Default: `./RandomForest/vega_model.skops`

---

### `-pca / --pca`
Path to the directory `./pca_files` containing the PCA lookup table JSON files. Used as an alternative to the Random Forest when `--disable-onlyacd` is set and `-rf` is not provided.

Default: None 

---

### `--debug`
Adds the true MC process label (`MC` field) to each event in the output `.etp` file, read from the simulation's interaction records. Useful for comparing classifier output against ground truth.

---

## Output format

Each event in the `.etp` file looks like this:

```
SE
ID <event_id>
ET <event_type>
TP <probability>
```

In debug mode, an extra `MC <process>` line is added between `ID` and `ET`.

Event types written: `CO` (Compton), `PA` (Pair Production), `PH` (Photoelectric), `MU` (background/muon), `UN` (unidentifiable).

---

## Example

ACD-only veto (default, fastest):
```bash
python3 EventType.py -i ./mini_test.sim -o . -g ..../Geometry/ComPair_23/ComPair23.geo.setup -m PointNetModels/test_torch_model_params_final_26-06.pth
```

With Random Forest background classifier:
```bash
python3 EventType.py -i ./mini_test.sim -o . -g ..../Geometry/ComPair_23/ComPair23.geo.setup -m PointNetModels/test_torch_model_params_final_26-06.pth --disable-onlyacd -rf RandomForest/vega_model.skops
```

With debug output:
```bash
python3 EventType.py -i ./mini_test.sim -o . -g ..../Geometry/ComPair_23/ComPair23.geo.setup -m PointNetModels/test_torch_model_params_final_26-06.pth --debug
```
