# On-device parity set (v2 model)

26 recordings used to compare the desktop pipeline (`src/inference/predict_wav.py`) with the Android application
(`digital-stethoscope-mobile`, `src/test/parityTest.ts`).

- `golden.json`: per file the source dataset, patient / record ID, true label, number of segments and the Python
  probabilities of the v2 TFLite model (`scripts/make_golden_v2.py`).
- `rn_results.json`: the on-device run (Samsung Galaxy S22+) after the app preprocessing was aligned with training
  (26/26 class agreement); `rn_results_before_preproc_fix.json`: the run before (25/26).

The WAV files are **not** in the repository because they are clips of third-party datasets whose licenses do not all
allow redistribution (BUET: CC BY-NC-ND; the license of the Yaseen and external "heart-sounds-public" clips is unclear).
Rebuild them from the original datasets: the file names in `golden.json` end with the original file name, and
`src/inference/make_parity_set.py` (v1 set in `test_parity/`) shows how the segments were taken; then run
`scripts/make_golden_v2.py`. Sources: PhysioNet/CinC 2016, PASCAL Dataset-B, CirCor DigiScope 2022, BUET, Yaseen et al. 2018.
