# Exploratory findings — CHI 2027

This folder is an **add-on** for the `historical-ai-interfaces` project. It does **not** duplicate the study data already present in the parent project.

## Folder layout

```text
historical-ai-interfaces/
├── data/
│   ├── study_data.csv
│   └── coded-messages-three-models.csv
├── analysis_output/
│   ├── analysis_merged_interaction_level.csv
│   └── knowledge_items.csv
└── exploratory_findings/
    ├── exploratory_findings_integrated.ipynb
    ├── exploratory_findings_integrated.py
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   ├── exposure_quiz_items_JOINED.csv
    │   └── INPUT_FILES_SHA256.json
    └── results/
        ├── 00_summary.json
        ├── *.csv
        └── *.png
```

**All four shared inputs are read from their existing project locations.** `exposure_quiz_items_JOINED.csv` is the only unique input retained under `exploratory_findings/data/`. This preserves the audit classifications used in the prior analyses.

## Run

From the **project root**:

```bash
python -m pip install -r exploratory_findings/requirements.txt
python exploratory_findings/exploratory_findings_integrated.py
```

For the notebook, open `exploratory_findings/exploratory_findings_integrated.ipynb` in JupyterLab and select **Run All Cells**. The notebook is also runnable with its working directory set to the project root. Alternatively, set `EXPLORATORY_FINDINGS_DIR` to the full path of the add-on folder.

Results and plots are always written to `exploratory_findings/results/`, not into `analysis_output/`.

## Analyses

The notebook includes the primary paired experience contrasts; prior preference, voice-input and emotional-tone subgroups; order, country, AI-experience, and collection-period checks; scale-item consistency; same-score and joint-high-experience summaries; equivalence sensitivity; exposure–accuracy GEE; individual knowledge-item tests; and multiple-testing adjustments.

All exploratory comparisons are reported, not only significant findings. The notebook distinguishes family-wise Holm adjustment from a more conservative Holm adjustment across the notebook's exploratory paired tests. Since hypotheses were developed post hoc, adjusted significance is not equivalent to preregistration or independent replication.

## Input-data coverage and limitations

- **146** retained participants, **292** original study interactions, and **1,460** knowledge-question responses.
- The deduplicated exposure audit contains **289** interaction-level records (**1,445** question items), allowing **143** complete participant pairs in audit-dependent paired analyses. Original paired experience and knowledge analyses use all **146** participants.
- The export does not contain a separate audit record for every study interaction. The analysis **does not fabricate** missing classifications or automatically treat repeated messages as missing participants.
- There is no pre-interaction knowledge test. The outcome is **immediate factual-recall performance**, not demonstrated learning gain.
- LLM-coded exposure and emotion labels require human validation. Subgroups defined by voice or emotions are post-treatment; the audiovisual interface components cannot be causally isolated.
- Equivalence bounds were chosen post hoc, and multiple exploratory tests were considered. See the notebook for all test families and statistical assumptions.

## Reproducibility

`data/INPUT_FILES_SHA256.json` records source file paths and SHA-256 hashes as they appeared in the uploaded project. It is a provenance record, **not** a restriction preventing you from updating the parent files. The analysis checks input existence and expected row counts before computing results.

## Jupyter working-directory permissions

In some Jupyter sessions, `Path.cwd()` raises `PermissionError` (for example, if the kernel was launched in a folder that has since been moved or access-restricted). The notebook now handles that exception and tries the folder supplied by `EXPLORATORY_FINDINGS_DIR`, its own script path (when available), `PWD` and other absolute path hints first.

If auto-discovery still fails, add the following line immediately after `import os` in the first notebook code cell, substituting your real absolute path:

```python
os.environ['EXPLORATORY_FINDINGS_DIR'] = '/absolute/path/to/historical-ai-interfaces/exploratory_findings'
```

Then restart the kernel and use **Run All Cells**. Alternatively, launch JupyterLab from a readable project directory. The command-line Python script also supports the environment variable.

## Fix for `PermissionError` inside `pd.read_csv`

If a traceback mentions `pandas.io.common._get_binary_io_classes`,
`import_optional_dependency("zstandard")`, and Python `importlib`, then the failure
is due to pandas checking an optional dependency while an inaccessible search
path is present. It is **not** evidence that the CSV contains invalid data.

The notebook and CLI script now open the CSV as a UTF-8 text stream and pass that
stream to pandas. This bypasses the optional dependency probe without monkeypatching
pandas, changing system permissions, or duplicating the study data. They also
switch to a known accessible working directory after resolving the project location.

Replace the old `exploratory_findings` folder with this version. Restart the Jupyter
kernel and run all cells. If you moved or renamed the project, set
`EXPLORATORY_FINDINGS_DIR` to the full path of this folder in the first cell.

If an error now points **directly to `Path(path).open('rb')`**, the operating system
is actually denying access to the study file; check permissions for that specific
file and its parent directories.
