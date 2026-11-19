# Landmark-Anchored and LLM-Driven Construction of Relative Locality Description Datasets

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B%20%2B%20PostGIS%203.x-336791.svg)](https://postgis.net/)
[![Code License](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/data%20license-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Records](https://img.shields.io/badge/records-532%2C285-orange.svg)](#dataset-at-a-glance)
[![Paper](https://img.shields.io/badge/paper-IJGIS%20(submitted)-lightgrey.svg)](#citation)

This repository contains the full construction pipeline for **AIGeoRel**, a large-scale bilingual
(Chinese–English) corpus of *relative locality descriptions* — natural-language locative expressions that
identify a target place by anchoring it to a nearby landmark ("往北京汽车压铸件厂南边看，嘉会湖地铁站A口大概在那儿940米远").

Unlike address-based geocoding corpora, every record here is paired with its **geometric ground truth**:
the target coordinate, the exact anchor-to-target distance, the azimuth, the discretized 8/16-sector
direction, the distance band, and one of six road-context topology classes. The corpus therefore works both
as training data for geocoding models and as a benchmark with objectively checkable spatial answers.

The pipeline covers seven stages: POI acquisition → cleaning → dual-dimensional landmark salience →
landmark selection under spatial-coverage control → multi-granularity spatial relation computation →
template/slot-filling generation with LLM refinement and geometric re-validation → bilingual export and
evaluation.

---

## Pipeline

```mermaid
flowchart TD
    A["Amap POI API<br/>GDCrawler.py"] --> B["Cleaning, dedup, type filtering<br/>POIDataPreprocessor"]
    B --> C["SIS — spatial salience<br/>edge centrality · exposure · isolation"]
    B --> D["CIS — cognitive salience<br/>knowledge · brand · uniqueness · category fame"]
    C --> E["LIS = σ(k·(0.2·SIS + 0.8·CIS − 0.5)), k = 8"]
    D --> E
    E --> F["Gridded quota selection (1 km grid)<br/>26,800 landmarks, priority P1–P4"]
    F --> G["Relation engine (PostGIS)<br/>distance · azimuth · 8/16-dir · 6 topologies"]
    G --> H["Multi-dimensional stratified sampling<br/>532,285 relation records"]
    H --> I["Template library + slot filling<br/>direction / distance granularity"]
    I --> J["LLM refinement (Qwen2.5)<br/>+ geometric re-validation, rule fallback"]
    J --> K["zh → en translation"]
    K --> L[("AIGeoRel CSV<br/>description · geometry")]
```

---

## Dataset at a glance

| Item | Value |
| --- | --- |
| Records | 532,285 aligned description–geometry pairs |
| Distinct anchor landmarks (`ref_name`) | 26,800 |
| Distinct target points | 204,564 (≈2.6 descriptions per target) |
| Spatial coverage | Beijing, all 16 districts (115.4413–117.4979 °E, 39.4650–40.9696 °N) |
| Languages | Chinese (`description`) + English (`description_en`) |
| Distance range | 0.08–1,000 m (candidate window capped at 1 km) |
| Direction bins | 8 and 16 sectors |
| Distance bands | 4 (`VeryClose` / `Close` / `Medium` / `Far`) |
| Topology classes | 6 (`Coincident`, `Abutting`, `Intersection`, `Opposite`, `SameSide`, `Isolated`) |
| Relation phrasing | 2 (`Explicit` / `Implicit`) |

The released data file, its field dictionary, per-class distributions and quick-start snippets are documented
in [`AIGeoRel_Release_Note.md`](../AIGeoRel_Release_Note.md) and attached to the
[v1.0.0 release](https://github.com/zccodehub/GeoDatasets/releases/tag/v1.0.0).

---

## Repository structure

```text
Geocoding_dataset/
├── README.md                     # This file
├── LICENSE                       # Apache License 2.0 (code); dataset is CC BY 4.0
├── NOTICE                        # Third-party attribution: Amap, OSM/ODbL, boundary data
├── requirements.txt              # Python dependencies (pip install -r)
├── GDCrawler.py                  # Amap (Gaode) POI 2.0 crawler: type-code sweep, paging, retry, ×lsx dump
├── main.py                       # Scratch script for WGS-84 → UTM 50N (EPSG:32650) sanity checks
├── config/
│   ├── template_library.yaml     # Sentence skeletons, grouped into scene/topology families (v3.0)
│   └── template_library_new.yaml # Extended library consumed by DescriptionGenerator
├── data/
│   ├── poi_types_all.xlsx        # Amap type-code dictionary used by the crawler
│   ├── poi_types_del.xlsx        # Type codes excluded from the landmark pool
│   ├── json/                     # Cognition lexicons: district prefixes, proper nouns, suffixes, facilities
│   └── *.csv                     # Intermediate POI score tables (git-ignored; regenerate locally)
├── src/
│   ├── dp/                       # Data preparation
│   │   ├── POIDataPreprocessor.py    # Dedup (ID + 1 m geodesic), address/coordinate cleaning, quality report
│   │   ├── Merge_files.py            # Merge per-district crawls
│   │   ├── Clean_type3_field.py      # Normalize third-level POI type strings
│   │   └── Diff_excel.py             # Row-level diff between two crawl snapshots
│   ├── landmark/                 # Dual-dimensional salience and landmark selection
│   │   ├── Calculate_edge_centrality.py  # Road-network betweenness/degree centrality by NetworkX
│   │   ├── Scaled_poi_centrality.py      # Percentile rescaling of centrality
│   │   ├── Scaled_poi_exposure.py        # Intersection / junction exposure normalization
│   │   ├── Calculate_poi_isolation.py    # Mean distance to the 3 nearest same-category POIs (cKDTree)
│   │   ├── Calculate_poi_cognition.py    # CIS from name morphology + lexicons (jieba, suffix/prefix rules)
│   │   ├── Calculate_poi_composite.py    # SIS aggregation, Sigmoid fusion → LIS
│   │   ├── Extract_poi_landmark.py       # Gridded quota Q_i, P1–P4 priority, coverage diagnostics
│   │   └── EDA_poi.py                    # Score distributions before selection
│   ├── relation/                 # Spatial relation computation
│   │   ├── Neatnet_roads.py              # Road-network topology cleaning (neatnet)
│   │   ├── Road_width_mapping.py         # OSM `highway` class → carriageway width (m)
│   │   ├── Spatial_relation_calculation.py  # Core PostGIS engine: distance, azimuth, 8/16-dir, 6 topologies
│   │   └── Spatial_relation_fractal.py   # Deviation angle + ambiguity level (EXACT/SLIGHT/MODERATE) backfill
│   ├── text/                     # Description generation
│   │   ├── TemplateGenerator.py      # LLM-assisted template-library expansion (design-time only)
│   │   ├── DirectionGenerator.py     # Direction phrase at 3 granularities (precise/vague/functional)
│   │   ├── DistanceGenerator.py      # Distance phrase at 4 granularities per band
│   │   ├── AmbiguityInjector.py      # Adverb pool selected by ambiguity level; granularity re-weighting
│   │   ├── SlotFiller.py             # Family lookup with hierarchical fallback → slot substitution
│   │   ├── DescriptionGenerator.py   # Runs the cascade over `spatial_relations` → writes `geo_desc`
│   │   ├── DescriptionOptimizer.py   # Qwen refinement + direction-consistency check + rule fallback
│   │   ├── DescriptionTranslator.py  # Batched zh → en translation (temperature 0.1)
│   │   └── DescriptionExport.py      # Streaming export to UTF-8-BOM CSV
│   ├── analysis/                 # Evaluation, one script per results section
│   │   ├── analyze_landmark_salience.py        # §5.1 univariate/bivariate salience distributions
│   │   ├── evaluate_composite_salience.py      # §5.2 fusion models, sensitivity, quadrant response
│   │   ├── separation_analysis.py              # §5.3 landmark vs. background separation (t, U, Cohen's d)
│   │   ├── evaluate_spatial_relation_results.py # §5.4 recomputation, entropy balance, expert sample, κ
│   │   ├── stats_5_5.py                        # §5.5 scale, language stats, BLEU/ROUGE, inverse geocoding
│   │   ├── linguistic_stats_5_5_2.py           # §5.5.2 TTR / Distinct-n / Self-BLEU / lexical entropy
│   │   └── gen_fig_5_6_1.py                    # §5.6 seven-dimension benchmark comparison figure
│   └── utils/
│       └── PercentileScaler.py   # Rank-based rescaling of positive values into [50, 100]
├── logs/                         # Runtime logs (git-ignored)
└── output/                       # Figures, CSV/JSON/XLSX results (git-ignored)
```

---

## Requirements

**Runtime.** Python ≥ 3.9, PostgreSQL ≥ 14 with PostGIS 3.x, and an LLM endpoint (the pipeline is written
against a local [Ollama](https://ollama.com/) server serving Qwen2.5; any OpenAI-compatible chat endpoint works
for template design).

**Python packages.** Everything is pinned to a working range in
[`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

Pinned floors matter in two places: `geopandas>=1.0` / `shapely>=2.0` (the selection script uses
`GeoSeries.union_all()` and the Shapely 2.x vectorised API) and `sqlalchemy>=1.4` (row access via
`Row._mapping`). `openai` is needed only by `TemplateGenerator.py`, at design time; the runtime pipeline
talks to Ollama over plain HTTP.

| Package | Used for |
| --- | --- |
| `pandas` `numpy` `scipy` `scikit-learn` | Tabular processing, statistics, k-NN |
| `geopandas` `shapely` `pyproj` | Geometry, projection, GeoPackage I/O |
| `networkx` `neatnet` | Road-network centrality and topology cleaning |
| `sqlalchemy` `psycopg2-binary` | PostGIS access (reads via SQLAlchemy, writes via psycopg2) |
| `sqlalchemy` + raw SQL / `ST_*` | All distance, azimuth and topology logic runs inside PostGIS |
| `PyYAML` `python-dotenv` `requests` | Template library, configuration, HTTP calls |
| `matplotlib` `seaborn` | Paper figures |
| `jieba` | Chinese tokenization (optional; falls back to character n-grams) |
| `openai` | Template-library expansion only, not required at runtime |

**Database schema.** The pipeline expects five PostGIS tables. `landmarks`, `road_centerlines` and
`road_intersects` are built once from the cleaned crawl; `spatial_relations` and `geo_desc` are created and
filled by the pipeline itself.

| Table | Key columns |
| --- | --- |
| `landmarks` | `poi_id`, `名称`, `地址`, `type_3`, `经度`, `纬度`, `geom`, `cognition_score`, `spatial_score`, `composite_score`, `priority` (0 = background, 1–4 = selected) |
| `road_centerlines` | `osm_id`, `name`, `highway`, `road_width`, `geom` |
| `road_intersects` | `name`, `geom` |
| `spatial_relations` | `landmark_id`, `target_id`, `exact_distance_m`, `dist_level`, `azimuth_deg`, `direction_8`, `direction_16`, `topology_type`, `combo_class`, `is_ambiguous`, `road_name`, `template_match_key`, `ambiguity_level`, … |
| `geo_desc` | `id`, `fid` (→ `spatial_relations.id`), `description_raw`, `description`, `description_en`, `ref_name`, `target_x`, `target_y` |

---

## Configuration

Create a `.env` file in the repository root. `.env` is git-ignored; never commit keys.

```dotenv
# Amap (Gaode) POI 2.0 API — required by GDCrawler.py
GD_API_KEY=your_amap_web_service_key

# PostGIS
DATABASE_URL=postgresql://user:password@localhost:5432/poi_db

# Local LLM (Ollama)
OLLAMA_URL=http://localhost:11434/api/generate
LLM_MODEL=qwen2.5:7b
LLM_EN_MODEL=qwen2.5:7b

# OpenAI-compatible endpoint — template-library expansion only
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

| Variable | Consumed by | Purpose |
| --- | --- | --- |
| `GD_API_KEY` | `GDCrawler.py` | Amap POI search |
| `DATABASE_URL` | all DB stages and analyses | PostGIS connection string |
| `OLLAMA_URL` | `DescriptionOptimizer.py`, `DescriptionTranslator.py` | `/api/generate` endpoint |
| `LLM_MODEL` | refiner (and translator, see note) | Chinese refiner model |
| `LLM_EN_MODEL` | `DescriptionTranslator.py` | Translation model (currently falls back to `LLM_MODEL`) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | `TemplateGenerator.py` | Template expansion |

---

## Usage

Each stage is an independent script rather than a single CLI, because the paper reports per-stage
intermediate statistics. **Run each stage from its own directory** — modules import siblings by bare name
(`from SlotFiller import SlotFiller`) and resolve paths as `../../config/...`.

### 1. Acquire POIs

```bash
python GDCrawler.py
```

Sweeps Amap type codes (`data/poi_types_all.xlsx`) for one administrative code (`AD_CODE`, default
Yanqing `110119`), 25 records per page with retry and 0.5 s throttling, then writes one workbook per district.
Repeat per district and merge with:

```bash
python src/dp/Merge_files.py
```

### 2. Clean and deduplicate

```bash
python src/dp/POIDataPreprocessor.py
```

Operates in batch: every workbook in `output/` is cleaned in turn and written back with a `_已清洗` suffix.
Per file it removes duplicate `POI ID`s, drops coordinates coinciding within 1 m (geodesic distance), filters
null / zero / out-of-range coordinates and null addresses, normalizes addresses, drops closed businesses,
parses the three-level category, removes meaningless names, and applies the excluded type-code list
(`data/poi_types_del.xlsx`). Every dropped record is logged with its reason, so the 428,354 → 285,413
reduction is fully auditable.

### 3. Score spatial salience (SIS)

```bash
python src/landmark/Calculate_edge_centrality.py     # road-network centrality  → centrality_scaled
python src/landmark/Scaled_poi_exposure.py           # junction exposure        → exposure_scaled
python src/landmark/Calculate_poi_isolation.py       # category isolation       → isolation_scaled
```

```python
SIS = 0.40 · exposure + 0.35 · centrality + 0.25 · isolation      # then min–max to [0, 1]
```

All three components are percentile-rescaled beforehand (`src/utils/PercentileScaler.py`), so each is
rank-based rather than raw-magnitude.

### 4. Score cognitive salience (CIS)

```bash
python src/landmark/Calculate_poi_cognition.py
```

Derives CIS from name morphology (proper nouns, functional suffixes, district prefixes) combined with
knowledge, brand, uniqueness and category-fame signals, weighted and penalized by `data/json/*.json` lexicons.
A `jieba`-based segmentation fallback keeps the script runnable without extra dictionaries.

### 5. Fuse into LIS and select landmarks

```bash
python src/landmark/Calculate_poi_composite.py   # LIS
python src/landmark/Extract_poi_landmark.py      # gridded quota + priority
```

```python
LIS = 1 / (1 + exp(-8 · (0.2 · SIS + 0.8 · CIS − 0.5)))
```

Landmark selection is coverage-controlled rather than globally top-k: POIs are binned into a 1 km grid, each
cell receives a quota `Q_i = max(1, ⌊0.2 · n_i^0.8⌋)`, and within each cell only POIs above the local
70th percentile of LIS are ranked by priority class:

| Priority | Condition | Meaning |
| --- | --- | --- |
| P1 | SIS ≥ P80 and CIS ≥ P80 | iconic in both dimensions |
| P2 | CIS ≥ P70 and SIS ≤ P60 | cognitively strong, spatially diffuse |
| P3 | SIS ≥ P70 and CIS ≤ P60 | spatially prominent, name-unremarkable |
| P4 | otherwise | fallback filler |
| 0 | not selected | background entity |

The script also reports 2 km buffer coverage, mean/max nearest-neighbour spacing, and the P1–P4 counts.

### 6. Compute spatial relations

```bash
python src/relation/Neatnet_roads.py            # clean road topology
python src/relation/Road_width_mapping.py       # OSM highway class → width
python src/relation/Spatial_relation_calculation.py
python src/relation/Spatial_relation_fractal.py # deviation angle + ambiguity level
```

The core engine runs entirely in PostGIS, in batches of 500 landmarks:

- candidate generation: all `priority = 0` POIs within 1,000 m of each landmark (`ST_DWithin`, `ST_Distance`, `ST_Azimuth`);
- distance bands: `< 50` → `VeryClose`, `< 200` → `Close`, `< 500` → `Medium`, `≤ 1000` → `Far`;
- direction: 8 sectors always; 16 sectors additionally for `Opposite`/`Abutting` under 200 m and for `Intersection`;
- topology, applied in priority order: `Coincident` (< 1 m) → `Abutting` (1–30 m, no road crossing the link)
  → `Intersection` (target within 60 m of a junction) → `Opposite` (both sides of one centerline, offset within
  ±10 m of carriageway width) → `SameSide` (same centerline, longitudinal offset ≤ 200 m) → `Isolated`;
- sampling: within each `(landmark, topology, distance band)` stratum, `max_keep` records are drawn at random,
  with `max_keep` set to 5 / 4 / 3 / 2 according to candidate density (≥ 1000 / ≥ 500 / ≥ 200 / else);
- `combo_class`: `Implicit` for `Coincident`/`Abutting`/`Intersection`, `Explicit` otherwise.

`Spatial_relation_fractal.py` then backfills `deviation_angle` (offset from the 8-sector centre) and
`ambiguity_level` (`EXACT` / `SLIGHT` / `MODERATE`), which drive adjective selection downstream.

### 7. Generate descriptions

```bash
cd src/text
python DescriptionGenerator.py    # template + slot filling      → geo_desc.description_raw
python DescriptionOptimizer.py    # LLM refinement + validation  → geo_desc.description
python DescriptionTranslator.py   # zh → en                      → geo_desc.description_en
python DescriptionExport.py       # CSV export                   → output/geo_desc.csv
```

Generation is a **cascade with a deterministic floor**:

1. **Template selection.** `SlotFiller` looks up `template_match_key`
   (`SCENE-B_Opposite_Close_East_EXACT`) in `config/template_library_new.yaml`, falling back from the exact key
   to scene+topology, then scene, then a default family.
2. **Slot filling.** `DirectionGenerator` renders the direction at three granularities
   (precise / vague / functional, weighted 0.20 / 0.50 / 0.30); `DistanceGenerator` renders the distance at four
   (precise / vague / ultra-vague / functional); `AmbiguityInjector` adds the adverb pool implied by
   `ambiguity_level` and re-weights granularity accordingly.
3. **LLM refinement.** Only records that need it are sent to Qwen — `Intersection` relations, `MODERATE`
   ambiguity, distances beyond 800 m, or long multi-clause sentences — in batches of 30 at temperature 0.7.
   The prompt forbids altering direction words, distances or place names, and requires de-duplicating repeated
   toponyms into "这条路 / 这儿 / 该处".
4. **Geometric re-validation.** `validate_spatial_consistency` re-checks the refined text against
   `direction_8`: if the expected cardinal root disappears and is not replaced by a functional phrase
   ("…方向", "…那边"), the record silently falls back to the rule-based text. Batch-level JSON parse failures
   degrade to per-record calls, and per-record failures degrade to the rule-based text.

The result is that no record depends on the LLM having behaved; the LLM can only improve style.

### 8. Evaluate

```bash
python src/analysis/analyze_landmark_salience.py         # §5.1
python src/analysis/evaluate_composite_salience.py       # §5.2
python src/analysis/separation_analysis.py               # §5.3
python src/analysis/evaluate_spatial_relation_results.py # §5.4
python src/analysis/stats_5_5.py                         # §5.5
python src/analysis/linguistic_stats_5_5_2.py            # §5.5.2
python src/analysis/gen_fig_5_6_1.py                     # §5.6 figure (no DB needed)
```

| Script | Output |
| --- | --- |
| `analyze_landmark_salience.py` | Univariate/bivariate salience figures, quadrant counts |
| `evaluate_composite_salience.py` | Fusion-model comparison, 25-run sensitivity grid, quadrant response |
| `separation_analysis.py` | Welch t, Mann–Whitney U, Cohen's d between landmark and background sets |
| `evaluate_spatial_relation_results.py` | `evaluation_results_5_2.json`, distribution workbook, expert re-labelling sample, Cohen's κ |
| `stats_5_5.py` | Scale/composition, BLEU-4 & ROUGE-L vs. template baseline, inverse-geocoding error |
| `linguistic_stats_5_5_2.py` | TTR, Distinct-1/2, Self-BLEU, lexical entropy (with toponym masking) |
| `gen_fig_5_6_1.py` | Seven-dimension benchmark heatmap + composite score bar chart |

Most analyses accept a `DATABASE_URL` override and write to `output/`; the JSON/CSV artefacts they produce
are the exact inputs behind the reported numbers, so any table in the paper can be regenerated independently.

---

## Output schema

`output/geo_desc.csv` (UTF-8 with BOM, so Excel reads Chinese correctly):

| Field | Type | Description |
| --- | --- | --- |
| `description` | string | Chinese relative locality description |
| `description_en` | string | Semantically parallel English version |
| `ref_name` | string | Anchor landmark name |
| `target_x`, `target_y` | float | Target longitude / latitude (GCJ-02 as crawled; converted to WGS-84 for distance and azimuth computation) |
| `exact_distance_m` | float | Anchor-to-target straight-line distance, metres |
| `dist_level` | enum | `VeryClose` / `Close` / `Medium` / `Far` |
| `azimuth_deg` | float | Bearing clockwise from north, (0, 360] |
| `direction_8`, `direction_16` | enum | Sector quantization of the azimuth |
| `topology_type` | enum | One of the six road-context classes |
| `combo_class` | enum | `Explicit` / `Implicit` |

---

## Reproducibility notes

- **Absolute paths.** Thirteen scripts still carry Windows paths from the research environment in their
  module-level constants (`Calculate_edge_centrality.py`, `Neatnet_roads.py`, `Extract_poi_landmark.py`,
  `Calculate_poi_cognition.py`, `EDA_poi.py`, `Scaled_poi_*.py`, `PercentileScaler.py`, `Clean_type3_field.py`,
  `gen_fig_5_6_1.py`, …). Edit the constants at the top of the file, or export the paths you need, before the
  first run.
- **GD crawler import.** `GDCrawler.py` calls `load_dotenv()` but does not import it; add
  `from dotenv import load_dotenv` before running.
- **Two CRS.** Distances and azimuths are computed in UTM 50N (`EPSG:32650`); published coordinates stay in
  GCJ-02 as returned by the Amap API. Convert to WGS-84 before comparing against OSM or GPS traces.
- **`data/*.csv` and `output/` are git-ignored.** Intermediate score tables and result artefacts are
  regenerated by the scripts above; only the released corpus is distributed, via GitHub Releases.
- **Rate limits.** The Amap free tier caps daily quota per key; the crawler throttles to ~2 requests/s and
  retries three times, but a full 16-district sweep needs several days or a paid quota.

---

## Citation

```bibtex
@dataset{aigeorel2026,
  title     = {AIGeoRel: A Landmark-Anchored Relative Spatial Description Corpus of Beijing},
  author    = {Zhuang, Can and Cui, Yi},
  year      = {2026},
  version   = {v1.0.0},
  publisher = {GitHub},
  url       = {https://github.com/zccodehub/GeoDatasets/releases/tag/v1.0.0}
}
```

If you use the construction pipeline, please also cite the accompanying manuscript:

> Zhuang, C., & Cui, Y. (2026). Landmark-anchored and LLM-driven construction of relative locality description
> datasets. *International Journal of Geographical Information Science* (submitted).

---

## License

Copyright 2026 Can Zhuang and Yi Cui.

**Code** in this repository is released under the [Apache License 2.0](LICENSE). You may use, modify and
redistribute it, including commercially, provided that you retain the copyright and attribution notices,
state any significant changes, and include a copy of the license. The license also grants an express patent
license from contributors. The software is provided on an "AS IS" basis, without warranties or conditions of
any kind.

**Dataset** (AIGeoRel, distributed through GitHub Releases) is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — share and adapt for any purpose, including
commercially, with attribution. Please cite the dataset as shown above.

The two licenses are deliberately different: Apache 2.0 fits reusable software and carries an explicit patent
grant, whereas CC BY 4.0 is the norm for open data and requires attribution rather than patent terms.

**Third-party material** is not covered by either license and remains subject to its own terms: POI records
are derived from the Amap (Gaode) Open Platform and follow its terms of service, OpenStreetMap road data
follow ODbL, and administrative boundaries follow the terms of their originating provider. If you redistribute
a derived dataset, check those terms first. The full attribution list is in [`NOTICE`](NOTICE).

If you build on this work, a citation (see above) is the expected form of attribution for the data; for the
code, retaining the `LICENSE` and `NOTICE` files satisfies Apache 2.0.

---

## Contact

Can Zhuang — School of Civil Engineering and Geomatics, Shandong University of Technology, Zibo, China.
Issues and pull requests are welcome; for data-related questions please open an issue rather than emailing,
so that answers stay searchable.

---

## 中文说明

本仓库是 **AIGeoRel v1.0.0** 数据集（532,285 条中英双语相对位置描述，锚定 26,800 个地标，覆盖北京 16 个行政区）的**完整构建代码**，按"数据采集 → 清洗 → 双维显著性 → 地标选取 → 空间关系计算 → 文本生成与校验 → 双语导出与评测"七个阶段组织。

- **环境**：Python 3.9+、PostgreSQL 14+ 与 PostGIS 3.x、本地 Ollama（Qwen2.5）。依赖已固定版本区间，直接 `pip install -r requirements.txt`；配置写入仓库根目录 `.env`。
- **运行方式**：各阶段为独立脚本，需在脚本所在目录下执行（`src/text/` 内的模块以裸名互相导入，并以 `../../` 解析配置与日志路径）。
- **注意**：部分脚本顶部仍保留研究环境的 Windows 绝对路径，首次运行前请改为本地路径；`data/*.csv` 与 `output/` 未纳入版本控制，需由脚本自行生成。
- **数据获取**：成品数据集通过 [Releases](https://github.com/zccodehub/GeoDatasets/releases/tag/v1.0.0) 发布，字段字典与分布统计见 `AIGeoRel_Release_Note.md`。
- **许可**：代码采用 [Apache License 2.0](LICENSE)（可商用、可修改、需保留版权与许可声明，含专利授权）；数据集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)（可商用、需署名引用）。高德 POI 与 OSM 路网数据另受各自服务条款约束。
