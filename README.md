# Landmark-Anchored and LLM-Driven Construction of Relative Locality Description Datasets

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B%20%2B%20PostGIS%203.x-336791.svg)
![Code License](https://img.shields.io/badge/code%20license-Apache--2.0-blue.svg)
![Data License](https://img.shields.io/badge/data%20license-CC%20BY%204.0-green.svg)
![Records](https://img.shields.io/badge/records-532%2C285-orange.svg)

This repository contains the construction pipeline for **AIGeoRel**, a large-scale bilingual  
(Chinese–English) corpus of *relative locality descriptions* — natural-language locative expressions that  
identify a target place by anchoring it to a nearby landmark ("往北京汽车压铸件厂南边看，嘉会湖地铁站A口大概在那儿940米远").



## Dataset at a glance

| Item                                   | Value                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| Records                                | 532,285 aligned description–geometry pairs                                       |
| Distinct anchor landmarks (`ref_name`) | 26,800                                                                           |
| Distinct target points                 | 204,564 (≈2.6 descriptions per target)                                           |
| Spatial coverage                       | Beijing, all 16 districts (115.4413–117.4979 °E, 39.4650–40.9696 °N)             |
| Languages                              | Chinese (`description`) + English (`description_en`)                             |
| Distance range                         | 0.08–1,000 m (candidate window capped at 1 km)                                   |
| Direction bins                         | 8 and 16 sectors                                                                 |
| Distance bands                         | 4 (`VeryClose` / `Close` / `Medium` / `Far`)                                     |
| Topology classes                       | 6 (`Coincident`, `Abutting`, `Intersection`, `Opposite`, `SameSide`, `Isolated`) |
| Relation phrasing                      | 2 (`Explicit` / `Implicit`)                                                      |

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

| Package                                 | Used for                                                         |
| --------------------------------------- | ---------------------------------------------------------------- |
| `pandas` `numpy` `scipy` `scikit-learn` | Tabular processing, statistics, k-NN                             |
| `geopandas` `shapely` `pyproj`          | Geometry, projection, GeoPackage I/O                             |
| `networkx` `neatnet`                    | Road-network centrality and topology cleaning                    |
| `sqlalchemy` `psycopg2-binary`          | PostGIS access (reads via SQLAlchemy, writes via psycopg2)       |
| `sqlalchemy` + raw SQL / `ST_*`         | All distance, azimuth and topology logic runs inside PostGIS     |
| `PyYAML` `python-dotenv` `requests`     | Template library, configuration, HTTP calls                      |
| `matplotlib` `seaborn`                  | Paper figures                                                    |
| `jieba`                                 | Chinese tokenization (optional; falls back to character n-grams) |
| `openai`                                | Template-library expansion only, not required at runtime         |

**Database schema.** The pipeline expects five PostGIS tables. `landmarks`, `road_centerlines` and  
`road_intersects` are built once from the cleaned crawl; `spatial_relations` and `geo_desc` are created and  
filled by the pipeline itself.

| Table               | Key columns                                                                                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `landmarks`         | `poi_id`, `名称`, `地址`, `type_3`, `经度`, `纬度`, `geom`, `cognition_score`, `spatial_score`, `composite_score`, `priority` (0 = background, 1–4 = selected)                                                              |
| `road_centerlines`  | `osm_id`, `name`, `highway`, `road_width`, `geom`                                                                                                                                                                   |
| `road_intersects`   | `name`, `geom`                                                                                                                                                                                                      |
| `spatial_relations` | `landmark_id`, `target_id`, `exact_distance_m`, `dist_level`, `azimuth_deg`, `direction_8`, `direction_16`, `topology_type`, `combo_class`, `is_ambiguous`, `road_name`, `template_match_key`, `ambiguity_level`, … |
| `geo_desc`          | `id`, `fid` (→ `spatial_relations.id`), `description_raw`, `description`, `description_en`, `ref_name`, `target_x`, `target_y`                                                                                      |


## Usage

Each stage is an independent script rather than a single CLI, because the paper reports per-stage  
intermediate statistics. **Run each stage from its own directory** — modules import siblings by bare name  
(`from SlotFiller import SlotFiller`) and resolve paths as `../../config/...`.

---

## Output schema

`output/geo_desc.csv` (UTF-8 with BOM, so Excel reads Chinese correctly):

| Field                         | Type   | Description                                                                                               |
| ----------------------------- | ------ | --------------------------------------------------------------------------------------------------------- |
| `description`                 | string | Chinese relative locality description                                                                     |
| `description_en`              | string | Semantically parallel English version                                                                     |
| `ref_name`                    | string | Anchor landmark name                                                                                      |
| `target_x`, `target_y`        | float  | Target longitude / latitude (GCJ-02 as crawled; converted to WGS-84 for distance and azimuth computation) |
| `exact_distance_m`            | float  | Anchor-to-target straight-line distance, metres                                                           |
| `dist_level`                  | enum   | `VeryClose` / `Close` / `Medium` / `Far`                                                                  |
| `azimuth_deg`                 | float  | Bearing clockwise from north, (0, 360]                                                                    |
| `direction_8`, `direction_16` | enum   | Sector quantization of the azimuth                                                                        |
| `topology_type`               | enum   | One of the six road-context classes                                                                       |
| `combo_class`                 | enum   | `Explicit` / `Implicit`                                                                                   |

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
- **Rate limits.** The Amap free tier caps daily quota per key; the crawler throttles to ~2 requests/s and  
  retries three times, but a full 16-district sweep needs several days or a paid quota.


## Contact

Can Zhuang — School of Civil Engineering and Geomatics, Shandong University of Technology, Zibo, China.  
Issues and pull requests are welcome; for data-related questions please open an issue rather than emailing,  
so that answers stay searchable.

---

## 中文说明

本仓库是 **AIGeoRel v1.0.0** 数据集（532,285 条中英双语相对位置描述，锚定 26,800 个地标，覆盖北京 16 个行政区）的**完整构建代码**。

- **环境**：Python 3.9+、PostgreSQL 14+ 与 PostGIS 3.x、本地 Ollama（Qwen2.5）。依赖已固定版本区间，直接 `pip install -r requirements.txt`；配置写入仓库根目录 `.env`。
- **运行方式**：各阶段为独立脚本，需在脚本所在目录下执行（`src/text/` 内的模块以裸名互相导入，并以 `../../` 解析配置与日志路径）。
- **注意**：部分脚本顶部仍保留研究环境的 Windows 绝对路径，首次运行前请改为本地路径；  `output/` 未纳入版本控制，需由脚本自行生成。
- **数据获取**：成品数据集通过 [Releases](https://github.com/zccodehub/GeoDatasets/releases/tag/v1.0.0) 发布。
- **许可**：代码采用 [Apache License 2.0](LICENSE)（可商用、可修改、需保留版权与许可声明，含专利授权）；数据集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)（可商用、需署名引用）。高德 POI 与 OSM 路网数据另受各自服务条款约束。
