```markdown
# TechSphere Analytics Engine v5.4

Production-ready job market analytics pipeline that transforms raw job data into actionable business intelligence for Power BI dashboards.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy spacy tqdm
python -m spacy download en_core_web_sm

# Run pipeline
python src/analytics_engine_spacy.py
```

## Input / Output

**Input:** `data/refined/final_refined_jobs.csv` (38,290 records, 7 domains)

**Output:** `data/analytics/` - 10 files including:
- `opportunity_scores_weighted.csv` - Risk-adjusted rankings
- `master_relative_scores.csv` - 0-10 scale scores for dashboard
- `layoff_exposure_scores.csv` - Layoff risk by domain
- `ai_resilience_scores.csv` - AI replacement risk
- `junior_bottleneck_index.csv` - Entry-level accessibility

## Key Features

| Feature | Description |
|---------|-------------|
| **Quality-weighted analytics** | Original data (weight=1.0), generated (0.5-0.7) |
| **Parallel NLP** | 7x faster keyword extraction across domains |
| **Smart caching** | Second run: 2-3 seconds (vs 15-20 sec first run) |
| **Relative scoring** | All metrics normalized to 0-10 scale |

## Opportunity Score Formula

```
demand_score = normalized(job_count)
salary_score = normalized(avg_salary)
diversity_score = normalized(unique_skills)

base_opportunity = (0.3 × demand) + (0.3 × salary) + (0.15 × diversity)
safety_score = (1 - layoff_exposure) × 0.6 + (ai_resilience) × 0.4
opportunity_score = (base_opportunity × 0.7) + (safety_score × 0.3)
```

## Output Files for Power BI

| File | Use |
|------|-----|
| `master_relative_scores.csv` | Main dashboard (0-10 scale) |
| `opportunity_scores_weighted.csv` | Rankings & recommendations |
| `layoff_heatmap_report.csv` | 4-quadrant matrix |
| `skill_frequency_weighted.csv` | Skill demand analysis |
| `salary_stats_weighted.csv` | Compensation benchmarks |

## Configuration

Edit at top of script:

```python
PARALLEL_WORKERS = 6          # CPU cores to use
USE_CACHE = True              # Enable/disable caching
SALARY_OUTLIER_CLIP = 0.99    # Remove top 1% outliers
```

## Performance

| Run | Time |
|-----|------|
| First run (no cache) | 15-20 seconds |
| Cached runs | 2-3 seconds |
| Memory usage | ~800 MB peak |

## Domain Coverage

| Domain | Job Count | Market Share |
|--------|-----------|--------------|
| Software Engineering | 16,074 | 42.0% |
| Data Science | 6,618 | 17.3% |
| DevOps | 4,636 | 12.1% |
| Cybersecurity | 3,132 | 8.2% |
| Web Development | 3,042 | 7.9% |
| AI/ML | 2,762 | 7.2% |
| Cloud Computing | 2,026 | 5.3% |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: spacy` | `pip install spacy && python -m spacy download en_core_web_sm` |
| Out of memory | Reduce `PARALLEL_WORKERS` to 2-3 |
| Cache corrupted | Delete `data/cache/*.pkl` files |
| Slow performance | Increase `SPACY_BATCH_SIZE` to 200 |

## Power BI Integration

1. Get Data → Text/CSV → Select all CSV files from `data/analytics/`
2. Create relationships using `domain` as key
3. Build visuals using relative scores (0-10 scale)

## License

UPES - TechSphere Analytics Team
```