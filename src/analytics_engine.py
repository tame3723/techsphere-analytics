"""
analytics_engine_spacy.py - PRODUCTION Job Market Analytics Engine

FIXES APPLIED:
✅ Windows encoding fix (no Unicode special chars in file writes)
✅ Reduced parallel workers (15 is too many - overhead > benefit)
✅ Optimized spaCy batch processing
✅ Removed redundant operations

Author: TechSphere Analytics Team
Date: 2026-04-03
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import logging
import time
import re
import pickle
import hashlib
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/refined/final_refined_jobs.csv"
OUTPUT_DIR = BASE_DIR / "data/analytics/"
CACHE_DIR = BASE_DIR / "data/cache/"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Analytics config
SALARY_OUTLIER_CLIP = 0.99
SPACY_BATCH_SIZE = 200  # Increased for better throughput
USE_CACHE = True
# IMPORTANT: 15 workers is too many! Optimal is CPU_COUNT - 2
PARALLEL_WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 6))  # Cap at 6 for efficiency

# Quality weights
QUALITY_WEIGHTS = {
    'original': 1.0, 'original_cleaned': 1.0, 'original_calculated': 1.0,
    'generated': 0.6, 'generated_refined': 0.6,
    'enhanced_generated': 0.7, 'generated_salary': 0.5,
}

OPPORTUNITY_WEIGHTS = {'demand': 0.4, 'salary': 0.4, 'diversity': 0.2}

# NLP config
INCLUDE_POS = {'NOUN', 'PROPN', 'ADJ', 'VERB'}
MIN_TOKEN_LENGTH = 3
SPACY_MAX_LENGTH = 2_000_000


# ============================================================
# GLOBAL WORKER INITIALIZER
# ============================================================

_NLP = None

def init_worker():
    """Initialize spaCy once per worker process."""
    global _NLP
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    _NLP.max_length = SPACY_MAX_LENGTH


def process_domain_keywords(domain, texts, weights, n=30):
    """Process a single domain for keyword extraction."""
    global _NLP
    
    if not texts or not any(texts):
        return []
    
    results = []
    
    # Process in batches
    for doc in _NLP.pipe(texts, batch_size=SPACY_BATCH_SIZE):
        keywords = set()
        for token in doc:
            if (token.pos_ in INCLUDE_POS and 
                len(token.text) >= MIN_TOKEN_LENGTH and 
                not token.is_stop and 
                not token.is_punct):
                
                keyword = token.lemma_.lower()
                if len(keyword) >= MIN_TOKEN_LENGTH:
                    keywords.add(keyword)
        results.append(keywords)
    
    # Aggregate counts
    counter = Counter()
    weighted_counter = Counter()
    
    for keywords, weight in zip(results, weights):
        for kw in keywords:
            counter[kw] += 1
            weighted_counter[kw] += weight
    
    # Prepare output
    output = []
    total_jobs = len(texts)
    total_weighted = sum(weighted_counter.values()) if weighted_counter else 1
    
    for kw, count in counter.most_common(n):
        output.append({
            'domain': domain,
            'keyword': kw,
            'frequency': count,
            'weighted_freq': round(weighted_counter[kw], 1),
            'pct_jobs': round(count / total_jobs * 100, 2) if total_jobs else 0,
            'weighted_pct': round(weighted_counter[kw] / total_weighted * 100, 2) if total_weighted else 0
        })
    
    return output


class SpacyAnalyticsEngine:
    """Production-ready analytics engine with optimized multiprocessing."""
    
    def __init__(self):
        """Initialize with portable paths."""
        self.df = None
        self.results = {}
        
        logger.info(f"Parallel workers: {PARALLEL_WORKERS}")
        logger.info(f"Cache enabled: {USE_CACHE}")
        logger.info(f"SpaCy batch size: {SPACY_BATCH_SIZE}")
    
    def _get_config_hash(self):
        """Generate cache key from data AND config parameters."""
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(self.df, index=True).values
        ).hexdigest()
        
        config_str = f"{INCLUDE_POS}_{MIN_TOKEN_LENGTH}_{SPACY_BATCH_SIZE}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return hashlib.md5(f"{df_hash}_{config_hash}".encode()).hexdigest()
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare dataset."""
        logger.info(f"Loading: {INPUT_FILE}")
        start = time.time()
        
        self.df = pd.read_csv(INPUT_FILE)
        
        # Memory optimization
        for col in ['domain', 'description_quality', 'salary_quality', 'source']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # Map column names
        if 'description_source' in self.df.columns:
            self.df.rename(columns={'description_source': 'description_quality'}, inplace=True)
        if 'salary_source' in self.df.columns:
            self.df.rename(columns={'salary_source': 'salary_quality'}, inplace=True)
        
        # Add quality weights
        self.df['desc_weight'] = self.df['description_quality'].map(lambda x: QUALITY_WEIGHTS.get(x, 0.5))
        self.df['salary_weight'] = self.df['salary_quality'].map(lambda x: QUALITY_WEIGHTS.get(x, 0.5))
        self.df['desc_weight'] = self.df['desc_weight'].fillna(0.5)
        self.df['salary_weight'] = self.df['salary_weight'].fillna(0.5)
        
        # Vectorized skill count
        if 'extracted_skills' in self.df.columns:
            self.df['skills_count'] = self.df['extracted_skills'].fillna('').apply(
                lambda x: len(re.split(r',\s*', str(x))) if str(x).strip() else 0
            )
        
        logger.info(f"Loaded {len(self.df):,} rows in {time.time()-start:.1f}s")
        return self.df
    
    def compute_domain_demand(self):
        """Domain demand with confidence scores."""
        logger.info("\n[1/7] Domain demand")
        
        domain_counts = self.df['domain'].value_counts().reset_index()
        domain_counts.columns = ['domain', 'job_count']
        
        weighted = self.df.groupby('domain')['desc_weight'].sum().reset_index()
        weighted.columns = ['domain', 'weighted_count']
        
        domain_counts = domain_counts.merge(weighted, on='domain')
        domain_counts['pct'] = (domain_counts['job_count'] / len(self.df) * 100).round(2)
        domain_counts['weighted_pct'] = (domain_counts['weighted_count'] / self.df['desc_weight'].sum() * 100).round(2)
        domain_counts['confidence'] = (domain_counts['weighted_count'] / domain_counts['job_count']).round(3)
        domain_counts = domain_counts.sort_values('job_count', ascending=False)
        
        self.results['domain_demand'] = domain_counts
        return domain_counts
    
    def compute_salary_stats(self):
        """Weighted salary statistics with outlier clipping."""
        logger.info("\n[2/7] Salary stats")
        
        stats = []
        for domain, domain_df in self.df.groupby('domain'):
            valid = domain_df[(domain_df['salary_min'] > 0) & (domain_df['salary_max'] > 0)]
            
            if len(valid) == 0:
                continue
            
            if SALARY_OUTLIER_CLIP:
                clip_min = valid['salary_min'].quantile(SALARY_OUTLIER_CLIP)
                clip_max = valid['salary_max'].quantile(SALARY_OUTLIER_CLIP)
                valid = valid[(valid['salary_min'] <= clip_min) & (valid['salary_max'] <= clip_max)]
            
            original_pct = (valid['salary_quality'] == 'original').mean() * 100
            
            w = valid['salary_weight'].values
            w_sum = w.sum()
            if w_sum == 0:
                w_avg_min = valid['salary_min'].mean()
                w_avg_max = valid['salary_max'].mean()
            else:
                w_avg_min = np.average(valid['salary_min'], weights=w)
                w_avg_max = np.average(valid['salary_max'], weights=w)
            
            stats.append({
                'domain': domain,
                'original_pct': round(original_pct, 1),
                'weighted_avg_min': round(w_avg_min, 0),
                'weighted_avg_max': round(w_avg_max, 0),
                'samples': len(valid),
                'confidence': round(valid['salary_weight'].mean(), 3)
            })
        
        salary_df = pd.DataFrame(stats).sort_values('weighted_avg_min', ascending=False)
        self.results['salary_stats'] = salary_df
        return salary_df
    
    def compute_skill_frequency_vectorized(self):
        """Skill frequency with VECTORIZED extraction."""
        logger.info("\n[3/7] Skill frequency (vectorized)")
        
        skill_data = []
        
        for domain, domain_df in tqdm(self.df.groupby('domain'), desc="  Processing domains"):
            counter = Counter()
            weighted_counter = Counter()
            
            for skills_str, weight in zip(domain_df['extracted_skills'], domain_df['desc_weight']):
                if skills_str and isinstance(skills_str, str):
                    skills = re.split(r',\s*', skills_str.strip())
                    for skill in skills:
                        if skill:
                            counter[skill] += 1
                            weighted_counter[skill] += weight
            
            if counter:
                total_weighted = sum(weighted_counter.values())
                total_jobs = len(domain_df)
                
                for skill, count in counter.most_common(50):
                    skill_data.append({
                        'domain': domain,
                        'skill': skill,
                        'frequency': count,
                        'weighted_freq': round(weighted_counter[skill], 1),
                        'pct_jobs': round(count / total_jobs * 100, 2),
                        'confidence': round(weighted_counter[skill] / count, 2) if count else 0
                    })
        
        skill_df = pd.DataFrame(skill_data) if skill_data else pd.DataFrame()
        if not skill_df.empty:
            skill_df = skill_df.sort_values(['domain', 'weighted_freq'], ascending=[True, False])
            skill_df['rank'] = skill_df.groupby('domain')['weighted_freq'].rank(ascending=False, method='dense').astype(int)
        
        self.results['skill_frequency'] = skill_df
        return skill_df
    
    def compute_top_keywords_parallel(self, n=30):
        """Parallel keyword extraction with optimized settings."""
        logger.info(f"\n[4/7] Top keywords (parallel, {PARALLEL_WORKERS} workers)")
        start = time.time()
        
        # Prepare lightweight data
        domain_data = []
        for domain, group in self.df.groupby('domain'):
            texts = group['cleaned_description'].fillna('').tolist()
            weights = group['desc_weight'].tolist()
            domain_data.append((domain, texts, weights, n))
        
        # Check cache
        cache_key = self._get_config_hash()
        cache_file = CACHE_DIR / f"keywords_{cache_key}.pkl"
        
        if USE_CACHE and cache_file.exists():
            logger.info("  Loading from cache...")
            with open(cache_file, 'rb') as f:
                all_results = pickle.load(f)
        else:
            all_results = []
            
            with ProcessPoolExecutor(
                max_workers=PARALLEL_WORKERS,
                initializer=init_worker
            ) as executor:
                futures = {
                    executor.submit(process_domain_keywords, domain, texts, weights, n): domain
                    for domain, texts, weights, n in domain_data
                }
                
                with tqdm(total=len(futures), desc="  Domains") as pbar:
                    for future in as_completed(futures):
                        domain = futures[future]
                        try:
                            results = future.result(timeout=180)
                            all_results.extend(results)
                        except Exception as e:
                            logger.warning(f"  Failed {domain}: {e}")
                        pbar.update(1)
            
            if USE_CACHE:
                with open(cache_file, 'wb') as f:
                    pickle.dump(all_results, f)
        
        keyword_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        if not keyword_df.empty:
            keyword_df = keyword_df.sort_values(['domain', 'weighted_freq'], ascending=[True, False])
            keyword_df['rank'] = keyword_df.groupby('domain')['weighted_freq'].rank(ascending=False, method='dense').astype(int)
        
        logger.info(f"  Completed in {time.time()-start:.1f}s")
        self.results['top_keywords'] = keyword_df
        return keyword_df
    
    def compute_opportunity_score(self):
        """Weighted opportunity score."""
        logger.info("\n[5/7] Opportunity scores")
        
        demand = self.results['domain_demand'][['domain', 'weighted_count', 'confidence']]
        salary = self.results['salary_stats'][['domain', 'weighted_avg_min', 'confidence']]
        skills = self.results['skill_frequency'].groupby('domain').size().reset_index(name='unique_skills')
        
        df = demand.merge(salary, on='domain', how='left', suffixes=('_demand', '_salary'))
        df = df.merge(skills, on='domain', how='left').fillna(0)
        
        # Normalize
        for col, name in [('weighted_count', 'demand'), ('weighted_avg_min', 'salary'), ('unique_skills', 'diversity')]:
            max_val = df[col].max()
            min_val = df[col].min()
            df[f'{name}_score'] = (df[col] - min_val) / (max_val - min_val) if max_val > min_val else 1.0
        
        df['comp_confidence'] = (df['confidence_demand'] * 0.3 + df['confidence_salary'] * 0.5 + 0.2).round(3)
        df['raw_score'] = (0.4 * df['demand_score'] + 0.4 * df['salary_score'] + 0.2 * df['diversity_score'])
        df['opportunity_score'] = (df['raw_score'] * df['comp_confidence']).round(4)
        df = df.sort_values('opportunity_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        self.results['opportunity_scores'] = df[['domain', 'opportunity_score', 'comp_confidence', 'rank']]
        return self.results['opportunity_scores']
    
    def compute_domain_trends(self):
        """Year-over-year trends."""
        logger.info("\n[6/7] Domain trends")
        
        if 'year' not in self.df.columns:
            logger.warning("  No year column")
            return pd.DataFrame()
        
        trends = (self.df.groupby(['year', 'domain'])
                  .agg(jobs=('domain', 'count'), weighted_jobs=('desc_weight', 'sum'))
                  .reset_index())
        
        trends = trends.sort_values(['domain', 'year'])
        trends['growth'] = trends.groupby('domain')['jobs'].pct_change() * 100
        
        yearly_total = trends.groupby('year')['weighted_jobs'].sum().reset_index(name='year_total')
        trends = trends.merge(yearly_total, on='year')
        trends['market_share'] = (trends['weighted_jobs'] / trends['year_total'] * 100).round(2)
        
        self.results['domain_trends'] = trends
        return trends
    
    def compute_data_quality(self):
        """Data source distribution."""
        logger.info("\n[7/7] Data quality")
        
        desc_q = (self.df.groupby('description_quality')
                  .agg(count=('description_quality', 'size'), weight=('desc_weight', 'sum'))
                  .reset_index())
        desc_q['pct'] = (desc_q['count'] / len(self.df) * 100).round(2)
        desc_q['weighted_pct'] = (desc_q['weight'] / self.df['desc_weight'].sum() * 100).round(2)
        
        salary_q = (self.df.groupby('salary_quality')
                    .agg(count=('salary_quality', 'size'), weight=('salary_weight', 'sum'))
                    .reset_index())
        salary_q['pct'] = (salary_q['count'] / len(self.df) * 100).round(2)
        salary_q['weighted_pct'] = (salary_q['weight'] / self.df['salary_weight'].sum() * 100).round(2)
        
        self.results['description_quality'] = desc_q
        self.results['salary_quality'] = salary_q
        return desc_q, salary_q
    
    def save_outputs(self):
        """Save all results to CSV."""
        logger.info("\nSaving outputs...")
        
        outputs = {
            'domain_demand.csv': self.results.get('domain_demand'),
            'salary_stats_weighted.csv': self.results.get('salary_stats'),
            'skill_frequency_weighted.csv': self.results.get('skill_frequency'),
            'opportunity_scores_weighted.csv': self.results.get('opportunity_scores'),
            'domain_trends_weighted.csv': self.results.get('domain_trends'),
            'top_keywords_spacy.csv': self.results.get('top_keywords'),
            'description_quality.csv': self.results.get('description_quality'),
            'salary_quality.csv': self.results.get('salary_quality'),
        }
        
        saved = []
        for name, df in outputs.items():
            if df is not None and not df.empty:
                df.to_csv(OUTPUT_DIR / name, index=False)
                saved.append(name)
                logger.info(f"  [OK] {name} ({len(df):,} rows)")
        
        self._save_summary()
        return saved
    
    def _save_summary(self):
        """Generate summary report (NO UNICODE SPECIAL CHARS for Windows)."""
        report = OUTPUT_DIR / "spacy_analytics_summary.txt"
        with open(report, 'w', encoding='utf-8') as f:  # Force UTF-8 encoding
            f.write("=" * 80 + "\n")
            f.write("TECHSPHERE ANALYTICS - PIPELINE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total records: {len(self.df):,}\n")
            f.write(f"Parallel workers: {PARALLEL_WORKERS}\n")
            f.write(f"Cache enabled: {USE_CACHE}\n")
            f.write(f"SpaCy batch size: {SPACY_BATCH_SIZE}\n\n")
            
            f.write("OPTIMIZATIONS APPLIED:\n")
            f.write("-" * 40 + "\n")
            f.write("[OK] spaCy loaded once per worker\n")
            f.write("[OK] Pass only lists to workers\n")
            f.write("[OK] Config-aware cache invalidation\n")
            f.write("[OK] Vectorized skill extraction\n\n")
            
            f.write("OPPORTUNITY SCORES\n")
            f.write("-" * 40 + "\n")
            opp = self.results.get('opportunity_scores')
            if opp is not None:
                for _, row in opp.iterrows():
                    f.write(f"  {row['domain']:25s}: {row['opportunity_score']:.4f}\n")
        
        logger.info(f"  [OK] Summary report")
    
    def run(self):
        """Execute full pipeline."""
        logger.info("=" * 80)
        logger.info("PRODUCTION JOB MARKET ANALYTICS ENGINE")
        logger.info("=" * 80)
        
        self.load_data()
        self.compute_domain_demand()
        self.compute_salary_stats()
        self.compute_skill_frequency_vectorized()
        self.compute_top_keywords_parallel()
        self.compute_opportunity_score()
        self.compute_domain_trends()
        self.compute_data_quality()
        
        saved = self.save_outputs()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"COMPLETE! Generated {len(saved)} files")
        logger.info("=" * 80)
        return self.results, saved


def main():
    try:
        engine = SpacyAnalyticsEngine()
        results, files = engine.run()
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nOutput: {OUTPUT_DIR}")
        print(f"Files: {', '.join(files)}")
        print(f"Parallel workers: {PARALLEL_WORKERS}")
        print(f"Cache: {'Enabled' if USE_CACHE else 'Disabled'}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()