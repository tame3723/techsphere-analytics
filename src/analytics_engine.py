"""
analytics_engine.py - Weighted Descriptive Analytics Engine for TechSphere Analytics

This script performs comprehensive descriptive analytics with weighted metrics that
prioritize original/real data over generated/imputed data.

Key Features:
- Quality-weighted statistics (original data weighted higher)
- Confidence scores for all metrics
- Document-frequency keyword extraction
- Weighted opportunity scores with configurable parameters
- Robust handling of zero-weight edge cases
- Outlier protection for salary statistics

Methodology:
- Original data weight = 1.0
- Generated data weight = 0.5-0.7
- Confidence score = weighted_count / raw_count

Author: TechSphere Analytics Team
Date: 2026-04-01
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import logging
import time
import re
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

INPUT_FILE = "D:/Code wala scene/techsphere-analytics/data/refined/final_refined_jobs.csv"
OUTPUT_DIR = "D:/Code wala scene/techsphere-analytics/data/analytics/"
RANDOM_SEED = 42

# Salary outlier clipping (99th percentile to prevent distortion)
# Set to None to disable, or to a percentile value (e.g., 0.99)
SALARY_OUTLIER_CLIP = 0.99

# Quality weights for different data sources
# These weights reflect how much we trust each data source
# Original/real data receives highest weight (1.0)
# Generated/imputed data receives lower weights (0.5-0.7)
QUALITY_WEIGHTS = {
    # Original data sources (highest trust)
    'original': 1.0,           # Original salary data
    'original_cleaned': 1.0,   # Original description (cleaned)
    'original_calculated': 1.0, # Original data
    
    # Generated/refined data (lower trust)
    'generated': 0.6,          # Generated salary/description
    'generated_refined': 0.6,  # Refined generated data
    'enhanced_generated': 0.7,  # Enhanced from original
    'generated_salary': 0.5,   # Completely generated salary
}

# Opportunity score weights (configurable for sensitivity analysis)
# These weights determine how much each factor contributes to the final score
OPPORTUNITY_WEIGHTS = {
    'demand': 0.4,      # Job market demand (40%)
    'salary': 0.4,      # Compensation potential (40%)
    'diversity': 0.2    # Skill variety (20%)
}

# Enhanced stop words for keyword extraction
# Removes common but non-informative words from keyword analysis
STOP_WORDS = {
    # Common English stop words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'should', 'could', 'may', 'might', 'can', 'must', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'our', 'your',
    
    # Job posting noise words (added for better keyword quality)
    'work', 'role', 'position', 'team', 'company', 'job', 'experience',
    'using', 'ability', 'strong', 'knowledge', 'requirements', 'develop',
    'developing', 'design', 'solutions', 'systems', 'skills', 'skill',
    'responsible', 'responsibilities', 'required', 'qualifications',
    'looking', 'seeking', 'candidate', 'ideal', 'opportunity'
}


class WeightedAnalyticsEngine:
    """
    Weighted analytics engine that prioritizes original/real data.
    All metrics are computed with quality-based weighting.
    
    Methodology:
    - Each record has a weight based on data source (original=1.0, generated=0.5-0.7)
    - Weighted statistics give more influence to original data
    - Confidence scores indicate data reliability
    - Opportunity scores are confidence-adjusted
    """
    
    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize the analytics engine.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output CSV files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.df = None
        self.opportunity_weights = OPPORTUNITY_WEIGHTS
        self.salary_outlier_clip = SALARY_OUTLIER_CLIP
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the finalized job market dataset.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        logger.info(f"Loading data from {self.input_path}...")
        start_time = time.time()
        
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(self.df):,} records in {time.time() - start_time:.2f} seconds")
            
            # Map column names
            self._map_column_names()
            
            # Add quality weight columns
            self._add_quality_weights()
            
            # Log data quality overview
            self._log_data_quality()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _map_column_names(self):
        """Map column names to standard format."""
        if 'description_source' in self.df.columns and 'description_quality' not in self.df.columns:
            self.df.rename(columns={'description_source': 'description_quality'}, inplace=True)
            logger.info("  Mapped 'description_source' to 'description_quality'")
        
        if 'salary_source' in self.df.columns and 'salary_quality' not in self.df.columns:
            self.df.rename(columns={'salary_source': 'salary_quality'}, inplace=True)
            logger.info("  Mapped 'salary_source' to 'salary_quality'")
        
        # Ensure quality columns exist with default values
        if 'description_quality' not in self.df.columns:
            self.df['description_quality'] = 'original_cleaned'
            logger.warning("  'description_quality' not found, defaulting to 'original_cleaned'")
        
        if 'salary_quality' not in self.df.columns:
            self.df['salary_quality'] = 'original'
            logger.warning("  'salary_quality' not found, defaulting to 'original'")
        
        # Calculate skills count
        if 'skills_count' not in self.df.columns and 'extracted_skills' in self.df.columns:
            self.df['skills_count'] = self.df['extracted_skills'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() != '' else 0
            )
    
    def _add_quality_weights(self):
        """Add weight columns based on data quality."""
        # Map quality to weight
        self.df['description_weight'] = self.df['description_quality'].map(
            lambda x: QUALITY_WEIGHTS.get(x, 0.5)
        )
        
        self.df['salary_weight'] = self.df['salary_quality'].map(
            lambda x: QUALITY_WEIGHTS.get(x, 0.5)
        )
        
        # Ensure valid weights (non-negative, not NaN)
        self.df['description_weight'] = self.df['description_weight'].fillna(0.5).clip(lower=0)
        self.df['salary_weight'] = self.df['salary_weight'].fillna(0.5).clip(lower=0)
    
    def _log_data_quality(self):
        """Log comprehensive data quality overview."""
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY OVERVIEW")
        logger.info("="*60)
        
        # Description quality distribution
        desc_counts = self.df['description_quality'].value_counts()
        logger.info("\nDescription Sources:")
        for source, count in desc_counts.items():
            weight = QUALITY_WEIGHTS.get(source, 0.5)
            logger.info(f"  {source:20s}: {count:>6,} ({count/len(self.df)*100:>5.1f}%) [weight={weight:.1f}]")
        
        # Salary quality distribution
        salary_counts = self.df['salary_quality'].value_counts()
        logger.info("\nSalary Sources:")
        for source, count in salary_counts.items():
            weight = QUALITY_WEIGHTS.get(source, 0.5)
            logger.info(f"  {source:20s}: {count:>6,} ({count/len(self.df)*100:>5.1f}%) [weight={weight:.1f}]")
        
        # Calculate effective data quality
        effective_desc_quality = self.df['description_weight'].mean()
        effective_salary_quality = self.df['salary_weight'].mean()
        
        logger.info(f"\nEffective Data Quality Scores:")
        logger.info(f"  Description Quality Index: {effective_desc_quality:.3f}")
        logger.info(f"  Salary Quality Index: {effective_salary_quality:.3f}")
    
    def compute_domain_demand(self) -> pd.DataFrame:
        """
        Compute domain demand with quality metrics.
        """
        logger.info("\n[1/8] Computing domain demand...")
        
        domain_counts = self.df['domain'].value_counts().reset_index()
        domain_counts.columns = ['domain', 'job_count']
        
        # Add weighted counts (based on description quality)
        weighted_counts = self.df.groupby('domain')['description_weight'].sum().reset_index()
        weighted_counts.columns = ['domain', 'weighted_job_count']
        
        # Merge
        domain_counts = domain_counts.merge(weighted_counts, on='domain')
        
        total_jobs = len(self.df)
        total_weighted = self.df['description_weight'].sum()
        
        domain_counts['percentage'] = (domain_counts['job_count'] / total_jobs * 100).round(2)
        domain_counts['weighted_percentage'] = (domain_counts['weighted_job_count'] / total_weighted * 100).round(2)
        
        # Confidence score = proportion of original-weighted data
        # Higher confidence = more original data
        domain_counts['confidence_score'] = (domain_counts['weighted_job_count'] / domain_counts['job_count']).round(3)
        
        domain_counts = domain_counts.sort_values('job_count', ascending=False)
        domain_counts['cumulative_percentage'] = domain_counts['percentage'].cumsum().round(2)
        
        logger.info(f"  Total jobs: {total_jobs:,}")
        logger.info(f"  Effective jobs (weighted): {total_weighted:.0f}")
        logger.info(f"  Top domain: {domain_counts.iloc[0]['domain']} ({domain_counts.iloc[0]['job_count']:,} jobs, confidence={domain_counts.iloc[0]['confidence_score']:.2f})")
        
        self.results['domain_demand'] = domain_counts
        return domain_counts
    
    def _safe_weighted_average(self, values, weights, default=None):
        """
        Safely compute weighted average with zero-weight guard.
        
        Args:
            values: Array of values
            weights: Array of weights
            default: Default value if sum of weights is zero
            
        Returns:
            Weighted average or default
        """
        weights = np.array(weights)
        weight_sum = weights.sum()
        
        if weight_sum == 0:
            return default if default is not None else values.mean()
        
        return np.average(values, weights=weights)
    
    def _weighted_median(self, values, weights):
        """
        Calculate weighted median with zero-weight guard.
        
        Args:
            values: Array of values
            weights: Array of weights
            
        Returns:
            Weighted median
        """
        if len(values) == 0:
            return 0
        
        # Sort by values
        sorted_idx = np.argsort(values)
        sorted_values = values.iloc[sorted_idx].values
        sorted_weights = weights.iloc[sorted_idx].values
        
        # Cumulative sum of weights
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        
        # FIXED: Handle zero total weight
        if total_weight == 0:
            return values.median()
        
        # Find median index
        median_idx = np.searchsorted(cumulative_weights, total_weight / 2)
        return sorted_values[median_idx]
    
    def _weighted_percentile(self, values, weights, percentile):
        """
        Calculate weighted percentile with zero-weight guard.
        
        Args:
            values: Array of values
            weights: Array of weights
            percentile: Percentile to calculate (0-1)
            
        Returns:
            Weighted percentile value
        """
        if len(values) == 0:
            return 0
        
        # Sort by values
        sorted_idx = np.argsort(values)
        sorted_values = values.iloc[sorted_idx].values
        sorted_weights = weights.iloc[sorted_idx].values
        
        # Cumulative sum of weights
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        
        # FIXED: Handle zero total weight
        if total_weight == 0:
            return values.quantile(percentile)
        
        # Find percentile index
        target_weight = total_weight * percentile
        idx = np.searchsorted(cumulative_weights, target_weight)
        
        return sorted_values[min(idx, len(sorted_values)-1)]
    
    def compute_salary_stats(self) -> pd.DataFrame:
        """
        Compute weighted salary statistics by domain.
        Original data gets higher weight in calculations.
        Includes optional outlier clipping to prevent distortion.
        """
        logger.info("\n[2/8] Computing weighted salary statistics...")
        if self.salary_outlier_clip:
            logger.info(f"  Outlier clipping: {self.salary_outlier_clip*100:.0f}th percentile")
        
        salary_stats = []
        
        for domain in self.df['domain'].unique():
            domain_df = self.df[self.df['domain'] == domain]
            
            # Filter valid salaries
            valid_salaries = domain_df[
                (domain_df['salary_min'] > 0) & 
                (domain_df['salary_max'] > 0) &
                (domain_df['salary_min'].notna()) &
                (domain_df['salary_max'].notna())
            ].copy()
            
            if len(valid_salaries) > 0:
                # Optional outlier clipping
                if self.salary_outlier_clip:
                    clip_threshold = valid_salaries['salary_min'].quantile(self.salary_outlier_clip)
                    original_count = len(valid_salaries)
                    valid_salaries = valid_salaries[valid_salaries['salary_min'] <= clip_threshold]
                    if len(valid_salaries) < original_count:
                        logger.debug(f"    {domain}: removed {original_count - len(valid_salaries)} outliers")
                
                # Separate original vs generated
                original_salaries = valid_salaries[valid_salaries['salary_quality'] == 'original']
                generated_salaries = valid_salaries[valid_salaries['salary_quality'] != 'original']
                
                # Weighted average with zero-weight guard
                weighted_avg_min = self._safe_weighted_average(
                    valid_salaries['salary_min'],
                    valid_salaries['salary_weight'],
                    default=valid_salaries['salary_min'].mean()
                )
                weighted_avg_max = self._safe_weighted_average(
                    valid_salaries['salary_max'],
                    valid_salaries['salary_weight'],
                    default=valid_salaries['salary_max'].mean()
                )
                
                # Weighted median
                weighted_median_min = self._weighted_median(
                    valid_salaries['salary_min'],
                    valid_salaries['salary_weight']
                )
                weighted_median_max = self._weighted_median(
                    valid_salaries['salary_max'],
                    valid_salaries['salary_weight']
                )
                
                stats = {
                    'domain': domain,
                    'original_count': len(original_salaries),
                    'generated_count': len(generated_salaries),
                    'original_pct': round(len(original_salaries) / len(valid_salaries) * 100, 2),
                    
                    # Unweighted statistics (for reference)
                    'avg_salary_min': valid_salaries['salary_min'].mean(),
                    'avg_salary_max': valid_salaries['salary_max'].mean(),
                    'median_salary_min': valid_salaries['salary_min'].median(),
                    'median_salary_max': valid_salaries['salary_max'].median(),
                    
                    # Weighted statistics (more reliable)
                    'weighted_avg_min': weighted_avg_min,
                    'weighted_avg_max': weighted_avg_max,
                    'weighted_median_min': weighted_median_min,
                    'weighted_median_max': weighted_median_max,
                    
                    'min_salary': valid_salaries['salary_min'].min(),
                    'max_salary': valid_salaries['salary_max'].max(),
                    'std_salary_min': valid_salaries['salary_min'].std(),
                    'salary_samples': len(valid_salaries),
                    'total_jobs': len(domain_df),
                    'coverage_pct': round(len(valid_salaries) / len(domain_df) * 100, 2),
                    'data_confidence': round(valid_salaries['salary_weight'].mean(), 3)
                }
                
                # Percentiles
                stats['p25_salary'] = valid_salaries['salary_min'].quantile(0.25)
                stats['p75_salary'] = valid_salaries['salary_min'].quantile(0.75)
                stats['weighted_p25'] = self._weighted_percentile(
                    valid_salaries['salary_min'], 
                    valid_salaries['salary_weight'], 
                    0.25
                )
                stats['weighted_p75'] = self._weighted_percentile(
                    valid_salaries['salary_min'], 
                    valid_salaries['salary_weight'], 
                    0.75
                )
                
                salary_stats.append(stats)
        
        salary_df = pd.DataFrame(salary_stats)
        if not salary_df.empty:
            salary_df = salary_df.sort_values('weighted_avg_min', ascending=False)
            
            # Round numeric columns
            numeric_cols = ['avg_salary_min', 'avg_salary_max', 'median_salary_min', 'median_salary_max',
                           'weighted_avg_min', 'weighted_avg_max', 'weighted_median_min', 'weighted_median_max',
                           'min_salary', 'max_salary', 'std_salary_min', 'p25_salary', 'p75_salary',
                           'weighted_p25', 'weighted_p75']
            for col in numeric_cols:
                if col in salary_df.columns:
                    salary_df[col] = salary_df[col].round(0)
            
            logger.info(f"  Highest paying domain (weighted): {salary_df.iloc[0]['domain']} (₹{salary_df.iloc[0]['weighted_avg_min']:,.0f})")
            logger.info(f"  Most reliable salary data: {salary_df.loc[salary_df['data_confidence'].idxmax(), 'domain']} (confidence={salary_df['data_confidence'].max():.2f})")
        
        self.results['salary_stats'] = salary_df
        return salary_df
    
    def compute_skill_frequency(self) -> pd.DataFrame:
        """
        Compute skill frequency with quality weighting.
        Each job contributes its weight to each skill it mentions.
        """
        logger.info("\n[3/8] Computing skill frequency...")
        
        skill_frequencies = []
        
        for domain in self.df['domain'].unique():
            domain_df = self.df[self.df['domain'] == domain]
            
            # Collect skills with weights
            skill_weighted_counter = Counter()
            skill_job_counter = Counter()
            
            for _, row in domain_df.iterrows():
                skills_str = row.get('extracted_skills', '')
                weight = row.get('description_weight', 1.0)
                
                if skills_str and isinstance(skills_str, str) and skills_str.strip():
                    skills = [s.strip() for s in skills_str.split(',') if s.strip()]
                    for skill in skills:
                        skill_weighted_counter[skill] += weight
                        skill_job_counter[skill] += 1
            
            if skill_weighted_counter:
                total_weighted = sum(skill_weighted_counter.values())
                total_jobs = len(domain_df)
                
                for skill, weighted_count in skill_weighted_counter.most_common(50):
                    # Confidence = weighted_count / raw_count (higher = more original data)
                    raw_count = skill_job_counter[skill]
                    confidence = round(weighted_count / raw_count, 2) if raw_count > 0 else 0
                    
                    skill_frequencies.append({
                        'domain': domain,
                        'skill': skill,
                        'frequency': raw_count,
                        'weighted_frequency': round(weighted_count, 1),
                        'pct_of_jobs': round(raw_count / total_jobs * 100, 2),
                        'weighted_pct': round(weighted_count / total_weighted * 100, 2),
                        'confidence': confidence
                    })
        
        if skill_frequencies:
            skill_df = pd.DataFrame(skill_frequencies)
            skill_df = skill_df.sort_values(['domain', 'weighted_frequency'], ascending=[True, False])
            skill_df['rank'] = skill_df.groupby('domain')['weighted_frequency'].rank(ascending=False, method='dense').astype(int)
            
            logger.info(f"  Total unique skills: {skill_df['skill'].nunique()}")
            if len(skill_df) > 0:
                logger.info(f"  Most confident skill: {skill_df.loc[skill_df['confidence'].idxmax(), 'skill']} (confidence={skill_df['confidence'].max():.2f})")
        else:
            skill_df = pd.DataFrame(columns=['domain', 'skill', 'frequency', 'weighted_frequency', 
                                            'pct_of_jobs', 'weighted_pct', 'confidence', 'rank'])
            logger.warning("  No skill data found")
        
        self.results['skill_frequency'] = skill_df
        return skill_df
    
    def compute_opportunity_score(self) -> pd.DataFrame:
        """
        Compute weighted opportunity score with configurable weights.
        
        Methodology:
        - Demand Score: Weighted job counts
        - Salary Score: Weighted salary averages
        - Diversity Score: Unique skills count
        
        Final score = Σ(weight_i × normalized_score_i) × confidence
        Confidence adjusts score based on data reliability.
        """
        logger.info("\n[4/8] Computing weighted opportunity scores...")
        logger.info(f"  Using weights: demand={self.opportunity_weights['demand']}, "
                   f"salary={self.opportunity_weights['salary']}, "
                   f"diversity={self.opportunity_weights['diversity']}")
        
        # Get required data
        demand_df = self.results.get('domain_demand')
        if demand_df is None:
            demand_df = self.compute_domain_demand()
        
        salary_df = self.results.get('salary_stats')
        if salary_df is None or salary_df.empty:
            salary_df = self.compute_salary_stats()
        
        skill_df = self.results.get('skill_frequency')
        
        # Merge dataframes
        opportunity_df = demand_df[['domain', 'job_count', 'weighted_job_count', 'confidence_score']].copy()
        opportunity_df = opportunity_df.merge(
            salary_df[['domain', 'weighted_avg_min', 'data_confidence', 'original_pct']],
            on='domain',
            how='left'
        )
        
        # Calculate skill diversity
        if skill_df is not None and not skill_df.empty:
            skill_diversity = skill_df.groupby('domain').agg({
                'skill': 'nunique',
                'weighted_frequency': 'sum'
            }).reset_index()
            skill_diversity.columns = ['domain', 'unique_skills', 'total_weighted_skill_mentions']
            opportunity_df = opportunity_df.merge(skill_diversity, on='domain', how='left')
        else:
            opportunity_df['unique_skills'] = 0
            opportunity_df['total_weighted_skill_mentions'] = 0
        
        # Fill NaN
        opportunity_df = opportunity_df.fillna(0)
        
        # Calculate composite confidence
        opportunity_df['composite_confidence'] = (
            opportunity_df['confidence_score'] * 0.3 +
            opportunity_df['data_confidence'] * 0.5 +
            (opportunity_df['original_pct'] / 100) * 0.2
        ).round(3)
        
        # Normalize scores (0-1 scale)
        # Demand Score (using weighted job counts)
        max_weighted_jobs = opportunity_df['weighted_job_count'].max()
        min_weighted_jobs = opportunity_df['weighted_job_count'].min()
        if max_weighted_jobs > min_weighted_jobs:
            opportunity_df['demand_score'] = (opportunity_df['weighted_job_count'] - min_weighted_jobs) / (max_weighted_jobs - min_weighted_jobs)
        else:
            opportunity_df['demand_score'] = 1.0
        
        # Salary Score (using weighted averages)
        max_salary = opportunity_df['weighted_avg_min'].max()
        min_salary = opportunity_df['weighted_avg_min'].min()
        if max_salary > min_salary:
            opportunity_df['salary_score'] = (opportunity_df['weighted_avg_min'] - min_salary) / (max_salary - min_salary)
        else:
            opportunity_df['salary_score'] = 1.0
        
        # Skill Diversity Score (using unique skills)
        max_skills = opportunity_df['unique_skills'].max()
        min_skills = opportunity_df['unique_skills'].min()
        if max_skills > min_skills:
            opportunity_df['diversity_score'] = (opportunity_df['unique_skills'] - min_skills) / (max_skills - min_skills)
        else:
            opportunity_df['diversity_score'] = 1.0
        
        # Weighted opportunity score with configurable weights
        opportunity_df['raw_opportunity_score'] = (
            self.opportunity_weights['demand'] * opportunity_df['demand_score'] +
            self.opportunity_weights['salary'] * opportunity_df['salary_score'] +
            self.opportunity_weights['diversity'] * opportunity_df['diversity_score']
        )
        
        # Apply confidence adjustment (higher confidence = higher score adjustment)
        opportunity_df['opportunity_score'] = (
            opportunity_df['raw_opportunity_score'] * opportunity_df['composite_confidence']
        ).round(4)
        
        # Sort and rank
        opportunity_df = opportunity_df.sort_values('opportunity_score', ascending=False)
        opportunity_df['rank'] = range(1, len(opportunity_df) + 1)
        
        logger.info(f"  Top opportunity domain: {opportunity_df.iloc[0]['domain']} (score: {opportunity_df.iloc[0]['opportunity_score']:.4f}, confidence: {opportunity_df.iloc[0]['composite_confidence']:.2f})")
        logger.info(f"  Most confident domain: {opportunity_df.loc[opportunity_df['composite_confidence'].idxmax(), 'domain']} (confidence: {opportunity_df['composite_confidence'].max():.3f})")
        
        self.results['opportunity_scores'] = opportunity_df
        return opportunity_df
    
    def compute_domain_trends(self) -> pd.DataFrame:
        """
        Compute weighted domain trends over time.
        Uses domain column for counting (most reliable).
        """
        logger.info("\n[5/8] Computing domain trends...")
        
        if 'year' not in self.df.columns:
            logger.warning("  'year' column not found - skipping trend analysis")
            return pd.DataFrame()
        
        # FIXED: Use domain for counting (most reliable, no nulls)
        weighted_trends = (
            self.df.groupby(['year', 'domain'])
            .agg(
                weighted_job_count=('description_weight', 'sum'),
                job_count=('domain', 'count')  # domain always has value
            )
            .reset_index()
        )
        
        # Calculate growth percentages
        weighted_trends = weighted_trends.sort_values(['domain', 'year'])
        weighted_trends['growth_pct'] = weighted_trends.groupby('domain')['job_count'].pct_change() * 100
        weighted_trends['weighted_growth_pct'] = weighted_trends.groupby('domain')['weighted_job_count'].pct_change() * 100
        
        # Market share
        yearly_totals = weighted_trends.groupby('year')['weighted_job_count'].sum().reset_index(name='year_total_weighted')
        weighted_trends = weighted_trends.merge(yearly_totals, on='year')
        weighted_trends['market_share_pct'] = (weighted_trends['weighted_job_count'] / weighted_trends['year_total_weighted'] * 100).round(2)
        
        logger.info(f"  Years covered: {sorted(weighted_trends['year'].unique())}")
        
        # Safe growth detection (handle NaN values)
        valid_growth = weighted_trends['growth_pct'].dropna()
        if not valid_growth.empty:
            fastest_idx = valid_growth.idxmax()
            fastest_domain = weighted_trends.loc[fastest_idx, 'domain']
            logger.info(f"  Fastest growing domain: {fastest_domain} ({valid_growth.max():.1f}% growth)")
        
        self.results['domain_trends'] = weighted_trends
        return weighted_trends
    
    def compute_top_keywords(self, n: int = 30) -> pd.DataFrame:
        """
        Extract weighted top keywords from descriptions.
        Uses document frequency (each job contributes once per unique word).
        Prevents bias from long descriptions.
        """
        logger.info("\n[6/8] Extracting weighted keywords...")
        
        keyword_data = []
        
        for domain in self.df['domain'].unique():
            domain_df = self.df[self.df['domain'] == domain]
            
            # Document frequency counter (each job contributes once per unique word)
            word_counter = Counter()
            weighted_counter = Counter()
            
            for _, row in domain_df.iterrows():
                desc = row.get('cleaned_description', '')
                weight = row.get('description_weight', 1.0)
                
                if desc and isinstance(desc, str) and desc.strip():
                    # Extract unique words from this description
                    desc_lower = desc.lower()
                    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', desc_lower))
                    
                    # Filter stop words
                    words = {w for w in words if w not in STOP_WORDS}
                    
                    # Count (each job contributes once per unique word)
                    for word in words:
                        word_counter[word] += 1
                        weighted_counter[word] += weight
            
            if word_counter:
                total_jobs = len(domain_df)
                total_weighted = sum(weighted_counter.values())
                
                for word, count in word_counter.most_common(n):
                    keyword_data.append({
                        'domain': domain,
                        'keyword': word,
                        'document_frequency': count,
                        'weighted_frequency': round(weighted_counter[word], 1),
                        'pct_of_jobs': round(count / total_jobs * 100, 2),
                        'weighted_pct': round(weighted_counter[word] / total_weighted * 100, 2) if total_weighted > 0 else 0
                    })
        
        if keyword_data:
            keyword_df = pd.DataFrame(keyword_data)
            keyword_df = keyword_df.sort_values(['domain', 'weighted_frequency'], ascending=[True, False])
            keyword_df['rank'] = keyword_df.groupby('domain')['weighted_frequency'].rank(ascending=False, method='dense').astype(int)
            
            logger.info(f"  Total unique keywords: {keyword_df['keyword'].nunique()}")
            if len(keyword_df) > 0:
                logger.info(f"  Most frequent keyword: '{keyword_df.iloc[0]['keyword']}' (in {keyword_df.iloc[0]['document_frequency']:,} jobs)")
        else:
            keyword_df = pd.DataFrame(columns=['domain', 'keyword', 'document_frequency', 'weighted_frequency', 
                                              'pct_of_jobs', 'weighted_pct', 'rank'])
            logger.warning("  No keyword data found")
        
        self.results['top_keywords'] = keyword_df
        return keyword_df
    
    def compute_data_quality_metrics(self) -> tuple:
        """
        Compute comprehensive data quality metrics.
        """
        logger.info("\n[7/8] Computing data quality metrics...")
        
        # Description quality with weights
        desc_quality = self.df.groupby('description_quality').agg({
            'job_id': 'count',
            'description_weight': 'sum'
        }).reset_index()
        desc_quality.columns = ['quality_source', 'count', 'total_weight']
        desc_quality['percentage'] = (desc_quality['count'] / len(self.df) * 100).round(2)
        desc_quality['weighted_percentage'] = (desc_quality['total_weight'] / self.df['description_weight'].sum() * 100).round(2)
        desc_quality = desc_quality.sort_values('count', ascending=False)
        
        # Salary quality with weights
        salary_quality = self.df.groupby('salary_quality').agg({
            'job_id': 'count',
            'salary_weight': 'sum'
        }).reset_index()
        salary_quality.columns = ['quality_source', 'count', 'total_weight']
        salary_quality['percentage'] = (salary_quality['count'] / len(self.df) * 100).round(2)
        salary_quality['weighted_percentage'] = (salary_quality['total_weight'] / self.df['salary_weight'].sum() * 100).round(2)
        salary_quality = salary_quality.sort_values('count', ascending=False)
        
        # Calculate effective original data percentages
        original_desc_pct = desc_quality[desc_quality['quality_source'] == 'original_cleaned']['weighted_percentage'].sum() if 'original_cleaned' in desc_quality['quality_source'].values else 0
        original_salary_pct = salary_quality[salary_quality['quality_source'] == 'original']['weighted_percentage'].sum() if 'original' in salary_quality['quality_source'].values else 0
        
        logger.info(f"  Effective original description data: {original_desc_pct:.1f}%")
        logger.info(f"  Effective original salary data: {original_salary_pct:.1f}%")
        
        self.results['description_quality'] = desc_quality
        self.results['salary_quality'] = salary_quality
        
        return desc_quality, salary_quality
    
    def compute_location_insights(self) -> tuple:
        """
        Compute weighted location insights.
        """
        logger.info("\n[8/8] Computing location insights...")
        
        # Top locations with weighting
        location_stats = self.df.groupby('location').agg({
            'job_id': 'count',
            'description_weight': 'sum'
        }).reset_index()
        location_stats.columns = ['location', 'job_count', 'weighted_job_count']
        location_stats['percentage'] = (location_stats['job_count'] / len(self.df) * 100).round(2)
        location_stats['weighted_percentage'] = (location_stats['weighted_job_count'] / self.df['description_weight'].sum() * 100).round(2)
        top_locations = location_stats.sort_values('weighted_job_count', ascending=False).head(20)
        
        # Top locations by domain
        location_by_domain = []
        for domain in self.df['domain'].unique():
            domain_df = self.df[self.df['domain'] == domain]
            domain_locations = domain_df.groupby('location').agg({
                'job_id': 'count',
                'description_weight': 'sum'
            }).reset_index()
            domain_locations.columns = ['location', 'job_count', 'weighted_job_count']
            domain_locations['domain'] = domain
            domain_locations['percentage'] = (domain_locations['job_count'] / len(domain_df) * 100).round(2)
            domain_locations = domain_locations.sort_values('weighted_job_count', ascending=False).head(10)
            location_by_domain.append(domain_locations)
        
        location_by_domain_df = pd.concat(location_by_domain, ignore_index=True) if location_by_domain else pd.DataFrame()
        
        logger.info(f"  Top location overall: {top_locations.iloc[0]['location']} ({top_locations.iloc[0]['job_count']:,} jobs, weighted: {top_locations.iloc[0]['weighted_job_count']:.0f})")
        
        self.results['top_locations'] = top_locations
        self.results['location_by_domain'] = location_by_domain_df
        
        return top_locations, location_by_domain_df
    
    def save_outputs(self):
        """
        Save all analytics outputs with weighted metrics.
        """
        logger.info(f"\nSaving weighted analytics outputs to {self.output_dir}...")
        
        output_files = {
            'domain_demand.csv': self.results.get('domain_demand'),
            'salary_stats_weighted.csv': self.results.get('salary_stats'),
            'skill_frequency_weighted.csv': self.results.get('skill_frequency'),
            'opportunity_scores_weighted.csv': self.results.get('opportunity_scores'),
            'domain_trends_weighted.csv': self.results.get('domain_trends'),
            'top_keywords_weighted.csv': self.results.get('top_keywords'),
            'description_quality.csv': self.results.get('description_quality'),
            'salary_quality.csv': self.results.get('salary_quality'),
            'top_locations_weighted.csv': self.results.get('top_locations'),
            'location_by_domain_weighted.csv': self.results.get('location_by_domain')
        }
        
        saved_files = []
        for filename, df in output_files.items():
            if df is not None and not df.empty:
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"  ✓ Saved {filename} ({len(df):,} rows)")
                saved_files.append(filename)
            else:
                logger.info(f"  ⊘ Skipped {filename} (no data)")
        
        self._save_summary_report()
        return saved_files
    
    def _save_summary_report(self):
        """Save comprehensive summary report with quality metrics."""
        summary_file = self.output_dir / "weighted_analytics_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TECHSPHERE ANALYTICS - WEIGHTED ANALYTICS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATA QUALITY FRAMEWORK\n")
            f.write("-" * 40 + "\n")
            f.write("Quality Weights (Higher = More Trustworthy):\n")
            for source, weight in sorted(QUALITY_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {source:25s}: {weight:.1f}\n")
            
            f.write(f"\nEffective Data Quality:\n")
            f.write(f"  Description Quality Index: {self.df['description_weight'].mean():.3f}\n")
            f.write(f"  Salary Quality Index: {self.df['salary_weight'].mean():.3f}\n\n")
            
            f.write("OPPORTUNITY SCORE CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Demand Weight: {self.opportunity_weights['demand']:.1f} (40% of raw score)\n")
            f.write(f"  Salary Weight: {self.opportunity_weights['salary']:.1f} (40% of raw score)\n")
            f.write(f"  Diversity Weight: {self.opportunity_weights['diversity']:.1f} (20% of raw score)\n")
            f.write(f"  Confidence Adjustment: Raw score × Confidence (0-1 scale)\n\n")
            
            if self.salary_outlier_clip:
                f.write(f"SALARY OUTLIER PROTECTION\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Outlier clipping: {self.salary_outlier_clip*100:.0f}th percentile\n")
                f.write(f"  Prevents extreme values from distorting averages\n\n")
            
            f.write("DOMAIN OPPORTUNITY SCORES (Weighted & Confidence-Adjusted)\n")
            f.write("-" * 40 + "\n")
            opportunity = self.results.get('opportunity_scores')
            if opportunity is not None and not opportunity.empty:
                for _, row in opportunity.iterrows():
                    f.write(f"  {row['domain']:25s}: Score={row['opportunity_score']:.4f} "
                           f"(Confidence={row['composite_confidence']:.2f}, "
                           f"Original Salary={row['original_pct']:.0f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Methodological Note:\n")
            f.write("- Original/real data receives weight = 1.0\n")
            f.write("- Generated/imputed data receives weight = 0.5-0.7\n")
            f.write("- Confidence scores reflect proportion of original data\n")
            f.write("- Opportunity scores = (weighted raw score) × confidence\n")
            f.write("- Document-frequency keyword extraction prevents bias\n")
            f.write("- Salary outliers clipped at 99th percentile for stability\n")
            f.write("- This ensures analytics reflect real job market trends\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"  ✓ Saved weighted_analytics_summary.txt")
    
    def run_pipeline(self):
        """
        Execute the complete weighted analytics pipeline.
        """
        try:
            logger.info("=" * 80)
            logger.info("WEIGHTED DESCRIPTIVE ANALYTICS ENGINE")
            logger.info("Prioritizing Original/Real Data Sources")
            logger.info("=" * 80)
            
            # Load data
            self.load_data()
            
            # Compute all analytics
            self.compute_domain_demand()
            self.compute_salary_stats()
            self.compute_skill_frequency()
            self.compute_opportunity_score()
            self.compute_domain_trends()
            self.compute_top_keywords()
            self.compute_data_quality_metrics()
            self.compute_location_insights()
            
            # Save outputs
            saved_files = self.save_outputs()
            
            logger.info("\n" + "=" * 80)
            logger.info("WEIGHTED ANALYTICS PIPELINE COMPLETED")
            logger.info("=" * 80)
            
            return self.results, saved_files
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """
    Main execution function.
    """
    try:
        engine = WeightedAnalyticsEngine(
            input_path=INPUT_FILE,
            output_dir=OUTPUT_DIR
        )
        
        results, saved_files = engine.run_pipeline()
        
        print("\n" + "=" * 80)
        print("✅ Weighted descriptive analytics pipeline complete.")
        print("=" * 80)
        print(f"\n📁 Output directory: {OUTPUT_DIR}")
        print(f"📊 Generated {len(saved_files)} analytics files:")
        for filename in saved_files:
            print(f"   ✓ {filename}")
        print(f"\n🎯 Key Features:")
        print(f"   • Original data weighted 1.0, generated data weighted 0.5-0.7")
        print(f"   • Confidence scores for all metrics (higher = more original data)")
        print(f"   • Weighted averages and percentiles with zero-weight protection")
        print(f"   • Configurable opportunity score weights")
        print(f"   • Document-frequency keyword extraction")
        if SALARY_OUTLIER_CLIP:
            print(f"   • Salary outlier clipping at {SALARY_OUTLIER_CLIP*100:.0f}th percentile")
        print("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("Please ensure the input file exists at: ../data/refined/final_refined_jobs.csv")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()