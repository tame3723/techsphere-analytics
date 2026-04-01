"""
domain_extractor.py - Domain Classification Module for TechSphere Analytics

This script processes cleaned job data and classifies Computer Science related jobs
into 7 technical domains using strict rule-based keyword matching with context awareness.

The script is optimized for large datasets (1.3M+ rows) using vectorized pandas operations.

IMPORTANT: This version includes strict validation to prevent misclassification
of non-technical roles (e.g., chefs, administrators, fundraisers) as CS jobs.

Author: TechSphere Analytics Team
Date: 2026-03-31
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# NON-TECHNICAL EXCLUSION KEYWORDS
# ============================================================
# These keywords indicate the job is NOT a CS/tech role
# If ANY of these appear in the job title, the job is rejected
# ============================================================

NON_TECH_KEYWORDS: Set[str] = {
    # Food service (false positives for DevOps "chef")
    'chef', 'sous chef', 'executive chef', 'pastry chef', 'line cook',
    'cook', 'kitchen manager', 'culinary', 'food service', 'restaurant',
    'dining', 'bakery', 'bartender', 'barista', 'waiter', 'waitress',
    'hostess', 'dishwasher', 'prep cook', 'grill cook',
    
    # Administrative (false positives for Cloud "admin")
    'administrator', 'administrative', 'admin assistant', 'office manager',
    'executive assistant', 'personal assistant', 'receptionist',
    'clerk', 'secretary', 'front desk', 'office coordinator',
    
    # Healthcare (non-tech)
    'nurse', 'doctor', 'physician', 'surgeon', 'dentist', 'veterinarian',
    'pharmacist', 'therapist', 'counselor', 'psychologist', 'radiologist',
    'medical assistant', 'caregiver', 'nursing', 'clinical',
    
    # Education (non-tech teaching roles)
    'teacher', 'professor', 'instructor', 'principal', 'superintendent',
    'librarian', 'coach', 'tutor', 'educator', 'school counselor',
    
    # Sales & Marketing (non-tech)
    'sales', 'marketing', 'account executive', 'business development',
    'recruiter', 'hr', 'human resources', 'talent acquisition',
    'real estate agent', 'broker', 'loan officer', 'financial advisor',
    
    # Operations (non-tech)
    'driver', 'delivery', 'warehouse', 'forklift', 'janitor', 'cleaner',
    'maintenance', 'security guard', 'cashier', 'retail', 'store manager',
    'customer service', 'call center', 'telemarketer',
    
    # Construction & Trades
    'plumber', 'electrician', 'carpenter', 'welder', 'mechanic',
    'technician non-it', 'repair', 'construction', 'contractor',
    'painter', 'roofer', 'mason', 'laborer',
    
    # Hospitality
    'housekeeper', 'housekeeping', 'maid', 'valet', 'bellman',
    'concierge', 'front desk agent', 'reservation', 'hotel manager',
    
    # Finance (non-tech)
    'accountant', 'bookkeeper', 'auditor', 'financial analyst non-it',
    'bank teller', 'loan processor', 'underwriter',
    
    # Legal
    'lawyer', 'attorney', 'paralegal', 'legal assistant', 'judge',
    
    # Creative (non-tech)
    'photographer', 'videographer', 'graphic designer non-tech',
    'artist', 'musician', 'writer non-tech', 'editor non-tech',
    
    # Social Services
    'social worker', 'case manager', 'counselor', 'psychiatrist',
    'therapist', 'psychologist',
}

# ============================================================
# CS JOB TITLE INDICATORS (must have at least one)
# ============================================================
# These keywords indicate the job is definitely CS/tech-related
# A job MUST match at least one of these to be considered
# ============================================================

CS_JOB_INDICATORS: Set[str] = {
    # Core CS roles
    'software engineer', 'software developer', 'programmer',
    'data scientist', 'data analyst', 'data engineer',
    'machine learning', 'ai engineer', 'ml engineer',
    'devops', 'site reliability', 'sre',
    'cloud engineer', 'cloud architect',
    'security engineer', 'cybersecurity',
    'frontend', 'backend', 'full stack',
    'systems engineer', 'systems architect',
    'application developer', 'web developer',
    
    # Engineering roles
    'engineer', 'developer', 'architect',
    
    # Technical specific
    'java', 'python', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
    'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring',
    'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform',
    'tensorflow', 'pytorch', 'pandas', 'sql', 'nosql',
    'api', 'microservices', 'ci/cd', 'jenkins', 'git',
}

# ============================================================
# DOMAIN KEYWORD DICTIONARY (with context-specific terms)
# ============================================================

DOMAIN_KEYWORDS = {
    # ---------------------------------------------------------
    # 1. AI / MACHINE LEARNING
    # ---------------------------------------------------------
    "AI/ML": [
        "artificial intelligence", "ai engineer", "ai developer",
        "machine learning", "ml engineer", "ml developer",
        "deep learning", "neural network", "cnn", "rnn", "lstm",
        "transformer", "bert", "gpt", "llm", "generative ai",
        "computer vision", "image processing", "object detection",
        "nlp", "natural language processing", "text mining",
        "speech recognition", "reinforcement learning",
        "model training", "model evaluation", "feature engineering",
        "tensorflow", "pytorch", "keras", "scikit-learn",
        "opencv", "hugging face", "tokenization", "embeddings",
        "predictive modeling", "ai research", "ai scientist",
        "ml ops", "ai pipeline", "algorithm engineer",
        "ai specialist", "ml specialist", "ai/ml engineer",
        "prompt engineer", "generative ai engineer", "llm engineer"
    ],

    # ---------------------------------------------------------
    # 2. DATA SCIENCE
    # ---------------------------------------------------------
    "Data Science": [
        "data scientist", "data science", "data analyst",
        "data analytics", "data mining", "data wrangling",
        "statistical analysis", "statistics", "data modeling",
        "predictive analytics", "business intelligence",
        "bi analyst", "bi developer", "data visualization",
        "tableau", "power bi", "data cleaning",
        "exploratory data analysis", "eda", "feature selection",
        "data pipeline", "data reporting", "dashboard developer",
        "quantitative analyst", "data researcher",
        "data engineer", "data architect", "analytics engineer",
        "insights analyst", "data ops", "etl developer"
    ],

    # ---------------------------------------------------------
    # 3. WEB DEVELOPMENT
    # ---------------------------------------------------------
    "Web Development": [
        "web developer", "web development", "frontend developer",
        "backend developer", "full stack developer", "fullstack",
        "frontend engineer", "backend engineer",
        "react developer", "angular developer", "vue developer",
        "javascript developer", "typescript developer",
        "node.js", "express.js", "django", "flask",
        "php developer", "laravel developer",
        "rest api", "graphql", "web application",
        "next.js", "nuxt.js", "web services",
        "html", "css", "responsive design",
        "ui developer", "ux developer"
    ],

    # ---------------------------------------------------------
    # 4. CYBERSECURITY
    # ---------------------------------------------------------
    "Cybersecurity": [
        "cybersecurity", "cyber security", "security analyst",
        "information security", "infosec", "security engineer",
        "penetration tester", "pen tester", "ethical hacker",
        "vulnerability assessment", "security audit",
        "network security", "application security",
        "soc analyst", "security operations center",
        "incident response", "threat intelligence",
        "malware analysis", "digital forensics",
        "siem", "splunk", "firewall engineer",
        "identity access management", "iam engineer",
        "risk analyst", "cyber defense", "security consultant",
        "security architect", "devsecops"
    ],

    # ---------------------------------------------------------
    # 5. CLOUD COMPUTING
    # ---------------------------------------------------------
    "Cloud Computing": [
        "cloud engineer", "cloud developer", "cloud architect",
        "cloud computing", "aws engineer", "azure engineer",
        "gcp engineer", "amazon web services", "microsoft azure",
        "google cloud platform", "cloud infrastructure",
        "cloud solutions architect", "cloud migration",
        "cloud security", "cloud deployment",
        "serverless", "lambda", "ec2", "s3",
        "kubernetes cloud", "cloud automation",
        "cloud networking", "cloud platform engineer",
        "terraform", "cloud provisioning", "cloud specialist",
        "aws architect", "azure architect", "gcp architect",
        "cloud native", "multi-cloud", "hybrid cloud"
    ],

    # ---------------------------------------------------------
    # 6. DEVOPS
    # ---------------------------------------------------------
    "DevOps": [
        "devops engineer", "devops developer", "devops specialist",
        "ci/cd", "continuous integration", "continuous deployment",
        "pipeline automation", "build engineer",
        "release engineer", "site reliability engineer",
        "sre", "infrastructure as code", "iac",
        "docker", "kubernetes", "containerization",
        "ansible", "terraform", "pulumi",
        "jenkins", "gitlab ci", "github actions",
        "monitoring", "prometheus", "grafana",
        "deployment automation", "configuration management",
        "platform engineer", "infrastructure engineer",
        "automation engineer", "gitops", "observability"
    ],

    # ---------------------------------------------------------
    # 7. SOFTWARE ENGINEERING
    # ---------------------------------------------------------
    "Software Engineering": [
        "software engineer", "software developer",
        "software engineering", "application developer",
        "systems developer", "programmer",
        "c++ developer", "java developer", "python developer",
        "c# developer", ".net developer",
        "software architect", "software tester",
        "qa engineer", "quality assurance",
        "unit testing", "integration testing",
        "oop", "object oriented programming",
        "software design", "design patterns",
        "agile developer", "scrum developer",
        "mobile app developer", "android developer",
        "ios developer", "cross platform developer",
        "embedded engineer", "firmware engineer",
        "game developer", "systems engineer",
        "distributed systems", "backend services",
        "scala developer", "kotlin developer", "swift developer",
        "go developer", "golang developer", "rust developer"
    ]
}

# Domain priority order for resolving conflicts
DOMAIN_PRIORITY = [
    "AI/ML",
    "Data Science",
    "Cybersecurity",
    "Cloud Computing",
    "DevOps",
    "Web Development",
    "Software Engineering"
]


class DomainExtractor:
    """
    Domain classification engine for job postings with strict validation.
    """
    
    def __init__(self, data_path: str = "data/cleaned/cleaned_jobs.csv"):
        """
        Initialize the domain extractor with data path.
        
        Args:
            data_path: Path to the cleaned jobs CSV file
        """
        self.data_path = Path(data_path)
        self.output_path = Path("data/refined")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        
        # Pre-compile regex patterns for faster matching
        self.non_tech_pattern = self._compile_non_tech_pattern()
        self.cs_indicator_pattern = self._compile_cs_indicator_pattern()
        self.domain_patterns = self._compile_domain_patterns()
        
    def _compile_non_tech_pattern(self) -> re.Pattern:
        """
        Compile regex pattern for non-technical keywords.
        
        Returns:
            Compiled regex pattern
        """
        # Sort by length (longest first) for better matching
        sorted_keywords = sorted(NON_TECH_KEYWORDS, key=len, reverse=True)
        escaped_keywords = [re.escape(kw) for kw in sorted_keywords]
        pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _compile_cs_indicator_pattern(self) -> re.Pattern:
        """
        Compile regex pattern for CS job indicators.
        
        Returns:
            Compiled regex pattern
        """
        sorted_keywords = sorted(CS_JOB_INDICATORS, key=len, reverse=True)
        escaped_keywords = [re.escape(kw) for kw in sorted_keywords]
        pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _compile_domain_patterns(self) -> Dict[str, re.Pattern]:
        """
        Pre-compile regex patterns for each domain.
        
        Returns:
            Dictionary mapping domains to compiled regex patterns
        """
        patterns = {}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            escaped_keywords = [re.escape(kw) for kw in sorted_keywords]
            pattern = r'\b(' + '|'.join(escaped_keywords) + r')\b'
            patterns[domain] = re.compile(pattern, re.IGNORECASE)
            
        return patterns
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the cleaned job dataset.
        
        Returns:
            DataFrame containing job data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}...")
        start_time = time.time()
        
        # Load all columns but be efficient
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df):,} records in {time.time() - start_time:.2f} seconds")
            
            # Log missing descriptions
            if 'job_description' in self.df.columns:
                desc_missing = self.df['job_description'].isna().sum()
                logger.info(f"Missing job descriptions: {desc_missing:,} ({desc_missing/len(self.df)*100:.1f}%)")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def is_cs_job(self, job_title: str) -> bool:
        """
        Determine if a job is CS-related using strict validation.
        
        Rules:
        1. Must NOT contain any non-tech keywords (chef, admin, etc.)
        2. Must contain at least one CS job indicator (engineer, developer, etc.)
        
        Args:
            job_title: The job title to validate
            
        Returns:
            True if the job is CS-related, False otherwise
        """
        if pd.isna(job_title) or job_title == '':
            return False
        
        title_lower = str(job_title).lower().strip()
        
        # Check for non-tech keywords (immediate rejection)
        if self.non_tech_pattern.search(title_lower):
            return False
        
        # Check for CS job indicators (must have at least one)
        if self.cs_indicator_pattern.search(title_lower):
            return True
        
        return False
    
    def prepare_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare text fields for classification with strict validation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with prepared text fields and validation flags
        """
        logger.info("Preparing text for classification...")
        
        df = df.copy()
        
        # Prepare job_title
        df['job_title_clean'] = df['job_title'].fillna('').str.lower().str.strip()
        
        # Prepare job_description if it exists
        if 'job_description' in df.columns:
            df['job_description_clean'] = df['job_description'].fillna('').str.lower().str.strip()
        else:
            df['job_description_clean'] = ''
        
        # First, validate if this is actually a CS job
        logger.info("Applying strict CS job validation...")
        df['is_cs_job'] = df['job_title_clean'].apply(self.is_cs_job)
        
        cs_job_count = df['is_cs_job'].sum()
        rejected_count = len(df) - cs_job_count
        logger.info(f"CS job validation results:")
        logger.info(f"  - Valid CS jobs: {cs_job_count:,} ({cs_job_count/len(df)*100:.1f}%)")
        logger.info(f"  - Rejected (non-CS): {rejected_count:,} ({rejected_count/len(df)*100:.1f}%)")
        
        # Only keep CS jobs for further processing
        df = df[df['is_cs_job']].copy()
        
        # Create combined text for classification
        df['combined_text'] = df['job_title_clean']
        
        # Add description only if not empty (to avoid diluting title signal)
        desc_not_empty = df['job_description_clean'].str.len() > 0
        df.loc[desc_not_empty, 'combined_text'] = (
            df.loc[desc_not_empty, 'job_title_clean'] + ' ' + 
            df.loc[desc_not_empty, 'job_description_clean']
        )
        
        logger.info(f"After validation: {len(df):,} CS jobs to classify")
        return df
    
    def classify_domain(self, text: str) -> Optional[str]:
        """
        Classify a single text record into a domain.
        
        Args:
            text: Combined text (title + description) for classification
            
        Returns:
            Domain name or None if no match found
        """
        if not text or len(text.strip()) == 0:
            return None
        
        matches = []
        
        # Check each domain for matches
        for domain, pattern in self.domain_patterns.items():
            if pattern.search(text):
                matches.append(domain)
        
        if not matches:
            return None
        
        # Return highest priority domain if multiple matches
        for priority_domain in DOMAIN_PRIORITY:
            if priority_domain in matches:
                return priority_domain
        
        return matches[0]
    
    def apply_domain_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain classification to all records.
        
        Args:
            df: DataFrame with prepared text fields
            
        Returns:
            DataFrame with domain classification
        """
        logger.info("Applying domain classification...")
        start_time = time.time()
        
        # Vectorized classification
        df['domain'] = df['combined_text'].apply(self.classify_domain)
        
        # Count classified and unclassified
        classified = df['domain'].notna().sum()
        unclassified = df['domain'].isna().sum()
        
        logger.info(f"Classification completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Classified: {classified:,} ({classified/len(df)*100:.1f}%)")
        logger.info(f"Unclassified (no domain match): {unclassified:,} ({unclassified/len(df)*100:.1f}%)")
        
        # Show domain distribution
        domain_counts = df['domain'].value_counts()
        if not domain_counts.empty:
            logger.info("\nDomain distribution among classified jobs:")
            for domain, count in domain_counts.items():
                logger.info(f"  {domain}: {count:,} ({count/classified*100:.1f}%)")
        
        return df
    
    def filter_cs_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to keep only classified CS jobs.
        
        Args:
            df: DataFrame with domain column
            
        Returns:
            Filtered DataFrame with only classified CS jobs
        """
        initial_count = len(df)
        
        # Keep only jobs that were both validated as CS AND classified into a domain
        df_filtered = df[df['domain'].notna()].copy()
        
        # Remove temporary columns
        temp_cols = ['job_title_clean', 'job_description_clean', 'combined_text', 'is_cs_job']
        df_filtered = df_filtered.drop(columns=[col for col in temp_cols if col in df_filtered.columns])
        
        removed = initial_count - len(df_filtered)
        logger.info(f"Filtering results:")
        logger.info(f"  - Validated CS jobs: {initial_count:,}")
        logger.info(f"  - Successfully classified: {len(df_filtered):,}")
        logger.info(f"  - Removed (no domain match): {removed:,}")
        
        return df_filtered
    
    def save_refined_data(self, df: pd.DataFrame) -> None:
        """
        Save the refined dataset with domain classification.
        
        Args:
            df: Refined DataFrame with domain column
        """
        output_file = self.output_path / "cs_jobs.csv"
        
        logger.info(f"Saving refined data to {output_file}...")
        start_time = time.time()
        
        try:
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Saved {len(df):,} records in {time.time() - start_time:.2f} seconds")
            
            # Save a summary report
            self.save_summary_report(df)
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def save_summary_report(self, df: pd.DataFrame) -> None:
        """
        Generate and save a detailed summary report.
        
        Args:
            df: Refined DataFrame
        """
        report_file = self.output_path / "domain_classification_summary.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TECHSPHERE ANALYTICS - DOMAIN CLASSIFICATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total CS jobs after validation: {len(df):,}\n")
            f.write(f"Columns: {len(df.columns)}\n\n")
            
            f.write("CLASSIFICATION APPROACH:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Strict validation against non-tech keywords (chef, admin, etc.)\n")
            f.write("2. CS job indicator check (engineer, developer, etc.)\n")
            f.write("3. Domain-specific keyword matching\n")
            f.write("4. Priority-based conflict resolution\n\n")
            
            f.write("DOMAIN DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            domain_counts = df['domain'].value_counts()
            for domain, count in domain_counts.items():
                f.write(f"  {domain:20s}: {count:>10,} ({count/len(df)*100:>5.1f}%)\n")
            
            if 'year' in df.columns:
                f.write("\nYEAR DISTRIBUTION BY DOMAIN:\n")
                f.write("-" * 40 + "\n")
                year_domain_counts = df.groupby(['year', 'domain']).size().unstack(fill_value=0)
                f.write(year_domain_counts.to_string())
            
            if 'salary_min' in df.columns:
                f.write("\n\nSALARY STATISTICS BY DOMAIN:\n")
                f.write("-" * 40 + "\n")
                for domain in DOMAIN_PRIORITY:
                    domain_df = df[df['domain'] == domain]
                    if not domain_df.empty:
                        f.write(f"\n{domain}:\n")
                        f.write(f"  Count: {len(domain_df):,}\n")
                        if domain_df['salary_min'].notna().any():
                            f.write(f"  Avg Salary (Min): ${domain_df['salary_min'].mean():,.2f}\n")
                            f.write(f"  Median Salary: ${domain_df['salary_min'].median():,.2f}\n")
            
            f.write("\n\nSAMPLE JOB TITLES BY DOMAIN:\n")
            f.write("-" * 40 + "\n")
            for domain in DOMAIN_PRIORITY:
                domain_df = df[df['domain'] == domain]
                if not domain_df.empty:
                    f.write(f"\n{domain}:\n")
                    sample_titles = domain_df['job_title'].dropna().head(10).tolist()
                    for i, title in enumerate(sample_titles, 1):
                        f.write(f"  {i}. {title}\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete domain classification pipeline.
        
        Returns:
            Refined DataFrame with domain classifications
        """
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Prepare text and validate CS jobs
            df_prepared = self.prepare_text(self.df)
            
            if len(df_prepared) == 0:
                logger.warning("No CS jobs found after validation!")
                return pd.DataFrame()
            
            # Step 3: Apply classification
            df_classified = self.apply_domain_classification(df_prepared)
            
            # Step 4: Filter to classified jobs only
            df_refined = self.filter_cs_jobs(df_classified)
            
            # Step 5: Save results
            self.save_refined_data(df_refined)
            
            logger.info("Domain classification pipeline completed successfully!")
            
            return df_refined
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """
    Main execution function.
    """
    try:
        extractor = DomainExtractor(data_path="data/cleaned/cleaned_jobs.csv")
        
        logger.info("=" * 80)
        logger.info("TECHSPHERE ANALYTICS - DOMAIN CLASSIFICATION PIPELINE")
        logger.info("=" * 80)
        logger.info("Classification Strategy:")
        logger.info("  1. STRICT VALIDATION: Reject non-tech jobs (chef, admin, etc.)")
        logger.info("  2. CS INDICATOR CHECK: Must contain 'engineer', 'developer', etc.")
        logger.info("  3. DOMAIN MATCHING: Keyword-based classification")
        logger.info("  4. PRIORITY RESOLUTION: AI/ML > Data Science > Cybersecurity...")
        logger.info("=" * 80)
        
        refined_df = extractor.run_pipeline()
        
        # Display sample results
        if not refined_df.empty:
            print("\n" + "=" * 80)
            print("SAMPLE CLASSIFIED JOBS (First 20 rows):")
            print("=" * 80)
            
            # Show only relevant columns
            display_cols = ['job_title', 'domain', 'company', 'location', 'year']
            available_cols = [col for col in display_cols if col in refined_df.columns]
            
            print(refined_df[available_cols].head(20).to_string())
            
            # Also show what was filtered out
            print("\n" + "=" * 80)
            print("NOTE: Non-technical jobs (chef, administrator, etc.) have been filtered out")
            print("due to strict validation rules. This ensures data quality for analysis.")
            print("=" * 80)
        
        logger.info("Pipeline execution completed!")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()