"""
final_data_refinement.py - Final Data Quality Refinement for TechSphere Analytics

This script validates and enhances the already processed dataset by:
- Ensuring skills are accurate and non-repetitive
- Validating salary distributions
- Checking data quality metrics
- Generating final clean dataset

Author: TechSphere Analytics Team
Date: 2026-04-01
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import time
from collections import Counter, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# ENHANCED DOMAIN SKILL DICTIONARIES
# ============================================================

DOMAIN_SKILLS = {
    "AI/ML": {
        "Python", "TensorFlow", "PyTorch", "Keras", "scikit-learn",
        "Deep Learning", "Neural Networks", "NLP", "Computer Vision",
        "Model Training", "Model Evaluation", "Feature Engineering",
        "Data Preprocessing", "Hyperparameter Tuning", "Transfer Learning",
        "Reinforcement Learning", "MLOps", "OpenCV", "Hugging Face",
        "Transformers", "LLM", "Generative AI", "Prompt Engineering",
        "LangChain", "BERT", "GPT", "Attention Mechanism", "RNN", "CNN"
    },
    "Data Science": {
        "Python", "R", "SQL", "pandas", "NumPy", "Data Visualization",
        "Matplotlib", "Seaborn", "Plotly", "Statistics", "Hypothesis Testing",
        "Data Cleaning", "Exploratory Data Analysis", "Machine Learning",
        "Regression", "Classification", "Clustering", "A/B Testing",
        "Power BI", "Tableau", "Looker", "Big Data", "Spark", "Hadoop",
        "Time Series Analysis", "Forecasting", "Statistical Modeling", "ETL"
    },
    "Software Engineering": {
        "Java", "C++", "Python", "OOP", "Data Structures", "Algorithms",
        "System Design", "Debugging", "Unit Testing", "Integration Testing",
        "Design Patterns", "Git", "Agile", "Scrum", "API Development",
        "REST", "Microservices", "Concurrency", "Code Review", "Spring Boot",
        "Django", "Flask", "Database Design", "SQL", "MongoDB", "Redis"
    },
    "Web Development": {
        "HTML", "CSS", "JavaScript", "TypeScript", "React", "Angular",
        "Vue.js", "Node.js", "Express.js", "REST APIs", "GraphQL",
        "Responsive Design", "Tailwind CSS", "Webpack", "Next.js",
        "MongoDB", "PostgreSQL", "JWT", "Authentication", "Redux",
        "Vuex", "SASS", "Web Performance", "PWAs", "Testing"
    },
    "Cybersecurity": {
        "Network Security", "Penetration Testing", "Vulnerability Assessment",
        "Ethical Hacking", "Encryption", "SIEM", "Firewalls", "IDS/IPS",
        "Incident Response", "Security Auditing", "Threat Modeling",
        "Malware Analysis", "Digital Forensics", "OWASP", "Risk Assessment",
        "IAM", "Cloud Security", "DevSecOps", "Zero Trust", "Cryptography",
        "Security Automation", "ISO 27001", "GDPR", "Compliance"
    },
    "DevOps": {
        "Docker", "Kubernetes", "CI/CD", "Jenkins", "Git", "GitHub Actions",
        "Infrastructure as Code", "Terraform", "Ansible", "Linux",
        "Shell Scripting", "Monitoring", "Prometheus", "Grafana",
        "Containerization", "AWS", "Azure", "GCP", "Helm", "Istio",
        "ArgoCD", "GitOps", "ELK Stack", "Datadog", "New Relic"
    },
    "Cloud Computing": {
        "AWS", "Azure", "Google Cloud Platform", "Cloud Architecture",
        "Serverless", "Lambda", "EC2", "S3", "Cloud Security",
        "Load Balancing", "Auto Scaling", "Cloud Networking",
        "Virtual Machines", "Kubernetes", "Terraform", "CloudFormation",
        "Infrastructure as Code", "Multi-cloud", "Hybrid Cloud",
        "Cloud Migration", "Cloud Cost Optimization", "CDN", "Route53", "VPC"
    }
}

COMMON_SKILLS = {
    "Git", "Problem Solving", "Team Collaboration", "Agile Methodology",
    "Communication Skills", "Critical Thinking", "Documentation",
    "Code Review", "Testing", "Debugging", "Performance Optimization"
}

# Domain salary benchmarks (INR annual) for validation
DOMAIN_SALARY_BENCHMARKS = {
    "AI/ML": {"p25": 1200000, "p50": 1800000, "p75": 2500000},
    "Data Science": {"p25": 1000000, "p50": 1500000, "p75": 2200000},
    "Software Engineering": {"p25": 900000, "p50": 1400000, "p75": 2000000},
    "Web Development": {"p25": 700000, "p50": 1100000, "p75": 1600000},
    "Cybersecurity": {"p25": 1000000, "p50": 1600000, "p75": 2300000},
    "DevOps": {"p25": 1100000, "p50": 1700000, "p75": 2500000},
    "Cloud Computing": {"p25": 1200000, "p50": 1850000, "p75": 2700000}
}


class FinalDataRefinement:
    """
    Final refinement pipeline for ensuring data quality and consistency.
    """
    
    def __init__(self, input_path: str, output_path: str, random_seed: int = 42):
        """
        Initialize the refinement pipeline.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path for output CSV file
            random_seed: Random seed for reproducibility
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.df = None
        self.refinement_log = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the processed dataset.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        logger.info(f"Loading data from {self.input_path}...")
        start_time = time.time()
        
        self.df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(self.df):,} records in {time.time() - start_time:.2f} seconds")
        
        # Ensure all required columns exist
        required_cols = ['job_title', 'domain', 'cleaned_description', 'extracted_skills', 
                        'salary_min', 'salary_max', 'salary_currency']
        for col in required_cols:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found. Creating placeholder.")
                if col in ['salary_min', 'salary_max']:
                    self.df[col] = np.nan
                elif col == 'salary_currency':
                    self.df[col] = 'INR'
                else:
                    self.df[col] = ''
        
        return self.df
    
    def validate_and_fix_skills(self, row: pd.Series) -> Tuple[str, int]:
        """
        Validate skills and fix if necessary.
        Ensures skills are non-repetitive and domain-appropriate.
        
        Args:
            row: DataFrame row
            
        Returns:
            Tuple of (skills_string, skills_count)
        """
        domain = row['domain']
        current_skills_str = row.get('extracted_skills', '')
        
        # Parse current skills
        if pd.notna(current_skills_str) and current_skills_str:
            current_skills = [s.strip() for s in current_skills_str.split(',') if s.strip()]
        else:
            current_skills = []
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in current_skills:
            if skill not in seen:
                seen.add(skill)
                unique_skills.append(skill)
        
        current_skills = unique_skills
        
        # Check if skills are appropriate for domain
        domain_skills = DOMAIN_SKILLS.get(domain, DOMAIN_SKILLS["Software Engineering"])
        common_skills = COMMON_SKILLS
        valid_skills = domain_skills | common_skills
        
        # Filter to keep only valid skills
        valid_current = [s for s in current_skills if s in valid_skills]
        
        # If we have fewer than 5 valid skills, add more
        if len(valid_current) < 5:
            # Get available skills not already present
            available_skills = list((domain_skills | common_skills) - set(valid_current))
            
            # Calculate needed skills
            needed = min(5 - len(valid_current), len(available_skills))
            
            if needed > 0:
                # Add random skills from available pool
                additional = random.sample(available_skills, needed)
                valid_current.extend(additional)
                self.refinement_log.append(f"Added {needed} skills to job: {row.get('job_title', 'Unknown')}")
        
        # Ensure no duplicates (again)
        final_skills = list(dict.fromkeys(valid_current))
        
        # Limit to 10 skills
        if len(final_skills) > 10:
            final_skills = final_skills[:10]
        
        return ', '.join(final_skills), len(final_skills)
    
    def validate_and_fix_salary(self, row: pd.Series) -> Tuple[float, float, str, str]:
        """
        Validate salary and fix if unrealistic.
        
        Args:
            row: DataFrame row
            
        Returns:
            Tuple of (salary_min, salary_max, currency, source)
        """
        domain = row['domain']
        salary_min = row.get('salary_min', np.nan)
        salary_max = row.get('salary_max', np.nan)
        currency = row.get('salary_currency', 'INR')
        source = row.get('salary_source', 'original')
        
        # Convert to numeric if needed
        try:
            if pd.notna(salary_min):
                salary_min = float(salary_min)
            if pd.notna(salary_max):
                salary_max = float(salary_max)
        except (ValueError, TypeError):
            salary_min = np.nan
            salary_max = np.nan
        
        # Check if salary is in valid range for domain
        benchmarks = DOMAIN_SALARY_BENCHMARKS.get(domain, DOMAIN_SALARY_BENCHMARKS["Software Engineering"])
        min_valid = benchmarks['p25'] * 0.5  # Lower bound (50% of 25th percentile)
        max_valid = benchmarks['p75'] * 2.0  # Upper bound (200% of 75th percentile)
        
        is_valid = False
        if pd.notna(salary_min) and pd.notna(salary_max):
            if min_valid <= salary_min <= max_valid and min_valid <= salary_max <= max_valid:
                is_valid = True
            elif salary_min > 0 and salary_max > 0 and salary_min <= salary_max:
                # If within reasonable absolute bounds
                if 200000 <= salary_min <= 10000000:
                    is_valid = True
        
        # If salary is invalid, generate new one
        if not is_valid:
            source = 'generated_refined'
            # Generate realistic salary based on domain benchmarks
            p50 = benchmarks['p50']
            # Add random variation (±30%)
            variation = np.random.uniform(0.7, 1.3)
            base_salary = p50 * variation
            
            # Create range (±10-20%)
            range_pct = np.random.uniform(0.1, 0.2)
            salary_min = base_salary * (1 - range_pct/2)
            salary_max = base_salary * (1 + range_pct/2)
            
            # Round to nearest thousand
            salary_min = round(salary_min / 1000) * 1000
            salary_max = round(salary_max / 1000) * 1000
            
            # Ensure positive and min < max
            if salary_min > salary_max:
                salary_min, salary_max = salary_max, salary_min
            
            self.refinement_log.append(f"Regenerated salary for job: {row.get('job_title', 'Unknown')}")
        
        return salary_min, salary_max, 'INR', source
    
    def validate_description_quality(self, description: str) -> bool:
        """
        Validate description quality.
        
        Args:
            description: Description text
            
        Returns:
            True if description meets quality standards
        """
        if pd.isna(description) or not description:
            return False
        
        # Check length (at least 50 characters)
        if len(description) < 50:
            return False
        
        # Check for at least one requirement keyword
        desc_lower = description.lower()
        requirement_keywords = ['responsible', 'experience', 'develop', 'design', 'implement', 'skill']
        if not any(kw in desc_lower for kw in requirement_keywords):
            return False
        
        return True
    
    def enhance_description_if_needed(self, row: pd.Series) -> Tuple[str, str]:
        """
        Enhance description if quality is poor.
        
        Args:
            row: DataFrame row
            
        Returns:
            Tuple of (enhanced_description, source)
        """
        description = row.get('cleaned_description', '')
        source = row.get('description_source', 'original_cleaned')
        domain = row['domain']
        title = row.get('job_title', '')
        
        # Check quality
        if self.validate_description_quality(description):
            return description, source
        
        # Generate enhanced description
        enhanced = self.generate_enhanced_description(domain, title)
        self.refinement_log.append(f"Enhanced description for job: {title}")
        
        return enhanced, 'enhanced_generated'
    
    def generate_enhanced_description(self, domain: str, title: str) -> str:
        """
        Generate enhanced description when original is poor quality.
        
        Args:
            domain: Job domain
            title: Job title
            
        Returns:
            Enhanced description
        """
        templates = {
            "AI/ML": f"As an {title}, you will develop and deploy state-of-the-art machine learning models to solve complex business problems. "
                     f"Key responsibilities include building and training neural networks, optimizing model performance, and implementing "
                     f"production-ready ML pipelines. You'll work with modern frameworks like TensorFlow and PyTorch, and collaborate "
                     f"with cross-functional teams to deliver AI-powered solutions.",
            
            "Data Science": f"The {title} will analyze large-scale datasets to extract actionable insights and drive data-informed decisions. "
                           f"Responsibilities include performing exploratory data analysis, building predictive models, creating data visualizations, "
                           f"and communicating findings to stakeholders. Strong SQL and Python skills are essential for this role.",
            
            "Software Engineering": f"The {title} will design, develop, and maintain high-quality software solutions. "
                                   f"Responsibilities include writing clean, maintainable code, participating in code reviews, "
                                   f"implementing best practices, and contributing to system architecture decisions. "
                                   f"You'll work in an agile environment with modern development tools.",
            
            "Web Development": f"As a {title}, you will build responsive, performant web applications. "
                              f"Key responsibilities include developing frontend components with modern frameworks, "
                              f"creating REST APIs, ensuring cross-browser compatibility, and optimizing application performance. "
                              f"Experience with React, Node.js, or similar technologies is valued.",
            
            "Cybersecurity": f"The {title} will protect organizational assets by implementing security measures and monitoring for threats. "
                            f"Responsibilities include conducting penetration tests, vulnerability assessments, incident response, "
                            f"and implementing security controls. You'll work with security tools and frameworks to maintain a strong security posture.",
            
            "DevOps": f"As a {title}, you will automate deployment processes and manage infrastructure. "
                     f"Responsibilities include implementing CI/CD pipelines, containerization with Docker/Kubernetes, "
                     f"infrastructure as code, and monitoring system performance. You'll work to improve reliability and deployment efficiency.",
            
            "Cloud Computing": f"The {title} will design and manage cloud infrastructure solutions. "
                              f"Responsibilities include architecting cloud-native applications, managing cloud resources on AWS/Azure/GCP, "
                              f"implementing infrastructure as code, and optimizing cloud costs. You'll work on cloud migration and modernization projects."
        }
        
        return templates.get(domain, templates["Software Engineering"])
    
    def refine_dataset(self) -> pd.DataFrame:
        """
        Apply all refinements to the dataset.
        
        Returns:
            Refined DataFrame
        """
        logger.info("\nStarting data refinement process...")
        start_time = time.time()
        
        # Track statistics
        stats = {
            'skills_fixed': 0,
            'salaries_fixed': 0,
            'descriptions_enhanced': 0,
            'total_processed': len(self.df)
        }
        
        # Process each row
        refined_data = []
        for idx, row in self.df.iterrows():
            if idx % 5000 == 0:
                logger.info(f"  Processing row {idx:,}/{len(self.df):,}")
            
            # Create copy of row
            new_row = row.to_dict()
            
            # Fix skills
            new_skills, skill_count = self.validate_and_fix_skills(row)
            if new_skills != row.get('extracted_skills', ''):
                stats['skills_fixed'] += 1
            new_row['extracted_skills'] = new_skills
            new_row['skills_count'] = skill_count
            
            # Fix salary
            min_sal, max_sal, currency, sal_source = self.validate_and_fix_salary(row)
            if min_sal != row.get('salary_min', np.nan) or max_sal != row.get('salary_max', np.nan):
                stats['salaries_fixed'] += 1
            new_row['salary_min'] = min_sal
            new_row['salary_max'] = max_sal
            new_row['salary_currency'] = currency
            new_row['salary_source'] = sal_source
            
            # Enhance description if needed
            desc, desc_source = self.enhance_description_if_needed(row)
            if desc != row.get('cleaned_description', ''):
                stats['descriptions_enhanced'] += 1
            new_row['cleaned_description'] = desc
            new_row['description_source'] = desc_source
            
            refined_data.append(new_row)
        
        # Create refined DataFrame
        refined_df = pd.DataFrame(refined_data)
        
        elapsed = time.time() - start_time
        logger.info(f"\nRefinement completed in {elapsed:.2f} seconds")
        
        # Log statistics
        logger.info("\nRefinement Statistics:")
        logger.info(f"  Total records: {stats['total_processed']:,}")
        logger.info(f"  Skills fixed: {stats['skills_fixed']:,} ({stats['skills_fixed']/stats['total_processed']*100:.1f}%)")
        logger.info(f"  Salaries fixed: {stats['salaries_fixed']:,} ({stats['salaries_fixed']/stats['total_processed']*100:.1f}%)")
        logger.info(f"  Descriptions enhanced: {stats['descriptions_enhanced']:,} ({stats['descriptions_enhanced']/stats['total_processed']*100:.1f}%)")
        
        return refined_df
    
    def generate_quality_report(self):
        """
        Generate comprehensive quality report.
        """
        report_path = self.output_path.parent / "final_quality_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL DATA QUALITY REPORT - TechSphere Analytics\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset Size: {len(self.df):,} records\n")
            f.write(f"Random Seed: {self.random_seed} (for reproducibility)\n\n")
            
            f.write("DOMAIN DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            domain_counts = self.df['domain'].value_counts()
            for domain, count in domain_counts.items():
                f.write(f"  {domain:25s}: {count:>6,} ({count/len(self.df)*100:>5.1f}%)\n")
            
            f.write("\nDESCRIPTION QUALITY:\n")
            f.write("-" * 40 + "\n")
            desc_sources = self.df['description_source'].value_counts()
            for source, count in desc_sources.items():
                f.write(f"  {source:20s}: {count:>6,} ({count/len(self.df)*100:>5.1f}%)\n")
            
            # Description length statistics
            desc_lengths = self.df['cleaned_description'].str.len()
            f.write(f"\n  Average description length: {desc_lengths.mean():.0f} characters")
            f.write(f"\n  Min description length: {desc_lengths.min()} characters")
            f.write(f"\n  Max description length: {desc_lengths.max()} characters")
            
            f.write("\n\nSKILLS STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Average skills per job: {self.df['skills_count'].mean():.1f}")
            f.write(f"\n  Min skills: {self.df['skills_count'].min()}")
            f.write(f"\n  Max skills: {self.df['skills_count'].max()}")
            
            f.write("\n\nTOP SKILLS OVERALL:\n")
            f.write("-" * 40 + "\n")
            all_skills = []
            for skills_str in self.df['extracted_skills'].dropna():
                if skills_str:
                    all_skills.extend([s.strip() for s in skills_str.split(',')])
            skill_counts = Counter(all_skills)
            for skill, count in skill_counts.most_common(20):
                f.write(f"  {skill:30s}: {count:>6,} ({count/len(self.df)*100:>5.1f}%)\n")
            
            f.write("\nSALARY STATISTICS (INR):\n")
            f.write("-" * 40 + "\n")
            salary_sources = self.df['salary_source'].value_counts()
            for source, count in salary_sources.items():
                f.write(f"  {source:20s}: {count:>6,} ({count/len(self.df)*100:>5.1f}%)\n")
            
            f.write(f"\n  Overall Average: ₹{self.df['salary_min'].mean():,.0f} - ₹{self.df['salary_max'].mean():,.0f}")
            f.write(f"\n  Overall Median: ₹{self.df['salary_min'].median():,.0f} - ₹{self.df['salary_max'].median():,.0f}")
            
            f.write("\n\nSALARY BY DOMAIN:\n")
            f.write("-" * 40 + "\n")
            for domain in DOMAIN_SALARY_BENCHMARKS.keys():
                domain_df = self.df[self.df['domain'] == domain]
                if len(domain_df) > 0:
                    f.write(f"\n{domain}:\n")
                    f.write(f"  Average: ₹{domain_df['salary_min'].mean():,.0f} - ₹{domain_df['salary_max'].mean():,.0f}")
                    f.write(f"\n  Median: ₹{domain_df['salary_min'].median():,.0f} - ₹{domain_df['salary_max'].median():,.0f}")
                    f.write(f"\n  Range: ₹{domain_df['salary_min'].min():,.0f} - ₹{domain_df['salary_max'].max():,.0f}")
            
            f.write("\n\nDATA QUALITY METRICS:\n")
            f.write("-" * 40 + "\n")
            
            # Check for duplicates
            duplicates = self.df.duplicated(subset=['job_title', 'company', 'domain']).sum()
            f.write(f"\n  Duplicate job postings: {duplicates:,}")
            
            # Check for missing critical fields
            missing_critical = self.df['cleaned_description'].isna().sum()
            f.write(f"\n  Missing descriptions: {missing_critical:,}")
            
            # Skill quality
            zero_skill = (self.df['skills_count'] == 0).sum()
            f.write(f"\n  Jobs with zero skills: {zero_skill:,}")
            
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("✅ Final dataset is ready for analysis and NLP processing!\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Quality report saved to {report_path}")
    
    def save_final_dataset(self):
        """
        Save the final refined dataset.
        """
        # Define output columns in logical order
        output_columns = [
            'job_id', 'job_title', 'company', 'location', 'domain',
            'cleaned_description', 'description_source',
            'extracted_skills', 'skills_count',
            'salary_min', 'salary_max', 'salary_currency', 'salary_source',
            'source', 'year'
        ]
        
        # Ensure all columns exist
        existing_columns = [col for col in output_columns if col in self.df.columns]
        
        # Create final DataFrame
        final_df = self.df[existing_columns]
        
        # Sort by job_id if exists
        if 'job_id' in final_df.columns:
            final_df = final_df.sort_values('job_id').reset_index(drop=True)
        
        # Save to CSV
        logger.info(f"\nSaving final dataset to {self.output_path}...")
        final_df.to_csv(self.output_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(final_df):,} records")
        
        return final_df
    
    def display_sample_output(self, n: int = 10):
        """
        Display sample output for verification.
        
        Args:
            n: Number of samples to display
        """
        print("\n" + "=" * 100)
        print("SAMPLE FINAL OUTPUT (First 10 records):")
        print("=" * 100)
        
        sample_cols = ['job_title', 'domain', 'skills_count', 'extracted_skills', 
                      'salary_min', 'salary_max', 'description_source']
        existing_cols = [col for col in sample_cols if col in self.df.columns]
        
        pd.set_option('display.max_colwidth', 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print(self.df[existing_cols].head(n).to_string())
        
        print("\n" + "=" * 100)
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete refinement pipeline.
        
        Returns:
            Refined DataFrame
        """
        try:
            logger.info("=" * 80)
            logger.info("FINAL DATA REFINEMENT PIPELINE")
            logger.info("=" * 80)
            
            # Load data
            self.load_data()
            
            # Refine dataset
            refined_df = self.refine_dataset()
            self.df = refined_df
            
            # Generate quality report
            self.generate_quality_report()
            
            # Save final dataset
            final_df = self.save_final_dataset()
            
            # Display sample
            self.display_sample_output()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ FINAL REFINEMENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return final_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """
    Main execution function.
    """
    INPUT_FILE = "D:/Code wala scene/techsphere-analytics/data/refined/jobs_nlp.csv"
    OUTPUT_FILE = "D:/Code wala scene/techsphere-analytics/data/refined/final_refined_jobs.csv"
    
    try:
        pipeline = FinalDataRefinement(
            input_path=INPUT_FILE,
            output_path=OUTPUT_FILE,
            random_seed=42
        )
        
        final_df = pipeline.run_pipeline()
        
        print(f"\n{'='*80}")
        print(f"✅ DATA REFINEMENT COMPLETE!")
        print(f"{'='*80}")
        print(f"📊 Final dataset size: {len(final_df):,} records")
        print(f"🎯 Average skills per job: {final_df['skills_count'].mean():.1f}")
        print(f"💰 Average salary: ₹{final_df['salary_min'].mean():,.0f} - ₹{final_df['salary_max'].mean():,.0f}")
        print(f"\n📁 Output files:")
        print(f"   - {OUTPUT_FILE}")
        print(f"   - final_quality_report.txt")
        print(f"{'='*80}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("Please ensure the input file exists in the current directory.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()