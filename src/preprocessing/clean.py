"""
clean.py - Data Cleaning Module for TechSphere Analytics

This module cleans and standardizes multiple raw job posting datasets from the data/raw/ directory,
combining them into a single unified dataset for further NLP processing.

Author: TechSphere Analytics Team
Date: 2026-03-31
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobDataCleaner:
    """Main class for cleaning and standardizing job posting datasets."""
    
    # Common schema for unified dataset
    COMMON_SCHEMA = [
        'job_id', 'job_title', 'company', 'location', 
        'salary_min', 'salary_max', 'salary_currency', 
        'job_description', 'source', 'year'
    ]
    
    # Column mapping for each dataset
    COLUMN_MAPPINGS = {
        'adzuna_global_job_listings_2025.csv': {
            'job_id': 'job_id',
            'job_title': 'title',
            'company': 'company',
            'location': 'location_display',
            'salary_min': 'salary_min',
            'salary_max': 'salary_max',
            'salary_currency': None,  # Will infer from dataset or set default
            'job_description': 'description',
            'latitude': 'latitude',
            'longitude': 'longitude'
        },
        'internship.csv': {
            'job_id': None,  # No job_id in this dataset
            'job_title': 'Job Role',
            'company': 'Company Name',
            'location': None,  # No location field
            'salary_min': 'Stipend',
            'salary_max': 'Stipend',
            'salary_currency': None,
            'job_description': None,  # No description field
        },
        'job_postings.csv': {
            'job_id': 'job_id',
            'job_title': 'title',
            'company': None,  # Company might be derived from company_id
            'location': 'location',
            'salary_min': 'min_salary',
            'salary_max': 'max_salary',
            'salary_currency': 'currency',
            'job_description': 'description',
        },
        'job_skills.csv': {
            'job_id': None,  # Using job_link as identifier
            'job_title': None,  # No job title field
            'company': None,  # No company field
            'location': None,  # No location field
            'salary_min': None,
            'salary_max': None,
            'salary_currency': None,
            'job_description': None,  # No description field
        },
        'linkedin_job_postings.csv': {
            'job_id': None,  # Using job_link as identifier
            'job_title': 'job_title',
            'company': 'company',
            'location': 'job_location',
            'salary_min': None,
            'salary_max': None,
            'salary_currency': None,
            'job_description': None,  # No description, but have got_summary and got_ner
        },
        'placement.csv': {
            'job_id': None,
            'job_title': 'Role',
            'company': 'Company Name',
            'location': None,
            'salary_min': 'Base',
            'salary_max': 'CTC',
            'salary_currency': None,
            'job_description': None,
        },
        'postings.csv': {
            'job_id': 'job_id',
            'job_title': 'title',
            'company': 'company_name',
            'location': 'location',
            'salary_min': 'min_salary',
            'salary_max': 'max_salary',
            'salary_currency': 'currency',
            'job_description': 'description',
        }
    }
    
    def __init__(self, raw_data_path: str = "data/raw", cleaned_data_path: str = "data/cleaned"):
        """
        Initialize the cleaner with paths to raw and cleaned data directories.
        
        Args:
            raw_data_path: Path to directory containing raw CSV files
            cleaned_data_path: Path where cleaned data will be saved
        """
        self.raw_data_path = Path(raw_data_path)
        self.cleaned_data_path = Path(cleaned_data_path)
        
        # Create cleaned data directory if it doesn't exist
        self.cleaned_data_path.mkdir(parents=True, exist_ok=True)
        
        self.dfs = []  # List to store loaded dataframes
        
    def extract_year_from_filename(self, filename: str) -> int:
        """
        Extract year from filename.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Year as integer, defaults to 2025 if not found
        """
        # Try to find a 4-digit year in the filename
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match:
            return int(year_match.group())
        return 2025  # Default year
    
    def parse_salary_string(self, salary_str: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Parse salary string to extract min, max, and currency.
        
        Args:
            salary_str: String containing salary information
            
        Returns:
            Tuple of (salary_min, salary_max, currency)
        """
        if pd.isna(salary_str) or salary_str == '':
            return None, None, None
        
        salary_str = str(salary_str).strip()
        
        # Extract currency symbols
        currency_map = {
            '$': 'USD', '€': 'EUR', '£': 'GBP', '₹': 'INR',
            'USD': 'USD', 'EUR': 'EUR', 'GBP': 'GBP', 'INR': 'INR'
        }
        
        currency = None
        for symbol, code in currency_map.items():
            if symbol in salary_str:
                currency = code
                salary_str = salary_str.replace(symbol, '')
                break
        
        # Extract numbers (including ranges)
        # Look for patterns like "10000-20000", "10,000 - 20,000", etc.
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', salary_str)
        
        if not numbers:
            return None, None, currency
        
        # Convert to numeric values
        numeric_values = []
        for num in numbers:
            # Remove commas and convert to float
            clean_num = float(num.replace(',', ''))
            numeric_values.append(clean_num)
        
        if len(numeric_values) == 1:
            # Single value - treat as both min and max
            return numeric_values[0], numeric_values[0], currency
        elif len(numeric_values) >= 2:
            # Range - take min and max
            return numeric_values[0], numeric_values[1], currency
        
        return None, None, currency
    
    def load_data(self) -> List[pd.DataFrame]:
        """
        Load all CSV files from the raw data directory.
        
        Returns:
            List of loaded dataframes with metadata
        """
        csv_files = list(self.raw_data_path.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.raw_data_path}")
            return []
        
        for csv_file in csv_files:
            try:
                logger.info(f"Loading {csv_file.name}...")
                
                # Handle different quoting and encoding issues
                df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
                
                # Add source and year columns
                df['_source'] = csv_file.name
                df['_year'] = self.extract_year_from_filename(csv_file.name)
                
                self.dfs.append({
                    'df': df,
                    'source': csv_file.name,
                    'year': df['_year'].iloc[0]
                })
                
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {str(e)}")
                continue
        
        return self.dfs
    
    def standardize_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Standardize column names to the common schema.
        
        Args:
            df: Input dataframe
            source: Source filename
            
        Returns:
            Dataframe with standardized columns
        """
        if source not in self.COLUMN_MAPPINGS:
            logger.warning(f"No column mapping found for {source}. Skipping...")
            return None
        
        mapping = self.COLUMN_MAPPINGS[source]
        standardized_df = pd.DataFrame()
        
        # Map columns to common schema
        for target_col, source_col in mapping.items():
            if source_col and source_col in df.columns:
                standardized_df[target_col] = df[source_col]
            elif target_col not in standardized_df.columns:
                standardized_df[target_col] = np.nan
        
        # Add source and year if not already present
        standardized_df['source'] = source
        standardized_df['year'] = self.extract_year_from_filename(source)
        
        # Special handling for datasets without direct salary fields
        if source == 'internship.csv':
            # Parse stipend field
            if 'Stipend' in df.columns:
                stipend_values = []
                for stipend in df['Stipend']:
                    min_sal, max_sal, currency = self.parse_salary_string(str(stipend))
                    stipend_values.append((min_sal, max_sal, currency))
                
                standardized_df['salary_min'] = [x[0] for x in stipend_values]
                standardized_df['salary_max'] = [x[1] for x in stipend_values]
                standardized_df['salary_currency'] = [x[2] for x in stipend_values]
        
        elif source == 'placement.csv':
            # Parse Base and CTC fields
            for col in ['Base', 'CTC']:
                if col in df.columns:
                    for idx, val in df[col].items():
                        min_sal, max_sal, currency = self.parse_salary_string(str(val))
                        if col == 'Base' and min_sal:
                            standardized_df.at[idx, 'salary_min'] = min_sal
                        elif col == 'CTC' and max_sal:
                            standardized_df.at[idx, 'salary_max'] = max_sal
        
        elif source == 'job_skills.csv':
            # This dataset only has skills, minimal info
            standardized_df['job_id'] = df.get('job_link', np.nan)
            # Keep job_skills for potential future use
            if 'job_skills' in df.columns:
                standardized_df['skills'] = df['job_skills']
        
        # For datasets with company_id but no company name, try to join if possible
        if source == 'job_postings.csv' and 'company_id' in df.columns:
            # We could potentially join with a companies table if available
            # For now, keep as is
            pass
        
        return standardized_df
    
    def clean_text(self, text: str) -> str:
        """
        Clean text fields by normalizing whitespace and converting to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ''
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        # text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\'\"]', ' ', text)
        
        return text
    
    def apply_text_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text cleaning to all text fields.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with cleaned text fields
        """
        text_columns = ['job_title', 'company', 'location', 'job_description']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
                # Convert empty strings to NaN for easier handling
                df[col] = df[col].replace('', np.nan)
        
        return df
    
    def parse_salary_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and clean salary fields.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with parsed salary fields
        """
        # Ensure salary columns are numeric
        for col in ['salary_min', 'salary_max']:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # If salary_min > salary_max, swap them
                if 'salary_min' in df.columns and 'salary_max' in df.columns:
                    mask = df['salary_min'] > df['salary_max']
                    df.loc[mask, ['salary_min', 'salary_max']] = df.loc[mask, ['salary_max', 'salary_min']].values
        
        # Set default currency if missing
        if 'salary_currency' in df.columns:
            df['salary_currency'] = df['salary_currency'].fillna('USD')
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate job postings based on key fields.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with duplicates removed
        """
        # Define columns to use for deduplication
        dedup_columns = ['job_title', 'company', 'location']
        
        # Check which columns are actually present
        available_columns = [col for col in dedup_columns if col in df.columns]
        
        if available_columns:
            # Keep first occurrence, drop duplicates
            initial_count = len(df)
            df = df.drop_duplicates(subset=available_columns, keep='first')
            logger.info(f"Removed {initial_count - len(df)} duplicate rows")
        else:
            logger.warning("Not enough columns available for deduplication")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in critical fields.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with missing values handled
        """
        initial_count = len(df)
        
        # Drop rows where job_title is missing
        if 'job_title' in df.columns:
            df = df.dropna(subset=['job_title'])
            
        # Drop rows where job_description is missing (only if it's a critical field)
        # But for some datasets (internships, LinkedIn), description might not exist
        # So we only drop if job_description exists and is empty
        if 'job_description' in df.columns:
            # Don't drop if description is the only field missing, but log it
            missing_desc = df['job_description'].isna().sum()
            if missing_desc > 0:
                logger.info(f"Rows with missing job_description: {missing_desc}")
        
        logger.info(f"Removed {initial_count - len(df)} rows with missing job_title")
        
        return df
    
    def process_all_datasets(self) -> pd.DataFrame:
        """
        Main processing pipeline to clean and combine all datasets.
        
        Returns:
            Combined and cleaned dataframe
        """
        # Load all data
        logger.info("Loading datasets...")
        self.load_data()
        
        if not self.dfs:
            logger.error("No data loaded. Exiting...")
            return pd.DataFrame()
        
        # Process each dataset
        cleaned_dfs = []
        
        for data_dict in self.dfs:
            df = data_dict['df']
            source = data_dict['source']
            
            logger.info(f"Processing {source}...")
            
            # Standardize columns
            standardized_df = self.standardize_columns(df, source)
            
            if standardized_df is None or standardized_df.empty:
                continue
            
            # Apply text cleaning
            standardized_df = self.apply_text_cleaning(standardized_df)
            
            # Parse salary fields
            standardized_df = self.parse_salary_fields(standardized_df)
            
            cleaned_dfs.append(standardized_df)
        
        # Combine all dataframes
        if not cleaned_dfs:
            logger.error("No dataframes to combine")
            return pd.DataFrame()
        
        logger.info("Combining all datasets...")
        combined_df = pd.concat(cleaned_dfs, ignore_index=True, sort=False)
        
        # Remove duplicates
        logger.info("Removing duplicates...")
        combined_df = self.remove_duplicates(combined_df)
        
        # Handle missing values
        logger.info("Handling missing values...")
        combined_df = self.handle_missing_values(combined_df)
        
        # Ensure only common schema columns are kept
        for col in self.COMMON_SCHEMA:
            if col not in combined_df.columns:
                combined_df[col] = np.nan
        
        # Select only common schema columns in the correct order
        combined_df = combined_df[self.COMMON_SCHEMA]
        
        logger.info(f"Final dataset has {len(combined_df)} rows")
        logger.info(f"Columns: {list(combined_df.columns)}")
        
        return combined_df
    
    def save_cleaned_data(self, df: pd.DataFrame, filename: str = "cleaned_jobs.csv") -> None:
        """
        Save the cleaned dataframe to CSV.
        
        Args:
            df: Cleaned dataframe to save
            filename: Output filename
        """
        if df.empty:
            logger.error("No data to save")
            return
        
        output_path = self.cleaned_data_path / filename
        
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Cleaned data saved to {output_path}")
            
            # Save a summary report
            self.save_summary_report(df)
            
        except Exception as e:
            logger.error(f"Error saving cleaned data: {str(e)}")
    
    def save_summary_report(self, df: pd.DataFrame) -> None:
        """
        Generate and save a summary report of the cleaned dataset.
        
        Args:
            df: Cleaned dataframe
        """
        report_path = self.cleaned_data_path / "cleaning_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("TECHSPHERE ANALYTICS - DATA CLEANING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total records: {len(df):,}\n")
            f.write(f"Total columns: {len(df.columns)}\n\n")
            
            f.write("DATA SOURCES:\n")
            f.write("-" * 40 + "\n")
            source_counts = df['source'].value_counts()
            for source, count in source_counts.items():
                f.write(f"  {source}: {count:,} records\n")
            
            f.write("\nMISSING VALUES:\n")
            f.write("-" * 40 + "\n")
            for col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    f.write(f"  {col}: {missing:,} ({missing/len(df)*100:.1f}%)\n")
            
            f.write("\nSALARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for col in ['salary_min', 'salary_max']:
                if col in df.columns:
                    stats = df[col].describe()
                    f.write(f"  {col}:\n")
                    f.write(f"    Count: {stats['count']:,.0f}\n")
                    f.write(f"    Mean: ${stats['mean']:,.2f}\n")
                    f.write(f"    Min: ${stats['min']:,.2f}\n")
                    f.write(f"    Max: ${stats['max']:,.2f}\n")
            
            f.write("\nYEAR DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            year_counts = df['year'].value_counts().sort_index()
            for year, count in year_counts.items():
                f.write(f"  {year}: {count:,} records\n")
        
        logger.info(f"Summary report saved to {report_path}")


def main():
    """Main execution function."""
    try:
        # Initialize the cleaner
        cleaner = JobDataCleaner(
            raw_data_path="data/raw",
            cleaned_data_path="data/cleaned"
        )
        
        # Process all datasets
        logger.info("Starting data cleaning pipeline...")
        cleaned_df = cleaner.process_all_datasets()
        
        # Save the cleaned data
        if not cleaned_df.empty:
            cleaner.save_cleaned_data(cleaned_df)
            logger.info("Data cleaning pipeline completed successfully!")
            
            # Display first few rows of cleaned data for verification
            print("\n" + "="*60)
            print("PREVIEW OF CLEANED DATA (First 5 rows):")
            print("="*60)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(cleaned_df.head())
            
        else:
            logger.error("No data was processed. Pipeline failed.")
            
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()