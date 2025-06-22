import os
import json
import zipfile
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kaggle_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KaggleDatasetDownloader:
    def __init__(self, output_dir: str):
        """
        Initialize the Kaggle dataset downloader.
        
        Args:
            output_dir (str): Directory where the dataset will be saved.
        """
        self.output_dir = Path(output_dir)
        self.api = KaggleApi()
        
        # Validate and create output directory
        self._prepare_output_directory()
        
    def _prepare_output_directory(self) -> None:
        """Ensure the output directory exists and is writable."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = self.output_dir / '.permission_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logger.error(f"Failed to prepare output directory: {e}")
            raise
            
    def _validate_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are properly configured."""
        # First check if kaggle.json exists in the current directory
        current_dir_kaggle = Path('kaggle.json')
        if current_dir_kaggle.exists():
            logger.info("Found kaggle.json in current directory")
            return True
            
        # Then check the default location
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            logger.error("Kaggle credentials not found. Please ensure kaggle.json exists in ~/.kaggle/ or current directory")
            return False
            
        try:
            with open(kaggle_json) as f:
                json.load(f)  # Validate JSON
            return True
        except Exception as e:
            logger.error(f"Invalid kaggle.json file: {e}")
            return False
            
    def download_dataset(self, dataset_name: str, unzip: bool = True, delete_zip: bool = True) -> bool:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_name (str): Kaggle dataset identifier in format 'owner/dataset-name'
            unzip (bool): Whether to unzip the downloaded file
            delete_zip (bool): Whether to delete the zip file after extraction
            
        Returns:
            bool: True if download and processing succeeded, False otherwise
        """
        if not self._validate_kaggle_credentials():
            return False
            
        try:
            logger.info(f"Initializing Kaggle API connection...")
            self.api.authenticate()
            
            logger.info(f"Downloading dataset: {dataset_name}")
            
            # Download with progress tracking
            zip_path = self.output_dir / f"{dataset_name.replace('/', '_')}.zip"
            
            # Use the correct API method
            self.api.dataset_download_files(
                dataset=dataset_name,
                path=self.output_dir,
                quiet=False,
                force=True,
                unzip=False
            )
            
            # Look for the downloaded zip file
            downloaded_files = list(self.output_dir.glob("*.zip"))
            if downloaded_files:
                # Use the most recently downloaded file
                zip_path = max(downloaded_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Found downloaded file: {zip_path}")
            else:
                raise FileNotFoundError(f"No zip file found in {self.output_dir}")
                
            logger.info(f"Successfully downloaded dataset to {zip_path}")
            
            if unzip:
                self._unzip_file(zip_path, delete_zip)
                
            return True
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error occurred: {e}")
            if e.response.status_code == 403:
                logger.error("Authentication failed. Please check your Kaggle API token.")
            elif e.response.status_code == 404:
                logger.error("Dataset not found. Please check the dataset name.")
        except Exception as e:
            logger.error(f"An error occurred while downloading dataset: {e}")
            
        return False
        
    def _unzip_file(self, zip_path: Path, delete_zip: bool = True) -> None:
        """
        Unzip a downloaded dataset.
        
        Args:
            zip_path (Path): Path to the zip file
            delete_zip (bool): Whether to delete the zip file after extraction
        """
        try:
            logger.info(f"Extracting {zip_path.name}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total files for progress bar
                file_count = len(zip_ref.infolist())
                
                # Extract with progress tracking
                for file in tqdm(zip_ref.infolist(), desc="Extracting", unit="files"):
                    try:
                        zip_ref.extract(file, self.output_dir)
                    except Exception as e:
                        logger.warning(f"Failed to extract {file.filename}: {e}")
                        continue
                        
            logger.info(f"Extraction complete to {self.output_dir}")
            
            if delete_zip:
                zip_path.unlink()
                logger.info(f"Deleted zip file: {zip_path.name}")
                
        except zipfile.BadZipFile:
            logger.error(f"File is not a zip file or is corrupted: {zip_path}")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            

def main():
    # Configuration
    DATASET_NAME = "olegbaryshnikov/rsna-roi-512x512-pngs"  # Example dataset
    OUTPUT_DIR = "/Volumes/KODAK/folder 02/Brest_cancer_prediction/data/raw_data"
    
    try:
        downloader = KaggleDatasetDownloader(OUTPUT_DIR)
        success = downloader.download_dataset(DATASET_NAME)
        
        if success:
            logger.info("Dataset download and processing completed successfully!")
        else:
            logger.error("Dataset download failed.")
            exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        exit(1)
        

if __name__ == "__main__":
    main() 