#!/usr/bin/env python3
"""
BaSTI Isochrone Downloader

A Python CLI application that interfaces with the BaSTI web service
to download stellar isochrones with support for multiple metallicities
and comprehensive configuration logging.

Note: BaSTI can only query 150 isochrones per request, so large grids
are automatically partitioned. Each request returns a tar.gz file which
is extracted and consolidated.
"""
try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False
import requests
import urllib3
import json
import os
import sys
import math
import time
import tarfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urljoin, urlencode
from bs4 import BeautifulSoup
import pandas as pd
import glob
import gzip
import pathlib

cwd = str(pathlib.Path(__file__).parent.resolve())

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASTI_BASE_URL = "http://basti-iac.oa-abruzzo.inaf.it"
BASTI_SUBMIT_URL = f"{BASTI_BASE_URL}/cgi-bin/isoc-get.py"
TEXTDATA_DIR = f"{BASTI_BASE_URL}/textdataISOCS"
MAX_ISOCHRONES_PER_REQUEST = 150


# ============================================================
# Data Classes
# ============================================================

@dataclass
class IsochroneRequest:
    """Represents a batch request to BaSTI."""
    ages: List[float]
    metallicities: List[float]
    use_log_age: bool

    @property
    def n_isochrones(self) -> int:
        """Total number of isochrones in this request."""
        return len(self.ages) * len(self.metallicities)


@dataclass
class DownloadedFile:
    """Information about a downloaded isochrone file."""
    filename: str
    ages: List[float]
    metallicities: List[float]
    use_log_age: bool
    n_isochrones: int
    timestamp: str


@dataclass
class RunConfiguration:
    """Complete run configuration for reproducibility."""
    timestamp: str
    heavy_element_mixture: str
    grid: str
    photometric_system: str
    age_min: float
    age_max: float
    age_step: float
    use_log_age: bool
    metallicity_mode: str  # 'Z' or '[Fe/H]'
    metallicity_min: float
    metallicity_max: float
    metallicity_step: float
    metallicities: List[float]
    output_directory: str
    total_requests: int
    total_files: int
    total_isochrones: int


# ============================================================
# Helper Functions
# ============================================================

def load_form_from_html_file(html_file: str) -> BeautifulSoup:
    """Load the BaSTI form from a local HTML file."""
    with open(html_file, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), "html.parser")


def choose_option(title: str, options: List[Tuple[str, str]], 
                  default_value: Optional[str] = None) -> str:
    """
    Present an interactive menu with options.
    
    Args:
        title: Menu title
        options: List of (label, value) tuples
        default_value: The default value to mark and accept on Enter
    
    Returns:
        Selected value
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    default_index = None
    for i, (label, value) in enumerate(options, start=1):
        marker = " [DEFAULT]" if value == default_value else ""
        print(f"{i:3d}) {label}{marker}")
        if value == default_value:
            default_index = i

    while True:
        try:
            user_input = input("\nChoice (press Enter for default): ").strip()

            # If user pressed Enter and there's a default, use it
            if user_input == "" and default_value is not None:
                for label, value in options:
                    if value == default_value:
                        print(f"Using: {label}")
                        return value

            # Otherwise, parse the choice
            if user_input == "":
                print("No default available. Please enter a number.")
                continue

            choice = int(user_input)
            if 1 <= choice <= len(options):
                return options[choice - 1][1]

        except (ValueError, IndexError):
            pass

        print("Invalid choice.")


def load_text_file(filename: str) -> List[Tuple[str, str]]:
    """
    Load options from a .txt file in the textdataISOCS directory.
    
    Format: each line is "label:value" or similar
    """
    options = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Try to parse as "label:value" or just use the whole line
                label = line.split('>')[1].split('<')[0]
                value = line.split('"')[1]
                options.append((label, value))
    except FileNotFoundError:
        print(f"Warning: File not found: {filename}")
    
    return options


def generate_grid(min_val: float, max_val: float, step: float) -> List[float]:
    """
    Generate a grid of values from min to max with given step.
    
    Returns:
        List of grid points
    """
    if step <= 0:
        return [min_val]
    
    grid = []
    current = min_val
    while current <= max_val + 1e-10:  # Small tolerance for floating point
        grid.append(current)
        current += step
    
    return grid


def load_isc_file(filename: str) -> List[Tuple[str, str]]:
    """
    Load options from an .isc file.
    
    Each line is an option (filename without extension).
    """
    options = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Remove any file extensions if present
                label = line.split('>')[1].split('<')[0]
                value = line.split('"')[1]
                options.append((label, value))
    except FileNotFoundError:
        print(f"Warning: File not found: {filename}")
    
    return options


def partition_requests(ages: List[float], metallicities: List[float],
                      use_log_age: bool) -> List[IsochroneRequest]:
    """
    Partition the age-metallicity grid into requests of ≤150 isochrones.
    
    Strategy: iterate through metallicities and create age chunks for each.
    
    Returns:
        List of IsochroneRequest objects
    """
    total = len(ages) * len(metallicities)
    
    if total <= MAX_ISOCHRONES_PER_REQUEST:
        return [IsochroneRequest(
            ages=ages,
            metallicities=metallicities,
            use_log_age=use_log_age
        )]
    
    requests = []
    
    # Partition by age chunks within each metallicity
    ages_per_request = MAX_ISOCHRONES_PER_REQUEST // len(metallicities)
    
    if ages_per_request == 0:
        # More metallicities than max per request; split metallicities instead
        mets_per_request = MAX_ISOCHRONES_PER_REQUEST // len(ages)
        if mets_per_request == 0:
            mets_per_request = 1  # At least one metallicity per request
        
        for i in range(0, len(metallicities), mets_per_request):
            met_chunk = metallicities[i:i+mets_per_request]
            requests.append(IsochroneRequest(
                ages=ages,
                metallicities=met_chunk,
                use_log_age=use_log_age
            ))
    else:
        # More ages than max per request; split ages for each metallicity
        for met in metallicities:
            for i in range(0, len(ages), ages_per_request):
                age_chunk = ages[i:i+ages_per_request]
                requests.append(IsochroneRequest(
                    ages=age_chunk,
                    metallicities=[met],
                    use_log_age=use_log_age
                ))
    
    return requests


# ============================================================
# Network Operations
# ============================================================

def submit_basti_request(session: requests.Session, 
                         alpha: str, grid: str, age_range: str, metal: str,
                         phot_system: str,
                         request_num: int, total_requests: int) -> Optional[bytes]:
    """
    Submit a request to the BaSTI service using GET with query parameters.
    
    Args:
        session: requests Session object
        alpha: heavy element mixture
        grid: grid name
        age_range: age range as "min--max,step" or single age
        metal: metallicity value(s)
        phot_system: photometric system
        request_num: current request number
        total_requests: total number of requests
    
    Returns:
        File content (bytes) or None on failure
    """
    
    try:
        # Build query parameters
        params = {
            "alpha": alpha,
            "grid": grid,
            "metal": "None",
            "imetal": "",
            "imetalh": metal,
            "iage": age_range,
            "bcsel": phot_system,
        }
        
        # Make GET request with query parameters
        r = requests.get(BASTI_SUBMIT_URL, params=params, verify=False, timeout=120)
        r.raise_for_status()
        
        # The response should contain the file data directly or a link
        if r.headers.get('content-type', '').startswith('text'):
            # HTML response - might contain error or redirect
            soup = BeautifulSoup(r.text, "html.parser")
            # Try to find a download link
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin(BASTI_BASE_URL, href)
                r2 = requests.get(full_url, verify=False, timeout=120)
                r2.raise_for_status()
                return r2.content
            error = soup.find("h1")
            if error:
                print(f"Error: {error.text}")
            return None
        else:
            # Direct file download
            return r.content
    
    except Exception as e:
        print(f"Error submitting request: {e}")
        return None


def extract_tar_gz(tar_path: Path, extract_to: Path) -> Optional[Path]:
    """
    Extract tar.gz file and return the directory containing the isochrones.
    
    Args:
        tar_path: path to tar.gz file
        extract_to: directory to extract to
    
    Returns:
        Path to extracted directory or None on failure
    """
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to,filter='fully_trusted')
        
        # Find the extracted directory (usually there's one main folder)
        extracted_items = list(extract_to.glob("*"))
        extracted_items = [item for item in extracted_items if item.name != tar_path.name]
        
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            extracted_dir = extracted_items[0]
            return extracted_dir
        else:
            print(f"Warning: Unexpected extraction structure. Found {len(extracted_items)} items")
            return extract_to
    
    except Exception as e:
        print(f"Error extracting tar.gz: {e}")
        return None


def move_files_up(source_dir: Path, target_dir: Path) -> int:
    """
    Move all files from source_dir to target_dir, removing source_dir after.
    
    Returns:
        Number of files moved
    """
    count = 0
    try:
        # Move all files
        for file_path in source_dir.glob("*"):
            target_path = target_dir / file_path.name
            
            # Handle duplicates by appending a suffix
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.move(str(file_path), str(target_path))
            count += 1
        
        # Remove empty source directory
        source_dir.rmdir()
        return count
    
    except Exception as e:
        print(f"Error moving files: {e}")
        return 0

def merge_isochrones(filename,output_path):
    files = glob.glob(str(output_path) + '/*.isc*')
    df_basti = pd.DataFrame()
    js_basti = {}

    for f in files:
        with open(f) as _f:
            lines = _f.readlines()
            metallicity = float(lines[4].split()[6])
            age = lines[4].split()[-1]

        df = pd.read_csv(f,
                            comment='#',
                            sep=r'\s+',
                            usecols=[0,4,5,6],
                            names=['Mass_init','G','G-BP','G-RP']
                        )
        
        df['Age'] = float(age)/1e3
        df['MoH'] = metallicity
        df['BP-RP'] = df['G-BP'] - df['G-RP']
        df['M'] = df['Mass_init']
        df.drop(columns=['G-BP','G-RP','Mass_init'],inplace=True)

        js_iso = {'age':float(age)/1e3}
        js_iso['MG'] = df['G'].values.tolist()
        js_iso['BP-RP'] = df['BP-RP'].tolist()
        js_iso['M'] = df['M'].values.tolist()
        if str(metallicity) in js_basti:
            js_basti[str(metallicity)].append(js_iso)
        else:
            js_basti[str(metallicity)] = [js_iso]

        df_basti = pd.concat([df_basti,df])

    df_basti.to_csv(str(output_path) + '/' + filename,index=False)

    with open(str(output_path) + '/' + filename.split('.')[0] + '.json','w') as f:
        f.write(str(js_basti).replace('\'','"'))


# ============================================================
# Main Application
# ============================================================

def interactive_isochrones_downloader():
    """Main application entry point."""

    # Load HTML form file
    html_file = cwd + '/http___basti-iac.oa-abruzzo.inaf.it_isocs.html'
    if not os.path.exists(html_file):
        print(f"Error: File {html_file} not found")
        sys.exit(1)
    
    soup = load_form_from_html_file(html_file)
    
    # ========================================================
    # Interactive Choices
    # ========================================================
    
    # Heavy element mixture (ALPHA)
    alpha_file = cwd + '/files/alphaoptions.txt'
    if os.path.exists(alpha_file):
        alpha_options = load_text_file(alpha_file)
        if alpha_options:
            alpha = choose_option("Heavy element mixture", alpha_options, alpha_options[1][1] if alpha_options else None)
        else:
            alpha = input("Heavy element mixture (ALPHA value): ").strip()
    else:
        print(f"Warning: {alpha_file} not found")
        alpha = input("Heavy element mixture (ALPHA value): ").strip()
    
    # Grid selection (depends on ALPHA)
    grid_file = cwd + f'/textdataISOCS/{alpha}.isc'
    if os.path.exists(grid_file):
        grid_options = load_isc_file(grid_file)
        if grid_options:
            grid = choose_option("Available grids", grid_options, grid_options[3][1] if grid_options else None)
            grid = grid.split('"')[1] if '"' in grid else grid
        else:
            grid = input("Grid name: ").strip()
    else:
        print(f"Warning: {grid_file} not found")
        grid = input("Grid name: ").strip()
    
    # Photometric system
    bolcor_file = cwd + '/files/bolcoroptions.txt'
    if os.path.exists(bolcor_file):
        bolcor_options = load_text_file(bolcor_file)
        if bolcor_options:
            phot_system = choose_option("Photometric system", bolcor_options, bolcor_options[8][1] if bolcor_options else None)
            phot_system = phot_system.split('"')[1] if '"' in phot_system else phot_system
        else:
            phot_system = input("Photometric system: ").strip()
    else:
        print(f"Warning: {bolcor_file} not found")
        phot_system = input("Photometric system: ").strip()
    
    # ========================================================
    # Age Selection
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Age Selection")
    print("=" * 70)
    use_log_age = input("Use log(age/yr)? [Y/n] (default n): ").strip().lower() == "Y"
    
    if use_log_age:
        age_min = float(input("log(age/yr) min: "))
        age_max = float(input("log(age/yr) max: "))
        age_step = float(input("log(age/yr) step: "))
    else:
        age_min = float(input("age (yr) min: "))
        age_max = float(input("age (yr) max: "))
        age_step = float(input("age (yr) step: "))
    
    # ========================================================
    # Metallicity Selection (Grid range)
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Metallicity Grid Range")
    print("=" * 70)

    met_min = float(input("[Fe/H] min: "))
    met_max = float(input("[Fe/H] max: "))
    met_step = float(input("[Fe/H] step: "))
    
    # ========================================================
    # Output Directory
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Output Directory")
    print("=" * 70)
    output_dir = input("Output directory [./BaSTI_isochrones]: ").strip() or "./BaSTI_isochrones"
    
    print("\n" + "=" * 70)

    input([output_dir,
        use_log_age,
        age_min,age_max,age_step,
        met_min,met_max,met_step,
        alpha,
        grid,
        phot_system])

    download_isochrones(
        output_dir,
        use_log_age,
        age_min,age_max,age_step,
        met_min,met_max,met_step,
        alpha,
        grid,
        phot_system
    )

def download_isochrones(
        output_dir,
        use_log_age,
        age_min,age_max,age_step,
        met_min,met_max,met_step,
        alpha="P00",
        grid="P00O1D1E1Y247",
        phot_system="GAIA-DR3",
        use_tqdm=True
    ):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ages = generate_grid(age_min, age_max, age_step)
    metallicities = generate_grid(met_min, met_max, met_step)
    
    files = glob.glob(str(output_path) + '/*.isc*')
    for f in files:
        if os.path.isdir(f):
            continue
        os.remove(f)
    if os.path.isdir(str(output_path) + '/temp'):
        shutil.rmtree(str(output_path) + '/temp')

    # Create temporary directory for tar.gz extraction
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # ========================================================
    # Partition Requests
    # ========================================================
    
    requests_list = partition_requests(ages, metallicities, use_log_age)
    
    # ========================================================
    # Execute Requests
    # ========================================================
    
    session = requests.Session()
    downloaded_files: List[DownloadedFile] = []
    successful_downloads = 0
    
    loop = enumerate(requests_list,1)
    if _has_tqdm and use_tqdm and len(requests_list) > 1:
        loop = tqdm(loop,total=len(requests_list))

    for req_idx, request in loop:
        # Format age range for query parameter
        if len(request.ages) == 1:
            age_range = str(request.ages[0])
        else:
            age_range = f"{request.ages[0]}--{request.ages[-1]},{request.ages[1] - request.ages[0] if len(request.ages) > 1 else 0}"
        
        # Format metallicity for query parameter
        metal_str = ",".join(f"{met:.10f}" for met in request.metallicities)
        
        content = submit_basti_request(
            session, alpha, grid, age_range, metal_str, phot_system,
            req_idx, len(requests_list)
        )
        
        if content:
            # Save tar.gz file temporarily
            tar_filename = f"batch_{req_idx:03d}.tar.gz"
            tar_file = temp_dir / tar_filename
            
            with open(tar_file, "wb") as f:
                f.write(content)
            
            # Extract tar.gz
            batch_extract_dir = temp_dir / f"batch_{req_idx:03d}"
            batch_extract_dir.mkdir(exist_ok=True)
            
            extracted_dir = extract_tar_gz(tar_file, batch_extract_dir)
            
            if extracted_dir:
                # Move files up to main output directory
                files_moved = move_files_up(extracted_dir, output_path)
                successful_downloads += request.n_isochrones
                
                # Record file info
                downloaded_files.append(DownloadedFile(
                    filename=tar_filename,
                    ages=request.ages,
                    metallicities=request.metallicities,
                    use_log_age=use_log_age,
                    n_isochrones=request.n_isochrones,
                    timestamp=datetime.now().isoformat()
                ))
                
                tar_file.unlink()
                
                time.sleep(1.0)
            else:
                print(f"Failed to extract batch {req_idx}")
        else:
            print(f"Failed to download batch {req_idx}")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    merge_isochrones('basti_all.csv',output_path)

if __name__ == "__main__":
    try:
        interactive_isochrones_downloader()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
