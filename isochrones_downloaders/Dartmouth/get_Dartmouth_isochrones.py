#!/usr/bin/env python3
"""
Dartmouth Isochrone Downloader

A Python CLI application that interfaces with the Dartmouth web service
to download stellar isochrones with support for multiple metallicities
and comprehensive configuration logging.
"""
try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False
try:
    import requests
    _has_requests = True
except ImportError:
    _has_requests = False
try:
    import urllib3
    _has_urllib3 = True
except ImportError:
    _has_urllib3 = False
import os
import sys
import time
import numpy as np
import tarfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin
try:
    from bs4 import BeautifulSoup
    _has_bs4 = True
except ImportError:
    _has_bs4 = False
try:
    import pandas as pd
    _has_pandas = True
except ImportError:
    _has_pandas = False
import glob
import pathlib
import warnings

cwd = str(pathlib.Path(__file__).parent.resolve())

if _has_urllib3:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MIST_BASE_URL = "https://rcweb.dartmouth.edu"
MIST_SUBMIT_URL = f"{MIST_BASE_URL}/stellar/isolf_new.php"
MAX_ISOCHRONES_PER_REQUEST = 50

# ============================================================
# Data Classes
# ============================================================

@dataclass
class IsochroneRequest:
    """Represents a batch request to MIST."""
    ages: List[float]
    metallicities: List[float]

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

def modules_missing_error():
    if not _has_urllib3:
        return ModuleNotFoundError("This function uses urllib3, which is not installed.")
    if not _has_bs4:
        return ModuleNotFoundError("This function uses BeautifulSoup4, which is not installed.")
    if not _has_pandas:
        return ModuleNotFoundError("This function uses pandas, which is not installed.")
    if not _has_requests:
        return ModuleNotFoundError("This function uses requests, which is not installed.")

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

def partition_requests(ages: List[float],metallicities: List[float],use_log_age: bool) -> List[IsochroneRequest]:
    """
    Partition the age-metallicity grid into requests of at most
    MAX_ISOCHRONES_PER_REQUEST isochrones.

    The API accepts only a single metallicity per request.
    """
    if not ages or not metallicities:
        return []

    requests = []

    for met in metallicities:
        for i in range(0, len(ages), MAX_ISOCHRONES_PER_REQUEST):
            requests.append(
                IsochroneRequest(
                    ages=ages[i:i + MAX_ISOCHRONES_PER_REQUEST],
                    metallicities=[met],
                    use_log_age=use_log_age,
                )
            )

    return requests

# ============================================================
# Network Operations
# ============================================================

def submit_mist_request(session, 
                         alpha: int, helium: int,
                         ages: List[float],
                         metal: int,
                         phot_system: int,
                         request_num: int, total_requests: int) -> Optional[bytes]:
    """
    Submit a request to the MIST service using GET with query parameters.
    
    Args:
        session: requests Session object
        alpha: heavy element mixture
        version: version name
        logage: "log10" if in log or "linear" if linear
        minage: minimum age value
        maxage: maximum age value
        deltaage: age step
        metal: metallicity value(s)
        phot_system: photometric system
        request_num: current request number
        total_requests: total number of requests
    
    Returns:
        File content (bytes) or None on failure
    """
    
    try:
        # Build query parameters
        ages = '+'.join(ages.round(2).astype(str))

        url = MIST_SUBMIT_URL + f'?int=1&out=1&age={ages}&feh={metal}&hel={helium}&afe={alpha}&clr={phot_system}&flt=&bin=&imf=1&pls=&lnm=&lns='
        
        # Make GET request with query parameters
        r = requests.get(url, verify=False, timeout=6000)
        r.raise_for_status()
        
        # The response should contain the file data directly or a link
        if r.headers.get('content-type', '').startswith('text'):
            # HTML response - might contain error or redirect
            soup = BeautifulSoup(r.text, "html.parser")
            # Try to find a download link
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin('https://rcweb.dartmouth.edu/stellar/', href)
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
    files = glob.glob(str(output_path) + '/*.iso')
    
    df_dartmouth = pd.DataFrame(columns=('Age','MoH','G','G-BP','G-RP','BP-RP','M'))

    for _f in files:
        df = pd.DataFrame(columns=('Age','MoH','G','G-BP','G-RP','BP-RP','M'))
        with open(_f,'r') as f:
            for i,line in enumerate(f.readlines()):
                if i == 8:
                    continue
                if i == 3:
                    feh = float(line.split()[5])
                if len(line.split()) < 2:
                    continue
                if line[0] == '#':
                    if line[:2] == '#A':
                        if len(line.split()) == 3:
                            age = float(line.split()[1])
                        else:
                            age = float(line.split()[0].split('=')[-1])
                    continue
                M = float(line.split()[1])
                G = float(line.split()[5])
                BP = float(line.split()[6])
                RP = float(line.split()[7])
                BPRP = BP-RP
                df.loc[i] = [age,feh,G,BP,RP,BPRP,M]
        if len(df_dartmouth) == 0:
            df_dartmouth = df
        else:
            df_dartmouth = pd.concat([df_dartmouth,df])

    df_dartmouth.to_csv(str(output_path) + '/' + filename,index=False)

    js_dartmouth = {}
    for moh in df_dartmouth['MoH'].unique():
        js_dartmouth[str(moh)] = []
        for age in df_dartmouth['Age'].unique()[::2]:
            df = df_dartmouth[df_dartmouth['Age'] == age]
            js_iso = {'age':float(age)}
            js_iso['MG'] = df['G'].values.tolist()
            js_iso['BP-RP'] = df['BP-RP'].tolist()
            js_iso['M'] = df['M'].values.tolist()
            js_dartmouth[str(moh)].append(js_iso)

    with open(str(output_path) + '/' + filename.split('.')[0] + '.json','w') as f:
        f.write(str(js_dartmouth).replace('\'','"'))

# ============================================================
# Main Application
# ============================================================

def interactive_isochrones_downloader():
    """Main application entry point."""
    
    # Photometric system

    photo_options = [
        ["UBV(RI)c + 2MASS + Kepler",1],
        ["Washington + DDO51 + Stromgren",2],
        ["HST/WFPC2",3],
        ["HST/ACS-WFC",4],
        ["HST/ACS-HRC",5],
        ["HST/WFC3",6],
        ["Spitzer-IRAC",7],
        ["UKIDSS",8],
        ["WISE",9],
        ["CFHT-MegaCam ugriz",10],
        ["SDSS ugriz",11],
        ["PanSTARRS",12],
        ["SkyMapper",13],
        ["DECam ",14],
        ["BV(RI)c+Stromgren",15],
        ["Gaia DR2 Revised",16]
    ]

    phot_system = choose_option("Photometric system", photo_options,16)

    # ========================================================
    # Alpha Selection
    # ========================================================
    
    alpha = choose_option('[⍺/Fe]',[['-0.2',1],['0.0',2],['0.2',3],['0.4',4],['0.6',5],['0.8',6]],2)

    # ========================================================
    # Helium Selection
    # ========================================================
    
    helium = choose_option('He',[['Y=0.245+1.5*Z',1],['Y=0.33',2],['Y=0.40',3]],1)
    
    # ========================================================
    # Age Selection
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Age Selection")
    print("=" * 70)
    use_log_age = input("Use log(age/yr)? [Y/n] (default n): ").strip().lower() == "Y"
    
    if use_log_age:
        age_min = float(input("log(age/Gyr) min (t>=1 Gyr): "))
        age_max = float(input("log(age/Gyr) max: (t<=15 Gyr)"))
        age_step = float(input("log(age/Gyr) step: "))
        ages = np.logspace(age_min,age_max,int((age_max-age_min)/age_step)+1)
        
    else:
        age_min = float(input("age (Gyr) min (t>=1 Gyr): "))
        age_max = float(input("age (Gyr) max (t<=15 Gyr): "))
        age_step = float(input("age (Gyr) step: "))
        ages = np.linspace(age_min,age_max,int((age_max-age_min)/age_step)+1)

    if age_min >= age_max:
        raise UserWarning("Age max has to be bigger than age min.")
    
    # ========================================================
    # Metallicity Selection (Grid range)
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Metallicity Grid Range")
    print("=" * 70)

    met_min = float(input("[Fe/H] min (>-4): "))
    met_max = float(input("[Fe/H] max (<0.5): "))
    met_step = float(input("[Fe/H] step: "))
    if met_step == 0:
        met_step = 1
    
    # ========================================================
    # Output Directory
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Output Directory")
    print("=" * 70)
    output_dir = input("Output directory [./Dartmouth_isochrones]: ").strip() or "./Dartmouth_isochrones"

    download_isochrones(
        output_dir,
        ages,
        met_min,met_max,met_step,
        helium,
        alpha,
        phot_system
    )

def download_isochrones(
        output_dir,
        ages,
        met_min,met_max,met_step,
        helium=1,
        alpha=2,
        phot_system=16,
        use_tqdm=True
    ):
    if _has_urllib3 * _has_bs4 * _has_pandas * _has_requests == 0:
        raise modules_missing_error()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metallicities = generate_grid(met_min, met_max, met_step)
    
    files = glob.glob(str(output_path) + '/*.iso*')
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
    # Execute Requests
    # ========================================================
    
    requests_list = []
    for met in metallicities:
        requests_list.append(
            IsochroneRequest(
                ages=ages,
                metallicities=[met]
            )
        )

    session = requests.Session()
    downloaded_files: List[DownloadedFile] = []
    successful_downloads = 0
    
    loop = enumerate(requests_list,1)
    if _has_tqdm and use_tqdm and len(requests_list) > 1:
        loop = tqdm(loop,total=len(requests_list))

    for req_idx, request in loop:        
        # Format metallicity for query parameter
        content = submit_mist_request(
            session, alpha, helium, request.ages, request.metallicities[0], phot_system,
            req_idx, len(requests_list)
        )

        if content:
            filename = output_path / f'batch_{req_idx:03d}.iso'
            with open(filename, "wb") as f:
                f.write(content)

            downloaded_files.append(DownloadedFile(
                filename=filename,
                ages=request.ages,
                metallicities=request.metallicities,
                n_isochrones=request.n_isochrones,
                timestamp=datetime.now().isoformat()
            ))
        else:
            print(f"Failed to download batch {req_idx}")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    #merge_isochrones('Dartmouth_all.csv',output_path,cut_evolutionary_phases)

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
