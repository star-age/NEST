#!/usr/bin/env python3
"""
MIST Isochrone Downloader

A Python CLI application that interfaces with the MIST web service
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
import zipfile
import pathlib
import warnings

cwd = str(pathlib.Path(__file__).parent.resolve())

if _has_urllib3:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MIST_BASE_URL = "https://mist.science"
MIST_SUBMIT_URL = f"{MIST_BASE_URL}/iso_form.php"
MAX_ISOCHRONES_PER_REQUEST = 150

# ============================================================
# Data Classes
# ============================================================

@dataclass
class IsochroneRequest:
    """Represents a batch request to MIST."""
    age_min: float
    age_max: float
    age_step: float
    metallicities: List[float]
    use_log_age: bool

    @property
    def n_isochrones(self) -> int:
        """Total number of isochrones in this request."""
        return int((self.age_max-self.age_min)/self.age_step) * len(self.metallicities)


@dataclass
class DownloadedFile:
    """Information about a downloaded isochrone file."""
    filename: str
    age_min: float
    age_max: float
    age_step: float
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
                         alpha: str, version: str,
                         logage: str, minage: float, maxage: float, deltaage: float,
                         metal: str,
                         phot_system: str,
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
        params = {
            'version':	version,
            'v_div_vcrit':	"vvcrit0.0",
            'age_scale':	logage,
            'age_value':	"",
            'age_type':	"range",
            'age_range_low':	minage,
            'age_range_high':	maxage,
            'age_range_delta':	deltaage,
            'age_list':	"",
            'FeH_value':	metal,
            'alpha_value':	alpha,
            'output_option':	"photometry",
            'output':	phot_system,
            'Av_value':	"0"
        }
        
        # Make GET request with query parameters
        r = requests.post(MIST_SUBMIT_URL, data=params, verify=False, timeout=6000)
        r.raise_for_status()
        
        # The response should contain the file data directly or a link
        if r.headers.get('content-type', '').startswith('text'):
            # HTML response - might contain error or redirect
            soup = BeautifulSoup(r.text, "html.parser")
            # Try to find a download link
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin(MIST_BASE_URL, href)
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

def extract_zip(zip_path: Path, extract_to: Path) -> Optional[Path]:
    """
    Extract .zip file and return the directory containing the isochrones.
    
    Args:
        zip_path: path to .zip file
        extract_to: directory to extract to
    
    Returns:
        Path to extracted directory or None on failure
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Find the extracted directory (usually there's one main folder)
        extracted_items = list(extract_to.glob("*"))
        extracted_items = [item for item in extracted_items if item.name != zip_path.name]
        
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            extracted_dir = extracted_items[0]
            return extracted_dir
        else:
            print(f"Warning: Unexpected extraction structure. Found {len(extracted_items)} items")
            return extract_to
    
    except Exception as e:
        print(f"Error extracting .zip: {e}")
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

def merge_isochrones(filename,output_path,cut_evolutionary_phases=True):
    files = glob.glob(str(output_path) + '/*.iso.*')
    df_mist = pd.DataFrame()
    
    for f in files:
        with open(f,'r') as _f:
            header = _f.readlines()[12].split()[1:]

        df = pd.read_csv(
            f,
            comment="#",
            sep=r"\s+",
            names=header
        )

        if 'log10_isochrone_age_yr' in df.columns:
            df['Age'] = 10**df['log10_isochrone_age_yr']/10**9
        else:
            df['Age'] = 10**df['isochrone_age_yr']/10**9
        df['G'] = df['Gaia_G_EDR3']
        df['G-BP'] = df['Gaia_BP_EDR3']
        df['G-RP'] = df['Gaia_RP_EDR3']
        df['BP-RP'] = df['Gaia_BP_EDR3'] - df['Gaia_RP_EDR3']
        df['MoH'] = df['[Fe/H]_init']
        df['M'] = df['initial_mass'].astype(float)

        if cut_evolutionary_phases:
            df = df[(df['phase'] != 6) & (df['phase'] != 5) & (df['Age'] > 1e-1)]
        df_mist = pd.concat([df_mist,df])

    df_mist.to_csv(str(output_path) + '/' + filename,index=False)

    js_mist = {}
    for moh in df_mist['MoH'].unique():
        js_mist[str(moh)] = []
        df = df_mist[df_mist['MoH'] == moh]
        for age in df['Age'].unique():
            dff = df[df['Age'] == age]
            js_iso = {'age':float(age)}
            js_iso['MG'] = dff['G'].values.tolist()
            js_iso['BP-RP'] = dff['BP-RP'].tolist()
            js_iso['M'] = dff['M'].values.tolist()
            js_mist[str(moh)].append(js_iso)    

    with open(str(output_path) + '/' + filename.split('.')[0] + '.json','w') as f:
        f.write(str(js_mist).replace('\'','"'))

# ============================================================
# Main Application
# ============================================================

def interactive_isochrones_downloader():
    """Main application entry point."""
    
    #Version
    version = choose_option("MIST version",[['1.2','MIST1'],['2.5','MIST2']],'MIST2')
    
    # Photometric system

    photo_options = [
        ["CFHT/MegaCam","CFHTugriz"],
        ["DECam","DECam"],
        ["HST ACS/HRC","HST_ACS_HRC"],
        ["HST ACS/SBC","HST_ACS_SBC"],
        ["HST ACS/WFC","HST_ACS_WFC"],
        ["HST WFC3/UVIS+IR","HST_WFC3"],
        ["HST WFPC2","HST_WFPC2"],
        ["INT / IPHAS","IPHAS"],
        ["GALEX","GALEX"],
        ["JWST NIRCAM","JWST"],
        ["JWST NIRISS","NIRISS"],
        ["PanSTARRS","PanSTARRS"],
        ["Roman (formerly WFIRST)","Roman"],
        ["Rubin / LSST","LSST"],
        ["SDSS","SDSSugriz"],
        ["SkyMapper","SkyMapper"],
        ["Spitzer IRAC","SPITZER"],
        ["S-PLUS","SPLUS"],
        ["Subaru Hyper Suprime-Cam","HSC"],
        ["Swift","Swift"],
        ["UBV(RI)c + 2MASS + Kepler + Hipparcos + Gaia + Tess","UBVRIplus"],
        ["UKIDSS","UKIDSS"],
        ["UVIT","UVIT"],
        ["VISTA","VISTA"],
        ["Washington + Strömgren + DDO51","WashDDOuvby"],
        ["WISE","WISE"]
    ]

    phot_system = choose_option("Photometric system", photo_options,'UBVRIplus')

    # ========================================================
    # Alpha Selection
    # ========================================================
    
    if version == '2.5':
        alpha = 'p0'
    else:
        alpha = choose_option('[⍺/Fe]',[['-0.2','m2'],['0.0','p0'],['0.2','p2'],['0.4','p4'],['0.6','p6']],'p0')
    
    # ========================================================
    # Age Selection
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Age Selection")
    print("=" * 70)
    use_log_age = input("Use log(age/yr)? [Y/n] (default n): ").strip().lower() == "Y"
    
    if use_log_age:
        age_min = float(input("log(age/yr) min (log(t)>5): "))
        age_max = float(input("log(age/yr) max: (log(t)<10.3)"))
        age_step = float(input("log(age/yr) step: "))
    else:
        age_min = float(input("age (yr) min (log(t)>5): "))
        age_max = float(input("age (yr) max (log(t)<10.3): "))
        age_step = float(input("age (yr) step: "))

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
    output_dir = input("Output directory [./MIST_isochrones]: ").strip() or "./MIST_isochrones"

    download_isochrones(
        output_dir,
        use_log_age,
        age_min,age_max,age_step,
        met_min,met_max,met_step,
        version,
        alpha,
        phot_system
    )

def download_isochrones(
        output_dir,
        use_log_age,
        age_min,age_max,age_step,
        met_min,met_max,met_step,
        version='2.5',
        alpha="p0",
        phot_system="UBVRIplus",
        cut_evolutionary_phases=False,
        use_tqdm=True
    ):
    if _has_urllib3 * _has_bs4 * _has_pandas * _has_requests == 0:
        raise modules_missing_error()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_log_age:
        use_log_age = 'log10'
    else:
        use_log_age = 'linear'

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
                age_min=age_min,
                age_max=age_max,
                age_step=age_step,
                metallicities=[met],
                use_log_age=use_log_age,
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
            session, alpha, version, request.use_log_age, request.age_min, request.age_max, request.age_step, request.metallicities[0], phot_system,
            req_idx, len(requests_list)
        )

        if content:
            # Save tar.gz file temporarily
            zip_filename = f"batch_{req_idx:03d}.zip"
            zip_file = temp_dir / zip_filename
            
            with open(zip_file, "wb") as f:
                f.write(content)
            
            # Extract .zip
            batch_extract_dir = temp_dir / f"batch_{req_idx:03d}"
            batch_extract_dir.mkdir(exist_ok=True)
            
            extracted_dir = extract_zip(zip_file, batch_extract_dir)
            
            if extracted_dir:
                # Move files up to main output directory
                files_moved = move_files_up(extracted_dir, output_path)
                successful_downloads += request.n_isochrones
                
                # Record file info
                downloaded_files.append(DownloadedFile(
                    filename=zip_filename,
                    age_min=request.age_min,
                    age_max=request.age_max,
                    age_step=request.age_step,
                    metallicities=request.metallicities,
                    use_log_age=use_log_age,
                    n_isochrones=request.n_isochrones,
                    timestamp=datetime.now().isoformat()
                ))
                
                zip_file.unlink()
                
                time.sleep(1.0)
            else:
                print(f"Failed to extract batch {req_idx}")
        else:
            print(f"Failed to download batch {req_idx}")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    merge_isochrones('MIST_all.csv',output_path,cut_evolutionary_phases)

if __name__ == "__main__":
    try:
        download_isochrones(
            "./MIST_isochrones",False,10000000.0,10000001.0,1.0,0.0,0.0,1,"MIST2","p0","UBVRIplus"
        )
        quit()
        interactive_isochrones_downloader()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
