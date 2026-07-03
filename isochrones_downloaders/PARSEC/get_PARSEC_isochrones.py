#!/usr/bin/env python3
"""
PARSEC CMD Isochrone Downloader

A Python CLI application that interfaces with the PARSEC CMD web service
to download stellar isochrones with automatic partitioning for large grids
and comprehensive configuration logging.
"""

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
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
import gzip
import shutil
import pathlib

if _has_urllib3:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CMD_SUBMIT_URL = "https://stev.oapd.inaf.it/cgi-bin/cmd_3.9"
MAX_ISOCHRONES_PER_REQUEST = 400


# ============================================================
# Data Classes
# ============================================================

@dataclass
class IsochroneRequest:
    """Represents a single request to the CMD service."""
    age_min: float
    age_max: float
    n_ages: int
    met_min: float
    met_max: float
    n_mets: int
    use_log_age: bool

    @property
    def n_isochrones(self) -> int:
        """Total number of isochrones in this request."""
        return self.n_ages * self.n_mets


@dataclass
class DownloadedFile:
    """Information about a downloaded isochrone file."""
    filename: str
    age_min: float
    age_max: float
    n_ages: int
    met_min: float
    met_max: float
    n_mets: int
    n_isochrones: int
    use_log_age: bool
    timestamp: str


@dataclass
class RunConfiguration:
    """Complete run configuration for reproducibility."""
    timestamp: str
    track_parsec: str
    track_colibri: str
    photsys_file: str
    photsys_version: str
    imf_file: str
    extinction_av: float
    output_directory: str
    total_requests: int
    total_isochrones: int
    grid_mode: str  # 'single', 'age', 'metallicity', 'age_metallicity'


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

def load_form_from_html_file(html_file: str):
    if _has_urllib3 * _has_bs4 * _has_pandas * _has_requests == 0:
        raise modules_missing_error()
    """Load the CMD form from a local HTML file."""
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


def extract_select_options(select) -> List[Tuple[str, str]]:
    """Extract options from an HTML select element."""
    options = []
    if select is None:
        return options
    for opt in select.find_all("option"):
        label = opt.get_text(strip=True)
        if not label:
            continue
        value = opt["value"]
        options.append((label, value))
    return options


def get_selected_option(select) -> Optional[str]:
    """Get the selected/default option value from a select element."""
    if select is None:
        return None
    selected = select.find("option", {"selected": "selected"})
    if selected:
        return selected.get("value")
    # Return first option if none explicitly selected
    first_opt = select.find("option")
    if first_opt:
        return first_opt.get("value")
    return None


def extract_radio_options_with_labels(soup, name: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """
    Extract radio button options with meaningful labels and find the default.
    
    Returns:
        Tuple of (options_list, default_value)
    """
    radios = soup.find_all("input", {"type": "radio", "name": name})
    options = []
    default_value = None
    seen = set()
    
    for radio in radios:
        value = radio.get("value")
        if not value or value in seen:
            continue
        seen.add(value)
        
        # Get label from nearby strong tag or text
        label = value
        parent = radio.parent
        if parent:
            strong_tag = parent.find("strong")
            if strong_tag:
                label = strong_tag.get_text(strip=True)
        
        options.append((label, value))
        
        # Check if this radio is checked
        if radio.get("checked"):
            default_value = value
    
    return options, default_value


# ============================================================
# Grid Calculation
# ============================================================

def calculate_grid_count(min_val: float, max_val: float, step: float) -> int:
    """
    Calculate number of points in a grid.
    
    Formula: N = floor((max - min) / step) + 1
    """
    if step == 0:
        return 1
    return int(round((max_val - min_val) / step)) + 1

def partition_grid(age_min: float, age_max: float, age_step: float,
                   met_min: float, met_max: float, met_step: float,
                   use_log_age: bool) -> List[IsochroneRequest]:
    """
    Partition the age-metallicity grid into requests of ≤400 isochrones.
    
    Ensures exact spacing is preserved across partitions.
    
    Returns:
        List of IsochroneRequest objects
    """
    n_ages = calculate_grid_count(age_min, age_max, age_step)
    n_mets = calculate_grid_count(met_min, met_max, met_step)
    total = n_ages * n_mets
    
    print(f"\nGrid Analysis:")
    print(f"  Ages: {n_ages} points (min={age_min}, max={age_max}, step={age_step})")
    print(f"  Metallicities: {n_mets} points (min={met_min}, max={met_max}, step={met_step})")
    print(f"  Total isochrones: {total}")
    
    if total <= MAX_ISOCHRONES_PER_REQUEST:
        return [IsochroneRequest(
            age_min=age_min,
            age_max=age_max,
            n_ages=n_ages,
            met_min=met_min,
            met_max=met_max,
            n_mets=n_mets,
            use_log_age=use_log_age
        )]
    
    # Partition by metallicity (age is typically the longer dimension)
    n_met_per_request = MAX_ISOCHRONES_PER_REQUEST // n_ages
    if n_met_per_request == 0:
        # Partition by age instead
        n_age_per_request = MAX_ISOCHRONES_PER_REQUEST // n_mets
        requests = []
        age_idx = 0
        while age_idx < n_ages:
            n_age_chunk = min(n_age_per_request, n_ages - age_idx)
            age_chunk_max = age_min + (age_idx + n_age_chunk - 1) * age_step
            requests.append(IsochroneRequest(
                age_min=age_min + age_idx * age_step,
                age_max=age_chunk_max,
                n_ages=n_age_chunk,
                met_min=met_min,
                met_max=met_max,
                n_mets=n_mets,
                use_log_age=use_log_age
            ))
            age_idx += n_age_chunk
        return requests
    else:
        # Partition by metallicity
        requests = []
        met_idx = 0
        while met_idx < n_mets:
            n_met_chunk = min(n_met_per_request, n_mets - met_idx)
            met_chunk_max = met_min + (met_idx + n_met_chunk - 1) * met_step
            requests.append(IsochroneRequest(
                age_min=age_min,
                age_max=age_max,
                n_ages=n_ages,
                met_min=met_min + met_idx * met_step,
                met_max=met_chunk_max,
                n_mets=n_met_chunk,
                use_log_age=use_log_age
            ))
            met_idx += n_met_chunk
        return requests


# ============================================================
# Payload Building
# ============================================================

def build_payload(track_parsec: str, track_colibri: str, photsys_file: str,
                  photsys_version: str, imf_file: str, extinction_av: float,
                  request: IsochroneRequest) -> Dict[str, str]:
    """Build the POST payload for a CMD request."""
    payload = {
        "submit_form": "Submit",
        "cmd_version": "3.9",
        "track_parsec": track_parsec,
        "track_colibri": track_colibri,
        "track_postagb": "no",
        "track_omegai": "0.00",
        "n_inTPC": "10",
        "eta_reimers": "0.2",
        "photsys_file": photsys_file,
        "photsys_version": photsys_version,
        "dust_sourceM": "dpmod60alox40",
        "dust_sourceC": "AMCSIC15",
        "extinction_av": str(extinction_av),
        "extinction_coeff": "constant",
        "extinction_curve": "cardelli",
        "kind_LPV": "4",
        "imf_file": imf_file,
        "output_kind": "0",
        "output_evstage": "1",
        "kind_interp": "1",
        "kind_postagb": "-1",
        "kind_mag": "2",
        "kind_dust": "0",
    }
    
    # Age parameters
    if request.use_log_age:
        payload["isoc_isagelog"] = "1"
        payload["isoc_lagelow"] = str(request.age_min)
        payload["isoc_lageupp"] = str(request.age_max)
        age_step = (request.age_max - request.age_min) / (request.n_ages - 1) if request.n_ages > 1 else 0
        payload["isoc_dlage"] = str(age_step)
    else:
        payload["isoc_isagelog"] = "0"
        payload["isoc_agelow"] = str(request.age_min)
        payload["isoc_ageupp"] = str(request.age_max)
        age_step = (request.age_max - request.age_min) / (request.n_ages - 1) if request.n_ages > 1 else 0
        payload["isoc_dage"] = str(age_step)
    
    # Metallicity parameters
    payload["isoc_ismetlog"] = "1"
    payload["isoc_metlow"] = str(request.met_min)
    payload["isoc_metupp"] = str(request.met_max)
    met_step = (request.met_max - request.met_min) / (request.n_mets - 1) if request.n_mets > 1 else 0
    payload["isoc_dmet"] = str(met_step)
    
    return payload


# ============================================================
# Network Operations
# ============================================================

def submit_cmd_request(session, payload: Dict[str, str]):
    """Submit a request to the CMD service and return the response page."""
    print(f"Submitting CMD request...")
    r = session.post(CMD_SUBMIT_URL, data=payload, verify=False, timeout=120)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_download_link(soup) -> Optional[str]:
    """Extract the download link from the CMD response page."""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".dat") or href.endswith(".dat.gz"):
            return href
    return None


def download_file(session, download_url: str, output_path: Path) -> None:
    """Download a file from the given URL."""
    print(f"Downloading: {download_url}")
    r = session.get(download_url, verify=False, timeout=120)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)
    print(f"Saved: {output_path}")


# ============================================================
# Main Application
# ============================================================

def interactive_isochrones_downloader():
    """Main application entry point."""

    # Load HTML form file
    cwd = str(pathlib.Path(__file__).parent.resolve())
    html_file = cwd + '/CMD_3.9_input_form.html'
    if not os.path.exists(html_file):
        print(f"Error: File {html_file} not found")
        sys.exit(1)
    
    soup = load_form_from_html_file(html_file)
    
    # ========================================================
    # Interactive Choices
    # ========================================================
    
    # PARSEC track set
    track_options, default_parsec = extract_radio_options_with_labels(soup, "track_parsec")
    track_parsec = choose_option("PARSEC track set", track_options, default_parsec)
    
    # COLIBRI extension
    colibri_options, default_colibri = extract_radio_options_with_labels(soup, "track_colibri")
    track_colibri = choose_option("COLIBRI extension", colibri_options, default_colibri)
    
    # Photometric system
    photsys_select = soup.find("select", {"name": "photsys_file"})
    photsys_options = extract_select_options(photsys_select)
    default_photsys = get_selected_option(photsys_select)
    photsys_file = choose_option("Photometric system", photsys_options, default_photsys)
    
    # Bolometric correction library
    photsys_version_options, default_photsys_version = extract_radio_options_with_labels(
        soup, "photsys_version"
    )
    photsys_version = choose_option(
        "Bolometric correction library",
        photsys_version_options,
        default_photsys_version
    )
    
    # Initial Mass Function
    imf_select = soup.find("select", {"name": "imf_file"})
    imf_options = extract_select_options(imf_select)
    default_imf = get_selected_option(imf_select)
    imf_file = choose_option("Initial Mass Function", imf_options, default_imf)
    
    # Extinction
    print("\n" + "=" * 70)
    print("Extinction")
    print("=" * 70)
    av_input = input("A_V extinction [0.0]: ").strip()
    extinction_av = float(av_input) if av_input else 0.0
    
    # ========================================================
    # Grid Mode Selection
    # ========================================================
    
    grid_modes = [
        ("Single isochrone", "single"),
        ("Age grid", "age"),
        ("Metallicity grid", "metallicity"),
        ("Age-Metallicity grid", "age_metallicity"),
    ]
    
    grid_mode = choose_option("Select grid mode", grid_modes)
    
    # ========================================================
    # Age and Metallicity Selection
    # ========================================================
    
    print("\nAge selection")
    age_mode = input("Use log(age/yr)? [Y/n]: ").strip().lower()
    use_log_age = age_mode != "n"
    
    if grid_mode in ["single", "age", "age_metallicity"]:
        if grid_mode == "single":
            if use_log_age:
                age_min = float(input("log(age/yr): "))
                age_max = age_min
                age_step = 0
            else:
                age_min = float(input("age (Myr): "))*1e6
                age_max = age_min
                age_step = 0
        else:  # age or age_metallicity
            if use_log_age:
                age_min = float(input("log(age/yr) min: "))
                age_max = float(input("log(age/yr) max: "))
                age_step = float(input("log(age/yr) step: "))
            else:
                age_min = float(input("age (Myr) min: "))*1e6
                age_max = float(input("age (Myr) max: "))*1e6
                age_step = float(input("age (Myr) step: "))*1e6
    else:
        age_min = 0
        age_max = 0
        age_step = 0
    
    print("\nMetallicity selection")

    if grid_mode in ["single", "metallicity", "age_metallicity"]:
        if grid_mode == "single":
            met_min = float(input("[M/H] = "))
            met_max = met_min
            met_step = 0
        else:
            met_min = float(input("[M/H] min: "))
            met_max = float(input("[M/H] max: "))
            met_step = float(input("[M/H] step: "))
    else:
        met_min = float(input("[M/H] = "))
        met_max = met_min
        met_step = 0
    
    # ========================================================
    # Output Directory
    # ========================================================
    
    print("\n" + "=" * 70)
    print("Output Directory")
    print("=" * 70)
    output_dir = input("Output directory [./CMD_isochrones]: ").strip() or "./CMD_isochrones"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.resolve()}")

    print("\n" + "=" * 70)
    print("Evolutionary phases")
    print("=" * 70)
    cut_evolutionary_phases = input("Cut evolutionary phases? [Y/n]") != 'n'
    
    download_isochrones(
        output_dir,
        use_log_age,age_min,age_max,age_step,
        met_min,met_max,met_step,
        track_parsec,track_colibri,
        photsys_file,photsys_version,
        imf_file,extinction_av,
        cut_evolutionary_phases
    )

def download_isochrones(
        output_dir,
        use_log_age,
        age_min,age_max,age_step,
        met_min,met_max,met_step,
        track_parsec="parsec_CAF09_v2.0",
        track_colibri="parsec_CAF09_v1.2S_S_LMC_08_web",
        photsys_file="YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat",
        photsys_version="YBCnewVega",
        imf_file="tab_imf/imf_kroupa_orig.dat",
        extinction_av=0.0,
        cut_evolutionary_phases=False
    ):

    if _has_urllib3 * _has_bs4 * _has_pandas * _has_requests == 0:
        raise modules_missing_error()
    files = glob.glob(str(output_dir) + '/*.dat')
    for f in files:
        if os.path.isdir(f):
            continue
        os.remove(f)
    if os.path.isdir(str(output_dir) + '/temp'):
        files = glob.glob(str(output_dir) + '/temp/*')
        for f in files:
            os.remove(f)
    
    requests_list = partition_grid(
        age_min, age_max, age_step,
        met_min, met_max, met_step,
        use_log_age
    )

    session = requests.Session()
    downloaded_files: List[DownloadedFile] = []
    total_isochrones = sum(r.n_isochrones for r in requests_list)
    
    for idx, request in enumerate(requests_list, 1):
        print(f"\n{'='*70}")
        print(f"Request {idx}/{len(requests_list)}")
        print(f"{'='*70}")
        print(f"Isochrones in this request: {request.n_isochrones}")
        
        # Build and submit payload
        payload = build_payload(
            track_parsec, track_colibri, photsys_file, photsys_version,
            imf_file, extinction_av, request
        )
        
        response_soup = submit_cmd_request(session, payload)
        
        # Extract download link
        download_link = extract_download_link(response_soup)
        if download_link is None:
            # Save response for debugging
            debug_file = output_dir / f"cmd_response_{idx}.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(response_soup.prettify())
            raise RuntimeError(
                f"Could not find output file in request {idx}. "
                f"Saved response as {debug_file}"
            )
        
        # Build meaningful filename
        age_label = f"logAge{request.age_min:.2f}to{request.age_max:.2f}" if request.use_log_age \
            else f"Age{request.age_min:.2e}to{request.age_max:.2e}"
        met_label = f"MH{request.met_min:.2f}to{request.met_max:.2f}"
        
        # Determine file extension
        file_ext = ".dat.gz" if download_link.endswith(".gz") else ".dat"
        output_filename = f"isochrones_{age_label}_{met_label}{file_ext}"
        output_file = output_dir + "/" + output_filename
        
        # Download file
        full_url = urljoin(CMD_SUBMIT_URL, download_link)
        download_file(session, full_url, output_file)
        
        # Record file info
        downloaded_files.append(DownloadedFile(
            filename=output_filename,
            age_min=request.age_min,
            age_max=request.age_max,
            n_ages=request.n_ages,
            met_min=request.met_min,
            met_max=request.met_max,
            n_mets=request.n_mets,
            n_isochrones=request.n_isochrones,
            use_log_age=request.use_log_age,
            timestamp=datetime.now().isoformat()
        ))
    
    files = glob.glob(str(output_dir) + '/*.gz')

    for file in files:
        with gzip.open(file, "rb") as zip:
            with open(file.replace('.gz',''),'wb') as out:
                shutil.copyfileobj(zip, out)
        os.remove(file)

    merge_isochrones('parsec_all.csv',output_dir,cut_evolutionary_phases)

def merge_isochrones(filename,output_path,cut_evolutionary_phases=True):
    files = glob.glob(str(output_path) + '/*.dat')
        
    df_parsec = pd.DataFrame()

    for f in files:
        with open(f,'r') as _f:
            lines = _f.readlines()
            iso_version = lines[1].split(' ')[6].replace('v','')
            header = lines[13][2:].split()

        df = pd.read_csv(f,
            skipfooter=1,
            sep=r'\s+',
            comment='#',
            names=header,
            engine='python'
        )

        df_parsec = pd.concat([df_parsec,df])
    
    if cut_evolutionary_phases:
        df_parsec = df_parsec[(df_parsec['label'] > 0) & (df_parsec['label'] < 4)]

    df_parsec['Age'] = 10**df_parsec['logAge']/1e9
    if iso_version == '2.0':
        df_parsec['G'] = df_parsec['G_i50']
        df_parsec['G-BP'] = df_parsec['G_BP_i50']
        df_parsec['G-RP'] = df_parsec['G_RP_i50']
        df_parsec['BP-RP'] = df_parsec['G_BP_i50'] - df_parsec['G_RP_i50']
    elif iso_version == '1.2S':
        df_parsec['G'] = df_parsec['Gmag']
        df_parsec['G-BP'] = df_parsec['G_BPmag']
        df_parsec['G-RP'] = df_parsec['G_RPmag']
        df_parsec['BP-RP'] = df_parsec['G_BPmag'] - df_parsec['G_RPmag']
    df_parsec['MoH'] = df_parsec['MH']
    df_parsec['M'] = df_parsec['Mass']
    df_parsec.to_csv(str(output_path) + '/' + filename,index=False)

    js_parsec = {}
    for moh in df_parsec['MoH'].unique():
        js_parsec[str(moh)] = []
        df = df_parsec[df_parsec['MoH'] == moh]
        for age in df_parsec['Age'].unique():
            dff = df[df['Age'] == age]
            js_iso = {'age':float(age)}
            js_iso['MG'] = dff['G'].values.tolist()
            js_iso['BP-RP'] = dff['BP-RP'].tolist()
            js_iso['M'] = dff['M'].values.tolist()
            js_parsec[str(moh)].append(js_iso)

    with open(str(output_path) + '/' + filename.split('.')[0] + '.json','w') as f:
        f.write(str(js_parsec).replace('\'','"'))

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