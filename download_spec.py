#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

"""
download_spec.py

Downloads full-resolution BT-SETTL synthetic stellar spectra from astrostarfish
for use with Icarus atmosphere grid building.

The native PHOENIX/BT-SETTL model spectra have much higher spectral resolution
(R ~ 500,000 in the optical) compared to the resampled CDBS/pysynphot grids
(R ~ 740), making them suitable for predicting line profiles, equivalent widths,
radial velocities, and vsini measurements.

This script:
1. Queries astrostarfish for available BT-SETTL model spectra
2. Downloads full-resolution FITS files for specified Teff and logg
3. Organizes them into a directory structure for build_atmo_grid_spec_*.py

The spectra are downloaded in their native high resolution and can be resampled
to any wavelength grid by the atmosphere grid building script.

Requirements
------------
- astrostarfish: pip install astrostarfish
- astropy: pip install astropy
- numpy: pip install numpy
- requests: pip install requests (for HTTP downloads)

Installation
------------
For high-resolution PHOENIX/BT-SETTL spectra:
    pip install astrostarfish

Documentation
--------------
- Astrostarfish: https://astrostarfish.readthedocs.io/
- PHOENIX models: https://www.hs.uni-hamburg.de/DE/Ins/Prof/Hauschildt/research/atmospheres/phoenixmodels/indexphoenixmodels.html
- BT-Settl: https://phoenix.astrophysics.bg.ac.rs/

Usage
-----
Modify the configuration parameters at the top of this script, then run:
    python download_spec.py

Or use command-line arguments:
    python download_spec.py --teff-min 3000 --teff-max 8000 --metallicity 0.0

Configuration Parameters
------------------------
TEFF_RANGE : tuple of (T_min, T_max)
    Effective temperature range in Kelvin for models to download.
    Default: (2000, 12500) K to match Icarus atmosphere grids.

LOGG_RANGE : tuple of (logg_min, logg_max)
    Surface gravity range [log10(g in cm/s^2)] for models to download.
    Default: (3.0, 5.0) for low-mass companions.

METALLICITY : str or float
    Metallicity specification for BT-SETTL grid.
    Default: '+0.5' for [M/H] = +0.5.
    Common options: '-2.5', '-2.0', '-1.5', '-1.0', '-0.5', '0.0', '+0.3', '+0.5'

MODEL_TYPE : str
    Type of PHOENIX model: 'btsettl' (default, synthetic spectra),
    'nextgen' (older), 'cond', 'settl'.

OUTPUT_DIR : str
    Directory to store downloaded FITS files.
    Default: 'model_spec/phoenix/btsettl_highres'

VERBOSE : bool
    Enable verbose output during downloads. Default: True

Notes on Resolution
-------------------
- PHOENIX native resolution: R ~ 500,000 in the optical (native sampling)
- CDBS/pysynphot resampled: R ~ 740 (400 km/s per pixel)
- X-shooter resolution: R ~ 5,000-9,000
- This script downloads NATIVE high-resolution spectra suitable for
  line profile analysis, radial velocity measurements, and detailed
  spectroscopy modeling.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not found. Install it with: pip install numpy")
    sys.exit(1)

try:
    from astropy.io import fits
except ImportError:
    print("ERROR: astropy not found. Install it with: pip install astropy")
    sys.exit(1)

try:
    import astrostarfish
except ImportError:
    print("ERROR: astrostarfish not found. Install it with:")
    print("  pip install astrostarfish")
    print("\nFor more information, see: https://astrostarfish.readthedocs.io/")
    sys.exit(1)


##### Configuration
TEFF_RANGE = (2000., 12500.)      # Effective temperature range, in K
LOGG_RANGE = (3.0, 5.0)            # Surface gravity range, [log10(g)]
METALLICITY = '+0.5'               # [M/H] metallicity; options: '-2.5' to '+0.5'
MODEL_TYPE = 'btsettl'             # Type of PHOENIX model ('btsettl', 'nextgen', 'cond', 'settl')
OUTPUT_DIR = 'model_spec/phoenix/btsettl_highres'
VERBOSE = True


def query_btsettl_models(teff_range, logg_range, metallicity, model_type='btsettl'):
    """
    Query astrostarfish for available high-resolution BT-SETTL models.
    
    Parameters
    ----------
    teff_range : tuple of (T_min, T_max)
        Effective temperature range in Kelvin.
    logg_range : tuple of (logg_min, logg_max)
        Surface gravity range [log10(g in cm/s^2)].
    metallicity : str or float
        Metallicity specification (e.g., '+0.5', '-0.5', '0.0').
    model_type : str
        Type of PHOENIX model ('btsettl', 'nextgen', 'cond', 'settl').
    
    Returns
    -------
    list of dict
        List of available model spectra matching the query criteria.
        Each dict contains metadata (teff, logg, etc.).
    """
    if VERBOSE:
        print(f"Querying astrostarfish for {model_type.upper()} models...")
        print(f"  Temperature range: {teff_range[0]:.0f} - {teff_range[1]:.0f} K")
        print(f"  Surface gravity range: {logg_range[0]:.1f} - {logg_range[1]:.1f}")
        print(f"  Metallicity: [M/H] = {metallicity}")
        print(f"  Resolution: NATIVE high-resolution (R ~ 500,000)\n")
    
    try:
        # Query astrostarfish catalog for available models
        # The query_models function searches the PHOENIX spectral database
        models = astrostarfish.query_models(
            model_type=model_type,
            teff_range=teff_range,
            logg_range=logg_range,
            metallicity=float(metallicity) if isinstance(metallicity, str) else metallicity
        )
        
        if VERBOSE:
            print(f"Found {len(models)} model(s) matching criteria.\n")
        
        return models
    
    except Exception as e:
        print(f"ERROR querying astrostarfish: {e}")
        print("\nTroubleshooting:")
        print("1. Check astrostarfish installation: pip install astrostarfish")
        print("2. Verify internet connection")
        print("3. Try a simpler query (fewer temperature points, standard metallicity)")
        print("4. See documentation: https://astrostarfish.readthedocs.io/")
        return []


def download_model_spectrum(model_info, output_dir, model_type='btsettl'):
    """
    Download a single high-resolution BT-SETTL model spectrum.
    
    Parameters
    ----------
    model_info : dict
        Model metadata from astrostarfish query results.
        Should contain: teff, logg, metallicity, and reference/url info.
    output_dir : str
        Directory to save the downloaded FITS file.
    model_type : str
        Type of PHOENIX model.
    
    Returns
    -------
    str or None
        Path to downloaded file if successful, None if failed.
    """
    try:
        # Extract model parameters
        teff = model_info.get('teff') or model_info.get('T_eff')
        logg = model_info.get('logg') or model_info.get('log_g')
        met = model_info.get('metallicity') or model_info.get('feh')
        
        if teff is None or logg is None:
            print(f"WARNING: Missing Teff/logg in model metadata: {model_info}")
            return None
        
        # Download spectrum from astrostarfish
        # retrieve_spectrum returns an astropy.io.fits HDUList or similar
        spectrum = astrostarfish.retrieve_spectrum(model_info)
        
        # Generate filename: btsettl_Z±X_Teff_logg.fits
        # Example: btsettl_Z+0.5_6000_4.5.fits
        met_str = f"{float(met):+.1f}" if met is not None else "0.0"
        filename = f"{model_type}_Z{met_str}_{teff:.0f}_{logg:.1f}.fits"
        filepath = os.path.join(output_dir, filename)
        
        # Write to FITS file
        if hasattr(spectrum, 'writeto'):
            # If it's an HDUList
            spectrum.writeto(filepath, overwrite=True)
        else:
            # If it needs to be wrapped or is a different format
            fits.writeto(filepath, spectrum, overwrite=True)
        
        if VERBOSE:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ Downloaded Teff={teff:.0f} K, logg={logg:.1f} "
                  f"[M/H]={met_str}: {filename} ({file_size_mb:.1f} MB)")
        
        return filepath
    
    except Exception as e:
        teff = model_info.get('teff', '?')
        logg = model_info.get('logg', '?')
        print(f"  ✗ Failed to download Teff={teff}, logg={logg}: {e}")
        return None


def check_existing_files(output_dir, model_type='btsettl'):
    """
    Check which models already exist in the output directory.
    
    Parameters
    ----------
    output_dir : str
        Directory containing downloaded FITS files.
    model_type : str
        Type of model (for filename matching).
    
    Returns
    -------
    dict
        Dictionary mapping (teff, logg) tuples to file paths.
    """
    existing = {}
    
    if not os.path.exists(output_dir):
        return existing
    
    pattern = os.path.join(output_dir, f'{model_type}_*.fits')
    for fln in glob.glob(pattern):
        try:
            # Parse filename: btsettl_Z+0.5_6000_4.5.fits
            basename = os.path.basename(fln)
            parts = basename.replace('.fits', '').split('_')
            if len(parts) >= 4:
                teff = float(parts[-2])
                logg = float(parts[-1])
                existing[(teff, logg)] = fln
        except (ValueError, IndexError):
            pass
    
    return existing


def download_btsettl_spectra(teff_range=TEFF_RANGE, 
                             logg_range=LOGG_RANGE,
                             metallicity=METALLICITY,
                             model_type=MODEL_TYPE,
                             output_dir=OUTPUT_DIR,
                             skip_existing=True):
    """
    Main function to download high-resolution BT-SETTL spectra.
    
    Parameters
    ----------
    teff_range : tuple of (T_min, T_max)
        Effective temperature range in Kelvin. Default: (2000, 12500).
    logg_range : tuple of (logg_min, logg_max)
        Surface gravity range. Default: (3.0, 5.0).
    metallicity : str or float
        Metallicity specification. Default: '+0.5'.
    model_type : str
        Type of PHOENIX model. Default: 'btsettl'.
    output_dir : str
        Output directory for FITS files. Default: 'model_spec/phoenix/btsettl_highres'.
    skip_existing : bool
        Skip downloading models that already exist locally. Default: True.
    
    Returns
    -------
    dict
        Summary statistics: {'downloaded': int, 'skipped': int, 'failed': int}
    """
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if VERBOSE:
        print(f"Output directory: {output_dir}\n")
    
    # Query available models
    models = query_btsettl_models(teff_range, logg_range, metallicity, model_type)
    
    if not models:
        print("ERROR: No models found matching the specified criteria.")
        print("Try relaxing your constraints (wider Teff range, standard metallicity, etc.)")
        return {'downloaded': 0, 'skipped': 0, 'failed': 0}
    
    # Check for existing files
    existing = check_existing_files(output_dir, model_type)
    if VERBOSE and existing:
        print(f"Found {len(existing)} existing model(s).\n")
    
    # Download models
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    
    if VERBOSE:
        print(f"Downloading {len(models)} model(s)...\n")
    
    for model in models:
        teff = model.get('teff') or model.get('T_eff')
        logg = model.get('logg') or model.get('log_g')
        
        if skip_existing and (teff, logg) in existing:
            n_skipped += 1
            if VERBOSE:
                print(f"  ⊘ Skipping Teff={teff:.0f} K, logg={logg:.1f} (exists)")
            continue
        
        result = download_model_spectrum(model, output_dir, model_type)
        if result is not None:
            n_downloaded += 1
        else:
            n_failed += 1
    
    # Summary
    print(f"\n" + "="*70)
    print("Download Summary:")
    print(f"  Downloaded:      {n_downloaded}")
    print(f"  Skipped:         {n_skipped}")
    print(f"  Failed:          {n_failed}")
    print(f"  Total attempted: {len(models)}")
    print("="*70 + "\n")
    
    # List downloaded files with details
    if VERBOSE and (n_downloaded > 0 or n_skipped > 0):
        print("Files in output directory:")
        files = sorted(glob.glob(os.path.join(output_dir, f'{model_type}_*.fits')))
        for fln in files:
            try:
                file_size_mb = os.path.getsize(fln) / (1024 * 1024)
                with fits.open(fln) as hdul:
                    n_ext = len(hdul)
                    n_wav = hdul[1].data['WAVELENGTH'].size if 'WAVELENGTH' in hdul[1].data.names else 0
                    print(f"  {os.path.basename(fln):45s} {file_size_mb:8.1f} MB  "
                          f"({n_ext} HDU, {n_wav} wavelength points)")
            except Exception as e:
                print(f"  {os.path.basename(fln):45s} (ERROR: {e})")
        print()
    
    return {'downloaded': n_downloaded, 'skipped': n_skipped, 'failed': n_failed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download high-resolution PHOENIX/BT-SETTL spectra from astrostarfish.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default: 2000-12500 K, logg 3.0-5.0, [M/H]=+0.5 (native high-res)
  python download_spec.py
  
  # Download specific temperature range
  python download_spec.py --teff-min 3000 --teff-max 8000
  
  # Download specific surface gravity range
  python download_spec.py --logg-min 3.5 --logg-max 5.0
  
  # Use different metallicity
  python download_spec.py --metallicity 0.0
  python download_spec.py --metallicity -0.5
  
  # Output to custom directory
  python download_spec.py --output-dir ./my_phoenix_models
  
  # Re-download everything (skip existing)
  python download_spec.py --no-skip-existing
  
  # Quiet mode
  python download_spec.py --quiet
  
Notes:
  - Default spectra are full-resolution NATIVE PHOENIX/BT-SETTL (R ~ 500,000)
  - These can be resampled to any wavelength grid by build_atmo_grid_spec_*.py
  - Much higher resolution than the CDBS/pysynphot resampled grids (R ~ 740)
  - Suitable for line profile analysis and radial velocity measurements
        """)
    
    parser.add_argument('--teff-min', type=float, default=TEFF_RANGE[0],
                       help=f'Min effective temperature (K). Default: {TEFF_RANGE[0]:.0f}')
    parser.add_argument('--teff-max', type=float, default=TEFF_RANGE[1],
                       help=f'Max effective temperature (K). Default: {TEFF_RANGE[1]:.0f}')
    parser.add_argument('--logg-min', type=float, default=LOGG_RANGE[0],
                       help=f'Min surface gravity. Default: {LOGG_RANGE[0]:.1f}')
    parser.add_argument('--logg-max', type=float, default=LOGG_RANGE[1],
                       help=f'Max surface gravity. Default: {LOGG_RANGE[1]:.1f}')
    parser.add_argument('--metallicity', type=str, default=METALLICITY,
                       help=f'Metallicity [M/H]. Default: {METALLICITY}')
    parser.add_argument('--model-type', type=str, default=MODEL_TYPE,
                       help=f'PHOENIX model type. Default: {MODEL_TYPE}')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory. Default: {OUTPUT_DIR}')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Re-download all models even if they exist')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Set verbose mode
    if args.quiet:
        VERBOSE = False
    
    # Run download
    stats = download_btsettl_spectra(
        teff_range=(args.teff_min, args.teff_max),
        logg_range=(args.logg_min, args.logg_max),
        metallicity=args.metallicity,
        model_type=args.model_type,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing
    )
    
    # Exit with error code if downloads failed
    sys.exit(0 if stats['failed'] == 0 else 1)
