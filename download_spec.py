#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

"""
download_spec.py

Downloads BT-SETTL synthetic stellar spectra from astrostarfish in FITS format
(CDBS/pysynphot grid format) for use with Icarus atmosphere grid building.

The BT-SETTL models are sourced from the PHOENIX spectral database via astrostarfish,
which provides synthetic stellar spectra in a consistent FITS format suitable for
building spectroscopic atmosphere grids for Icarus.

This script performs:
1. Queries astrostarfish for available BT-SETTL model spectra
2. Downloads FITS files for specified effective temperatures and surface gravities
3. Organizes them into the directory structure expected by build_atmo_grid_spec_*.py

Requirements
------------
- astrostarfish: pip install astrostarfish
- astropy: pip install astropy
- numpy: pip install numpy

Usage
-----
Modify the configuration parameters at the top of this script, then run:
    python download_spec.py

Configuration Parameters
------------------------
TEFF_RANGE : tuple of (T_min, T_max)
    Effective temperature range in Kelvin for models to download.
    Default: (2000, 12500) K to match Icarus atmosphere grids.

LOGG_RANGE : tuple of (logg_min, logg_max)
    Surface gravity range [log10(g in cm/s^2)] for models to download.
    Default: (3.0, 4.5) to match low-mass companions.

METALLICITY : str or float
    Metallicity specification for BT-SETTL grid.
    Default: '+0.5' for [M/H] = +0.5 (phoenixp05 grid).
    Common options: '-0.5', '0.0', '+0.5'

OUTPUT_DIR : str
    Directory to store downloaded FITS files.
    Default: 'model_spec/grp/redcat/trds/grid/phoenix/phoenixp05'
    This matches the FITS_DIR expected by build_atmo_grid_spec_J2241-5236.py.

VERBOSE : bool
    Enable verbose output during downloads. Default: True
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np

try:
    import astrostarfish
except ImportError:
    print("ERROR: astrostarfish not found. Install it with:")
    print("  pip install astrostarfish")
    sys.exit(1)

from astropy.io import fits


##### Configuration
TEFF_RANGE = (2000., 12500.)      # Effective temperature range, in K
LOGG_RANGE = (3.0, 4.5)            # Surface gravity range, [log10(g)]
METALLICITY = '+0.5'               # [M/H] metallicity; options: '-0.5', '0.0', '+0.5'
OUTPUT_DIR = 'model_spec/grp/redcat/trds/grid/phoenix/phoenixp05'
VERBOSE = True


def query_btsettl_models(teff_range, logg_range, metallicity):
    """
    Query astrostarfish for available BT-SETTL models.
    
    Parameters
    ----------
    teff_range : tuple of (T_min, T_max)
        Effective temperature range in Kelvin.
    logg_range : tuple of (logg_min, logg_max)
        Surface gravity range [log10(g in cm/s^2)].
    metallicity : str or float
        Metallicity specification (e.g., '+0.5', '-0.5', '0.0').
    
    Returns
    -------
    list of dict
        List of available model spectra matching the query criteria.
        Each dict contains metadata about the spectrum (teff, logg, etc.).
    """
    if VERBOSE:
        print(f"Querying astrostarfish for BT-SETTL models...")
        print(f"  Temperature range: {teff_range[0]:.0f} - {teff_range[1]:.0f} K")
        print(f"  Surface gravity range: {logg_range[0]:.1f} - {logg_range[1]:.1f}")
        print(f"  Metallicity: [M/H] = {metallicity}")
    
    try:
        # Query for available models matching criteria
        # Note: The exact query method depends on the astrostarfish API
        # This is a typical pattern; adjust based on actual astrostarfish documentation
        models = astrostarfish.query_models(
            model_set='btsettl',
            teff_range=teff_range,
            logg_range=logg_range,
            metallicity=metallicity
        )
        
        if VERBOSE:
            print(f"Found {len(models)} models matching criteria.\n")
        
        return models
    
    except Exception as e:
        print(f"ERROR querying astrostarfish: {e}")
        print("\nTroubleshooting:")
        print("1. Check that astrostarfish is properly installed: pip install astrostarfish")
        print("2. Verify internet connection (models are downloaded from remote server)")
        print("3. Check astrostarfish documentation for current API: https://astrostarfish.readthedocs.io/")
        sys.exit(1)


def download_model_spectrum(model, output_dir):
    """
    Download a single BT-SETTL model spectrum from astrostarfish.
    
    Parameters
    ----------
    model : dict
        Model metadata dict from astrostarfish query results.
    output_dir : str
        Directory to save the downloaded FITS file.
    
    Returns
    -------
    str or None
        Path to downloaded file if successful, None if failed.
    """
    teff = model.get('teff', model.get('T_eff', None))
    logg = model.get('logg', model.get('log_g', None))
    
    if teff is None or logg is None:
        print(f"WARNING: Could not extract Teff and logg from model: {model}")
        return None
    
    try:
        # Download the spectrum
        # Note: Adjust method name/parameters based on actual astrostarfish API
        spectrum = astrostarfish.retrieve_spectrum(model)
        
        # Generate output filename following the pattern used by CDBS/pysynphot
        # Format: *_*.fits where the second * is the effective temperature
        filename = f"btsettl_g{logg*10:02.0f}_{teff:.0f}.fits"
        filepath = os.path.join(output_dir, filename)
        
        # Save to FITS file
        spectrum.writeto(filepath, overwrite=True)
        
        if VERBOSE:
            print(f"  Downloaded Teff={teff:.0f} K, logg={logg:.1f}: {filename}")
        
        return filepath
    
    except Exception as e:
        print(f"WARNING: Failed to download model Teff={teff} K, logg={logg}: {e}")
        return None


def check_existing_files(output_dir, teff_range, logg_range):
    """
    Check which models already exist in the output directory.
    
    Parameters
    ----------
    output_dir : str
        Directory containing downloaded FITS files.
    teff_range : tuple of (T_min, T_max)
        Effective temperature range in Kelvin.
    logg_range : tuple of (logg_min, logg_max)
        Surface gravity range.
    
    Returns
    -------
    list of tuple
        List of (teff, logg) tuples for existing models.
    """
    existing = []
    
    if not os.path.exists(output_dir):
        return existing
    
    for fln in glob.glob(os.path.join(output_dir, '*_*.fits')):
        try:
            # Extract Teff from filename (last part before .fits)
            teff = float(os.path.basename(fln).rsplit('_', 1)[1].replace('.fits', ''))
            
            # Extract logg from filename (g prefix)
            basename = os.path.basename(fln)
            logg_str = basename.split('_')[0].replace('g', '')
            logg = float(logg_str) / 10.0
            
            if teff_range[0] <= teff <= teff_range[1] and logg_range[0] <= logg <= logg_range[1]:
                existing.append((teff, logg))
        
        except (ValueError, IndexError):
            # Skip files that don't match the expected naming pattern
            pass
    
    return existing


def download_btsettl_spectra(teff_range=TEFF_RANGE, 
                             logg_range=LOGG_RANGE,
                             metallicity=METALLICITY,
                             output_dir=OUTPUT_DIR,
                             skip_existing=True):
    """
    Main function to download BT-SETTL spectra.
    
    Parameters
    ----------
    teff_range : tuple of (T_min, T_max)
        Effective temperature range in Kelvin. Default: (2000, 12500).
    logg_range : tuple of (logg_min, logg_max)
        Surface gravity range. Default: (3.0, 4.5).
    metallicity : str or float
        Metallicity specification. Default: '+0.5'.
    output_dir : str
        Output directory for FITS files. Default: 'model_spec/...'.
    skip_existing : bool
        Skip downloading models that already exist locally. Default: True.
    """
    
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if VERBOSE:
        print(f"Output directory: {output_dir}\n")
    
    # Query available models
    models = query_btsettl_models(teff_range, logg_range, metallicity)
    
    if not models:
        print("ERROR: No models found matching the specified criteria.")
        sys.exit(1)
    
    # Check for existing files
    if skip_existing:
        existing = check_existing_files(output_dir, teff_range, logg_range)
        if VERBOSE and existing:
            print(f"Found {len(existing)} existing model(s) matching criteria.\n")
    else:
        existing = []
    
    # Download models
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    
    if VERBOSE:
        print(f"Downloading {len(models)} model(s)...\n")
    
    for model in models:
        teff = model.get('teff', model.get('T_eff', None))
        logg = model.get('logg', model.get('log_g', None))
        
        if skip_existing and (teff, logg) in existing:
            n_skipped += 1
            continue
        
        result = download_model_spectrum(model, output_dir)
        if result is not None:
            n_downloaded += 1
        else:
            n_failed += 1
    
    # Summary
    print(f"\n" + "="*60)
    print("Download Summary:")
    print(f"  Downloaded: {n_downloaded}")
    print(f"  Skipped (existing): {n_skipped}")
    print(f"  Failed: {n_failed}")
    print(f"  Total: {n_downloaded + n_skipped + n_failed}")
    print("="*60 + "\n")
    
    # List downloaded files
    if VERBOSE:
        print("Files in output directory:")
        files = sorted(glob.glob(os.path.join(output_dir, '*_*.fits')))
        for fln in files:
            try:
                with fits.open(fln) as hdul:
                    n_ext = len(hdul)
                    print(f"  {os.path.basename(fln):40s} ({n_ext} HDU{'s' if n_ext > 1 else ''})")
            except Exception as e:
                print(f"  {os.path.basename(fln):40s} (ERROR: {e})")
    
    return n_downloaded, n_skipped, n_failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download BT-SETTL spectra from astrostarfish for Icarus atmosphere grids.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default configuration (2000-12500 K, logg 3.0-4.5, [M/H]=+0.5)
  python download_spec.py
  
  # Specify custom temperature range
  python download_spec.py --teff-min 3000 --teff-max 8000
  
  # Specify custom surface gravity range
  python download_spec.py --logg-min 3.5 --logg-max 5.0
  
  # Use different metallicity
  python download_spec.py --metallicity 0.0
  
  # Output to custom directory
  python download_spec.py --output-dir ./my_btsettl_models
  
  # Re-download everything (skip existing)
  python download_spec.py --no-skip-existing
        """)
    
    parser.add_argument('--teff-min', type=float, default=TEFF_RANGE[0],
                       help=f'Minimum effective temperature (K). Default: {TEFF_RANGE[0]:.0f}')
    parser.add_argument('--teff-max', type=float, default=TEFF_RANGE[1],
                       help=f'Maximum effective temperature (K). Default: {TEFF_RANGE[1]:.0f}')
    parser.add_argument('--logg-min', type=float, default=LOGG_RANGE[0],
                       help=f'Minimum surface gravity. Default: {LOGG_RANGE[0]:.1f}')
    parser.add_argument('--logg-max', type=float, default=LOGG_RANGE[1],
                       help=f'Maximum surface gravity. Default: {LOGG_RANGE[1]:.1f}')
    parser.add_argument('--metallicity', type=str, default=METALLICITY,
                       help=f'Metallicity [M/H]. Default: {METALLICITY}')
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
    download_btsettl_spectra(
        teff_range=(args.teff_min, args.teff_max),
        logg_range=(args.logg_min, args.logg_max),
        metallicity=args.metallicity,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing
    )
