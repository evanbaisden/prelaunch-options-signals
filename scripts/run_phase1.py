#!/usr/bin/env python3
"""
Phase 1 Analysis Runner Script

A convenient wrapper script for running the complete Phase 1 prelaunch options signals analysis.
This script handles environment setup, provides command-line interface, and offers different
execution modes for development and production use.

Usage:
    python scripts/run_phase1.py                    # Run with default settings
    python scripts/run_phase1.py --config .env.dev  # Use custom config file
    python scripts/run_phase1.py --verbose          # Enable verbose logging
    python scripts/run_phase1.py --dry-run          # Validate setup without running analysis
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase1 import Phase1Analyzer
from src.common.utils import load_env_file, setup_logging
from src.config import Config


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 prelaunch options signals analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default analysis
  %(prog)s --verbose                # Verbose output
  %(prog)s --config .env.prod       # Custom config
  %(prog)s --data-dir data/custom   # Custom data directory
  %(prog)s --dry-run                # Validate setup only
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to environment configuration file (default: .env)'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        help='Directory containing raw stock data CSV files'
    )
    
    parser.add_argument(
        '--results-dir', '-r', 
        type=str,
        help='Directory for output results and reports'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and data availability without running analysis'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true', 
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def validate_data_files(data_dir: Path) -> dict:
    """
    Validate that required data files exist.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with validation results
    """
    required_files = [
        "microsoft_xbox_series_x-s_raw.csv",
        "nvidia_rtx_30_series_raw.csv", 
        "nvidia_rtx_40_series_raw.csv",
        "nvidia_rtx_40_super_raw.csv",
        "apple_iphone_12_raw.csv",
        "apple_iphone_13_raw.csv",
        "apple_iphone_14_raw.csv",
        "apple_iphone_15_raw.csv"
    ]
    
    validation = {
        'data_dir_exists': data_dir.exists(),
        'missing_files': [],
        'existing_files': [],
        'total_required': len(required_files)
    }
    
    if validation['data_dir_exists']:
        for filename in required_files:
            filepath = data_dir / filename
            if filepath.exists():
                validation['existing_files'].append(filename)
            else:
                validation['missing_files'].append(filename)
    
    validation['files_found'] = len(validation['existing_files'])
    validation['complete'] = len(validation['missing_files']) == 0
    
    return validation


def print_validation_results(validation: dict, quiet: bool = False):
    """Print data validation results."""
    if quiet:
        return
    
    print("=" * 60)
    print("DATA VALIDATION RESULTS")
    print("=" * 60)
    
    if not validation['data_dir_exists']:
        print("❌ Data directory does not exist")
        return
    
    print(f"📁 Data directory: EXISTS")
    print(f"📊 Data files: {validation['files_found']}/{validation['total_required']} found")
    
    if validation['complete']:
        print("✅ All required data files are available")
    else:
        print(f"⚠️  Missing {len(validation['missing_files'])} data files:")
        for filename in validation['missing_files']:
            print(f"   - {filename}")
    
    print()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load environment configuration
        if args.config:
            if not load_env_file(args.config):
                print(f"Warning: Config file {args.config} not found, using defaults")
        else:
            load_env_file()  # Load default .env if exists
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else ("ERROR" if args.quiet else "INFO")
        setup_logging(log_level)
        
        # Initialize configuration
        config = Config()
        
        # Override directories if specified
        data_dir = Path(args.data_dir) if args.data_dir else Path(config.data_raw_dir)
        results_dir = Path(args.results_dir) if args.results_dir else Path(config.results_dir)
        
        # Validate data files
        validation = validate_data_files(data_dir)
        print_validation_results(validation, args.quiet)
        
        if not validation['data_dir_exists']:
            print(f"❌ Error: Data directory '{data_dir}' does not exist")
            return 1
        
        if not validation['complete'] and not args.dry_run:
            print("⚠️  Warning: Some data files are missing. Analysis may fail for those products.")
            
            response = input("Continue anyway? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Analysis cancelled.")
                return 1
        
        # Dry run - just validate and exit
        if args.dry_run:
            if validation['complete']:
                print("✅ Dry run complete: All systems ready for analysis")
                return 0
            else:
                print("⚠️  Dry run complete: Missing data files detected")
                return 1
        
        # Initialize and run analyzer
        if not args.quiet:
            print("🚀 Starting Phase 1 Analysis...")
        
        analyzer = Phase1Analyzer(data_dir=str(data_dir), results_dir=str(results_dir))
        results = analyzer.run_complete_analysis()
        
        # Print results summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print("PHASE 1 ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"✅ Products analyzed: {results['successful_analyses']}/{results['total_products_configured']}")
            print(f"📈 Results saved to: {results['output_files']['summary_csv']}")
            print(f"📋 Report saved to: {results['output_files']['analysis_report']}")
            
            if results['errors']:
                print(f"\n⚠️  Errors encountered: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"   - {error['product']}: {error['error']}")
            else:
                print("\n🎉 All analyses completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())