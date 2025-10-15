#!/usr/bin/env python3
"""
SPMS CLI interface.

Modern command-line interface for Spherical Projection Molecular Surface (SPMS) 
descriptors with rich output formatting.
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.json import JSON
import warnings

from spms.numpy_compat import ensure_numpy_compatibility, validate_molecular_data
from spms.desc import SPMS

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    SPMS: Spherical Projection Molecular Surface descriptors.
    
    Generate molecular descriptors from 3D conformers using spherical projection.
    """
    # Check NumPy compatibility on startup
    if not ensure_numpy_compatibility():
        console.print("[yellow]Warning: NumPy compatibility issues detected[/yellow]")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for descriptors (default: input_file_descriptors.json)')
@click.option('--sphere-radius', '-r', type=float,
              help='Radius of the spherical surface enclosing the molecule')
@click.option('--desc-n', '-n', default=40, type=int,
              help='Descriptor latitudinal resolution (default: 40)')
@click.option('--desc-m', '-m', default=40, type=int,
              help='Descriptor longitudinal resolution (default: 40)')
@click.option('--key-atoms', '-k', multiple=True, type=int,
              help='Key atomic indices for constraining molecular orientation (1-indexed)')
@click.option('--no-standardize', is_flag=True,
              help='Disable molecular orientation standardization')
@click.option('--format', 'output_format', default='json',
              type=click.Choice(['json', 'csv', 'npy']),
              help='Output format')
def calculate(input_file, output, sphere_radius, desc_n, desc_m, key_atoms, 
              no_standardize, output_format):
    """
    Calculate SPMS descriptors from molecular structure.
    
    INPUT_FILE: SDF or XYZ file containing molecular structure
    """
    try:
        # Determine file type
        file_path = Path(input_file)
        if file_path.suffix.lower() == '.sdf':
            sdf_file = str(input_file)
            xyz_file = None
        elif file_path.suffix.lower() == '.xyz':
            sdf_file = None
            xyz_file = str(input_file)
        else:
            console.print(f"[red]Error: Unsupported file format: {file_path.suffix}[/red]")
            console.print("[yellow]Supported formats: .sdf, .xyz[/yellow]")
            sys.exit(1)
        
        # Set up output path
        if output is None:
            output = file_path.parent / f"{file_path.stem}_descriptors.{output_format}"
        
        console.print(f"[bold blue]Calculating SPMS descriptors[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output}")
        console.print(f"Resolution: {desc_n} × {desc_m}")
        
        # Prepare key atoms (convert to 0-indexed)
        key_atom_list = list(key_atoms) if key_atoms else None
        
        # Create SPMS object
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing SPMS...", total=None)
            
            spms = SPMS(
                sdf_file=sdf_file,
                xyz_file=xyz_file,
                key_atom_num=key_atom_list,
                sphere_radius=sphere_radius,
                desc_n=desc_n,
                desc_m=desc_m,
                orientation_standard=not no_standardize
            )
            
            progress.update(task, description="Computing sphere descriptors...")
            sphere_descriptors = spms.GetSphereDescriptors()
            
            progress.update(task, description="Computing quarter descriptors...")
            quarter_descriptors = spms.GetQuaterDescriptors()
            
            progress.update(task, description="✅ Descriptors computed successfully!")
        
        # Prepare results
        results = {
            'input_file': str(input_file),
            'sphere_radius': float(spms.sphere_radius),
            'desc_n': desc_n,
            'desc_m': desc_m,
            'orientation_standardized': not no_standardize,
            'key_atoms': key_atom_list,
            'sphere_descriptors': sphere_descriptors.tolist(),
            'quarter_descriptors': {
                'left_top': float(quarter_descriptors[0]),
                'right_top': float(quarter_descriptors[1]),
                'left_bottom': float(quarter_descriptors[2]),
                'right_bottom': float(quarter_descriptors[3])
            },
            'descriptor_statistics': {
                'mean': float(np.mean(sphere_descriptors)),
                'std': float(np.std(sphere_descriptors)),
                'min': float(np.min(sphere_descriptors)),
                'max': float(np.max(sphere_descriptors)),
                'sum': float(np.sum(sphere_descriptors))
            }
        }
        
        # Save results
        if output_format == 'json':
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_format == 'csv':
            import csv
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['descriptor_type', 'value'])
                writer.writerow(['sphere_radius', results['sphere_radius']])
                for key, value in results['quarter_descriptors'].items():
                    writer.writerow([f'quarter_{key}', value])
                for key, value in results['descriptor_statistics'].items():
                    writer.writerow([f'stat_{key}', value])
        elif output_format == 'npy':
            np.save(output, sphere_descriptors)
        
        # Display results summary
        table = Table(title="SPMS Calculation Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input file", str(input_file))
        table.add_row("Molecule atoms", str(len(spms.atom_types)))
        table.add_row("Sphere radius", f"{spms.sphere_radius:.3f}")
        table.add_row("Resolution", f"{desc_n} × {desc_m}")
        table.add_row("Orientation standardized", str(not no_standardize))
        table.add_row("Descriptor mean", f"{results['descriptor_statistics']['mean']:.6f}")
        table.add_row("Descriptor std", f"{results['descriptor_statistics']['std']:.6f}")
        table.add_row("Output file", str(output))
        
        console.print(table)
        
        # Quarter descriptors summary
        quarter_table = Table(title="Quarter Descriptors")
        quarter_table.add_column("Quarter", style="cyan")
        quarter_table.add_column("Value", style="green")
        quarter_table.add_column("Percentage", style="yellow")
        
        total = sum(results['quarter_descriptors'].values())
        for quarter, value in results['quarter_descriptors'].items():
            percentage = (value / total * 100) if total > 0 else 0
            quarter_table.add_row(quarter.replace('_', ' ').title(), 
                                f"{value:.6f}", f"{percentage:.2f}%")
        
        console.print(quarter_table)
        
    except ImportError as e:
        console.print(f"[red]Error: Required modules not available: {e}[/red]")
        console.print("[yellow]Please ensure RDKit and ASE are properly installed[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error calculating descriptors: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for batch results')
@click.option('--desc-n', '-n', default=40, type=int,
              help='Descriptor latitudinal resolution (default: 40)')
@click.option('--desc-m', '-m', default=40, type=int,
              help='Descriptor longitudinal resolution (default: 40)')
@click.option('--format', 'output_format', default='json',
              type=click.Choice(['json', 'csv', 'npy']),
              help='Output format')
def batch(input_file, output, desc_n, desc_m, output_format):
    """
    Process multiple conformers from a single file.
    
    INPUT_FILE: Multi-conformer SDF file
    """
    try:
        from rdkit import Chem
        
        file_path = Path(input_file)
        
        if output is None:
            output = file_path.parent / f"{file_path.stem}_batch_results"
            output.mkdir(exist_ok=True)
        else:
            output = Path(output)
            output.mkdir(exist_ok=True)
        
        console.print(f"[bold blue]Batch processing conformers[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output directory: {output}")
        
        # Read multi-conformer SDF
        supplier = Chem.SDMolSupplier(str(input_file), removeHs=False, sanitize=False)
        conformers = [mol for mol in supplier if mol is not None]
        
        if not conformers:
            console.print("[red]Error: No valid conformers found in file[/red]")
            sys.exit(1)
        
        console.print(f"Found {len(conformers)} conformers")
        
        batch_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing conformers...", total=len(conformers))
            
            for i, mol in enumerate(conformers):
                try:
                    # Save temporary SDF for this conformer
                    temp_sdf = output / f"temp_conformer_{i}.sdf"
                    writer = Chem.SDWriter(str(temp_sdf))
                    writer.write(mol)
                    writer.close()
                    
                    # Calculate SPMS descriptors
                    spms = SPMS(
                        sdf_file=str(temp_sdf),
                        desc_n=desc_n,
                        desc_m=desc_m,
                        orientation_standard=True
                    )
                    
                    sphere_descriptors = spms.GetSphereDescriptors()
                    quarter_descriptors = spms.GetQuaterDescriptors()
                    
                    # Store results
                    result = {
                        'conformer_id': i,
                        'sphere_radius': float(spms.sphere_radius),
                        'sphere_descriptors': sphere_descriptors.tolist(),
                        'quarter_descriptors': {
                            'left_top': float(quarter_descriptors[0]),
                            'right_top': float(quarter_descriptors[1]),
                            'left_bottom': float(quarter_descriptors[2]),
                            'right_bottom': float(quarter_descriptors[3])
                        },
                        'statistics': {
                            'mean': float(np.mean(sphere_descriptors)),
                            'std': float(np.std(sphere_descriptors)),
                            'min': float(np.min(sphere_descriptors)),
                            'max': float(np.max(sphere_descriptors))
                        }
                    }
                    
                    batch_results.append(result)
                    
                    # Clean up temporary file
                    temp_sdf.unlink()
                    
                    progress.update(task, advance=1, 
                                  description=f"Processed conformer {i+1}/{len(conformers)}")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to process conformer {i}: {e}[/yellow]")
                    continue
            
            progress.update(task, description="✅ Batch processing complete!")
        
        # Save batch results
        output_file = output / f"batch_results.{output_format}"
        
        if output_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(batch_results, f, indent=2)
        elif output_format == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['conformer_id', 'sphere_radius', 'quarter_left_top', 
                         'quarter_right_top', 'quarter_left_bottom', 'quarter_right_bottom',
                         'stat_mean', 'stat_std', 'stat_min', 'stat_max']
                writer.writerow(header)
                
                for result in batch_results:
                    row = [
                        result['conformer_id'],
                        result['sphere_radius'],
                        result['quarter_descriptors']['left_top'],
                        result['quarter_descriptors']['right_top'],
                        result['quarter_descriptors']['left_bottom'],
                        result['quarter_descriptors']['right_bottom'],
                        result['statistics']['mean'],
                        result['statistics']['std'],
                        result['statistics']['min'],
                        result['statistics']['max']
                    ]
                    writer.writerow(row)
        elif output_format == 'npy':
            # Save as a 3D array: (n_conformers, desc_n, desc_m)
            all_descriptors = [result['sphere_descriptors'] for result in batch_results]
            np.save(output_file, np.array(all_descriptors))
        
        # Display summary
        table = Table(title="Batch Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total conformers", str(len(conformers)))
        table.add_row("Successfully processed", str(len(batch_results)))
        table.add_row("Failed", str(len(conformers) - len(batch_results)))
        table.add_row("Resolution", f"{desc_n} × {desc_m}")
        table.add_row("Output file", str(output_file))
        
        console.print(table)
        
    except ImportError as e:
        console.print(f"[red]Error: RDKit not available: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error in batch processing: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('descriptors_file', type=click.Path(exists=True))
@click.option('--method', default='correlation',
              type=click.Choice(['correlation', 'pca', 'variance']),
              help='Analysis method')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for analysis results')
def analyze(descriptors_file, method, output):
    """
    Analyze computed SPMS descriptors.
    
    DESCRIPTORS_FILE: JSON file containing computed descriptors
    """
    try:
        # Load descriptors
        with open(descriptors_file, 'r') as f:
            data = json.load(f)
        
        if output is None:
            input_path = Path(descriptors_file)
            output = input_path.parent / f"{input_path.stem}_analysis_{method}.json"
        
        console.print(f"[bold blue]Analyzing SPMS descriptors[/bold blue]")
        console.print(f"Input: {descriptors_file}")
        console.print(f"Method: {method}")
        console.print(f"Output: {output}")
        
        if isinstance(data, list):
            # Batch results
            all_descriptors = np.array([item['sphere_descriptors'] for item in data])
            quarter_data = [item['quarter_descriptors'] for item in data]
        else:
            # Single result
            all_descriptors = np.array([data['sphere_descriptors']])
            quarter_data = [data['quarter_descriptors']]
        
        analysis_results = {'method': method}
        
        if method == 'correlation':
            # Analyze quarter descriptor correlations
            quarters = ['left_top', 'right_top', 'left_bottom', 'right_bottom']
            quarter_matrix = np.array([[q[quarter] for quarter in quarters] for q in quarter_data])
            
            if quarter_matrix.shape[0] > 1:
                corr_matrix = np.corrcoef(quarter_matrix.T)
                analysis_results['quarter_correlations'] = {
                    f"{quarters[i]}_{quarters[j]}": float(corr_matrix[i, j])
                    for i in range(len(quarters)) for j in range(i+1, len(quarters))
                }
            else:
                analysis_results['quarter_correlations'] = "Insufficient data for correlation"
        
        elif method == 'pca':
            # Principal component analysis
            from sklearn.decomposition import PCA
            
            # Flatten descriptors for PCA
            n_conformers, n_desc, m_desc = all_descriptors.shape
            flat_descriptors = all_descriptors.reshape(n_conformers, -1)
            
            pca = PCA(n_components=min(5, n_conformers, flat_descriptors.shape[1]))
            pca_result = pca.fit_transform(flat_descriptors)
            
            analysis_results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components': pca.n_components_,
                'principal_components': pca_result.tolist()
            }
        
        elif method == 'variance':
            # Variance analysis
            if all_descriptors.shape[0] > 1:
                variance_map = np.var(all_descriptors, axis=0)
                analysis_results['variance_analysis'] = {
                    'variance_map': variance_map.tolist(),
                    'max_variance': float(np.max(variance_map)),
                    'min_variance': float(np.min(variance_map)),
                    'mean_variance': float(np.mean(variance_map)),
                    'std_variance': float(np.std(variance_map))
                }
            else:
                analysis_results['variance_analysis'] = "Single conformer - no variance"
        
        # Save analysis results
        with open(output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Display summary
        table = Table(title="Analysis Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Analysis method", method)
        table.add_row("Input conformers", str(all_descriptors.shape[0]))
        table.add_row("Descriptor dimensions", f"{all_descriptors.shape[1]} × {all_descriptors.shape[2]}")
        table.add_row("Output file", str(output))
        
        if method == 'pca' and 'pca' in analysis_results:
            total_var = sum(analysis_results['pca']['explained_variance_ratio'])
            table.add_row("Total variance explained", f"{total_var:.3f}")
        
        console.print(table)
        
    except ImportError as e:
        console.print(f"[red]Error: Required analysis modules not available: {e}[/red]")
        console.print("[yellow]Install with: pip install scikit-learn[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error in analysis: {e}[/red]")
        sys.exit(1)


@cli.command()
def info():
    """Display system information and SPMS capabilities."""
    
    # System info panel
    info_table = Table(title="SPMS System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Version/Status", style="green")
    
    # Python and package versions
    info_table.add_row("Python", f"{sys.version.split()[0]}")
    
    try:
        import numpy as np
        info_table.add_row("NumPy", np.__version__)
        compat_status = "✅ Compatible" if ensure_numpy_compatibility() else "⚠️ Issues detected"
        info_table.add_row("NumPy Compatibility", compat_status)
    except ImportError:
        info_table.add_row("NumPy", "❌ Not installed")
    
    try:
        import ase
        info_table.add_row("ASE", ase.__version__)
    except ImportError:
        info_table.add_row("ASE", "❌ Not installed")
    
    try:
        import rdkit
        info_table.add_row("RDKit", rdkit.__version__)
    except ImportError:
        info_table.add_row("RDKit", "❌ Not installed")
    
    try:
        import sklearn
        info_table.add_row("Scikit-learn", sklearn.__version__)
    except ImportError:
        info_table.add_row("Scikit-learn", "❌ Not installed (optional)")
    
    console.print(info_table)
    
    # Features
    features_panel = Panel(
        """
🧬 [bold]SPMS Features:[/bold]
• Spherical Projection Molecular Surface descriptors
• Support for SDF and XYZ molecular formats
• Configurable spherical resolution (n × m grid)
• Molecular orientation standardization
• Quarter descriptor analysis
• Batch processing for multiple conformers
• Descriptor analysis (PCA, correlation, variance)
• NumPy 2.x compatibility
• Rich CLI interface with progress tracking

📚 [bold]Usage Examples:[/bold]
• spms calculate molecule.sdf --desc-n 50 --desc-m 50
• spms batch conformers.sdf --format npy
• spms analyze descriptors.json --method pca
• spms info

📖 [bold]Algorithm:[/bold]
SPMS generates molecular descriptors by projecting the molecular surface
onto a spherical grid, capturing 3D structural information in a 
rotation-invariant descriptor format suitable for machine learning.

🔧 [bold]Requirements:[/bold]
• Core: numpy, ase, rdkit-pypi
• Analysis: scikit-learn (optional)
• CLI: click, rich
        """,
        title="SPMS Capabilities",
        border_style="blue"
    )
    
    console.print(features_panel)


if __name__ == '__main__':
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    cli()