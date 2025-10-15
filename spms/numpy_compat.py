"""
NumPy compatibility utilities for SPMS.

This module provides compatibility functions to handle differences between 
NumPy versions and ensure optimal performance with molecular descriptors.
"""

import warnings
from typing import Optional, Union, Tuple, List
import numpy as np


def ensure_numpy_compatibility() -> bool:
    """
    Check NumPy version compatibility and setup optimizations.
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        # Check NumPy version
        np_version = tuple(map(int, np.__version__.split('.')[:2]))
        
        if np_version < (1, 20):
            warnings.warn(
                f"NumPy version {np.__version__} detected. "
                "Please upgrade to NumPy >=1.20.0 for optimal performance.",
                UserWarning
            )
            return False
        
        # Test basic functionality
        test_array = np.random.random((100, 100))
        _ = np.mean(test_array, axis=0)
        _ = np.std(test_array, axis=0)
        
        return True
        
    except Exception as e:
        warnings.warn(f"NumPy compatibility check failed: {e}", UserWarning)
        return False


def safe_array_conversion(data, dtype=None) -> np.ndarray:
    """
    Safely convert data to NumPy array with compatibility across versions.
    
    Args:
        data: Input data to convert
        dtype: Target data type
        
    Returns:
        NumPy array
    """
    try:
        if dtype is not None:
            return np.array(data, dtype=dtype)
        return np.array(data)
    except Exception as e:
        warnings.warn(f"Array conversion failed: {e}", UserWarning)
        return np.asarray(data)


def safe_meshgrid(*arrays, indexing='xy', sparse=False) -> List[np.ndarray]:
    """
    Safely create meshgrid with NumPy compatibility.
    
    Args:
        arrays: Input arrays for meshgrid
        indexing: Indexing mode ('xy' or 'ij')
        sparse: Whether to return sparse arrays
        
    Returns:
        List of meshgrid arrays
    """
    try:
        return np.meshgrid(*arrays, indexing=indexing, sparse=sparse)
    except TypeError:
        # Fallback for older NumPy versions
        return np.meshgrid(*arrays)


def safe_cross_product(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate cross product with NumPy compatibility.
    
    Args:
        a: First array
        b: Second array
        axis: Axis along which to compute cross product
        
    Returns:
        Cross product array
    """
    try:
        return np.cross(a, b, axis=axis)
    except Exception:
        # Fallback for different array shapes
        if a.ndim == 1 and b.ndim == 1:
            return np.cross(a, b)
        elif a.ndim == 2 and b.ndim == 2:
            result = []
            for i in range(a.shape[0]):
                result.append(np.cross(a[i], b[i]))
            return np.array(result)
        else:
            # Generic fallback
            return np.cross(a, b)


def safe_arccos(x: np.ndarray, clip_values: bool = True) -> np.ndarray:
    """
    Calculate arccos with numerical stability.
    
    Args:
        x: Input array
        clip_values: Whether to clip values to valid range
        
    Returns:
        Arccos values
    """
    if clip_values:
        x = np.clip(x, -1.0, 1.0)
    
    try:
        return np.arccos(x)
    except Exception as e:
        warnings.warn(f"Arccos calculation failed: {e}", UserWarning)
        # Fallback with more aggressive clipping
        x_safe = np.clip(x, -0.9999, 0.9999)
        return np.arccos(x_safe)


def safe_arctan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculate arctan2 with numerical stability.
    
    Args:
        y: Y coordinate array
        x: X coordinate array
        
    Returns:
        Arctan2 values
    """
    try:
        return np.arctan2(y, x)
    except Exception as e:
        warnings.warn(f"Arctan2 calculation failed: {e}", UserWarning)
        # Element-wise fallback
        result = np.zeros_like(y)
        for i in range(y.size):
            try:
                result.flat[i] = np.arctan2(y.flat[i], x.flat[i])
            except:
                result.flat[i] = 0.0
        return result


def safe_norm(x: np.ndarray, axis=None, ord=2, keepdims=False) -> np.ndarray:
    """
    Calculate norm with NumPy compatibility.
    
    Args:
        x: Input array
        axis: Axis along which to compute norm
        ord: Order of norm
        keepdims: Whether to keep dimensions
        
    Returns:
        Norm values
    """
    try:
        return np.linalg.norm(x, axis=axis, ord=ord, keepdims=keepdims)
    except Exception:
        # Fallback for older NumPy versions
        if ord == 2:  # L2 norm
            if axis is None:
                return np.sqrt(np.sum(x**2))
            else:
                return np.sqrt(np.sum(x**2, axis=axis, keepdims=keepdims))
        elif ord == 1:  # L1 norm
            if axis is None:
                return np.sum(np.abs(x))
            else:
                return np.sum(np.abs(x), axis=axis, keepdims=keepdims)
        else:
            # Generic fallback
            if axis is None:
                return np.sum(np.abs(x)**ord)**(1/ord)
            else:
                return np.sum(np.abs(x)**ord, axis=axis, keepdims=keepdims)**(1/ord)


def safe_concatenate(arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
    """
    Safely concatenate arrays with compatibility across NumPy versions.
    
    Args:
        arrays: List of arrays to concatenate
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated array
    """
    try:
        return np.concatenate(arrays, axis=axis)
    except Exception as e:
        warnings.warn(f"Concatenation failed: {e}", UserWarning)
        # Fallback: try different methods
        try:
            return np.vstack(arrays) if axis == 0 else np.hstack(arrays)
        except:
            return np.array(arrays[0]) if len(arrays) == 1 else arrays[0]


def optimize_molecular_arrays(positions: np.ndarray, 
                             precision: int = 8) -> np.ndarray:
    """
    Optimize molecular coordinate arrays for memory and precision.
    
    Args:
        positions: Molecular coordinates
        precision: Decimal precision for rounding
        
    Returns:
        Optimized coordinate array
    """
    try:
        # Round to specified precision
        positions_rounded = np.round(positions, decimals=precision)
        
        # Optimize data type if possible
        if positions_rounded.dtype == np.float64:
            # Check if float32 precision is sufficient
            max_val = np.max(np.abs(positions_rounded))
            if max_val < 1e6:  # Float32 has ~7 decimal digits of precision
                return positions_rounded.astype(np.float32)
        
        return positions_rounded
        
    except Exception as e:
        warnings.warn(f"Array optimization failed: {e}", UserWarning)
        return positions


def validate_molecular_data(positions: np.ndarray, 
                           atom_types: Optional[List] = None,
                           name: str = "molecular_data") -> bool:
    """
    Validate molecular data arrays for common issues.
    
    Args:
        positions: Atomic coordinates
        atom_types: List of atomic numbers or symbols
        name: Name for error messages
        
    Returns:
        bool: True if valid
    """
    if not isinstance(positions, np.ndarray):
        warnings.warn(f"{name} positions is not a NumPy array", UserWarning)
        return False
    
    if positions.size == 0:
        warnings.warn(f"{name} positions array is empty", UserWarning)
        return False
    
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        warnings.warn(f"{name} positions should be (N, 3) array", UserWarning)
        return False
    
    if np.any(np.isnan(positions)):
        warnings.warn(f"{name} positions contains NaN values", UserWarning)
        return False
    
    if np.any(np.isinf(positions)):
        warnings.warn(f"{name} positions contains infinite values", UserWarning)
        return False
    
    if atom_types is not None:
        if len(atom_types) != positions.shape[0]:
            warnings.warn(
                f"{name} atom_types length doesn't match positions", 
                UserWarning
            )
            return False
    
    return True


def batch_process_conformers(conformer_list: List[np.ndarray], 
                           func: callable,
                           batch_size: int = 100) -> List:
    """
    Process conformers in batches to manage memory usage.
    
    Args:
        conformer_list: List of conformer coordinate arrays
        func: Function to apply to each conformer
        batch_size: Size of processing batches
        
    Returns:
        List of processed results
    """
    results = []
    n_conformers = len(conformer_list)
    
    for i in range(0, n_conformers, batch_size):
        end_idx = min(i + batch_size, n_conformers)
        batch = conformer_list[i:end_idx]
        
        batch_results = []
        for conformer in batch:
            try:
                result = func(conformer)
                batch_results.append(result)
            except Exception as e:
                warnings.warn(f"Error processing conformer: {e}", UserWarning)
                batch_results.append(None)
        
        results.extend(batch_results)
    
    return results


def safe_spherical_coordinates(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to spherical coordinates with numerical stability.
    
    Args:
        positions: Cartesian coordinates (N, 3)
        
    Returns:
        Tuple of (r, theta, phi) arrays
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    # Radius
    r = safe_norm(positions, axis=1)
    
    # Polar angle (theta): 0 to pi
    theta = safe_arccos(z / np.where(r == 0, 1, r))
    
    # Azimuthal angle (phi): 0 to 2pi
    phi = safe_arctan2(y, x)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    
    return r, theta, phi