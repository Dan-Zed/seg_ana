"""
This module patches matplotlib to use the Agg backend.
Import this module before any other imports to ensure matplotlib works in server environments.
"""
import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Force matplotlib to use non-interactive Agg backend
os.environ['MPLBACKEND'] = 'Agg'
logger.info("Forcing matplotlib to use Agg backend")

# Try to patch any existing matplotlib imports
try:
    import matplotlib
    if hasattr(matplotlib, 'use'):
        matplotlib.use('Agg', force=True)
        logger.info("Successfully patched matplotlib to use Agg backend")
except Exception as e:
    logger.warning(f"Failed to patch matplotlib: {e}")
