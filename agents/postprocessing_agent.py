"""
Rule-based Postprocessing Agent.

Extracts structured property values from the VASP outputs of a completed
workflow. Pure file parsing; no LLM calls.
"""
from __future__ import annotations

import os
import re
import logging
from typing import Dict, Any, Optional

from common_types.workflow_models import WorkflowStep, ExecutionContext
from execution.job_manager import run_command
from tools.file_parsers import (
    extract_space_group,
    extract_magnetic_moment,
    extract_total_energy,
    get_composition_from_poscar,
    calculate_heat_of_formation,
    load_elemental_references,
    extract_raman_spectra
)
from tools.postprocessing import parse_bandgap_from_vaspkit


class PostprocessingAgent:
    """
    Agent specialized in extracting and analyzing VASP outputs.
    
    Extracts multiple properties from VASP calculations:
    - Bandgap (existing)
    - Space group number (new)
    - Magnetic moment (new)
    - Heat of formation (new)
    """
    
    def __init__(self):
        logging.info("PostprocessingAgent initialized")
        # Pre-load elemental references for heat of formation calculations
        self._elemental_references = None
    
    def execute_postprocessing(
        self,
        step: WorkflowStep,
        context: ExecutionContext,
        work_dir: str
    ) -> Dict[str, Any]:
        """
        Execute postprocessing actions defined in the step.
        
        Args:
            step: WorkflowStep with postprocessing instructions
            context: Execution context
            work_dir: Working directory
            
        Returns:
            Dictionary of extracted data
        """
        logging.info(f"[Postprocessing] Executing postprocessing for {step.name}")
        
        extracted_data = {}
        
        # Execute vaspkit postprocessing commands
        for cmd in step.vaspkit_actions:
            try:
                self._execute_vaspkit_postprocessing(cmd, work_dir)
            except Exception as e:
                logging.warning(f"Postprocessing command failed: {cmd}, error: {e}")
        
        # Extract data based on task
        if "bandgap" in step.name.lower() or "band" in step.name.lower():
            bandgap_data = self._extract_bandgap(work_dir, context.material_id)
            extracted_data.update(bandgap_data)
        
        if "dos" in step.name.lower():
            dos_data = self._extract_dos_info(work_dir)
            extracted_data.update(dos_data)
        
        if "energy" in step.name.lower():
            energy_data = self._extract_energy(work_dir)
            extracted_data.update(energy_data)
        
        return extracted_data
    
    def extract_all_properties(
        self,
        work_dir: str,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Extract all available properties from a completed calculation.
        
        This method extracts space group, magnetic moment, and heat of formation
        in addition to bandgap. It should be called after the main workflow completes.
        
        Args:
            work_dir: Working directory containing VASP outputs
            context: Execution context with material information
            
        Returns:
            Dictionary of all extracted properties
        """
        logging.info(f"[Postprocessing] Extracting all properties for {context.material_id}")
        
        extracted_data = {}
        
        # Find the relaxation directory (contains converged structure and energy)
        relax_dir = self._find_relaxation_dir(work_dir, context)
        
        # 1. Extract space group from relaxed structure
        spacegroup_data = self._extract_space_group(relax_dir)
        extracted_data.update(spacegroup_data)
        
        # 2. Extract magnetic moment from OUTCAR
        magmom_data = self._extract_magnetic_moment(relax_dir)
        extracted_data.update(magmom_data)
        
        # 3. Calculate heat of formation
        hof_data = self._calculate_heat_of_formation(relax_dir, context)
        extracted_data.update(hof_data)
        
        # 4. Extract Raman spectra data (if DFPT/phonon calculation was performed)
        raman_data = self._extract_raman_data(work_dir, context)
        extracted_data.update(raman_data)

        # 5. Bandgap from the most recent step with eigenvalues (band-structure steps preferred)
        band_dir = self._find_band_dir(context)
        if band_dir:
            extracted_data.update(self._extract_bandgap(band_dir, context.material_id))
        
        return extracted_data
    
    def _find_band_dir(self, context: ExecutionContext) -> Optional[str]:
        """Return the latest completed step directory with EIGENVAL, preferring band steps."""
        candidates = [
            (name, d) for name, d in (context.completed_steps or {}).items()
            if d and os.path.exists(os.path.join(d, "EIGENVAL"))
        ]
        if not candidates:
            return None
        band_steps = [d for name, d in candidates if "band" in name.lower()]
        return band_steps[-1] if band_steps else candidates[-1][1]

    def _find_relaxation_dir(self, work_dir: str, context: ExecutionContext) -> str:
        """
        Find the relaxation calculation directory.
        
        Args:
            work_dir: Base working directory
            context: Execution context
            
        Returns:
            Path to relaxation directory, or work_dir if not found
        """
        # Check completed steps for relaxation
        if hasattr(context, 'completed_steps') and context.completed_steps:
            for step_name, step_dir in context.completed_steps.items():
                if 'relax' in step_name.lower():
                    if os.path.exists(step_dir):
                        logging.info(f"Found relaxation directory: {step_dir}")
                        return step_dir
        
        # Try to find relaxation directory by name pattern
        material_dir = os.path.join(work_dir, context.material_id) if work_dir else context.material_id
        if os.path.exists(material_dir):
            for item in sorted(os.listdir(material_dir)):
                if 'relax' in item.lower():
                    relax_path = os.path.join(material_dir, item)
                    if os.path.isdir(relax_path):
                        logging.info(f"Found relaxation directory: {relax_path}")
                        return relax_path
        
        # Return work_dir if no specific relaxation directory found
        logging.warning(f"Relaxation directory not found, using: {work_dir}")
        return work_dir
    
    def _extract_space_group(self, work_dir: str) -> Dict[str, Any]:
        """
        Extract space group information from relaxed structure.
        
        Args:
            work_dir: Directory containing CONTCAR/POSCAR
            
        Returns:
            Dictionary with space group data
        """
        result = {
            'space_group_number': None,
            'space_group_symbol': None,
            'has_inversion_symmetry': None
        }
        
        # Prefer CONTCAR (relaxed structure), fall back to POSCAR
        contcar_path = os.path.join(work_dir, 'CONTCAR')
        poscar_path = os.path.join(work_dir, 'POSCAR')
        
        structure_file = None
        if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
            structure_file = contcar_path
        elif os.path.exists(poscar_path):
            structure_file = poscar_path
        
        if structure_file:
            try:
                spacegroup_data = extract_space_group(structure_file)
                result.update(spacegroup_data)
                logging.info(f"  ✓ Space group: {result.get('space_group_symbol')} "
                           f"({result.get('space_group_number')})")
            except Exception as e:
                logging.warning(f"Failed to extract space group: {e}")
        else:
            logging.warning(f"No structure file found in {work_dir}")
        
        return result
    
    def _extract_magnetic_moment(self, work_dir: str) -> Dict[str, Any]:
        """
        Extract magnetic moment from OUTCAR.
        
        Args:
            work_dir: Directory containing OUTCAR
            
        Returns:
            Dictionary with magnetic moment data
        """
        result = {
            'total_magnetic_moment': None,
            'is_magnetic': None
        }
        
        outcar_path = os.path.join(work_dir, 'OUTCAR')
        if os.path.exists(outcar_path):
            try:
                magmom_data = extract_magnetic_moment(outcar_path)
                result.update(magmom_data)
                if result.get('total_magnetic_moment') is not None:
                    logging.info(f"  ✓ Magnetic moment: {result['total_magnetic_moment']:.4f} μ_B")
            except Exception as e:
                logging.warning(f"Failed to extract magnetic moment: {e}")
        else:
            logging.warning(f"OUTCAR not found in {work_dir}")
        
        return result
    
    def _calculate_heat_of_formation(
        self,
        work_dir: str,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Calculate heat of formation from DFT results.
        
        Args:
            work_dir: Directory containing OUTCAR and POSCAR/CONTCAR
            context: Execution context
            
        Returns:
            Dictionary with heat of formation data
        """
        result = {
            'heat_of_formation': None
        }
        
        # Load elemental references (cache for efficiency)
        if self._elemental_references is None:
            self._elemental_references = load_elemental_references()
        
        if not self._elemental_references:
            logging.warning("Elemental references not available, skipping heat of formation")
            return result
        
        # Extract total energy
        outcar_path = os.path.join(work_dir, 'OUTCAR')
        total_energy = extract_total_energy(outcar_path) if os.path.exists(outcar_path) else None
        
        if total_energy is None:
            logging.warning(f"Could not extract total energy from {work_dir}")
            return result
        
        # Get composition from structure file
        contcar_path = os.path.join(work_dir, 'CONTCAR')
        poscar_path = os.path.join(work_dir, 'POSCAR')
        
        structure_file = None
        if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
            structure_file = contcar_path
        elif os.path.exists(poscar_path):
            structure_file = poscar_path
        
        if structure_file is None:
            logging.warning(f"No structure file found in {work_dir}")
            return result
        
        composition = get_composition_from_poscar(structure_file)
        if not composition:
            logging.warning("Could not extract composition")
            return result
        
        # Calculate heat of formation
        try:
            hof_data = calculate_heat_of_formation(
                total_energy=total_energy,
                composition=composition,
                elemental_references=self._elemental_references
            )
            
            if hof_data.get('heat_of_formation') is not None:
                result['heat_of_formation'] = hof_data['heat_of_formation']
                logging.info(f"  ✓ Heat of formation: {result['heat_of_formation']:.4f} eV/atom")
            
        except Exception as e:
            logging.warning(f"Failed to calculate heat of formation: {e}")
        
        return result
    
    def _extract_raman_data(self, work_dir: str, context: ExecutionContext) -> Dict[str, Any]:
        """
        Extract Raman spectra data from DFPT/phonon calculation.
        
        Searches for Raman/DFPT/phonon calculation directories and extracts
        Raman frequencies and activities.
        
        Args:
            work_dir: Base working directory
            context: Execution context
            
        Returns:
            Dictionary with Raman data (or empty fields if not found)
        """
        result = {
            'has_raman_data': False,
            'raman_frequencies': [],
            'raman_activities': [],
            'n_raman_modes': 0,
            'ir_frequencies': [],
            'n_ir_modes': 0
        }
        
        # Find DFPT/Raman/phonon calculation directory
        raman_dir = self._find_raman_dir(work_dir, context)
        
        if not raman_dir:
            logging.info("  - No Raman/DFPT calculation directory found, skipping Raman extraction")
            return result
        
        logging.info(f"  Extracting Raman data from: {raman_dir}")
        
        try:
            raman_data = extract_raman_spectra(raman_dir, context.material_id)
            result.update(raman_data)
            
            if result.get('has_raman_data'):
                logging.info(f"  ✓ Raman data: {result.get('n_raman_modes', 0)} Raman-active modes, "
                           f"{result.get('n_ir_modes', 0)} IR-active modes")
            else:
                logging.info("  - No Raman data extracted (calculation may not have completed)")
                
        except Exception as e:
            logging.warning(f"Failed to extract Raman data: {e}")
        
        return result
    
    def _find_raman_dir(self, work_dir: str, context: ExecutionContext) -> Optional[str]:
        """
        Find the DFPT/Raman/phonon calculation directory containing actual phonon data.
        
        The key is to find a directory that contains DFPT calculation results:
        - Has DYNMAT file, OR
        - Has OUTCAR with phonon frequencies, OR  
        - Has IBRION=7/8 in INCAR (actual DFPT calculation)
        
        Uses keyword-based pre-filtering then validates with actual file checks.
        
        Args:
            work_dir: Base working directory
            context: Execution context
            
        Returns:
            Path to Raman/DFPT directory, or None if not found
        """
        # Keywords for identifying potential DFPT directories
        dfpt_keywords = ['dfpt', 'phonon', 'vibration', 'raman']
        
        # Keywords that indicate post-processing or preparatory steps (not actual DFPT)
        exclude_keywords = ['postprocess', 'post_process', 'validation', 'analysis', 
                          'scf_for', 'prep', 'static_scf']
        
        def is_excluded_step(name_lower: str) -> bool:
            """Check if this is a post-processing or prep step."""
            for excl in exclude_keywords:
                if excl in name_lower:
                    return True
            return False
        
        def has_phonon_data(dir_path: str) -> bool:
            """Check if directory contains actual phonon/DFPT calculation results."""
            # Check for DYNMAT file (strongest indicator of completed DFPT)
            if os.path.exists(os.path.join(dir_path, 'DYNMAT')):
                return True
            
            # Check INCAR for IBRION=7 or 8
            incar_path = os.path.join(dir_path, 'INCAR')
            if os.path.exists(incar_path):
                try:
                    with open(incar_path, 'r') as f:
                        content = f.read().upper()
                        if 'IBRION' in content:
                            import re
                            match = re.search(r'IBRION\s*=\s*(\d+)', content)
                            if match:
                                ibrion = int(match.group(1))
                                if ibrion in [7, 8]:
                                    return True
                except Exception:
                    pass
            
            # Check OUTCAR for phonon frequency patterns
            outcar_path = os.path.join(dir_path, 'OUTCAR')
            if os.path.exists(outcar_path):
                try:
                    with open(outcar_path, 'r') as f:
                        # Read last 50KB to check for phonon output
                        f.seek(0, 2)  # Go to end
                        size = f.tell()
                        f.seek(max(0, size - 50000))  # Read last 50KB
                        content = f.read()
                        # Look for phonon frequency output pattern
                        if 'f  =' in content and 'THz' in content and 'cm-1' in content:
                            return True
                except Exception:
                    pass
            
            return False
        
        def find_dfpt_dir_in(search_dir: str) -> Optional[str]:
            """Find DFPT directory within a search directory."""
            if not os.path.exists(search_dir):
                return None
            
            candidates = []
            for item in sorted(os.listdir(search_dir)):
                item_lower = item.lower()
                path = os.path.join(search_dir, item)
                
                if not os.path.isdir(path):
                    continue
                
                # Skip excluded directories
                if is_excluded_step(item_lower):
                    continue
                
                # Check if name contains relevant keywords
                has_keyword = any(kw in item_lower for kw in dfpt_keywords)
                if not has_keyword:
                    continue
                
                # Verify it has actual phonon data
                if has_phonon_data(path):
                    # Extract step number for sorting
                    step_num = 0
                    parts = item.split('_')
                    if parts and parts[0].isdigit():
                        step_num = int(parts[0])
                    candidates.append((step_num, path))
            
            if candidates:
                # Prefer highest step number (latest calculation)
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
            
            return None
        
        # Strategy 1: Check completed_steps from context
        if hasattr(context, 'completed_steps') and context.completed_steps:
            candidates = []
            for step_name, step_dir in context.completed_steps.items():
                step_name_lower = step_name.lower()
                
                # Skip excluded steps
                if is_excluded_step(step_name_lower):
                    continue
                
                # Check if name contains relevant keywords
                has_keyword = any(kw in step_name_lower for kw in dfpt_keywords)
                if not has_keyword:
                    continue
                
                # Verify directory exists and has phonon data
                if os.path.exists(step_dir) and has_phonon_data(step_dir):
                    # Extract step number from directory name
                    step_num = 0
                    dir_name = os.path.basename(step_dir)
                    parts = dir_name.split('_')
                    if parts and parts[0].isdigit():
                        step_num = int(parts[0])
                    candidates.append((step_num, step_dir))
            
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                result = candidates[0][1]
                logging.debug(f"Found Raman/DFPT directory from completed_steps: {result}")
                return result
        
        # Strategy 2: Search in material folder
        material_dir = os.path.join(work_dir, context.material_id) if work_dir else context.material_id
        result = find_dfpt_dir_in(material_dir)
        if result:
            logging.debug(f"Found Raman/DFPT directory in material folder: {result}")
            return result
        
        # Strategy 3: Search in work_dir directly
        result = find_dfpt_dir_in(work_dir)
        if result:
            logging.debug(f"Found Raman/DFPT directory in work_dir: {result}")
            return result
        
        return None
    
    def _execute_vaspkit_postprocessing(self, cmd: str, work_dir: str):
        """Execute a VASPKIT postprocessing command."""
        # Parse command description and map to vaspkit tasks
        if "extract eigenvalues" in cmd.lower() or "band" in cmd.lower():
            # Use vaspkit to process band structure
            # Task 211: extract band eigenvalues
            run_command("vaspkit -task 211", cwd=work_dir, shell=True)
            logging.info("  Extracted band eigenvalues")
        
        elif "dos" in cmd.lower():
            # Task 111: total DOS
            run_command("vaspkit -task 111", cwd=work_dir, shell=True)
            logging.info("  Extracted DOS")
    
    def _extract_bandgap(self, work_dir: str, material_id: str) -> Dict[str, Any]:
        """Extract bandgap information from VASP outputs."""
        try:
            # Use existing parser
            bandgap_result = parse_bandgap_from_vaspkit(work_dir, material_id)
            return bandgap_result
        except Exception as e:
            logging.warning(f"Could not extract bandgap: {e}")
            return {"calculated_gap": None, "calculated_character": "None"}
    
    def _extract_dos_info(self, work_dir: str) -> Dict[str, Any]:
        """Extract DOS information."""
        dos_data = {}
        
        # Check for DOSCAR
        doscar_path = os.path.join(work_dir, "DOSCAR")
        if os.path.exists(doscar_path):
            dos_data["doscar_present"] = True
            # Could parse DOSCAR for more details
        
        return dos_data
    
    def _extract_energy(self, work_dir: str) -> Dict[str, Any]:
        """Extract final energy from calculation."""
        energy_data = {}
        
        outcar_path = os.path.join(work_dir, "OUTCAR")
        if os.path.exists(outcar_path):
            try:
                with open(outcar_path, 'r') as f:
                    lines = f.readlines()
                
                # Find last occurrence of "free  energy"
                for line in reversed(lines):
                    if "free  energy" in line and "TOTEN" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "TOTEN":
                                try:
                                    energy = float(parts[i+2])
                                    energy_data["final_energy"] = energy
                                    energy_data["energy_units"] = "eV"
                                    break
                                except (IndexError, ValueError):
                                    pass
                        break
            except Exception as e:
                logging.warning(f"Could not extract energy: {e}")
        
        return energy_data
