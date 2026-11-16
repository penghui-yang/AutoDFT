"""
Postprocessing modules for extracting and analyzing VASP outputs.

This package provides modular tools for:
- Bandgap analysis and electronic type classification (bandgap.py)
- DOS extraction (dos.py)
- Dielectric function analysis (dielectric.py)
- Phonon/Raman spectra and thermodynamics (phonon.py)
- MD trajectory analysis (md_analysis.py)
- NEB reaction path analysis (neb.py)
- Work function extraction (workfunction.py)
- Charged defect calculations (charged_defects.py)
"""

from .bandgap import (
    parse_bandgap_from_vaspkit,
    extract_bandgap_from_eigenval,
    classify_electronic_type,
    extract_electronic_type_from_calculation,
    BandgapResult,
    ElectronicTypeResult,
)
from .dos import (
    extract_dos_data,
    extract_pdos_data,
    DOSData
)
from .dielectric import (
    extract_dielectric_tensor,
    extract_optical_properties,
    extract_born_effective_charges,
    extract_frequency_dependent_dielectric,
    get_average_dielectric_constant,
    DielectricResult
)
from .phonon import (
    extract_phonon_frequencies,
    extract_raman_spectra,
    extract_ir_spectra,
    calculate_thermodynamic_properties,
    calculate_thermodynamic_properties_range,
    extract_thermodynamic_from_calculation,
    get_gibbs_free_energy,
    PhononResult,
    ThermodynamicResult,
)
from .md_analysis import (
    parse_xdatcar,
    extract_md_statistics,
    MDTrajectory,
    MDStatistics
)
from .neb import (
    extract_neb_path,
    calculate_barrier,
    interpolate_images,
    prepare_neb_directories,
    get_neb_incar_settings,
    plot_neb_path,
    extract_neb_forces,
    NEBResult
)
from .workfunction import (
    extract_workfunction,
    parse_locpot,
    WorkfunctionResult
)
from .charged_defects import (
    parse_charge_state_from_text,
    calculate_nelect,
    prepare_charged_defect_calculation,
    get_charged_defect_incar_settings,
    extract_defect_formation_energy,
    ChargedDefectSettings,
)
from .magnetic import (
    extract_magnetic_moments,
    compare_magnetic_configurations,
    calculate_exchange_coupling,
    generate_magmom_string,
    identify_magnetic_atoms,
    MagneticResult,
    MagneticComparisonResult,
)
from .gw import (
    extract_gw_results,
    extract_qp_corrections,
    get_gw_incar_settings,
    get_gw_dft_incar_settings,
    GWResult,
    QPEnergy,
)

__all__ = [
    # Bandgap and Electronic Type
    'parse_bandgap_from_vaspkit',
    'extract_bandgap_from_eigenval',
    'classify_electronic_type',
    'extract_electronic_type_from_calculation',
    'BandgapResult',
    'ElectronicTypeResult',
    # DOS
    'extract_dos_data',
    'extract_pdos_data',
    'DOSData',
    # Dielectric
    'extract_dielectric_tensor',
    'extract_optical_properties',
    'extract_born_effective_charges',
    'extract_frequency_dependent_dielectric',
    'get_average_dielectric_constant',
    'DielectricResult',
    # Phonon and Thermodynamics
    'extract_phonon_frequencies',
    'extract_raman_spectra',
    'extract_ir_spectra',
    'calculate_thermodynamic_properties',
    'calculate_thermodynamic_properties_range',
    'extract_thermodynamic_from_calculation',
    'get_gibbs_free_energy',
    'PhononResult',
    'ThermodynamicResult',
    # MD
    'parse_xdatcar',
    'extract_md_statistics',
    'MDTrajectory',
    'MDStatistics',
    # NEB
    'extract_neb_path',
    'calculate_barrier',
    'interpolate_images',
    'prepare_neb_directories',
    'get_neb_incar_settings',
    'plot_neb_path',
    'extract_neb_forces',
    'NEBResult',
    # Workfunction
    'extract_workfunction',
    'parse_locpot',
    'WorkfunctionResult',
    # Charged Defects
    'parse_charge_state_from_text',
    'calculate_nelect',
    'prepare_charged_defect_calculation',
    'get_charged_defect_incar_settings',
    'extract_defect_formation_energy',
    'ChargedDefectSettings',
    # Magnetic
    'extract_magnetic_moments',
    'compare_magnetic_configurations',
    'calculate_exchange_coupling',
    'generate_magmom_string',
    'identify_magnetic_atoms',
    'MagneticResult',
    'MagneticComparisonResult',
    # GW
    'extract_gw_results',
    'extract_qp_corrections',
    'get_gw_incar_settings',
    'get_gw_dft_incar_settings',
    'GWResult',
    'QPEnergy',
]
