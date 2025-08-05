"""
Thermal Models for InGaN/GaN HEMT Simulation
Includes heat equation solvers, thermal resistance networks, and coupled electro-thermal models
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Callable, Optional
import sys
sys.path.append('../../materials')
from gan_properties import (GaNProperties, InGaNProperties, AlNProperties,
                           SiCProperties, SiProperties, SapphireProperties,
                           thermal_boundary_resistance)


class ThermalModel:
    """Base class for thermal modeling of HEMT devices"""
    
    def __init__(self, device_structure, ambient_temperature: float = 300):
        """
        Initialize thermal model
        
        Args:
            device_structure: HEMTStructure object
            ambient_temperature: Ambient temperature in K
        """
        self.device = device_structure
        self.T_ambient = ambient_temperature
        self.material_properties = self._load_material_properties()
        
    def _load_material_properties(self) -> Dict:
        """Load material properties for all layers"""
        properties = {}
        
        # Load properties for each unique material
        for layer in self.device.layers:
            if layer.material not in properties:
                if layer.material == "GaN":
                    properties["GaN"] = GaNProperties.get_properties()
                elif layer.material == "InGaN" and layer.composition:
                    x = layer.composition.get('In', 0.0)
                    properties["InGaN"] = InGaNProperties(x).get_properties()
                elif layer.material == "AlN":
                    properties["AlN"] = AlNProperties.get_properties()
                elif layer.material == "SiC":
                    properties["SiC"] = SiCProperties.get_properties()
                elif layer.material == "Si":
                    properties["Si"] = SiProperties.get_properties()
                elif layer.material == "Sapphire":
                    properties["Sapphire"] = SapphireProperties.get_properties()
                # Add more materials as needed
                
        return properties


class HeatEquationSolver(ThermalModel):
    """
    Finite difference solver for the heat equation
    ∇·(k∇T) + Q = ρc∂T/∂t
    """
    
    def __init__(self, device_structure, ambient_temperature: float = 300,
                 nx: int = 100, ny: int = 50, nz: int = 100):
        """
        Initialize heat equation solver with mesh
        
        Args:
            nx, ny, nz: Number of mesh points in x, y, z directions
        """
        super().__init__(device_structure, ambient_temperature)
        
        # Mesh parameters
        self.nx, self.ny, self.nz = nx, ny, nz
        self.setup_mesh()
        self.setup_material_map()
        
    def setup_mesh(self):
        """Create computational mesh"""
        # Device dimensions
        Lx = self.device.device_dimensions['device_length'] * 1e-6  # m
        Ly = self.device.device_dimensions['gate_width'] * 1e-6    # m
        Lz = self.device.get_total_thickness() * 1e-9              # m
        
        # Create mesh
        self.x = np.linspace(0, Lx, self.nx)
        self.y = np.linspace(0, Ly, self.ny)
        self.z = np.linspace(0, Lz, self.nz)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        # Create 3D mesh grid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
    def setup_material_map(self):
        """Map materials to mesh points"""
        self.material_map = np.zeros((self.nx, self.ny, self.nz), dtype=int)
        self.thermal_conductivity = np.zeros((self.nx, self.ny, self.nz))
        self.specific_heat = np.zeros((self.nx, self.ny, self.nz))
        self.density = np.zeros((self.nx, self.ny, self.nz))
        
        # Map layers to z-indices
        z_boundaries = np.array(self.device.get_layer_boundaries()) * 1e-9  # m
        
        for k in range(self.nz):
            z_pos = self.z[k]
            
            # Find which layer this z-position belongs to
            layer_idx = 0
            for i, z_bound in enumerate(z_boundaries[1:]):
                if z_pos <= z_bound:
                    layer_idx = i
                    break
            
            # Get material properties for this layer
            layer = list(reversed(self.device.layers))[layer_idx]
            if layer.material in self.material_properties:
                mat_props = self.material_properties[layer.material]
                
                # Assign properties (assuming uniform temperature initially)
                for i in range(self.nx):
                    for j in range(self.ny):
                        self.thermal_conductivity[i, j, k] = mat_props.thermal_conductivity(self.T_ambient)
                        self.specific_heat[i, j, k] = mat_props.specific_heat(self.T_ambient)
                        self.density[i, j, k] = mat_props.density
    
    def build_system_matrix(self, dt: float = None) -> sparse.csr_matrix:
        """
        Build finite difference matrix for heat equation
        
        Args:
            dt: Time step for transient analysis (None for steady-state)
        
        Returns:
            System matrix A for AT = b
        """
        n_total = self.nx * self.ny * self.nz
        
        # Pre-allocate lists for sparse matrix construction
        row_indices = []
        col_indices = []
        values = []
        
        # Helper function to convert 3D indices to 1D
        def idx_3d_to_1d(i, j, k):
            return i + j * self.nx + k * self.nx * self.ny
        
        # Build matrix row by row
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    idx = idx_3d_to_1d(i, j, k)
                    
                    # Get thermal conductivity at interfaces
                    k_curr = self.thermal_conductivity[i, j, k]
                    
                    # Diagonal element
                    diag_val = 0.0
                    
                    # X-direction
                    if i > 0:
                        k_minus = self.thermal_conductivity[i-1, j, k]
                        k_interface = 2 * k_curr * k_minus / (k_curr + k_minus)
                        coeff = k_interface / self.dx**2
                        
                        row_indices.append(idx)
                        col_indices.append(idx_3d_to_1d(i-1, j, k))
                        values.append(coeff)
                        diag_val -= coeff
                    
                    if i < self.nx - 1:
                        k_plus = self.thermal_conductivity[i+1, j, k]
                        k_interface = 2 * k_curr * k_plus / (k_curr + k_plus)
                        coeff = k_interface / self.dx**2
                        
                        row_indices.append(idx)
                        col_indices.append(idx_3d_to_1d(i+1, j, k))
                        values.append(coeff)
                        diag_val -= coeff
                    
                    # Y-direction
                    if j > 0:
                        k_minus = self.thermal_conductivity[i, j-1, k]
                        k_interface = 2 * k_curr * k_minus / (k_curr + k_minus)
                        coeff = k_interface / self.dy**2
                        
                        row_indices.append(idx)
                        col_indices.append(idx_3d_to_1d(i, j-1, k))
                        values.append(coeff)
                        diag_val -= coeff
                    
                    if j < self.ny - 1:
                        k_plus = self.thermal_conductivity[i, j+1, k]
                        k_interface = 2 * k_curr * k_plus / (k_curr + k_plus)
                        coeff = k_interface / self.dy**2
                        
                        row_indices.append(idx)
                        col_indices.append(idx_3d_to_1d(i, j+1, k))
                        values.append(coeff)
                        diag_val -= coeff
                    
                    # Z-direction
                    if k > 0:
                        k_minus = self.thermal_conductivity[i, j, k-1]
                        k_interface = 2 * k_curr * k_minus / (k_curr + k_minus)
                        coeff = k_interface / self.dz**2
                        
                        row_indices.append(idx)
                        col_indices.append(idx_3d_to_1d(i, j, k-1))
                        values.append(coeff)
                        diag_val -= coeff
                    
                    if k < self.nz - 1:
                        k_plus = self.thermal_conductivity[i, j, k+1]
                        k_interface = 2 * k_curr * k_plus / (k_curr + k_plus)
                        coeff = k_interface / self.dz**2
                        
                        row_indices.append(idx)
                        col_indices.append(idx_3d_to_1d(i, j, k+1))
                        values.append(coeff)
                        diag_val -= coeff
                    
                    # Add time derivative term for transient analysis
                    if dt is not None:
                        rho_c = self.density[i, j, k] * self.specific_heat[i, j, k]
                        diag_val -= rho_c / dt
                    
                    # Add diagonal element
                    row_indices.append(idx)
                    col_indices.append(idx)
                    values.append(-diag_val)
        
        # Create sparse matrix
        A = sparse.csr_matrix((values, (row_indices, col_indices)), 
                             shape=(n_total, n_total))
        
        return A
    
    def apply_boundary_conditions(self, A: sparse.csr_matrix, b: np.ndarray,
                                 bc_type: str = "convection") -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Apply boundary conditions to the system
        
        Args:
            A: System matrix
            b: Right-hand side vector
            bc_type: 'isothermal', 'adiabatic', or 'convection'
        
        Returns:
            Modified A and b
        """
        n_total = self.nx * self.ny * self.nz
        
        def idx_3d_to_1d(i, j, k):
            return i + j * self.nx + k * self.nx * self.ny
        
        # Bottom boundary (substrate bottom) - isothermal
        for i in range(self.nx):
            for j in range(self.ny):
                idx = idx_3d_to_1d(i, j, self.nz-1)
                
                # Set row to identity
                A_data = A.data
                A_indices = A.indices
                A_indptr = A.indptr
                
                # Zero out the row
                for ptr in range(A_indptr[idx], A_indptr[idx+1]):
                    if A_indices[ptr] != idx:
                        A_data[ptr] = 0
                    else:
                        A_data[ptr] = 1
                
                b[idx] = self.T_ambient
        
        # Side boundaries - adiabatic or convection
        if bc_type == "convection":
            h_conv = 10  # W/(m²·K) - convection coefficient
            
            # X boundaries
            for j in range(self.ny):
                for k in range(self.nz-1):  # Exclude bottom
                    # x = 0
                    idx = idx_3d_to_1d(0, j, k)
                    k_thermal = self.thermal_conductivity[0, j, k]
                    A[idx, idx] += h_conv * self.dy * self.dz / (k_thermal * self.dx)
                    b[idx] += h_conv * self.dy * self.dz * self.T_ambient / (k_thermal * self.dx)
                    
                    # x = Lx
                    idx = idx_3d_to_1d(self.nx-1, j, k)
                    k_thermal = self.thermal_conductivity[self.nx-1, j, k]
                    A[idx, idx] += h_conv * self.dy * self.dz / (k_thermal * self.dx)
                    b[idx] += h_conv * self.dy * self.dz * self.T_ambient / (k_thermal * self.dx)
        
        return A, b
    
    def add_heat_source(self, b: np.ndarray, power_density: Callable) -> np.ndarray:
        """
        Add heat source term to right-hand side
        
        Args:
            b: Right-hand side vector
            power_density: Function returning power density at (x, y, z)
        
        Returns:
            Modified b vector
        """
        def idx_3d_to_1d(i, j, k):
            return i + j * self.nx + k * self.nx * self.ny
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    Q = power_density(x, y, z)
                    
                    if Q > 0:
                        idx = idx_3d_to_1d(i, j, k)
                        b[idx] -= Q  # Negative because moved to RHS
        
        return b
    
    def solve_steady_state(self, power_density: Callable) -> np.ndarray:
        """
        Solve steady-state heat equation
        
        Args:
            power_density: Function returning power density at (x, y, z)
        
        Returns:
            3D temperature field
        """
        # Build system matrix
        A = self.build_system_matrix()
        
        # Initialize right-hand side
        b = np.zeros(self.nx * self.ny * self.nz)
        
        # Add heat source
        b = self.add_heat_source(b, power_density)
        
        # Apply boundary conditions
        A, b = self.apply_boundary_conditions(A, b)
        
        # Solve system
        T_flat = spsolve(A, b)
        
        # Reshape to 3D
        T = T_flat.reshape((self.nx, self.ny, self.nz))
        
        # Update temperature-dependent properties
        self.update_material_properties(T)
        
        return T
    
    def update_material_properties(self, T: np.ndarray):
        """Update temperature-dependent material properties"""
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Find material at this point
                    z_pos = self.z[k]
                    z_boundaries = np.array(self.device.get_layer_boundaries()) * 1e-9
                    
                    layer_idx = 0
                    for idx, z_bound in enumerate(z_boundaries[1:]):
                        if z_pos <= z_bound:
                            layer_idx = idx
                            break
                    
                    layer = list(reversed(self.device.layers))[layer_idx]
                    if layer.material in self.material_properties:
                        mat_props = self.material_properties[layer.material]
                        temp = T[i, j, k]
                        
                        self.thermal_conductivity[i, j, k] = mat_props.thermal_conductivity(temp)
                        self.specific_heat[i, j, k] = mat_props.specific_heat(temp)


class ThermalResistanceNetwork(ThermalModel):
    """
    Simplified thermal resistance network model for fast computation
    """
    
    def __init__(self, device_structure, ambient_temperature: float = 300):
        super().__init__(device_structure, ambient_temperature)
        self.build_network()
    
    def build_network(self):
        """Build thermal resistance network"""
        self.thermal_resistances = {}
        self.thermal_capacitances = {}
        
        # Calculate thermal resistances for each layer
        area = (self.device.device_dimensions['device_length'] * 
                self.device.device_dimensions['gate_width'] * 1e-12)  # m²
        
        for layer in self.device.layers:
            if layer.material in self.material_properties:
                mat_props = self.material_properties[layer.material]
                k = mat_props.thermal_conductivity(self.T_ambient)
                thickness = layer.thickness * 1e-9  # m
                
                # Thermal resistance
                R_th = thickness / (k * area)
                self.thermal_resistances[layer.name] = R_th
                
                # Thermal capacitance
                volume = thickness * area
                C_th = mat_props.density * mat_props.specific_heat(self.T_ambient) * volume
                self.thermal_capacitances[layer.name] = C_th
        
        # Add interface thermal resistances
        self.interface_resistances = {}
        layers = list(self.device.layers)
        
        for i in range(len(layers) - 1):
            interface_name = f"{layers[i].name}-{layers[i+1].name}"
            R_tbr = thermal_boundary_resistance(self.T_ambient, 
                                              layers[i].material, 
                                              layers[i+1].material) / area
            self.interface_resistances[interface_name] = R_tbr
    
    def calculate_junction_temperature(self, power: float) -> float:
        """
        Calculate junction temperature for given power dissipation
        
        Args:
            power: Power dissipation in W
        
        Returns:
            Junction temperature in K
        """
        # Sum all thermal resistances
        R_total = sum(self.thermal_resistances.values())
        R_total += sum(self.interface_resistances.values())
        
        # Calculate temperature rise
        delta_T = power * R_total
        
        return self.T_ambient + delta_T
    
    def calculate_thermal_impedance(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate frequency-dependent thermal impedance
        
        Args:
            frequencies: Array of frequencies in Hz
        
        Returns:
            Complex thermal impedance
        """
        omega = 2 * np.pi * frequencies
        
        # Build RC ladder network
        Z_th = np.zeros_like(omega, dtype=complex)
        
        # Simple RC model for each layer
        for layer_name in self.thermal_resistances:
            R = self.thermal_resistances[layer_name]
            C = self.thermal_capacitances[layer_name]
            
            # Impedance of RC parallel combination
            Z_layer = R / (1 + 1j * omega * R * C)
            Z_th += Z_layer
        
        # Add interface resistances (purely resistive)
        Z_th += sum(self.interface_resistances.values())
        
        return Z_th


class ElectroThermalModel(HeatEquationSolver):
    """
    Coupled electro-thermal model for self-consistent simulation
    """
    
    def __init__(self, device_structure, electrical_model=None, **kwargs):
        super().__init__(device_structure, **kwargs)
        self.electrical_model = electrical_model
        
    def calculate_joule_heating(self, current_density: np.ndarray, 
                               electric_field: np.ndarray) -> np.ndarray:
        """
        Calculate Joule heating power density
        
        Args:
            current_density: 3D array of current density (A/m²)
            electric_field: 3D array of electric field (V/m)
        
        Returns:
            3D array of power density (W/m³)
        """
        return current_density * electric_field
    
    def self_consistent_solve(self, V_gs: float, V_ds: float, 
                            max_iterations: int = 10, tolerance: float = 1.0):
        """
        Self-consistent electro-thermal solution
        
        Args:
            V_gs: Gate-source voltage
            V_ds: Drain-source voltage
            max_iterations: Maximum number of iterations
            tolerance: Temperature convergence tolerance in K
        
        Returns:
            Converged temperature field
        """
        # Initial temperature guess
        T = np.ones((self.nx, self.ny, self.nz)) * self.T_ambient
        
        for iteration in range(max_iterations):
            T_old = T.copy()
            
            # Update material properties with temperature
            self.update_material_properties(T)
            
            # Solve electrical problem with current temperature
            if self.electrical_model:
                J, E = self.electrical_model.solve(V_gs, V_ds, T)
                
                # Calculate power density
                def power_density(x, y, z):
                    i = np.argmin(np.abs(self.x - x))
                    j = np.argmin(np.abs(self.y - y))
                    k = np.argmin(np.abs(self.z - z))
                    return self.calculate_joule_heating(J[i,j,k], E[i,j,k])
            else:
                # Simple power density model for testing
                def power_density(x, y, z):
                    # Power concentrated near 2DEG
                    z_2deg = self.device.get_2deg_position() * 1e-9
                    if abs(z - z_2deg) < 5e-9:  # Within 5 nm of 2DEG
                        # Power under gate
                        gate_start = 1.5e-6
                        gate_end = gate_start + self.device.device_dimensions['gate_length'] * 1e-6
                        if gate_start <= x <= gate_end:
                            return 1e10 * V_ds * V_gs  # W/m³
                    return 0.0
            
            # Solve thermal problem
            T = self.solve_steady_state(power_density)
            
            # Check convergence
            max_change = np.max(np.abs(T - T_old))
            print(f"Iteration {iteration + 1}: Max temperature change = {max_change:.2f} K")
            
            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return T


# Example usage functions
def create_example_power_distribution(device, V_ds: float = 10.0, I_ds: float = 0.1):
    """Create example power distribution for HEMT"""
    
    def power_density(x, y, z):
        """Power density function (W/m³)"""
        # Power is concentrated in 2DEG channel
        z_2deg = device.get_2deg_position() * 1e-9  # m
        
        # Gaussian distribution around 2DEG
        sigma_z = 2e-9  # 2 nm spread
        power_z = np.exp(-(z - z_2deg)**2 / (2 * sigma_z**2))
        
        # Power distribution along channel (highest under gate)
        gate_start = 1.5e-6
        gate_end = gate_start + device.device_dimensions['gate_length'] * 1e-6
        
        if gate_start <= x <= gate_end:
            # Under gate - highest power
            power_x = 1.0
        elif x < gate_start:
            # Source side
            power_x = 0.3
        else:
            # Drain side - moderate power
            power_x = 0.5 * np.exp(-(x - gate_end) / 1e-6)
        
        # Total power density
        P_total = V_ds * I_ds  # W
        channel_volume = (device.device_dimensions['device_length'] * 1e-6 * 
                         device.device_dimensions['gate_width'] * 1e-6 * 
                         10e-9)  # Approximate active volume
        
        P_density_max = P_total / channel_volume
        
        return P_density_max * power_x * power_z
    
    return power_density


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../../device_structures')
    from hemt_structure import HEMTStructure
    
    # Create device structure
    device = HEMTStructure(substrate_type="SiC")
    
    # Quick thermal resistance calculation
    print("Thermal Resistance Network Analysis")
    print("=" * 50)
    trn = ThermalResistanceNetwork(device)
    
    P_diss = 1.0  # W
    T_j = trn.calculate_junction_temperature(P_diss)
    print(f"Power dissipation: {P_diss} W")
    print(f"Junction temperature: {T_j:.1f} K ({T_j-273.15:.1f} °C)")
    print(f"Temperature rise: {T_j - trn.T_ambient:.1f} K")
    
    # Detailed heat equation solution
    print("\n\nHeat Equation Solution")
    print("=" * 50)
    solver = HeatEquationSolver(device, nx=50, ny=20, nz=50)
    
    # Create power distribution
    power_func = create_example_power_distribution(device, V_ds=20, I_ds=0.05)
    
    # Solve steady-state
    print("Solving steady-state heat equation...")
    T_field = solver.solve_steady_state(power_func)
    
    # Extract key temperatures
    T_max = np.max(T_field)
    T_junction = T_field[25, 10, np.argmin(np.abs(solver.z - device.get_2deg_position()*1e-9))]
    
    print(f"Maximum temperature: {T_max:.1f} K ({T_max-273.15:.1f} °C)")
    print(f"Junction temperature: {T_junction:.1f} K ({T_junction-273.15:.1f} °C)")
    print(f"Substrate temperature: {T_field[25, 10, -1]:.1f} K")