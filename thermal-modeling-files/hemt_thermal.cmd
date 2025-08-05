# Sentaurus Device Command File
# Thermal Simulation of InGaN/GaN HEMT
# Including self-heating effects and temperature-dependent properties

Device HEMT {

# Grid specification
File {
    Grid = "hemt_structure.grd"
    Plot = "hemt_thermal"
    Current = "hemt_thermal"
    Output = "hemt_thermal"
}

# Electrode definitions
Electrode {
    { Name="Source" Voltage=0.0 }
    { Name="Drain" Voltage=0.0 }
    { Name="Gate" Voltage=0.0 WorkFunction=5.1 }
    { Name="Substrate" Voltage=0.0 Temperature=300 }
}

# Physics models
Physics {
    # Temperature-dependent models
    Temperature(Dependence)
    
    # Carrier transport
    Mobility(
        DopingDep
        HighFieldSaturation
        Enormal
        Temperature
    )
    
    # Recombination models
    Recombination(
        SRH(DopingDep TempDependence)
        Auger
        Radiative
    )
    
    # Piezoelectric polarization
    Piezo(
        Model=Strain
        PolarizationCharge
    )
    
    # Thermal models
    Thermodynamic
    HeatCapacity
    ThermalConductivity(
        Model=PowerLaw
        Aniso
    )
    
    # Self-heating
    SelfHeating(
        CoupledThermodynamic
        MaxTemperatureChange=1
    )
    
    # Interface models
    Thermionic
}

# Physics interface specifications
Physics(MaterialInterface="GaN/InGaN") {
    Piezo(PolarizationCharge)
    Thermionic
    ThermalResistance(Value=3e-9)  # m²K/W
}

Physics(MaterialInterface="GaN/AlN") {
    Piezo(PolarizationCharge)
    Thermionic
    ThermalResistance(Value=1e-9)
}

Physics(MaterialInterface="GaN/SiC") {
    ThermalResistance(Value=2e-9)
}

# Numerical parameters
Math {
    # Solver controls
    Method=Blocked
    SubMethod=ILS
    ILSrc=1.2
    
    # Newton solver
    Notdamped=50
    Iterations=20
    RelErrControl
    ErrRef(Electron)=1e8
    ErrRef(Hole)=1e8
    
    # Temperature solver
    TempLinearSolver=ILS
    TemperatureDamping=0.5
    
    # Coupled iterations
    CoupledIterations=10
    CoupledTolerance=1e-3
    
    # Breakdown detection
    BreakCriteria {
        Current(Contact="Drain" Absval=1e-3)
    }
    
    # Extrapolation
    Extrapolate
    Derivatives
    RelativeStepControl
    
    # Parallel computation
    Number_of_Threads=4
}

# Plot definitions
Plot {
    # Solution variables
    eDensity hDensity
    eCurrent hCurrent
    Current TotalCurrent
    
    # Fields
    ElectricField
    Potential
    BandGap
    
    # Temperature
    Temperature
    ThermalConductivity
    HeatGeneration
    HeatFlux
    
    # Material properties
    Doping DonorConcentration AcceptorConcentration
    
    # Band structure
    ConductionBandEnergy ValenceBandEnergy
    eQuasiFermiEnergy hQuasiFermiEnergy
    
    # Mobility
    eMobility hMobility
    
    # Recombination
    SRHRecombination AugerRecombination
    TotalRecombination
}

# Current plot
CurrentPlot {
    ModelParameter="Drain Voltage"
    ModelParameter="Gate Voltage"
    ModelParameter="Max Temperature"
    ModelParameter="Junction Temperature"
}

}

# Solve section
Solve {

# Initial solution
Coupled { Poisson }
Coupled { Poisson Electron Hole }

# Gate voltage sweep at low Vds
Quasistationary(
    InitialStep=0.1 Increment=1.2
    MinStep=1e-5 MaxStep=0.5
    Goal { Name="Gate" Voltage=-5 }
) { Coupled { Poisson Electron Hole } }

Quasistationary(
    InitialStep=0.1 Increment=1.2
    MinStep=1e-5 MaxStep=0.5
    Goal { Name="Drain" Voltage=0.1 }
) { Coupled { Poisson Electron Hole } }

# Transfer characteristics
NewCurrentFile="IdVg_Vds0.1V"
Quasistationary(
    InitialStep=0.1 Increment=1.2
    MinStep=1e-5 MaxStep=0.2
    Goal { Name="Gate" Voltage=2 }
) { 
    Coupled { Poisson Electron Hole Temperature }
    CurrentPlot(Time=(Range=(0 1) Intervals=20))
}

# Output characteristics with self-heating
Load(FilePrefix="hemt_thermal_Vg0V")
NewCurrentFile="IdVd_Vg0V_thermal"

Quasistationary(
    InitialStep=0.1 Increment=1.2
    MinStep=1e-5 MaxStep=1.0
    Goal { Name="Drain" Voltage=20 }
) { 
    Coupled { Poisson Electron Hole Temperature }
    CurrentPlot(Time=(Range=(0 1) Intervals=40))
}

# High power simulation
Load(FilePrefix="hemt_thermal_Vg0V")
Quasistationary(
    InitialStep=0.01 Increment=1.1
    MinStep=1e-6 MaxStep=0.5
    Goal { Name="Drain" Voltage=40 }
) { 
    Coupled { 
        Poisson Electron Hole Temperature
        Iterations=15
    }
    CurrentPlot(Time=(Range=(0 1) Intervals=80))
}

# Transient thermal analysis
Load(FilePrefix="hemt_thermal_Vds20V")
NewCurrentFile="thermal_transient"

Transient(
    InitialTime=0 FinalTime=1e-3
    InitialStep=1e-9 MaxStep=1e-5
    Increment=1.2
) {
    Coupled { Poisson Electron Hole Temperature }
    CurrentPlot(
        Time=(
            Range=(1e-9 1e-8) Intervals=9;
            Range=(1e-8 1e-7) Intervals=9;
            Range=(1e-7 1e-6) Intervals=9;
            Range=(1e-6 1e-5) Intervals=9;
            Range=(1e-5 1e-4) Intervals=9;
            Range=(1e-4 1e-3) Intervals=9;
        )
    )
}

}

# End of command file