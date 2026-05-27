import finesse

base_kat = finesse.Model()
base_kat.modes(maxtem=4)
base_kat.parse(f"""
# Add a Laser named L0 with a power of 1 W.
l L0 P=1

s s1 portA=L0.p1 portB=eom1.p1 L=10

modulator eom1 9M 0.1 order=1

s s2 portA=eom1.p2 portB=ITM.p1 L=10

# Input mirror of cavity.
m ITM L=0 T=0.014 Rc=-1934

# Intra-cavity space with length of 4 km
s CAV ITM.p2 ETM.p1 L=3994.47

# End mirror of cavity.
m ETM L=0 T=5u  Rc=2245

cavity cavArm source=ITM.p2.o

# Power detectors on reflection, circulation and transmission.
pd circ ETM.p1.i

pd1 pdhI node=ITM.p1.o f=eom1.f phase=0 # In phase demodulated signal
pd1 pdhQ node=ITM.p1.o f=eom1.f phase=90 # Quadrature phase demodulated signal

# dof ETMz ETM.dofs.z
# readout_rf pdh_readout ITM.p1.o f=eom1.f output_detectors=true phase=0

# Add a lock
lock lock_length pdhI ETM.phi -1.0673950644453318 1e-12
""")