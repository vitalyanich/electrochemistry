# Electrochemistry

The project aims to develop a code for processing DFT calculations to
calculate the rate constant of the heterogeneous electron transfer from the 
cathode surface to the redox particle in the electrolyte. The project is based 
on the theories of Marcus, Landau-Zener, the concept of quantum capacity. 
Part of the project uses Gerischer's approximations.

To date, the developed code allows to process the output DFT data from VASP and calculate:
1) 2D and 3D STM (scanning tunneling microscopy) images in various approximations of the acceptor orbital
(Tersoff-Hamann, Chen, numerical molecular orbitals calculated using cluster DFT: e.g. O2, IrCl6, Ru(NH3)6)
2) 2D ECSTM (electrochemical scanning tunneling microscopy) images taking into account various parameters of the system
3) DOS graphs

# Citing

If you use stm/GerischerMarkus.py file to calculate the alignment of energy levels between an electrolyte and
a redox couple, please cite: Kislenko V.A., Pavlov. S.V., Kislenko. S.A. 
“Influence of defects in graphene on electron transfer kinetics: The role of the surface electronic structure”, 
Electrochimica Acta, 341 (2020), 136011. DOI: 10.1016/j.electacta.2020.136011

If you use this project to generate stm images or HET rate constant with spatial resolution. please cite in addition:
Pavlov S.V., Kislenko V.A., Kislenko S.A. “Fast Method for Calculating Spatially Resolved Heterogeneous Electron
 Transfer Kinetics and Application for Graphene With Defects”, 
 J. Phys. Chem. C 124(33) (2020), 18147-18155. DOI: 10.1021/acs.jpcc.0c05376

# Acknowledgments

1) A part of this project is supported by the by grant 18-03-00773A of Russian Foundation for Basic Research
2) We acknowledge QijingZheng (github.com/QijingZheng) for the VaspBandUnfolding project, which is very useful
for processing WAVECAR