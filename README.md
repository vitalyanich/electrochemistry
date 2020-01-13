# Electrochemistry

The project aims to develop a code for processing DFT calculations to
 calculate the rate constant of the heterogeneous electron transfer from the 
 cathode surface to the redox particle in the electrolyte. The project is based 
 on the theories of Marcus, Landau-Zener, the concept of quantum capacity. 
 The part of the project uses Gerisher's approximations.

To date, the developed code allows to process the output DFT data from VASP and calculate:
1) 2D and 3D STM (scanning tunneling microscopy) images in various approximations of the acceptor orbital
(Tersoff-Hamann, Chen, numerical molecular orbitals calculated using cluster DFT: e.g. O2, IrCl6, Ru(NH3)6)
2) 2D ECSTM (electrochemical scanning tunneling microscopy) images taking into account various parameters of the system
3) DOS graphs

# Acknowledgments

1) A part of this project is supported by the by grant 18-03-00773A of Russian Foundation for Basic Research
2) We acknowledge QijingZheng (github.com/QijingZheng) for the VaspBandUnfolding project, which is very useful for processing WAVECAR
