''' Save the visible gamut of CIELAB D65 2° in a vtk file. '''

import colorio

illuminant = colorio.illuminants.d65()
observer = colorio.observers.cie_1931_2()

colorspace = colorio.CIELAB()
colorspace.save_visible_gamut(observer, illuminant, "cielab_d65_2_visible.vtk")
