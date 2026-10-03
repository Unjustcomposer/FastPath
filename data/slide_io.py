import os
import openslide
import numpy as np

class SlideReader:
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(slide_path)
        
    def get_level_for_mag(self, target_mag=20.0):
        # Infer best level. Typically level 0 is 40x in Camelyon.
        base_mag = float(self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40.0))
        downsample_needed = base_mag / target_mag
        level = self.slide.get_best_level_for_downsample(downsample_needed)
        return level

    def read_region(self, location, level, size):
        """Read RGB image from slide"""
        img = self.slide.read_region(location, level, size)
        return np.array(img.convert('RGB'))
    
    def get_thumbnail(self, level=None):
        if level is None:
            level = self.slide.level_count - 1
        return np.array(self.slide.read_region((0, 0), level, self.slide.level_dimensions[level]).convert('RGB'))
    
    def close(self):
        self.slide.close()
