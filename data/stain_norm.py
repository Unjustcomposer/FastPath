import staintools

def get_macenko_normalizer(reference_image):
    """
    Fit a Macenko normalizer on a reference RGB image (numpy array).
    """
    normalizer = staintools.MacenkoNormalizer()
    normalizer.fit(reference_image)
    return normalizer

def normalize_tile(normalizer, tile):
    """
    Apply normalisation to an RGB tile.
    """
    return normalizer.transform(tile)
