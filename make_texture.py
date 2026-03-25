import numpy as np
from PIL import Image

def generate_splat_texture(size=64, filename="splat_tex.png"):
    center = size / 2.0
    
    # Create a coordinate grid
    y, x = np.ogrid[:size, :size]
    
    # Calculate distance from the center for every pixel
    # We add 0.5 to target the exact pixel centers
    dist_from_center = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
    
    # Calculate Gaussian alpha (opacity)
    # sigma controls the "softness" of the edge. size/4 is a standard 3DGS ratio.
    sigma = size / 4.0
    alpha = np.exp(-0.5 * (dist_from_center / sigma)**2)
    
    # Scale alpha to 0-255 and convert to uint8
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    
    # Create pure white RGB channels (255, 255, 255)
    r = np.full((size, size), 255, dtype=np.uint8)
    g = np.full((size, size), 255, dtype=np.uint8)
    b = np.full((size, size), 255, dtype=np.uint8)
    
    # Stack them together into a 4-channel RGBA array
    rgba = np.dstack((r, g, b, alpha))
    
    # Save the image
    img = Image.fromarray(rgba, 'RGBA')
    img.save(filename)
    print(f"✓ Generated perfect {filename} ({size}x{size})")

if __name__ == "__main__":
    generate_splat_texture()