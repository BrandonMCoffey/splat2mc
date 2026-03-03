"""Core converter: PLY → mcfunction."""

import numpy as np
from plyfile import PlyData
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GaussianSplat:
    """A single Gaussian splat with position, color, opacity, and scale."""
    x: float
    y: float
    z: float
    r: float  # 0-1
    g: float  # 0-1
    b: float  # 0-1
    opacity: float  # 0-1
    scale: float  # average scale


def load_ply(path: Path) -> list[GaussianSplat]:
    """Load Gaussian splats from a PLY file.
    
    3DGS PLY files typically contain:
    - x, y, z: position
    - f_dc_0, f_dc_1, f_dc_2: spherical harmonics DC component (color)
    - opacity: opacity (logit space, needs sigmoid)
    - scale_0, scale_1, scale_2: log scale
    - rot_0, rot_1, rot_2, rot_3: rotation quaternion
    """
    plydata = PlyData.read(str(path))
    vertex = plydata['vertex']
    
    # Extract positions
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])
    
    # Extract colors from spherical harmonics DC component
    # SH coefficients need conversion: color = 0.5 + SH_DC * C0
    # where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    
    if 'f_dc_0' in vertex.data.dtype.names:
        # 3DGS format with spherical harmonics
        r = 0.5 + np.array(vertex['f_dc_0']) * C0
        g = 0.5 + np.array(vertex['f_dc_1']) * C0
        b = 0.5 + np.array(vertex['f_dc_2']) * C0
    elif 'red' in vertex.data.dtype.names:
        # Standard PLY with RGB
        r = np.array(vertex['red']) / 255.0
        g = np.array(vertex['green']) / 255.0
        b = np.array(vertex['blue']) / 255.0
    else:
        # Fallback: white
        r = np.ones_like(x)
        g = np.ones_like(x)
        b = np.ones_like(x)
    
    # Clamp colors to valid range
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    # Extract opacity (stored as logit, apply sigmoid)
    if 'opacity' in vertex.data.dtype.names:
        opacity_logit = np.array(vertex['opacity'])
        opacity = 1 / (1 + np.exp(-opacity_logit))
    else:
        opacity = np.ones_like(x)
    
    # Extract scale (stored as log, take exp and average)
    if 'scale_0' in vertex.data.dtype.names:
        scale_0 = np.exp(np.array(vertex['scale_0']))
        scale_1 = np.exp(np.array(vertex['scale_1']))
        scale_2 = np.exp(np.array(vertex['scale_2']))
        scale = (scale_0 + scale_1 + scale_2) / 3
    else:
        scale = np.ones_like(x) * 0.01
    
    # Build splat list
    splats = []
    for i in range(len(x)):
        splats.append(GaussianSplat(
            x=float(x[i]),
            y=float(y[i]),
            z=float(z[i]),
            r=float(r[i]),
            g=float(g[i]),
            b=float(b[i]),
            opacity=float(opacity[i]),
            scale=float(scale[i]),
        ))
    
    return splats


def normalize_splats(
    splats: list[GaussianSplat],
    target_size: float = 10.0,
    center: bool = True,
    flip_y: bool = False,
) -> list[GaussianSplat]:
    """Normalize splat positions to fit in a reasonable Minecraft space."""
    if not splats:
        return splats
    
    # Flip the up vector if requested
    if flip_y:
        for s in splats:
            s.y = -s.y
            
    # Get bounds
    xs = [s.x for s in splats]
    ys = [s.y for s in splats]
    zs = [s.z for s in splats]
    
    # 3DGS often has "floaters" (stray particles far from the center). 
    # Using strict min/max includes these floaters and shrinks the main object.
    # We use the 2nd and 98th percentiles to calculate bounds based on the dense core.
    min_x, max_x = np.percentile(xs, 2), np.percentile(xs, 98)
    min_y, max_y = np.percentile(ys, 2), np.percentile(ys, 98)
    min_z, max_z = np.percentile(zs, 2), np.percentile(zs, 98)
    
    # Calculate scale factor based on the trimmed range
    range_x = max_x - min_x or 1
    range_y = max_y - min_y or 1
    range_z = max_z - min_z or 1
    max_range = max(range_x, range_y, range_z)
    scale_factor = target_size / max_range
    
    # Calculate center offset based on the dense core
    if center:
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
    else:
        center_x = center_y = center_z = 0
    
    # Normalize
    normalized = []
    for s in splats:
        normalized.append(GaussianSplat(
            x=(s.x - center_x) * scale_factor,
            y=(s.y - center_y) * scale_factor,
            z=(s.z - center_z) * scale_factor,
            r=s.r,
            g=s.g,
            b=s.b,
            opacity=s.opacity,
            scale=s.scale * scale_factor,
        ))
    
    return normalized


def downsample_splats(
    splats: list[GaussianSplat],
    max_count: int = 5000,
    method: str = "opacity",
) -> list[GaussianSplat]:
    """Downsample splats to fit Minecraft particle limits.
    
    Args:
        splats: List of splats
        max_count: Maximum number of splats to keep
        method: "opacity" (keep most opaque) or "random"
    """
    if len(splats) <= max_count:
        return splats
    
    if method == "opacity":
        # Sort by opacity descending, keep top N
        sorted_splats = sorted(splats, key=lambda s: s.opacity, reverse=True)
        return sorted_splats[:max_count]
    else:
        # Random sample
        indices = np.random.choice(len(splats), max_count, replace=False)
        return [splats[i] for i in indices]


def generate_mcfunction(
    splats: list[GaussianSplat],
    relative: bool = True,
    min_opacity: float = 0.1,
    use_absolute_pos: tuple[float, float, float] | None = None,
    particle_scale_multiplier: float = 1.0,
    particle_type: str = "dust",
) -> str:
    """Generate mcfunction content with particle commands."""
    lines = [
        "# Generated by splat2mc",
        f"# {len(splats)} Gaussian splats",
        "",
    ]
    
    for s in splats:
        if s.opacity < min_opacity:
            continue
        
        # Format position
        if use_absolute_pos:
            pos = f"{use_absolute_pos[0] + s.x:.3f} {use_absolute_pos[1] + s.y:.3f} {use_absolute_pos[2] + s.z:.3f}"
        elif relative:
            pos = f"~{s.x:.3f} ~{s.y:.3f} ~{s.z:.3f}"
        else:
            pos = f"{s.x:.3f} {s.y:.3f} {s.z:.3f}"
        
        # Clamp and format color
        r = max(0, min(1, s.r))
        g = max(0, min(1, s.g))
        b = max(0, min(1, s.b))
        
        # Scale particle size (clamp to Minecraft's 0.01-4 range)
        particle_scale = max(0.1, min(4.0, s.scale * 50 * particle_scale_multiplier))
        
        # Generate chosen particle command
        if particle_type == "entity_effect":
            # entity_effect uses RGB for dx dy dz, doesn't support size scaling
            line = f"particle entity_effect {pos} {r:.3f} {g:.3f} {b:.3f} 1 0 force"
        elif particle_type == "dust_color_transition":
            # Fades from splat color to black
            line = f"particle dust_color_transition {r:.3f} {g:.3f} {b:.3f} {particle_scale:.2f} 0 0 0 {pos} 0 0 0 0 1 force"
        else:
            # Default dust
            line = f"particle dust {r:.3f} {g:.3f} {b:.3f} {particle_scale:.2f} {pos} 0 0 0 0 1 force"
            
        lines.append(line)
    
    return "\n".join(lines)


def generate_datapack(
    name: str,
    mcfunction_content: str,
    output_dir: Path,
    loop: bool = True,
) -> Path:
    """Generate a complete Minecraft datapack."""
    # Sanitize name for filesystem/minecraft
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    
    # Create datapack structure
    datapack_dir = output_dir / f"splat_{safe_name}"
    functions_dir = datapack_dir / "data" / "splats" / "functions"
    functions_dir.mkdir(parents=True, exist_ok=True)
    
    # Write pack.mcmeta
    pack_mcmeta = datapack_dir / "pack.mcmeta"
    pack_mcmeta.write_text('''{
  "pack": {
    "pack_format": 26,
    "description": "3D Gaussian Splat: ''' + name + '''"
  }
}
''')
    
    # Write main particle function
    mcfunction_file = functions_dir / f"{safe_name}.mcfunction"
    mcfunction_file.write_text(mcfunction_content)
    
    # Write start function (enables the loop)
    start_content = f"""# Start displaying the splat
scoreboard objectives add splat_active dummy
scoreboard players set {safe_name} splat_active 1
tellraw @s {{"text":"Started splat: {safe_name}","color":"green"}}
tellraw @s {{"text":"Run /function splats:stop to stop","color":"gray"}}"""
    (functions_dir / "start.mcfunction").write_text(start_content)
    
    # Write stop function
    stop_content = f"""# Stop displaying all splats
scoreboard players set {safe_name} splat_active 0
tellraw @s {{"text":"Stopped splat display","color":"yellow"}}"""
    (functions_dir / "stop.mcfunction").write_text(stop_content)
    
    # Write tick function that checks if active (Now fires every 5 ticks to stop flickering)
    tick_content = f"""# Check if splat should be displayed (runs every tick)
execute if score {safe_name} splat_active matches 1 run scoreboard players add {safe_name}_timer splat_active 1
execute if score {safe_name}_timer splat_active matches 5.. run function splats:{safe_name}
execute if score {safe_name}_timer splat_active matches 5.. run scoreboard players set {safe_name}_timer splat_active 0"""
    (functions_dir / "tick.mcfunction").write_text(tick_content)
    
    # Write load function (Added the timer initialization)
    load_content = f"""# Initialize scoreboard on load
scoreboard objectives add splat_active dummy
scoreboard players set {safe_name} splat_active 0
scoreboard players set {safe_name}_timer splat_active 0
tellraw @a {{"text":"[splat2mc] Loaded {safe_name}. Run /function splats:start","color":"gold"}}"""
    (functions_dir / "load.mcfunction").write_text(load_content)
    
    # Write help function
    help_content = f"""tellraw @s {{"text":"=== splat2mc ===","color":"gold","bold":true}}
tellraw @s {{"text":"Available commands:","color":"white"}}
tellraw @s {{"text":"  /function splats:start","color":"green"}}
tellraw @s {{"text":"    Start displaying particles (loops)","color":"gray"}}
tellraw @s {{"text":"  /function splats:stop","color":"red"}}  
tellraw @s {{"text":"    Stop displaying particles","color":"gray"}}
tellraw @s {{"text":"  /function splats:{safe_name}","color":"yellow"}}
tellraw @s {{"text":"    Show particles once","color":"gray"}}"""
    (functions_dir / "help.mcfunction").write_text(help_content)
    
    # Create minecraft namespace for tick/load tags
    mc_tags_dir = datapack_dir / "data" / "minecraft" / "tags" / "functions"
    mc_tags_dir.mkdir(parents=True, exist_ok=True)
    
    # tick.json - runs every tick
    (mc_tags_dir / "tick.json").write_text('''{
  "values": ["splats:tick"]
}
''')
    
    # load.json - runs on /reload
    (mc_tags_dir / "load.json").write_text('''{
  "values": ["splats:load"]
}
''')
    
    return datapack_dir


def convert_ply_to_datapack(
    ply_path: Path,
    output_dir: Path,
    max_particles: int = 5000,
    target_size: float = 10.0,
    min_opacity: float = 0.1,
    particle_scale_multiplier: float = 1.0,
    particle_type: str = "dust",
    flip_y: bool = False,
) -> Path:
    """Full pipeline: PLY → datapack."""
    # Load
    print(f"Loading {ply_path}...")
    splats = load_ply(ply_path)
    print(f"  Loaded {len(splats)} splats")
    
    # Normalize
    print(f"Normalizing to {target_size} blocks...")
    splats = normalize_splats(splats, target_size=target_size, flip_y=flip_y)
    
    # Downsample
    if len(splats) > max_particles:
        print(f"Downsampling to {max_particles} particles...")
        splats = downsample_splats(splats, max_count=max_particles)
    
    # Generate mcfunction
    print("Generating mcfunction...")
    mcfunction = generate_mcfunction(
        splats, 
        min_opacity=min_opacity,
        particle_scale_multiplier=particle_scale_multiplier,
        particle_type=particle_type,
    )
    
    # Generate datapack
    name = ply_path.stem
    print(f"Creating datapack '{name}'...")
    datapack_path = generate_datapack(name, mcfunction, output_dir)
    
    print(f"Done! Datapack at: {datapack_path}")
    return datapack_path
