"""CLI interface for splat2mc."""

import click
from pathlib import Path
from .converter import convert_ply_to_datapack, load_ply


@click.group()
@click.version_option()
def main():
    """Convert 3D Gaussian Splats to Minecraft particle datapacks."""
    pass


@main.command()
@click.argument("ply_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for datapack",
)
@click.option(
    "-n", "--max-particles",
    type=int,
    default=5000,
    help="Maximum number of particles (default: 5000)",
)
@click.option(
    "-s", "--size",
    type=float,
    default=10.0,
    help="Target size in Minecraft blocks (default: 10)",
)
@click.option(
    "--min-opacity",
    type=float,
    default=0.1,
    help="Minimum opacity threshold (default: 0.1)",
)
@click.option(
    "-p", "--particle-scale",
    type=float,
    default=1.0,
    help="Multiplier for individual particle sizes (default: 1.0)",
)
@click.option(
    "-t", "--particle-type",
    type=click.Choice(["dust", "entity_effect", "dust_color_transition"]),
    default="dust",
    help="Minecraft particle type to use (default: dust)",
)
@click.option(
    "--flip-y",
    is_flag=True,
    help="Flip the Y-axis (up vector) to fix upside-down imports",
)
def convert(
    ply_file: Path, 
    output: Path, 
    max_particles: int, 
    size: float, 
    min_opacity: float, 
    particle_scale: float, 
    particle_type: str, 
    flip_y: bool
):
    """Convert a PLY file to a Minecraft datapack.
    
    Example:
        splat2mc convert scene.ply -o ./datapacks --flip-y -t entity_effect
    """
    output.mkdir(parents=True, exist_ok=True)
    datapack_path = convert_ply_to_datapack(
        ply_path=ply_file,
        output_dir=output,
        max_particles=max_particles,
        target_size=size,
        min_opacity=min_opacity,
        particle_scale_multiplier=particle_scale,
        particle_type=particle_type,
        flip_y=flip_y,
    )
    click.echo(f"\n✓ Datapack created: {datapack_path}")
    click.echo(f"  Copy to your world's datapacks/ folder")
    click.echo(f"  Then: /reload and /function splats:{ply_file.stem.lower()}")


@main.command()
@click.argument("ply_file", type=click.Path(exists=True, path_type=Path))
def info(ply_file: Path):
    """Show info about a PLY file.
    
    Example:
        splat2mc info scene.ply
    """
    splats = load_ply(ply_file)
    
    if not splats:
        click.echo("No splats found in file.")
        return
    
    # Calculate stats
    opacities = [s.opacity for s in splats]
    scales = [s.scale for s in splats]
    
    click.echo(f"File: {ply_file}")
    click.echo(f"Splats: {len(splats):,}")
    click.echo(f"Opacity: min={min(opacities):.3f}, max={max(opacities):.3f}, avg={sum(opacities)/len(opacities):.3f}")
    click.echo(f"Scale: min={min(scales):.6f}, max={max(scales):.6f}, avg={sum(scales)/len(scales):.6f}")
    
    # Bounds
    xs = [s.x for s in splats]
    ys = [s.y for s in splats]
    zs = [s.z for s in splats]
    click.echo(f"Bounds X: [{min(xs):.3f}, {max(xs):.3f}]")
    click.echo(f"Bounds Y: [{min(ys):.3f}, {max(ys):.3f}]")
    click.echo(f"Bounds Z: [{min(zs):.3f}, {max(zs):.3f}]")


@main.command()
@click.argument("splat_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for datapacks",
)
@click.option(
    "-n", "--max-particles",
    type=int,
    default=5000,
    help="Maximum number of particles (default: 5000)",
)
def batch(splat_dir: Path, output: Path, max_particles: int):
    """Convert all PLY files in a directory.
    
    Example:
        splat2mc batch ./splats -o ./datapacks
    """
    ply_files = list(splat_dir.glob("*.ply"))
    
    if not ply_files:
        click.echo(f"No PLY files found in {splat_dir}")
        return
    
    click.echo(f"Found {len(ply_files)} PLY files")
    output.mkdir(parents=True, exist_ok=True)
    
    for ply_file in ply_files:
        click.echo(f"\nProcessing {ply_file.name}...")
        try:
            convert_ply_to_datapack(
                ply_path=ply_file,
                output_dir=output,
                max_particles=max_particles,
            )
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
    
    click.echo(f"\n✓ Done! Datapacks in: {output}")


if __name__ == "__main__":
    main()
