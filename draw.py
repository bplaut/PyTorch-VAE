import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def create_side_by_side_image(params, original, reconstruction, total_loss, total_norm_loss):
    """
    Create a side-by-side comparison of original and reconstructed images with loss annotation.

    Args:
        original: Original image tensor
        reconstruction: Reconstructed image tensor
        total_loss: Loss value
        total_norm_loss: Normalized loss (0-1). If None, will be calculated using global min/max if available.

    Returns:
        PIL Image with side-by-side comparison and annotations
    """
    # Create side-by-side comparison
    comparison = torch.cat([original, reconstruction], dim=3)

    # Convert to PIL for annotation
    comparison_np = comparison.numpy()
    comparison_np = np.transpose(comparison_np[0], (1, 2, 0))
    comparison_np = (comparison_np - comparison_np.min()) / (comparison_np.max() - comparison_np.min()) * 255.0
    comparison_pil = Image.fromarray(comparison_np.astype(np.uint8))


    # Create annotated image if requested
    if params['dont_annotate_loss']:
        final_img = comparison_pil
    else:
        final_img = create_annotated_image(comparison_pil, total_loss, total_norm_loss)

    return final_img

def save_loss_histogram(params, test_data):
    """
    Generate and save a histogram of total reconstruction error across individual frames
    with vertical lines at 50th, 75th, 90th, and 95th percentiles, and the mean
    """

    # Create directory for histogram
    histogram_dir = os.path.join(params['test_output_dir'], "histograms")
    os.makedirs(histogram_dir, exist_ok=True)

    # Extract total losses from test data
    total_losses = [data['total_loss'] for data in test_data]

    # Calculate percentiles
    percentiles = [50, 75, 90, 95]
    percentile_values = np.percentile(total_losses, percentiles)
    mean = np.mean(total_losses)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(total_losses, bins=40, alpha=0.8, color='blue')
    plt.title('Histogram of loss across time steps')
    plt.xlabel('Loss (x1000)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Add vertical lines for percentiles
    colors = ['green', 'orange', 'purple', 'red']
    for i, (percentile, value) in enumerate(zip(percentiles, percentile_values)):
        plt.axvline(x=value, color=colors[i], linestyle='--', 
                   label=f'{percentile}th percentile: {value:.4f}')
    mean_color = 'black'
    plt.axvline(x=mean, color=mean_color, linestyle='-', label=f'Mean: {mean:.4f}')

    plt.legend()

    # Save the histogram
    last_sep = params['test_output_dir'].rfind('/')
    overall_output_dir = params['test_output_dir'][:last_sep]
    exp_name = params['test_output_dir'][last_sep+1:]
    histogram_path = os.path.join(overall_output_dir, f"{exp_name}_loss_histogram.png")
    plt.savefig(histogram_path, dpi=300)
    plt.close()

    # Calculate and print statistics
    print("\nTotal Loss Statistics:")
    print(f"  Mean: {np.mean(total_losses):.4f}")
    print(f"  Median: {np.median(total_losses):.4f}")
    print(f"  Std Dev: {np.std(total_losses):.4f}")
    print(f"  Min: {np.min(total_losses):.4f}")
    print(f"  Max: {np.max(total_losses):.4f}")
    print("\nPercentiles:")
    for percentile, value in zip(percentiles, percentile_values):
        print(f"  {percentile}th: {value:.4f}")

    print(f"Loss histogram saved to: {histogram_path}")

def create_annotated_image(comparison_img, loss, norm_loss):
    img_width, img_height = comparison_img.size
    header_height = 28  # Adjusted for one line with larger text

    new_img = Image.new('RGB', (img_width, img_height + header_height), color=(240, 240, 240))
    new_img.paste(comparison_img, (0, header_height))

    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except:
            font = ImageFont.load_default()

    color = get_color_from_score(norm_loss)

    text = f"Loss: {loss:.3f} ({int(norm_loss*100)}%)"

    left_margin = img_width // 6
    draw.text((left_margin, (header_height - 18) // 2), text, fill=color, font=font)

    # Meter bar on the right
    meter_start_x = img_width // 2 - 15
    meter_width = img_width // 3 + 15
    meter_height = 13
    meter_y = (header_height - meter_height) // 2

    # Background of meter
    draw.rectangle(
        [(meter_start_x, meter_y), (meter_start_x + meter_width, meter_y + meter_height)],
        fill=(220, 220, 220), outline=(180, 180, 180)
    )

    # Filled part of meter
    filled_width = int(meter_width * norm_loss)
    if filled_width > 0:
        draw.rectangle(
            [(meter_start_x, meter_y), (meter_start_x + filled_width, meter_y + meter_height)],
            fill=color
        )

    return new_img

def get_color_from_score(percentile):
    """
    Map a percentile (0-1) to a color: blue (0) -> purple (0.5) -> red (1)
    """
    if percentile < 0.5:
        # Blue to Purple (0 to 0.5)
        normalized = percentile * 2  # Scale 0-0.5 to 0-1
        r = int(128 * normalized)
        g = 0
        b = int(255 - 127 * normalized)
    else:
        # Purple to Red (0.5 to 1)
        normalized = (percentile - 0.5) * 2  # Scale 0.5-1 to 0-1
        r = int(128 + 127 * normalized)
        g = 0
        b = int(128 - 128 * normalized)

    return (r, g, b)

